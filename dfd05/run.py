from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import (
    RunConfig,
    effective_session_gate,
    load_config,
    validate_pine16_strategy_parity,
)
from .data import load_bars_for_symbol, timeframe_to_minutes
from .dashboard import build_horizon_dashboard, merge_events_forward
from .forward import (
    compute_forward_for_events,
    horizon_columns,
    validate_forward_outcomes,
)
from .indicators import rolling_sum_via_sma
from .strategy import EVENT_COLUMNS, EVENT_INT8_COLUMNS, extract_dfd05_events
from .pine16_truth import TruthLabel


KEY_COLS = ["symbol", "timeframe", "signal_id", "event_time_ms"]
PINE16_PRESET_PATH = Path("configs/dfd05_pine16_baseline.yaml")


def _horizon_suffix(hours: float | int) -> str:
    if float(hours).is_integer():
        return f"{int(hours)}h"
    return f"{str(hours).replace('.', 'p')}h"


def _build_cvd_proxy(
    config: RunConfig,
    symbol: str,
    timeframe: str,
    bars: pd.DataFrame,
) -> np.ndarray:
    toggles = config.strategy.toggles
    src = bars
    if timeframe.lower() != "m1":
        try:
            src = load_bars_for_symbol(config, symbol=symbol, timeframe="m1")
            src = src[(src["time"] >= bars["time"].iloc[0]) & (src["time"] <= bars["time"].iloc[-1])]
            if src.empty:
                src = bars
        except FileNotFoundError:
            src = bars

    signed_vol = np.sign(src["close"].to_numpy(dtype=float) - src["open"].to_numpy(dtype=float)) * src[
        "volume"
    ].to_numpy(dtype=float)
    cvd = rolling_sum_via_sma(signed_vol, max(1, toggles.cvd_len))
    s = pd.Series(cvd, index=src["time"]).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    aligned = s.reindex(bars["time"], method="ffill").to_numpy(dtype=float)
    return aligned


def _validate_unique_keys(df: pd.DataFrame, name: str) -> None:
    if df.empty:
        return
    dup = df.duplicated(KEY_COLS, keep=False)
    if dup.any():
        raise ValueError(f"{name} contains duplicate keys for {KEY_COLS}")


def _parse_horizons_csv(raw: str) -> List[int]:
    out: List[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    if not out:
        raise ValueError("No valid horizons parsed from --horizons")
    return sorted(set(out))


def _resolve_config_path(config_arg: Optional[str], preset_arg: Optional[str]) -> Path:
    if config_arg:
        return Path(config_arg)
    if (preset_arg or "").strip().lower() == "pine16":
        return PINE16_PRESET_PATH
    raise SystemExit("Provide --config PATH or --config-preset pine16.")


def _validate_parity_or_exit(config: RunConfig, parity_mode: Optional[str]) -> None:
    mode = (parity_mode or "").strip().lower()
    if mode == "":
        return
    if mode != "pine16":
        raise SystemExit(f"Unsupported parity mode: {parity_mode}")

    diffs = validate_pine16_strategy_parity(config)
    if not diffs:
        print("[parity] pine16 OK")
        return

    lines = ["[parity] pine16 mismatch:"]
    for path, actual, expected in diffs:
        lines.append(
            f"  - {path}: actual={json.dumps(actual, sort_keys=True)} expected={json.dumps(expected, sort_keys=True)}"
        )
    print("\n".join(lines), file=sys.stderr)
    raise SystemExit(2)


def _print_columns_and_assert(events_df: pd.DataFrame) -> None:
    print("[columns] event dtypes:")
    for c in events_df.columns:
        print(f"[columns] {c}: {events_df[c].dtype}")

    required_numeric = [
        "atr_entry",
        "osc_change_pct",
        "bars_gap",
        "loc_pivot",
        "vol_ratio_pivot",
        "rsi_pivot",
        "macd_pivot",
        "session_ok_pivot",
        "vol_ratio_entry",
        "atr_ratio_entry",
        "daily_adx",
        "daily_plus_di",
        "daily_minus_di",
        "daily_ema_ok",
        "daily_slope_ok",
        "daily_di_ok",
        "cvd_proxy_entry",
        "vol_behavior_ok_entry",
        "vol_spike_ok_entry",
        "session_ok_entry",
    ]
    missing = [c for c in required_numeric if c not in events_df.columns]
    if missing:
        raise ValueError(f"Missing required event feature columns: {missing}")
    if "cvd_z_entry" not in events_df.columns and "cvd_pct_entry" not in events_df.columns:
        raise ValueError("Expected at least one normalized CVD feature: cvd_z_entry or cvd_pct_entry")

    if len(events_df) == 0:
        print("[columns] no event rows; existence checks passed")
        return

    non_numeric = [
        c for c in required_numeric if not pd.api.types.is_numeric_dtype(events_df[c])
    ]
    if non_numeric:
        raise ValueError(f"Required event columns are not numeric: {non_numeric}")

    bool_cols = [
        c for c in required_numeric if c in EVENT_INT8_COLUMNS
    ]
    bad_bool = [c for c in bool_cols if str(events_df[c].dtype) != "int8"]
    if bad_bool:
        raise ValueError(f"Expected int8 dtype for boolean feature columns: {bad_bool}")


def run(
    config: RunConfig,
    horizons_override: Optional[List[int]] = None,
    print_columns: bool = False,
) -> dict[str, Path]:
    print(
        "[legacy-warning] dfd05.run is a legacy research path. "
        "Outputs are not exact Pine-exported truth."
    )
    timeframe = config.timeframe.lower()
    tf_minutes = timeframe_to_minutes(timeframe)
    horizons = (
        sorted(set(int(h) for h in horizons_override))
        if horizons_override is not None
        else config.forward.normalized_horizons_hours()
    )
    tie_break = config.forward.normalized_tie_break()
    emit_resolved = bool(config.forward.emit_resolved)
    forward_mode = config.forward.normalized_mode()
    ret_thresholds_bps = config.forward.normalized_ret_thresholds_bps()
    percentile_targets = config.forward.normalized_percentile_targets()
    mfe_thresholds_bps = config.forward.normalized_mfe_thresholds_bps()
    mae_thresholds_bps = config.forward.normalized_mae_thresholds_bps()
    mfe_percentile_targets = config.forward.normalized_mfe_percentile_targets()
    mfe_atr_thresholds = config.forward.normalized_mfe_atr_thresholds()
    mae_atr_thresholds = config.forward.normalized_mae_atr_thresholds()
    mfeatr_percentile_targets = config.forward.normalized_mfeatr_percentile_targets()
    quality_mfe_threshold = config.forward.normalized_quality_mfe_threshold()
    quality_mae_threshold = config.forward.normalized_quality_mae_threshold()
    truncate_policy = str(config.forward.truncate_policy)

    run_id = config.output.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(config.output.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("data/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    session_spec = effective_session_gate(config.strategy)
    print(f"[session_gate] {json.dumps(session_spec, sort_keys=True)}")
    print(
        "[forward_cfg] "
        f"mode={forward_mode} "
        f"horizons={horizons} "
        f"ret_thresholds_bps={ret_thresholds_bps} "
        f"percentile_targets={percentile_targets} "
        f"mfe_thresholds_bps={mfe_thresholds_bps} "
        f"mae_thresholds_bps={mae_thresholds_bps} "
        f"mfe_percentile_targets={mfe_percentile_targets} "
        f"mfe_atr_thresholds={mfe_atr_thresholds} "
        f"mae_atr_thresholds={mae_atr_thresholds} "
        f"mfeatr_percentile_targets={mfeatr_percentile_targets} "
        f"quality_mfe_threshold={quality_mfe_threshold} "
        f"quality_mae_threshold={quality_mae_threshold} "
        f"truncate_policy={truncate_policy} "
        f"emit_barrier={bool(config.forward.emit_barrier)} "
        f"tie_break={tie_break} emit_resolved={emit_resolved}"
    )

    all_events: List[pd.DataFrame] = []
    all_forward: List[pd.DataFrame] = []

    for symbol in config.symbols:
        t0 = perf_counter()
        bars = load_bars_for_symbol(config, symbol=symbol, timeframe=timeframe)
        if bars.empty:
            print(f"[run] symbol={symbol} bars=0 events=0 seconds={perf_counter() - t0:.2f}")
            continue
        cvd_proxy = _build_cvd_proxy(config, symbol=symbol, timeframe=timeframe, bars=bars)
        events = extract_dfd05_events(
            bars=bars, symbol=symbol, timeframe=timeframe, config=config, cvd_proxy=cvd_proxy
        )
        if events.empty:
            print(
                f"[run] symbol={symbol} bars={len(bars)} events=0 "
                f"seconds={perf_counter() - t0:.2f}"
            )
            continue
        forward = compute_forward_for_events(
            bars_df=bars,
            events_df=events,
            horizons_hours=horizons,
            tf_minutes=tf_minutes,
            tie_break=tie_break,
            emit_resolved=emit_resolved,
            sl_atr_mult=config.risk.sl_atr_mult,
            rr_mult=config.risk.rr_mult,
            warn_cb=print,
            mode=forward_mode,
            ret_thresholds_bps=ret_thresholds_bps,
            percentile_targets=percentile_targets,
            mfe_thresholds_bps=mfe_thresholds_bps,
            mae_thresholds_bps=mae_thresholds_bps,
            mfe_percentile_targets=mfe_percentile_targets,
            mfe_atr_thresholds=mfe_atr_thresholds,
            mae_atr_thresholds=mae_atr_thresholds,
            mfeatr_percentile_targets=mfeatr_percentile_targets,
            quality_mfe_threshold=quality_mfe_threshold,
            quality_mae_threshold=quality_mae_threshold,
            truncate_policy=truncate_policy,
            emit_barrier=bool(config.forward.emit_barrier),
        )
        print(
            f"[run] symbol={symbol} bars={len(bars)} events={len(events)} "
            f"seconds={perf_counter() - t0:.2f}"
        )
        all_events.append(events)
        all_forward.append(forward)

    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    forward_df = pd.concat(all_forward, ignore_index=True) if all_forward else pd.DataFrame()
    _validate_unique_keys(events_df, "events_df")
    _validate_unique_keys(forward_df, "forward_df")

    events_out = out_dir / f"events_dfd05_{timeframe}_{run_id}.parquet"
    forward_out = out_dir / f"forward_dfd05_{timeframe}_{run_id}.parquet"
    labeled_out = out_dir / f"labeled_dfd05_{timeframe}_{run_id}.parquet"

    event_drop = ["event_index"]
    event_cols = [c for c in EVENT_COLUMNS if c not in event_drop]
    if events_df.empty:
        events_write = pd.DataFrame(columns=event_cols)
    else:
        events_write = events_df.drop(columns=[c for c in event_drop if c in events_df.columns])
    events_write["truth_label"] = TruthLabel.UNVERIFIED_PYTHON_APPROXIMATION.value

    fwd_cols = [
        "symbol",
        "timeframe",
        "signal_id",
        "event_time_ms",
        "entry_time_ms",
        "entry_price",
        "pivot_time_ms",
        "pivot_price",
        "mode",
        "trade_mode",
        "toggles_json",
    ]
    if forward_mode == "barrier" or bool(config.forward.emit_barrier):
        fwd_cols.extend(["sl_price", "tp_price"])
    for h in horizons:
        fwd_cols.extend(
            horizon_columns(
                hours=h,
                emit_resolved=emit_resolved,
                mode=forward_mode,
                ret_thresholds_bps=ret_thresholds_bps,
                percentile_targets=percentile_targets,
                mfe_thresholds_bps=mfe_thresholds_bps,
                mae_thresholds_bps=mae_thresholds_bps,
                mfe_percentile_targets=mfe_percentile_targets,
                mfe_atr_thresholds=mfe_atr_thresholds,
                mae_atr_thresholds=mae_atr_thresholds,
                mfeatr_percentile_targets=mfeatr_percentile_targets,
                quality_mfe_threshold=quality_mfe_threshold,
                quality_mae_threshold=quality_mae_threshold,
            )
        )
        if forward_mode == "time_only" and bool(config.forward.emit_barrier):
            fwd_cols.extend(
                horizon_columns(
                    hours=h,
                    emit_resolved=emit_resolved,
                    mode="barrier",
                )
            )
    if forward_df.empty:
        forward_write = pd.DataFrame(columns=fwd_cols)
    else:
        forward_write = forward_df
    forward_write["truth_label"] = TruthLabel.UNVERIFIED_PYTHON_APPROXIMATION.value

    if print_columns:
        _print_columns_and_assert(events_write)

    events_write.to_parquet(events_out, index=False)
    forward_write.to_parquet(forward_out, index=False)

    outputs = {"events": events_out, "forward": forward_out}

    # Required validation gate before labeled output.
    summaries = validate_forward_outcomes(
        forward_df=forward_write,
        horizons_hours=horizons,
        rr_mult=config.risk.rr_mult,
        emit_resolved=emit_resolved,
        mode=forward_mode,
    )
    for s in summaries:
        if forward_mode == "time_only":
            print(
                "[forward] "
                f"h={int(s['horizon_h'])}h "
                f"n_total={int(s['n_events'])} "
                f"n_valid={int(s['n_valid'])} "
                f"mean_ret={s['mean_ret']:.6f} "
                f"median_ret={s['median_ret']:.6f} "
                f"up_rate={s['up_rate']:.6f} "
                f"mean_mfe={s['mean_mfe']:.6f} "
                f"mean_mae={s['mean_mae']:.6f} "
                f"p25={s['p25_ret']:.6f} "
                f"p50={s['p50_ret']:.6f} "
                f"p75={s['p75_ret']:.6f} "
                f"trunc={s['trunc_rate']:.6f}"
            )
        else:
            print(
                "[forward] "
                f"h={int(s['horizon_h'])}h "
                f"n={int(s['n_events'])} "
                f"tp={s['tp_rate']:.6f} "
                f"sl={s['sl_rate']:.6f} "
                f"same={s['samebar_rate']:.6f} "
                f"no_hit={s['no_hit_rate']:.6f} "
                f"tp_res={s['tp_resolved_rate']:.6f} "
                f"sl_res={s['sl_resolved_rate']:.6f} "
                f"hit={s['hit_rate']:.6f} "
                f"trunc={s['trunc_rate']:.6f} "
                f"expR_total={s['expR_total']:.6f} "
                f"expR_hit={s['expR_hit_only']:.6f}"
            )

    labeled: Optional[pd.DataFrame] = None
    if config.labels.write_labeled and not events_write.empty and not forward_write.empty:
        labeled = events_write.merge(
            forward_write,
            on=[
                "symbol",
                "timeframe",
                "signal_id",
                "event_time_ms",
                "entry_time_ms",
                "entry_price",
                "pivot_time_ms",
                "pivot_price",
                "mode",
                "trade_mode",
                "toggles_json",
            ],
            how="left",
        )
        lh = config.labels.label_horizon_hours
        if lh is not None:
            suffix = _horizon_suffix(lh)
            if forward_mode == "time_only":
                src_col = config.labels.label_column
                if src_col not in labeled.columns:
                    fallback = [
                        f"worked_{suffix}_ge_0bps",
                        f"up_{suffix}",
                    ]
                    src_col = next((c for c in fallback if c in labeled.columns), "")
            else:
                src_col = f"tp_first_resolved_{suffix}"
                if src_col not in labeled.columns:
                    src_col = f"tp_first_{suffix}"
            if src_col in labeled.columns:
                labeled[config.labels.label_column] = labeled[src_col].fillna(0).astype(np.int8)
        labeled.to_parquet(labeled_out, index=False)
        outputs["labeled"] = labeled_out

    if labeled is not None:
        dashboard_src = labeled
    else:
        dashboard_src = merge_events_forward(events_df=events_write, forward_df=forward_write)
    dashboard = build_horizon_dashboard(
        frame=dashboard_src,
        horizons_hours=horizons,
        rr_mult=config.risk.rr_mult,
        mode=forward_mode,
    )
    if forward_mode == "time_only":
        dashboard_out = reports_dir / f"dfd05_dashboard_timeonly_{timeframe}_{run_id}.csv"
    else:
        dashboard_out = reports_dir / f"dfd05_dashboard_{timeframe}_{run_id}.csv"
    dashboard.to_csv(dashboard_out, index=False)
    outputs["dashboard"] = dashboard_out

    if dashboard.empty:
        return outputs

    overall = dashboard[dashboard["section"] == "overall"].sort_values("horizon_h")
    for _, row in overall.iterrows():
        if forward_mode == "time_only":
            print(
                "[dashboard] "
                f"h={int(row['horizon_h'])}h "
                f"n_total={int(row['n_total'])} "
                f"n_valid={int(row['n_valid'])} "
                f"mean_ret={row['mean_ret']:.6f} "
                f"median_ret={row['median_ret']:.6f} "
                f"up_rate={row['up_rate']:.6f} "
                f"mean_mfe={row['mean_mfe']:.6f} "
                f"mean_mae={row['mean_mae']:.6f} "
                f"p25={row['p25_ret']:.6f} "
                f"p50={row['p50_ret']:.6f} "
                f"p75={row['p75_ret']:.6f} "
                f"worst_year_mean_ret={row['worst_year_mean_ret']:.6f} "
                f"worst_year_up_rate={row['worst_year_up_rate']:.6f}"
            )
        else:
            print(
                "[dashboard] "
                f"h={int(row['horizon_h'])}h "
                f"n={int(row['n'])} "
                f"tp={row['tp_rate']:.6f} "
                f"sl={row['sl_rate']:.6f} "
                f"same={row['samebar_rate']:.6f} "
                f"no_hit={row['no_hit_rate']:.6f} "
                f"tp_res={row['tp_resolved_rate']:.6f} "
                f"sl_res={row['sl_resolved_rate']:.6f} "
                f"hit={row['hit_rate']:.6f} "
                f"trunc={row['trunc_rate']:.6f} "
                f"expR_total={row['expR_total']:.6f} "
                f"expR_hit={row['expR_hit_only']:.6f}"
            )
    ranking = dashboard[dashboard["section"] == "ranking"].sort_values("rank")
    if not ranking.empty:
        top = ranking.iloc[0]
        if forward_mode == "time_only":
            print(
                "[dashboard] "
                f"best_h={int(top['horizon_h'])}h "
                f"worst_year_mean_ret={top['worst_year_mean_ret']:.6f} "
                f"overall_mean_ret={top['overall_mean_ret']:.6f}"
            )
        else:
            print(
                "[dashboard] "
                f"best_h={int(top['horizon_h'])}h "
                f"worst_year_expR={top['worst_year_expR']:.6f} "
                f"overall_expR={top['overall_expR']:.6f}"
            )

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="DFD05 parity simulator + forward horizon labeler")
    parser.add_argument("--config", required=False, help="Path to YAML config")
    parser.add_argument(
        "--config-preset",
        required=False,
        choices=["pine16"],
        help="Convenience preset. pine16 -> configs/dfd05_pine16_baseline.yaml",
    )
    parser.add_argument(
        "--parity",
        required=False,
        choices=["pine16"],
        help="Fail-fast parity check against Pine baseline defaults",
    )
    parser.add_argument(
        "--horizons",
        required=False,
        help="Override forward horizons in hours. Example: 4,24,72",
    )
    parser.add_argument(
        "--print-columns",
        action="store_true",
        help="Print event feature columns + dtypes and assert required numeric schema",
    )
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config, args.config_preset)
    cfg = load_config(str(config_path))
    _validate_parity_or_exit(cfg, args.parity)
    horizons_override = _parse_horizons_csv(args.horizons) if args.horizons else None
    outputs = run(cfg, horizons_override=horizons_override, print_columns=bool(args.print_columns))
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
