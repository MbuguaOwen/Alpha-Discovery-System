from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfd05.config import RunConfig, load_config, validate_pine16_strategy_parity
from dfd05.data import timeframe_to_minutes
from dfd05.run import run as run_dfd05


BPS_THRESHOLDS = [10, 25, 50, 100]
EVAL_FLOAT_COLS = [
    "win_rate_up",
    "win_rate_ge_10bps",
    "win_rate_ge_25bps",
    "win_rate_ge_50bps",
    "win_rate_ge_100bps",
    "win_rate_top_q80",
    "opportunity_rate_mfeatr_top_q80",
    "quality_rate_good",
    "mean_ret",
    "median_ret",
    "p25_ret",
    "p50_ret",
    "p75_ret",
    "mean_mfe",
    "mean_mae",
    "mean_ret_atr",
    "mean_mfe_atr",
    "mean_mae_atr",
    "trunc_rate",
]


def _parse_int_csv(raw: str, field_name: str) -> List[int]:
    vals: List[int] = []
    for token in str(raw).split(","):
        tok = token.strip()
        if tok == "":
            continue
        vals.append(int(tok))
    if not vals:
        raise ValueError(f"No valid values parsed from --{field_name}")
    out = sorted(set(vals))
    if any(v <= 0 for v in out):
        raise ValueError(f"--{field_name} must be positive integers.")
    return out


def _parse_timeframes_csv(raw: str) -> List[str]:
    vals: List[str] = []
    for token in str(raw).split(","):
        tok = token.strip().lower()
        if tok == "":
            continue
        _ = timeframe_to_minutes(tok)  # fail-fast for unsupported timeframe
        vals.append(tok)
    if not vals:
        raise ValueError("No valid values parsed from --timeframes")
    return sorted(set(vals), key=lambda tf: timeframe_to_minutes(tf))


def _horizon_suffix(hours: int) -> str:
    return f"{int(hours)}h"


def _threshold_label(value: float) -> str:
    token = f"{float(value):.10g}"
    if token.startswith("-"):
        token = f"m{token[1:]}"
    return token.replace(".", "p")


def _validate_parity_or_exit(config: RunConfig, parity_mode: str | None) -> None:
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
    raise SystemExit("\n".join(lines))


def _set_session_mode(config: RunConfig, mode: str) -> None:
    if mode == "session_on":
        if config.strategy.session_gate_source == "legacy":
            config.strategy.toggles.enable_session_gate = True
        else:
            config.strategy.session_gate.enabled = True
            config.strategy.session_gate.tz = "Etc/GMT-3"
            config.strategy.session_gate.ny = True
            config.strategy.session_gate.london = False
            config.strategy.session_gate.tokyo = False
            config.strategy.session_gate.sydney = False
        return
    if mode == "session_off":
        if config.strategy.session_gate_source == "legacy":
            config.strategy.toggles.enable_session_gate = False
        else:
            config.strategy.session_gate.enabled = False
        return
    raise ValueError(f"Unsupported session mode: {mode}")


def _validate_horizon_bar_conversion(timeframe: str, horizons: Sequence[int]) -> None:
    tf_minutes = timeframe_to_minutes(timeframe)
    for h in horizons:
        total_mins = int(h) * 60
        bars = total_mins // tf_minutes
        rem = total_mins % tf_minutes
        if bars <= 0:
            raise ValueError(f"horizon={h}h produces zero bars for timeframe={timeframe}")
        if rem != 0:
            print(
                f"[warn] timeframe={timeframe} horizon={h}h not divisible by {tf_minutes}m; using floor bars={bars}."
            )


def _coerce_binary(series: pd.Series, name: str, allow_na: bool) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype("float64")
    mask = np.isfinite(x.to_numpy(dtype=float))
    if not allow_na and not mask.all():
        raise ValueError(f"{name} contains NaN/invalid values.")
    if mask.any():
        good = np.isin(x.to_numpy(dtype=float)[mask], np.array([0.0, 1.0], dtype=float))
        if not bool(np.all(good)):
            raise ValueError(f"{name} contains values outside {{0,1}}.")
    return x


def _prepare_horizon_frames(
    labeled: pd.DataFrame,
    horizon_h: int,
    quality_mfe_thr: float,
    quality_mae_thr: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    suffix = _horizon_suffix(horizon_h)
    good_col = (
        f"good_{suffix}_mfe{_threshold_label(quality_mfe_thr)}_mae{_threshold_label(quality_mae_thr)}"
    )
    req = {
        "symbol": "symbol",
        "event_time_ms": "event_time_ms",
        "ret": f"ret_{suffix}",
        "mfe": f"mfe_{suffix}",
        "mae": f"mae_{suffix}",
        "ret_atr": f"ret_atr_{suffix}",
        "mfe_atr": f"mfe_atr_{suffix}",
        "mae_atr": f"mae_atr_{suffix}",
        "up": f"up_{suffix}",
        "win_top_q80": f"worked_{suffix}_top_q80",
        "opp_mfeatr_top_q80": f"worked_mfeatr_{suffix}_top_q80",
        "quality_good": good_col,
        "is_truncated": f"is_truncated_{suffix}",
    }
    missing = [src for src in req.values() if src not in labeled.columns]
    if missing:
        raise ValueError(f"Missing required labeled columns for {suffix}: {missing}")

    out = pd.DataFrame()
    out["symbol"] = labeled[req["symbol"]].astype(str)
    dt = pd.to_datetime(labeled[req["event_time_ms"]], unit="ms", utc=True, errors="coerce")
    if dt.isna().any():
        raise ValueError(f"Invalid event_time_ms values found for {suffix}.")
    out["year"] = dt.dt.year.astype("int64")
    for key in ["ret", "mfe", "mae", "ret_atr", "mfe_atr", "mae_atr"]:
        out[key] = pd.to_numeric(labeled[req[key]], errors="coerce").astype("float64")
    for key in ["up", "win_top_q80", "opp_mfeatr_top_q80", "quality_good"]:
        out[key] = _coerce_binary(labeled[req[key]], name=req[key], allow_na=True).astype("Int8")
    out["is_truncated"] = _coerce_binary(
        labeled[req["is_truncated"]],
        name=req["is_truncated"],
        allow_na=False,
    ).astype("int8")

    full = out
    valid = out[out["is_truncated"] == 0].copy()
    for key in ["up", "win_top_q80", "opp_mfeatr_top_q80", "quality_good"]:
        if valid[key].isna().any():
            raise ValueError(
                f"Non-truncated rows have NaN in required binary column {req[key]} for {suffix}."
            )
        valid[key] = valid[key].astype("int8")
    return full, valid


def _summarize_valid_group(group: pd.DataFrame) -> pd.Series:
    n = int(len(group))
    if n == 0:
        return pd.Series(
            {
                "n_valid": 0,
                "win_rate_up": np.nan,
                "win_rate_ge_10bps": np.nan,
                "win_rate_ge_25bps": np.nan,
                "win_rate_ge_50bps": np.nan,
                "win_rate_ge_100bps": np.nan,
                "win_rate_top_q80": np.nan,
                "opportunity_rate_mfeatr_top_q80": np.nan,
                "quality_rate_good": np.nan,
                "mean_ret": np.nan,
                "median_ret": np.nan,
                "p25_ret": np.nan,
                "p50_ret": np.nan,
                "p75_ret": np.nan,
                "mean_mfe": np.nan,
                "mean_mae": np.nan,
                "mean_ret_atr": np.nan,
                "mean_mfe_atr": np.nan,
                "mean_mae_atr": np.nan,
            }
        )

    ret = group["ret"].to_numpy(dtype=float)
    ret_atr = group["ret_atr"].to_numpy(dtype=float)
    mfe_atr = group["mfe_atr"].to_numpy(dtype=float)
    mae_atr = group["mae_atr"].to_numpy(dtype=float)
    out = {
        "n_valid": n,
        "win_rate_up": float(group["up"].to_numpy(dtype=float).mean()),
        "win_rate_top_q80": float(group["win_top_q80"].to_numpy(dtype=float).mean()),
        "opportunity_rate_mfeatr_top_q80": float(
            group["opp_mfeatr_top_q80"].to_numpy(dtype=float).mean()
        ),
        "quality_rate_good": float(group["quality_good"].to_numpy(dtype=float).mean()),
        "mean_ret": float(np.mean(ret)),
        "median_ret": float(np.median(ret)),
        "p25_ret": float(np.quantile(ret, 0.25)),
        "p50_ret": float(np.quantile(ret, 0.50)),
        "p75_ret": float(np.quantile(ret, 0.75)),
        "mean_mfe": float(group["mfe"].to_numpy(dtype=float).mean()),
        "mean_mae": float(group["mae"].to_numpy(dtype=float).mean()),
        "mean_ret_atr": float(np.nanmean(ret_atr)) if np.isfinite(ret_atr).any() else np.nan,
        "mean_mfe_atr": float(np.nanmean(mfe_atr)) if np.isfinite(mfe_atr).any() else np.nan,
        "mean_mae_atr": float(np.nanmean(mae_atr)) if np.isfinite(mae_atr).any() else np.nan,
    }
    for bps in BPS_THRESHOLDS:
        out[f"win_rate_ge_{bps}bps"] = float((ret >= (float(bps) / 10000.0)).mean())
    return pd.Series(out)


def _aggregate_eval(
    full: pd.DataFrame,
    valid: pd.DataFrame,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    if len(group_cols) == 0:
        trunc_rate = float(full["is_truncated"].to_numpy(dtype=float).mean()) if len(full) > 0 else np.nan
        base = pd.DataFrame([{"n_total": int(len(full)), "trunc_rate": trunc_rate}])
        met = _summarize_valid_group(valid).to_frame().T
        out = pd.concat([base.reset_index(drop=True), met.reset_index(drop=True)], axis=1)
    else:
        base = (
            full.groupby(list(group_cols), dropna=False)
            .agg(
                n_total=("is_truncated", "size"),
                trunc_rate=("is_truncated", "mean"),
            )
            .reset_index()
        )
        if valid.empty:
            met = pd.DataFrame(columns=list(group_cols) + ["n_valid"])
        else:
            metric_cols = [
                "ret",
                "mfe",
                "mae",
                "ret_atr",
                "mfe_atr",
                "mae_atr",
                "up",
                "win_top_q80",
                "opp_mfeatr_top_q80",
                "quality_good",
            ]
            met = (
                valid.groupby(list(group_cols), dropna=False)[metric_cols]
                .apply(_summarize_valid_group)
                .reset_index()
            )
        out = base.merge(met, on=list(group_cols), how="left")

    out["n_valid"] = pd.to_numeric(out.get("n_valid"), errors="coerce").fillna(0).astype("int64")
    out["n_total"] = pd.to_numeric(out.get("n_total"), errors="coerce").fillna(0).astype("int64")
    for col in EVAL_FLOAT_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out


def _evaluate_labeled_run(
    labeled: pd.DataFrame,
    timeframe: str,
    session_mode: str,
    horizons: Sequence[int],
    quality_mfe_thr: float,
    quality_mae_thr: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows: List[pd.DataFrame] = []
    per_symbol_year_rows: List[pd.DataFrame] = []

    for h in horizons:
        full, valid = _prepare_horizon_frames(
            labeled=labeled,
            horizon_h=int(h),
            quality_mfe_thr=quality_mfe_thr,
            quality_mae_thr=quality_mae_thr,
        )
        overall = _aggregate_eval(full=full, valid=valid, group_cols=[])
        overall["timeframe"] = timeframe
        overall["session_mode"] = session_mode
        overall["horizon_h"] = int(h)
        overall_rows.append(overall)

        per_sy = _aggregate_eval(full=full, valid=valid, group_cols=["symbol", "year"])
        per_sy["timeframe"] = timeframe
        per_sy["session_mode"] = session_mode
        per_sy["horizon_h"] = int(h)
        per_symbol_year_rows.append(per_sy)

    overall_df = pd.concat(overall_rows, ignore_index=True, sort=False) if overall_rows else pd.DataFrame()
    per_sy_df = (
        pd.concat(per_symbol_year_rows, ignore_index=True, sort=False)
        if per_symbol_year_rows
        else pd.DataFrame()
    )
    return overall_df, per_sy_df


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    if not mask.any():
        return float("nan")
    return float(np.average(v[mask], weights=w[mask]))


def _best_horizon_selection(
    overall: pd.DataFrame,
    per_symbol_year: pd.DataFrame,
    include_session_mode: bool,
) -> pd.DataFrame:
    if overall.empty:
        return pd.DataFrame()

    key_cols = ["timeframe"] + (["session_mode"] if include_session_mode else [])
    out_rows: List[Dict[str, object]] = []

    for key_vals, cand in overall.groupby(key_cols, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        key_map = dict(zip(key_cols, key_vals))

        key_mask = np.ones(len(per_symbol_year), dtype=bool)
        for k in key_cols:
            key_mask = key_mask & (per_symbol_year[k] == key_map[k]).to_numpy(dtype=bool)
        sy = per_symbol_year.loc[key_mask].copy()
        sy = sy[sy["n_valid"] > 0].copy()

        yearly_rows: List[Dict[str, object]] = []
        for (h, y), g in sy.groupby(["horizon_h", "year"], dropna=False):
            yearly_rows.append(
                {
                    "horizon_h": int(h),
                    "year": int(y),
                    "year_mean_ret": _weighted_mean(g["mean_ret"], g["n_valid"]),
                    "year_quality_rate_good": _weighted_mean(g["quality_rate_good"], g["n_valid"]),
                }
            )
        yearly = pd.DataFrame(yearly_rows)

        work = cand.copy()
        work["worst_year_mean_ret"] = np.nan
        work["worst_year_mean_ret_year"] = np.nan
        work["worst_year_quality_rate_good"] = np.nan
        work["worst_year_quality_rate_good_year"] = np.nan

        if not yearly.empty:
            for h, g in yearly.groupby("horizon_h", dropna=False):
                ret_series = pd.to_numeric(g["year_mean_ret"], errors="coerce").astype(float)
                qual_series = pd.to_numeric(g["year_quality_rate_good"], errors="coerce").astype(float)
                idx_ret = ret_series.idxmin() if np.isfinite(ret_series).any() else None
                idx_q = qual_series.idxmin() if np.isfinite(qual_series).any() else None
                if idx_ret is not None:
                    work.loc[work["horizon_h"] == int(h), "worst_year_mean_ret"] = float(
                        g.loc[idx_ret, "year_mean_ret"]
                    )
                    work.loc[work["horizon_h"] == int(h), "worst_year_mean_ret_year"] = float(
                        g.loc[idx_ret, "year"]
                    )
                if idx_q is not None:
                    work.loc[work["horizon_h"] == int(h), "worst_year_quality_rate_good"] = float(
                        g.loc[idx_q, "year_quality_rate_good"]
                    )
                    work.loc[work["horizon_h"] == int(h), "worst_year_quality_rate_good_year"] = float(
                        g.loc[idx_q, "year"]
                    )

        ranked = work.sort_values(
            [
                "worst_year_mean_ret",
                "worst_year_quality_rate_good",
                "mean_ret",
                "horizon_h",
            ],
            ascending=[False, False, False, True],
            na_position="last",
        ).reset_index(drop=True)
        ranked["selection_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
        best = ranked.iloc[0]
        out_rows.append(
            {
                **key_map,
                "selected_horizon_h": int(best["horizon_h"]),
                "worst_year_mean_ret": float(best["worst_year_mean_ret"])
                if pd.notna(best["worst_year_mean_ret"])
                else np.nan,
                "worst_year_mean_ret_year": int(best["worst_year_mean_ret_year"])
                if pd.notna(best["worst_year_mean_ret_year"])
                else pd.NA,
                "worst_year_quality_rate_good": float(best["worst_year_quality_rate_good"])
                if pd.notna(best["worst_year_quality_rate_good"])
                else np.nan,
                "worst_year_quality_rate_good_year": int(best["worst_year_quality_rate_good_year"])
                if pd.notna(best["worst_year_quality_rate_good_year"])
                else pd.NA,
                "overall_mean_ret": float(best["mean_ret"]) if pd.notna(best["mean_ret"]) else np.nan,
                "overall_quality_rate_good": float(best["quality_rate_good"])
                if pd.notna(best["quality_rate_good"])
                else np.nan,
                "selection_rank": int(best["selection_rank"]),
                "selection_primary": "max worst_year_mean_ret",
                "selection_secondary": "max worst_year_quality_rate_good",
                "selection_tertiary": "max overall_mean_ret",
                "selection_rule": (
                    "Sort DESC by worst_year_mean_ret, then worst_year_quality_rate_good, "
                    "then overall_mean_ret; final tie: smallest horizon."
                ),
            }
        )

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out
    out["selected_horizon_h"] = pd.to_numeric(out["selected_horizon_h"], errors="coerce").astype("int64")
    out["selection_rank"] = pd.to_numeric(out["selection_rank"], errors="coerce").astype("int64")
    if "worst_year_mean_ret_year" in out.columns:
        out["worst_year_mean_ret_year"] = out["worst_year_mean_ret_year"].astype("Int64")
    if "worst_year_quality_rate_good_year" in out.columns:
        out["worst_year_quality_rate_good_year"] = out["worst_year_quality_rate_good_year"].astype("Int64")
    for col in [
        "worst_year_mean_ret",
        "worst_year_quality_rate_good",
        "overall_mean_ret",
        "overall_quality_rate_good",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out


def _fmt_md_cell(v: object) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
        return ""
    if isinstance(v, (np.integer, int)):
        return str(int(v))
    if isinstance(v, (np.floating, float)):
        return f"{float(v):.6f}"
    return str(v)


def _markdown_table(df: pd.DataFrame, columns: Sequence[str]) -> str:
    if df.empty:
        return "_No rows._"
    use_cols = [c for c in columns if c in df.columns]
    header = "| " + " | ".join(use_cols) + " |"
    sep = "| " + " | ".join(["---"] * len(use_cols)) + " |"
    lines = [header, sep]
    for _, row in df[use_cols].iterrows():
        lines.append("| " + " | ".join(_fmt_md_cell(row[c]) for c in use_cols) + " |")
    return "\n".join(lines)


def _write_summary_md(
    path: Path,
    args: argparse.Namespace,
    run_logs: pd.DataFrame,
    overall: pd.DataFrame,
    best: pd.DataFrame,
    include_session_mode: bool,
) -> None:
    key_cols = ["timeframe"] + (["session_mode"] if include_session_mode else [])
    overall_top = (
        overall.sort_values(key_cols + ["horizon_h"]).reset_index(drop=True)
        if not overall.empty
        else overall
    )
    lines: List[str] = []
    lines.append("# DFD05 Timeframe + Horizon Evaluation")
    lines.append("")
    lines.append(f"- Generated UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Config: `{args.config}`")
    lines.append(f"- Parity: `{args.parity or 'none'}`")
    lines.append(f"- Timeframes: `{args.timeframes}`")
    lines.append(f"- Horizons (hours): `{args.horizons}`")
    lines.append(f"- Compare sessions: `{bool(args.compare_sessions)}`")
    lines.append("")
    lines.append("## Runtime")
    lines.append("")
    lines.append(
        _markdown_table(
            run_logs,
            [
                "timeframe",
                "session_mode",
                "run_id",
                "seconds",
                "events_n",
                "forward_n",
                "labeled_n",
            ],
        )
    )
    lines.append("")
    lines.append("## Overall By Timeframe/Horizon")
    lines.append("")
    lines.append(
        _markdown_table(
            overall_top,
            key_cols
            + [
                "horizon_h",
                "n_valid",
                "win_rate_up",
                "quality_rate_good",
                "mean_ret",
                "mean_ret_atr",
                "trunc_rate",
            ],
        )
    )
    lines.append("")
    lines.append("## Best Horizon Selection")
    lines.append("")
    lines.append(
        _markdown_table(
            best,
            key_cols
            + [
                "selected_horizon_h",
                "worst_year_mean_ret",
                "worst_year_quality_rate_good",
                "overall_mean_ret",
                "selection_rule",
            ],
        )
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run DFD05 time-only pipeline across multiple timeframes and build master evaluation reports."
    )
    ap.add_argument("--config", required=True, help="Path to baseline DFD05 YAML config")
    ap.add_argument(
        "--parity",
        required=False,
        choices=["pine16"],
        help="Optional fail-fast parity validation for baseline config",
    )
    ap.add_argument("--timeframes", required=True, help="CSV list, e.g. m15,h1,h4")
    ap.add_argument("--horizons", required=True, help="CSV list in hours, e.g. 4,24,72")
    ap.add_argument("--outdir", default="data/reports/dfd05_timeframe_eval", help="Output report directory")
    ap.add_argument(
        "--compare_sessions",
        action="store_true",
        help="If set, run each timeframe with session gate ON and OFF and add session_mode in outputs.",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg_base = load_config(args.config)
    _validate_parity_or_exit(cfg_base, args.parity)
    if cfg_base.forward.normalized_mode() != "time_only":
        raise SystemExit("This evaluator requires forward.mode=time_only.")

    timeframes = _parse_timeframes_csv(args.timeframes)
    horizons = _parse_int_csv(args.horizons, field_name="horizons")
    for tf in timeframes:
        _validate_horizon_bar_conversion(tf, horizons)

    # Ensure required q80 targets exist for required columns.
    cfg_base.forward.percentile_targets = sorted(set(list(cfg_base.forward.percentile_targets) + [0.8]))
    cfg_base.forward.atr_targets.pct_targets = sorted(set(list(cfg_base.forward.atr_targets.pct_targets) + [0.8]))
    quality_mfe_thr = cfg_base.forward.normalized_quality_mfe_threshold()
    quality_mae_thr = cfg_base.forward.normalized_quality_mae_threshold()

    session_modes = ["session_on", "session_off"] if args.compare_sessions else ["config_default"]
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    overall_runs: List[pd.DataFrame] = []
    per_sy_runs: List[pd.DataFrame] = []
    run_logs: List[Dict[str, object]] = []

    for tf in timeframes:
        for session_mode in session_modes:
            cfg_run = copy.deepcopy(cfg_base)
            cfg_run.timeframe = tf
            cfg_run.forward.horizons_hours = list(horizons)
            cfg_run.forward.horizons_hours_legacy = None
            cfg_run.forward.forward_horizons_hours = []
            cfg_run.labels.write_labeled = True
            if args.compare_sessions:
                _set_session_mode(cfg_run, session_mode)
                if args.parity and session_mode == "session_off":
                    print("[parity] skipped for session_off mode.")

            run_id = f"{run_stamp}_{tf}_{session_mode}"
            cfg_run.output.run_id = run_id

            t0 = perf_counter()
            outputs = run_dfd05(cfg_run, horizons_override=list(horizons), print_columns=False)
            elapsed = perf_counter() - t0

            events_path = outputs["events"]
            forward_path = outputs["forward"]
            labeled_path = outputs.get("labeled")
            dashboard_path = outputs["dashboard"]
            if labeled_path is None:
                raise RuntimeError("run() did not produce labeled output. Ensure labels.write_labeled=true.")

            events_n = len(pd.read_parquet(events_path, columns=["symbol"]))
            forward_n = len(pd.read_parquet(forward_path, columns=["symbol"]))
            labeled = pd.read_parquet(labeled_path)
            labeled_n = len(labeled)

            print(
                f"[tf_eval] timeframe={tf} session_mode={session_mode} run_id={run_id} "
                f"seconds={elapsed:.2f} events={events_n} forward={forward_n} labeled={labeled_n}"
            )

            ov, sy = _evaluate_labeled_run(
                labeled=labeled,
                timeframe=tf,
                session_mode=session_mode,
                horizons=horizons,
                quality_mfe_thr=quality_mfe_thr,
                quality_mae_thr=quality_mae_thr,
            )
            overall_runs.append(ov)
            per_sy_runs.append(sy)
            run_logs.append(
                {
                    "timeframe": tf,
                    "session_mode": session_mode,
                    "run_id": run_id,
                    "seconds": float(elapsed),
                    "events_n": int(events_n),
                    "forward_n": int(forward_n),
                    "labeled_n": int(labeled_n),
                    "events_path": str(events_path),
                    "forward_path": str(forward_path),
                    "labeled_path": str(labeled_path),
                    "dashboard_path": str(dashboard_path),
                }
            )

    overall_raw = pd.concat(overall_runs, ignore_index=True, sort=False) if overall_runs else pd.DataFrame()
    per_sy_raw = pd.concat(per_sy_runs, ignore_index=True, sort=False) if per_sy_runs else pd.DataFrame()
    if overall_raw.empty or per_sy_raw.empty:
        raise SystemExit("No evaluation rows were produced.")

    include_session_mode = bool(args.compare_sessions)
    if not include_session_mode:
        overall_raw = overall_raw.drop(columns=["session_mode"], errors="ignore")
        per_sy_raw = per_sy_raw.drop(columns=["session_mode"], errors="ignore")

    # Output schemas (requested fields, plus p50_ret to explicitly report p25/p50/p75).
    overall_cols = [
        "timeframe",
        "horizon_h",
        "n_valid",
        "win_rate_up",
        "win_rate_ge_10bps",
        "win_rate_ge_25bps",
        "win_rate_ge_50bps",
        "win_rate_ge_100bps",
        "win_rate_top_q80",
        "opportunity_rate_mfeatr_top_q80",
        "quality_rate_good",
        "mean_ret",
        "median_ret",
        "p25_ret",
        "p50_ret",
        "p75_ret",
        "mean_mfe",
        "mean_mae",
        "mean_ret_atr",
        "mean_mfe_atr",
        "mean_mae_atr",
        "trunc_rate",
    ]
    per_sy_cols = [
        "timeframe",
        "horizon_h",
        "symbol",
        "year",
        "n_valid",
        "win_rate_up",
        "win_rate_ge_10bps",
        "win_rate_ge_25bps",
        "win_rate_ge_50bps",
        "win_rate_ge_100bps",
        "win_rate_top_q80",
        "opportunity_rate_mfeatr_top_q80",
        "quality_rate_good",
        "mean_ret",
        "median_ret",
        "p25_ret",
        "p50_ret",
        "p75_ret",
        "mean_mfe",
        "mean_mae",
        "mean_ret_atr",
        "mean_mfe_atr",
        "mean_mae_atr",
        "trunc_rate",
    ]
    if include_session_mode:
        overall_cols.insert(1, "session_mode")
        per_sy_cols.insert(1, "session_mode")

    overall = overall_raw[overall_cols].copy()
    per_sy = per_sy_raw[per_sy_cols].copy()

    overall["horizon_h"] = pd.to_numeric(overall["horizon_h"], errors="coerce").astype("int64")
    overall["n_valid"] = pd.to_numeric(overall["n_valid"], errors="coerce").fillna(0).astype("int64")
    per_sy["horizon_h"] = pd.to_numeric(per_sy["horizon_h"], errors="coerce").astype("int64")
    per_sy["n_valid"] = pd.to_numeric(per_sy["n_valid"], errors="coerce").fillna(0).astype("int64")
    per_sy["year"] = pd.to_numeric(per_sy["year"], errors="coerce").astype("Int64")
    for col in EVAL_FLOAT_COLS:
        if col in overall.columns:
            overall[col] = pd.to_numeric(overall[col], errors="coerce").astype("float64")
        if col in per_sy.columns:
            per_sy[col] = pd.to_numeric(per_sy[col], errors="coerce").astype("float64")

    sort_overall = ["timeframe"] + (["session_mode"] if include_session_mode else []) + ["horizon_h"]
    sort_sy = sort_overall + ["symbol", "year"]
    overall = overall.sort_values(sort_overall).reset_index(drop=True)
    per_sy = per_sy.sort_values(sort_sy).reset_index(drop=True)

    best = _best_horizon_selection(
        overall=overall_raw,
        per_symbol_year=per_sy_raw,
        include_session_mode=include_session_mode,
    )
    if not include_session_mode and "session_mode" in best.columns:
        best = best.drop(columns=["session_mode"])

    overall_path = outdir / "overall_by_tf_h.csv"
    per_sy_path = outdir / "per_symbol_year_by_tf_h.csv"
    best_path = outdir / "best_horizon_selection.csv"
    summary_path = outdir / "summary.md"
    overall.to_csv(overall_path, index=False)
    per_sy.to_csv(per_sy_path, index=False)
    best.to_csv(best_path, index=False)

    run_logs_df = pd.DataFrame(run_logs).sort_values(
        ["timeframe", "session_mode"],
        ascending=[True, True],
    )
    _write_summary_md(
        path=summary_path,
        args=args,
        run_logs=run_logs_df,
        overall=overall,
        best=best,
        include_session_mode=include_session_mode,
    )

    print(f"overall: {overall_path}")
    print(f"per_symbol_year: {per_sy_path}")
    print(f"best_horizon_selection: {best_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
