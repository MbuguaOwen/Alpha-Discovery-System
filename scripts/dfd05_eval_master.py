from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfd05.config import RunConfig, effective_session_gate, load_config, validate_pine16_strategy_parity
from dfd05.data import timeframe_to_minutes
from dfd05.run import run as run_dfd05
from dfd05.trade_sim import select_executed_trades


VARIANT_PRESETS: Dict[str, str] = {
    "baseline": "configs/dfd05_pine16_baseline.yaml",
    "gate_only": "configs/dfd05_ablate_gate_only.yaml",
    "bos_only": "configs/dfd05_ablate_bos_only.yaml",
    "bos_plus_vol": "configs/dfd05_ablate_bos_plus_vol.yaml",
    "prod": "configs/dfd05_pine16_prod_bos_vol.yaml",
}
BPS_THRESHOLDS = [10, 25, 50, 100]
VALID_LEVELS = ("signal", "executed")
VALID_SESSION_REGIONS = ("ny", "london", "tokyo", "sydney")
PREFER_HORIZON_RET_EPS = 5e-5
PREFER_HORIZON_RATE_EPS = 5e-3


def _parse_csv_tokens(raw: str) -> List[str]:
    out: List[str] = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if t:
            out.append(t)
    return out


def _parse_int_csv(raw: str, field_name: str) -> List[int]:
    vals: List[int] = []
    for token in _parse_csv_tokens(raw):
        vals.append(int(token))
    if not vals:
        raise ValueError(f"No valid values parsed from --{field_name}")
    out = sorted(set(vals))
    if any(v <= 0 for v in out):
        raise ValueError(f"--{field_name} must contain positive integers.")
    return out


def _parse_timeframes_csv(raw: str) -> List[str]:
    vals: List[str] = []
    for token in _parse_csv_tokens(raw):
        tf = token.lower()
        _ = timeframe_to_minutes(tf)
        vals.append(tf)
    if not vals:
        raise ValueError("No valid values parsed from --timeframes")
    return sorted(set(vals), key=lambda tf: timeframe_to_minutes(tf))


def _parse_variants_csv(raw: str) -> List[Tuple[str, str]]:
    variants: List[Tuple[str, str]] = []
    for token in _parse_csv_tokens(raw):
        key = token.strip()
        low = key.lower()
        if low in VARIANT_PRESETS:
            variants.append((low, VARIANT_PRESETS[low]))
            continue
        p = Path(key)
        if p.exists():
            variants.append((p.stem.lower(), str(p)))
            continue
        raise ValueError(f"Unknown variant token: {token}. Use one of {list(VARIANT_PRESETS)} or a config path.")
    if not variants:
        raise ValueError("No variants resolved from --configs")
    dedup: Dict[str, str] = {}
    for name, path in variants:
        dedup[name] = path
    return [(k, v) for k, v in dedup.items()]


def _parse_selection_levels_csv(raw: str) -> List[str]:
    levels: List[str] = []
    for token in _parse_csv_tokens(raw):
        level = token.strip().lower()
        if level not in VALID_LEVELS:
            raise ValueError(f"Unknown selection level: {token}. Valid values: {list(VALID_LEVELS)}")
        levels.append(level)
    if not levels:
        raise ValueError("No selection levels resolved from --selection_levels")
    return sorted(set(levels), key=lambda x: 0 if x == "executed" else 1)


def _parse_session_regions_csv(raw: Optional[str]) -> List[str]:
    token = "ny" if raw is None else str(raw)
    picked: List[str] = []
    for item in _parse_csv_tokens(token):
        key = item.strip().lower()
        if key not in VALID_SESSION_REGIONS:
            raise ValueError(
                f"Unknown session region: {item}. Valid values: {list(VALID_SESSION_REGIONS)}"
            )
        picked.append(key)
    if not picked:
        raise ValueError("--session_on_regions must include at least one region.")
    chosen = set(picked)
    return [k for k in VALID_SESSION_REGIONS if k in chosen]


def _normalized_gate_vol_entry_at(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    token = str(raw).strip().lower()
    if token not in {"signal", "trigger"}:
        raise ValueError("--gate_vol_entry_at must be one of: signal,trigger")
    return token


def _apply_gate_vol_entry_at_override(config: RunConfig, gate_vol_entry_at: Optional[str]) -> None:
    if gate_vol_entry_at is None:
        return
    config.strategy.gate_vol_entry_at = gate_vol_entry_at


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
    raise SystemExit("\n".join(lines))


def _set_session_mode(
    config: RunConfig,
    mode: str,
    *,
    session_on_regions: Sequence[str],
    session_tz: str,
) -> None:
    if mode == "session_on":
        if config.strategy.session_gate_source == "legacy":
            config.strategy.toggles.enable_session_gate = True
        else:
            selected = set([str(x).strip().lower() for x in session_on_regions if str(x).strip()])
            if not selected:
                raise ValueError("session_on_regions cannot be empty for session_on mode.")
            config.strategy.session_gate.enabled = True
            config.strategy.session_gate.tz = str(session_tz).strip() or "Etc/GMT-3"
            config.strategy.session_gate.ny = "ny" in selected
            config.strategy.session_gate.london = "london" in selected
            config.strategy.session_gate.tokyo = "tokyo" in selected
            config.strategy.session_gate.sydney = "sydney" in selected
        return
    if mode == "session_off":
        if config.strategy.session_gate_source == "legacy":
            config.strategy.toggles.enable_session_gate = False
        else:
            config.strategy.session_gate.enabled = False
        return
    if mode == "config_default":
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


def _parse_time_scale_params(raw: Optional[str]) -> Optional[Dict[str, str]]:
    if raw is None or str(raw).strip() == "":
        return None
    items = _parse_csv_tokens(raw)
    parsed: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError("Invalid --time_scale_params format. Use: reference_tf=m15")
        k, v = item.split("=", 1)
        parsed[k.strip().lower()] = v.strip()
    if "reference_tf" not in parsed:
        raise ValueError("--time_scale_params must include reference_tf=<tf>")
    _ = timeframe_to_minutes(parsed["reference_tf"].lower())
    return parsed


def _apply_time_scaling(
    cfg_run: RunConfig,
    cfg_ref: RunConfig,
    target_timeframe: str,
    time_scale: Optional[Dict[str, str]],
) -> Dict[str, int]:
    if time_scale is None:
        return {
            "reference_tf_minutes": timeframe_to_minutes(cfg_ref.timeframe),
            "don_len": int(cfg_run.strategy.don_len),
            "pivot_len": int(cfg_run.strategy.pivot_len),
            "osc_len": int(cfg_run.strategy.osc_len),
            "scaled": 0,
        }

    ref_tf = time_scale["reference_tf"].lower()
    ref_minutes = timeframe_to_minutes(ref_tf)
    tf_minutes = timeframe_to_minutes(target_timeframe)

    don_minutes = int(cfg_ref.strategy.don_len) * ref_minutes
    pivot_minutes = int(cfg_ref.strategy.pivot_len) * ref_minutes
    scaled_don = int(np.clip(np.round(don_minutes / tf_minutes), 20, 500))
    scaled_pivot = int(np.clip(np.round(pivot_minutes / tf_minutes), 2, 20))

    cfg_run.strategy.don_len = int(scaled_don)
    cfg_run.strategy.pivot_len = int(scaled_pivot)
    return {
        "reference_tf_minutes": ref_minutes,
        "don_len": int(cfg_run.strategy.don_len),
        "pivot_len": int(cfg_run.strategy.pivot_len),
        "osc_len": int(cfg_run.strategy.osc_len),
        "scaled": 1,
    }


def _horizon_suffix(hours: int) -> str:
    return f"{int(hours)}h"


def _threshold_label(value: float) -> str:
    token = f"{float(value):.10g}"
    if token.startswith("-"):
        token = f"m{token[1:]}"
    return token.replace(".", "p")


MASTER_OVERALL_COLUMNS = [
    "variant",
    "timeframe",
    "session_mode",
    "level",
    "horizon_h",
    "n_total",
    "n_valid",
    "min_symbol_year_n_valid",
    "selection_eligible",
    "trunc_rate",
    "up_rate",
    "ge_10bps_rate",
    "ge_25bps_rate",
    "ge_50bps_rate",
    "ge_100bps_rate",
    "worked_top_q80_rate",
    "worked_top_q70_rate",
    "worked_top_q90_rate",
    "worked_mfeatr_top_q80_rate",
    "worked_mfeatr_top_q70_rate",
    "worked_mfeatr_top_q90_rate",
    "good_rate",
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
    "worst_year_mean_ret",
    "worst_year_good_rate",
    "worst_year_up_rate",
    "eligible_year_count",
    "eligible_symbol_year_rows",
]
MASTER_PER_SY_COLUMNS = [
    "variant",
    "timeframe",
    "session_mode",
    "level",
    "horizon_h",
    "symbol",
    "year",
    "n_total",
    "n_valid",
    "trunc_rate",
    "up_rate",
    "ge_10bps_rate",
    "ge_25bps_rate",
    "ge_50bps_rate",
    "ge_100bps_rate",
    "worked_top_q80_rate",
    "worked_top_q70_rate",
    "worked_top_q90_rate",
    "worked_mfeatr_top_q80_rate",
    "worked_mfeatr_top_q70_rate",
    "worked_mfeatr_top_q90_rate",
    "good_rate",
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
]


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
    frame: pd.DataFrame,
    horizon_h: int,
    quality_mfe_thr: float,
    quality_mae_thr: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        "worked_top_q80": f"worked_{suffix}_top_q80",
        "worked_top_q70": f"worked_{suffix}_top_q70",
        "worked_top_q90": f"worked_{suffix}_top_q90",
        "worked_mfeatr_top_q80": f"worked_mfeatr_{suffix}_top_q80",
        "worked_mfeatr_top_q70": f"worked_mfeatr_{suffix}_top_q70",
        "worked_mfeatr_top_q90": f"worked_mfeatr_{suffix}_top_q90",
        "good": good_col,
        "is_truncated": f"is_truncated_{suffix}",
    }
    required_keys = {
        "symbol",
        "event_time_ms",
        "ret",
        "mfe",
        "mae",
        "ret_atr",
        "mfe_atr",
        "mae_atr",
        "up",
        "worked_top_q80",
        "worked_mfeatr_top_q80",
        "good",
        "is_truncated",
    }
    missing = [req[k] for k in required_keys if req[k] not in frame.columns]
    if missing:
        raise ValueError(f"Missing required labeled columns for {suffix}: {missing}")
    optional_binary_keys = [
        "worked_top_q70",
        "worked_top_q90",
        "worked_mfeatr_top_q70",
        "worked_mfeatr_top_q90",
    ]

    out = pd.DataFrame()
    out["symbol"] = frame[req["symbol"]].astype(str)
    dt = pd.to_datetime(frame[req["event_time_ms"]], unit="ms", utc=True, errors="coerce")
    if dt.isna().any():
        raise ValueError(f"Invalid event_time_ms values found for {suffix}.")
    out["year"] = dt.dt.year.astype("int64")
    for key in ["ret", "mfe", "mae", "ret_atr", "mfe_atr", "mae_atr"]:
        out[key] = pd.to_numeric(frame[req[key]], errors="coerce").astype("float64")
    for key in ["up", "worked_top_q80", "worked_mfeatr_top_q80", "good"]:
        out[key] = _coerce_binary(frame[req[key]], name=req[key], allow_na=True).astype("Int8")
    for key in optional_binary_keys:
        src = req[key]
        if src in frame.columns:
            out[key] = _coerce_binary(frame[src], name=src, allow_na=True).astype("Int8")
        else:
            out[key] = pd.Series([pd.NA] * len(frame), index=frame.index, dtype="Int8")
    out["is_truncated"] = _coerce_binary(
        frame[req["is_truncated"]],
        name=req["is_truncated"],
        allow_na=False,
    ).astype("int8")

    full = out
    valid = out[out["is_truncated"] == 0].copy()
    for key in ["up", "worked_top_q80", "worked_mfeatr_top_q80", "good"]:
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
                "up_rate": np.nan,
                "ge_10bps_rate": np.nan,
                "ge_25bps_rate": np.nan,
                "ge_50bps_rate": np.nan,
                "ge_100bps_rate": np.nan,
                "worked_top_q80_rate": np.nan,
                "worked_top_q70_rate": np.nan,
                "worked_top_q90_rate": np.nan,
                "worked_mfeatr_top_q80_rate": np.nan,
                "worked_mfeatr_top_q70_rate": np.nan,
                "worked_mfeatr_top_q90_rate": np.nan,
                "good_rate": np.nan,
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

    def _opt_float_col(name: str) -> np.ndarray:
        if name not in group.columns:
            return np.array([], dtype=float)
        return pd.to_numeric(group[name], errors="coerce").to_numpy(dtype=float)

    worked_top_q70 = _opt_float_col("worked_top_q70")
    worked_top_q80 = _opt_float_col("worked_top_q80")
    worked_top_q90 = _opt_float_col("worked_top_q90")
    worked_mfeatr_top_q70 = _opt_float_col("worked_mfeatr_top_q70")
    worked_mfeatr_top_q80 = _opt_float_col("worked_mfeatr_top_q80")
    worked_mfeatr_top_q90 = _opt_float_col("worked_mfeatr_top_q90")

    out = {
        "n_valid": n,
        "up_rate": float(group["up"].to_numpy(dtype=float).mean()),
        "worked_top_q70_rate": float(np.nanmean(worked_top_q70)) if np.isfinite(worked_top_q70).any() else np.nan,
        "worked_top_q80_rate": float(np.nanmean(worked_top_q80)) if np.isfinite(worked_top_q80).any() else np.nan,
        "worked_top_q90_rate": float(np.nanmean(worked_top_q90)) if np.isfinite(worked_top_q90).any() else np.nan,
        "worked_mfeatr_top_q70_rate": float(np.nanmean(worked_mfeatr_top_q70))
        if np.isfinite(worked_mfeatr_top_q70).any()
        else np.nan,
        "worked_mfeatr_top_q80_rate": float(np.nanmean(worked_mfeatr_top_q80))
        if np.isfinite(worked_mfeatr_top_q80).any()
        else np.nan,
        "worked_mfeatr_top_q90_rate": float(np.nanmean(worked_mfeatr_top_q90))
        if np.isfinite(worked_mfeatr_top_q90).any()
        else np.nan,
        "good_rate": float(group["good"].to_numpy(dtype=float).mean()),
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
        out[f"ge_{bps}bps_rate"] = float((ret >= (float(bps) / 10000.0)).mean())
    return pd.Series(out)


def _aggregate_eval(full: pd.DataFrame, valid: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if len(group_cols) == 0:
        trunc_rate = float(full["is_truncated"].to_numpy(dtype=float).mean()) if len(full) > 0 else np.nan
        base = pd.DataFrame([{"n_total": int(len(full)), "trunc_rate": trunc_rate}])
        met = _summarize_valid_group(valid).to_frame().T
        out = pd.concat([base.reset_index(drop=True), met.reset_index(drop=True)], axis=1)
    else:
        base = (
            full.groupby(list(group_cols), dropna=False)
            .agg(n_total=("is_truncated", "size"), trunc_rate=("is_truncated", "mean"))
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
                "worked_top_q70",
                "worked_top_q80",
                "worked_top_q90",
                "worked_mfeatr_top_q70",
                "worked_mfeatr_top_q80",
                "worked_mfeatr_top_q90",
                "good",
            ]
            metric_cols = [c for c in metric_cols if c in valid.columns]
            met = (
                valid.groupby(list(group_cols), dropna=False)[metric_cols]
                .apply(_summarize_valid_group)
                .reset_index()
            )
        out = base.merge(met, on=list(group_cols), how="left")

    out["n_total"] = pd.to_numeric(out["n_total"], errors="coerce").fillna(0).astype("int64")
    out["n_valid"] = pd.to_numeric(out.get("n_valid"), errors="coerce").fillna(0).astype("int64")
    float_cols = [
        "trunc_rate",
        "up_rate",
        "ge_10bps_rate",
        "ge_25bps_rate",
        "ge_50bps_rate",
        "ge_100bps_rate",
        "worked_top_q80_rate",
        "worked_top_q70_rate",
        "worked_top_q90_rate",
        "worked_mfeatr_top_q80_rate",
        "worked_mfeatr_top_q70_rate",
        "worked_mfeatr_top_q90_rate",
        "good_rate",
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
    ]
    for col in float_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out


def _evaluate_level(
    labeled: pd.DataFrame,
    horizons: Sequence[int],
    quality_mfe_thr: float,
    quality_mae_thr: float,
    selected_by_h: Optional[Dict[int, pd.DataFrame]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows: List[pd.DataFrame] = []
    per_sy_rows: List[pd.DataFrame] = []
    for h in horizons:
        src = selected_by_h[int(h)] if selected_by_h is not None else labeled
        full, valid = _prepare_horizon_frames(
            frame=src,
            horizon_h=int(h),
            quality_mfe_thr=quality_mfe_thr,
            quality_mae_thr=quality_mae_thr,
        )
        ov = _aggregate_eval(full=full, valid=valid, group_cols=[])
        ov["horizon_h"] = int(h)
        overall_rows.append(ov)

        sy = _aggregate_eval(full=full, valid=valid, group_cols=["symbol", "year"])
        sy["horizon_h"] = int(h)
        per_sy_rows.append(sy)

    overall_df = pd.concat(overall_rows, ignore_index=True, sort=False) if overall_rows else pd.DataFrame()
    per_sy_df = pd.concat(per_sy_rows, ignore_index=True, sort=False) if per_sy_rows else pd.DataFrame()
    return overall_df, per_sy_df


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    if not mask.any():
        return np.nan
    return float(np.average(v[mask], weights=w[mask]))


def _attach_worst_year_metrics(
    overall: pd.DataFrame,
    per_symbol_year: pd.DataFrame,
    min_n_valid_per_symbol_year: int = 0,
) -> pd.DataFrame:
    key_cols = ["variant", "timeframe", "session_mode", "level", "horizon_h"]
    rows: List[Dict[str, object]] = []
    if per_symbol_year.empty:
        out = overall.copy()
        out["worst_year_mean_ret"] = np.nan
        out["worst_year_good_rate"] = np.nan
        out["worst_year_up_rate"] = np.nan
        out["eligible_year_count"] = np.nan
        out["eligible_symbol_year_rows"] = np.nan
        return out

    for key_vals, g in per_symbol_year.groupby(key_cols, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        key_map = dict(zip(key_cols, key_vals))
        yearly_rows: List[Dict[str, object]] = []
        eligible_sy_rows = 0
        for y, gy in g.groupby("year", dropna=False):
            gy_work = gy.copy()
            if "n_valid" in gy_work.columns:
                gy_work["n_valid"] = pd.to_numeric(gy_work["n_valid"], errors="coerce")
            if "mean_ret" in gy_work.columns:
                gy_work["mean_ret"] = pd.to_numeric(gy_work["mean_ret"], errors="coerce")
            if "good_rate" in gy_work.columns:
                gy_work["good_rate"] = pd.to_numeric(gy_work["good_rate"], errors="coerce")
            if "up_rate" in gy_work.columns:
                gy_work["up_rate"] = pd.to_numeric(gy_work["up_rate"], errors="coerce")
            gy_work = gy_work[gy_work["n_valid"] >= float(max(0, int(min_n_valid_per_symbol_year)))].copy()
            if gy_work.empty:
                continue
            eligible_sy_rows += int(len(gy_work))
            yearly_rows.append(
                {
                    "year": int(y) if pd.notna(y) else pd.NA,
                    "year_mean_ret": _weighted_mean(gy_work["mean_ret"], gy_work["n_valid"]),
                    "year_good_rate": _weighted_mean(gy_work["good_rate"], gy_work["n_valid"]),
                    "year_up_rate": _weighted_mean(gy_work["up_rate"], gy_work["n_valid"]),
                }
            )
        yf = pd.DataFrame(yearly_rows)
        rows.append(
            {
                **key_map,
                "worst_year_mean_ret": float(yf["year_mean_ret"].min())
                if ("year_mean_ret" in yf and len(yf) > 0)
                else np.nan,
                "worst_year_good_rate": float(yf["year_good_rate"].min())
                if ("year_good_rate" in yf and len(yf) > 0)
                else np.nan,
                "worst_year_up_rate": float(yf["year_up_rate"].min())
                if ("year_up_rate" in yf and len(yf) > 0)
                else np.nan,
                "eligible_year_count": int(len(yf)),
                "eligible_symbol_year_rows": int(eligible_sy_rows),
            }
        )
    worst_df = pd.DataFrame(rows)
    out = overall.merge(worst_df, on=key_cols, how="left")
    for col in [
        "worst_year_mean_ret",
        "worst_year_good_rate",
        "worst_year_up_rate",
        "eligible_year_count",
        "eligible_symbol_year_rows",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out


def _attach_min_symbol_year_n_valid(overall: pd.DataFrame, per_symbol_year: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["variant", "timeframe", "session_mode", "level", "horizon_h"]
    out = overall.copy()
    if per_symbol_year.empty:
        out["min_symbol_year_n_valid"] = np.nan
        return out
    min_sy = (
        per_symbol_year.groupby(key_cols, dropna=False)["n_valid"]
        .min()
        .reset_index()
        .rename(columns={"n_valid": "min_symbol_year_n_valid"})
    )
    out = out.merge(min_sy, on=key_cols, how="left")
    out["min_symbol_year_n_valid"] = pd.to_numeric(
        out["min_symbol_year_n_valid"], errors="coerce"
    ).astype("float64")
    return out


def _prioritize_prefer_horizon(
    ranked: pd.DataFrame,
    prefer_horizon: Optional[int],
) -> pd.DataFrame:
    if prefer_horizon is None or ranked.empty:
        return ranked
    candidates = ranked[ranked["horizon_h"] == int(prefer_horizon)]
    if candidates.empty:
        return ranked

    best = ranked.iloc[0]
    close_mask = (
        (ranked["worst_year_mean_ret"] >= float(best["worst_year_mean_ret"]) - PREFER_HORIZON_RET_EPS)
        & (ranked["worst_year_good_rate"] >= float(best["worst_year_good_rate"]) - PREFER_HORIZON_RATE_EPS)
        & (ranked["mean_ret"] >= float(best["mean_ret"]) - PREFER_HORIZON_RET_EPS)
        & (ranked["good_rate"] >= float(best["good_rate"]) - PREFER_HORIZON_RATE_EPS)
    )
    preferred_close = ranked[close_mask & (ranked["horizon_h"] == int(prefer_horizon))]
    if preferred_close.empty:
        return ranked
    chosen_idx = preferred_close.index[0]
    if int(chosen_idx) == int(ranked.index[0]):
        return ranked
    promoted = pd.concat([ranked.loc[[chosen_idx]], ranked.drop(index=chosen_idx)], axis=0)
    return promoted.reset_index(drop=True)


def _best_horizon_selection(
    overall: pd.DataFrame,
    selection_levels: Sequence[str],
    min_n_valid_global: int,
    min_n_valid_per_symbol_year: int,
    prefer_horizon: Optional[int],
) -> pd.DataFrame:
    if overall.empty:
        return pd.DataFrame()
    key_cols = ["variant", "timeframe", "session_mode", "level"]
    rows: List[Dict[str, object]] = []
    levels = {str(x).strip().lower() for x in selection_levels}
    scoped = overall[overall["level"].astype(str).str.lower().isin(levels)].copy()
    if scoped.empty:
        return pd.DataFrame()
    for key_vals, g in scoped.groupby(key_cols, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        key_map = dict(zip(key_cols, key_vals))
        ranked_all = g.sort_values(
            ["worst_year_mean_ret", "worst_year_good_rate", "mean_ret", "good_rate", "horizon_h"],
            ascending=[False, False, False, False, True],
            na_position="last",
        ).reset_index(drop=True)
        ranked_all["eligible_global"] = ranked_all["n_valid"].astype(float) >= float(min_n_valid_global)
        ranked_all["eligible_per_symbol_year"] = (
            pd.to_numeric(ranked_all.get("eligible_year_count"), errors="coerce")
            .fillna(0.0)
            .astype(float)
            > 0.0
        )
        ranked_all["selection_eligible"] = (
            ranked_all["eligible_global"] & ranked_all["eligible_per_symbol_year"]
        )
        eligible = ranked_all[ranked_all["selection_eligible"]].copy().reset_index(drop=True)
        if eligible.empty:
            rows.append(
                {
                    **key_map,
                    "selected_horizon_h": pd.NA,
                    "worst_year_mean_ret": np.nan,
                    "worst_year_good_rate": np.nan,
                    "worst_year_up_rate": np.nan,
                    "overall_mean_ret": np.nan,
                    "overall_good_rate": np.nan,
                    "overall_up_rate": np.nan,
                    "selection_rank": pd.NA,
                    "selection_candidates": int(len(ranked_all)),
                    "selection_eligible_candidates": 0,
                    "min_n_valid_global": int(min_n_valid_global),
                    "min_n_valid_per_symbol_year": int(min_n_valid_per_symbol_year),
                    "selection_status": "no_eligible_rows",
                    "selection_primary": "max worst_year_mean_ret",
                    "selection_secondary": "max worst_year_good_rate",
                    "selection_tertiary": "max overall_mean_ret",
                    "selection_quaternary": "max overall_good_rate",
                    "selection_rule": (
                        "Eligible rows only; sort DESC by worst_year_mean_ret, worst_year_good_rate, "
                        "overall_mean_ret, overall_good_rate; final tie: smallest horizon."
                    ),
                }
            )
            continue
        ranked = _prioritize_prefer_horizon(
            ranked=eligible.sort_values(
                ["worst_year_mean_ret", "worst_year_good_rate", "mean_ret", "good_rate", "horizon_h"],
                ascending=[False, False, False, False, True],
                na_position="last",
            ).reset_index(drop=True),
            prefer_horizon=prefer_horizon,
        )
        ranked["selection_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
        best = ranked.iloc[0]
        rows.append(
            {
                **key_map,
                "selected_horizon_h": int(best["horizon_h"]),
                "worst_year_mean_ret": float(best["worst_year_mean_ret"])
                if pd.notna(best["worst_year_mean_ret"])
                else np.nan,
                "worst_year_good_rate": float(best["worst_year_good_rate"])
                if pd.notna(best["worst_year_good_rate"])
                else np.nan,
                "worst_year_up_rate": float(best["worst_year_up_rate"])
                if pd.notna(best["worst_year_up_rate"])
                else np.nan,
                "overall_mean_ret": float(best["mean_ret"]) if pd.notna(best["mean_ret"]) else np.nan,
                "overall_good_rate": float(best["good_rate"]) if pd.notna(best["good_rate"]) else np.nan,
                "overall_up_rate": float(best["up_rate"]) if pd.notna(best["up_rate"]) else np.nan,
                "selection_rank": int(best["selection_rank"]),
                "selection_candidates": int(len(ranked_all)),
                "selection_eligible_candidates": int(len(ranked)),
                "min_n_valid_global": int(min_n_valid_global),
                "min_n_valid_per_symbol_year": int(min_n_valid_per_symbol_year),
                "selection_status": "selected",
                "selection_primary": "max worst_year_mean_ret",
                "selection_secondary": "max worst_year_good_rate",
                "selection_tertiary": "max overall_mean_ret",
                "selection_quaternary": "max overall_good_rate",
                "selection_rule": (
                    "Eligible rows only; sort DESC by worst_year_mean_ret, worst_year_good_rate, "
                    "overall_mean_ret, overall_good_rate; final tie: smallest horizon."
                ),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["selected_horizon_h"] = pd.to_numeric(out["selected_horizon_h"], errors="coerce").astype("Int64")
    out["selection_rank"] = pd.to_numeric(out["selection_rank"], errors="coerce").astype("Int64")
    for col in [
        "worst_year_mean_ret",
        "worst_year_good_rate",
        "worst_year_up_rate",
        "overall_mean_ret",
        "overall_good_rate",
        "overall_up_rate",
    ]:
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
    cols = [c for c in columns if c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df[cols].iterrows():
        lines.append("| " + " | ".join(_fmt_md_cell(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_master_outputs(
    overall: pd.DataFrame,
    per_symbol_year: pd.DataFrame,
    best: pd.DataFrame,
    run_logs: pd.DataFrame,
    outdir: Path,
    min_n_valid_global: int = 200,
    min_n_valid_per_symbol_year: int = 10,
) -> Dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    overall_out = overall.copy()
    per_sy_out = per_symbol_year.copy()
    best_out = best.copy()

    for col in MASTER_OVERALL_COLUMNS:
        if col not in overall_out.columns:
            overall_out[col] = np.nan
    overall_out = overall_out[MASTER_OVERALL_COLUMNS].copy()
    for col in MASTER_PER_SY_COLUMNS:
        if col not in per_sy_out.columns:
            per_sy_out[col] = np.nan
    per_sy_out = per_sy_out[MASTER_PER_SY_COLUMNS].copy()

    overall_path = outdir / "overall_by_variant_tf_h.csv"
    per_sy_path = outdir / "per_symbol_year_by_variant_tf_h.csv"
    best_path = outdir / "best_horizon_selection.csv"
    run_logs_path = outdir / "run_logs.csv"
    summary_path = outdir / "summary.md"

    overall_out.to_csv(overall_path, index=False)
    per_sy_out.to_csv(per_sy_path, index=False)
    best_out.to_csv(best_path, index=False)
    run_logs.to_csv(run_logs_path, index=False)

    lines: List[str] = []
    lines.append("# DFD05 Master Evaluation")
    lines.append("")
    lines.append(f"- Generated UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Best Horizon By Variant/Timeframe/Session/Level")
    lines.append("")
    lines.append(
        _markdown_table(
            best_out.sort_values(["variant", "timeframe", "session_mode", "level"])
            if not best_out.empty
            else best_out,
            [
                "variant",
                "timeframe",
                "session_mode",
                "level",
                "selected_horizon_h",
                "worst_year_mean_ret",
                "worst_year_good_rate",
                "overall_mean_ret",
                "selection_status",
                "selection_eligible_candidates",
            ],
        )
    )
    lines.append("")
    lines.append("## Run Sample Sizes")
    lines.append("")
    lines.append(
        _markdown_table(
            run_logs.sort_values(["variant", "timeframe", "session_mode"])
            if not run_logs.empty
            else run_logs,
            ["variant", "timeframe", "session_mode", "seconds", "events_n", "forward_n", "labeled_n"],
        )
    )
    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    warn_rows: List[str] = []
    for _, r in overall_out.iterrows():
        if pd.notna(r.get("n_valid")) and int(r["n_valid"]) < int(min_n_valid_global):
            warn_rows.append(
                f"- Low global sample (ineligible): variant={r['variant']} tf={r['timeframe']} "
                f"session={r['session_mode']} level={r['level']} h={int(r['horizon_h'])} "
                f"n_valid={int(r['n_valid'])} < min_n_valid_global={int(min_n_valid_global)}"
            )
        if pd.notna(r.get("eligible_year_count")) and float(r["eligible_year_count"]) <= 0.0:
            warn_rows.append(
                f"- No symbol-year buckets met per-cell minimum (ineligible): variant={r['variant']} "
                f"tf={r['timeframe']} session={r['session_mode']} level={r['level']} h={int(r['horizon_h'])} "
                f"min_n_valid_per_symbol_year={int(min_n_valid_per_symbol_year)}"
            )
        if pd.notna(r.get("trunc_rate")) and float(r["trunc_rate"]) > 0.05:
            warn_rows.append(
                f"- High truncation: variant={r['variant']} tf={r['timeframe']} session={r['session_mode']} "
                f"level={r['level']} h={int(r['horizon_h'])} trunc_rate={float(r['trunc_rate']):.4f}"
            )
    lines.extend(warn_rows if warn_rows else ["- None"])
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append(f"- `{overall_path}`")
    lines.append(f"- `{per_sy_path}`")
    lines.append(f"- `{best_path}`")
    lines.append(f"- `{run_logs_path}`")
    lines.append(f"- `{summary_path}`")
    summary_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    return {
        "overall": overall_path,
        "per_symbol_year": per_sy_path,
        "best_horizon_selection": best_path,
        "run_logs": run_logs_path,
        "summary": summary_path,
    }


def run_master_evaluation(args: argparse.Namespace) -> Dict[str, Path]:
    variants = _parse_variants_csv(args.configs)
    timeframes = _parse_timeframes_csv(args.timeframes)
    horizons = _parse_int_csv(args.horizons, field_name="horizons")
    selection_levels = _parse_selection_levels_csv(args.selection_levels)
    gate_vol_entry_at = _normalized_gate_vol_entry_at(args.gate_vol_entry_at)
    session_on_regions = _parse_session_regions_csv(getattr(args, "session_on_regions", "ny"))
    session_tz = str(getattr(args, "session_tz", "Etc/GMT-3")).strip() or "Etc/GMT-3"
    time_scale = _parse_time_scale_params(args.time_scale_params)
    min_n_valid_global = int(args.min_n_valid_global)
    min_n_valid_per_symbol_year = int(args.min_n_valid_per_symbol_year)
    prefer_horizon = int(args.prefer_horizon) if args.prefer_horizon is not None else None
    if min_n_valid_global < 1:
        raise ValueError("--min_n_valid_global must be >= 1")
    if min_n_valid_per_symbol_year < 1:
        raise ValueError("--min_n_valid_per_symbol_year must be >= 1")
    if prefer_horizon is not None and prefer_horizon < 1:
        raise ValueError("--prefer_horizon must be >= 1 when provided")
    for tf in timeframes:
        _validate_horizon_bar_conversion(tf, horizons)

    session_modes = ["session_on", "session_off"] if bool(args.compare_sessions) else ["config_default"]
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    overall_rows: List[pd.DataFrame] = []
    per_sy_rows: List[pd.DataFrame] = []
    run_logs: List[Dict[str, object]] = []

    for variant, config_path in variants:
        cfg_variant = load_config(config_path)
        cfg_variant.forward.percentile_targets = sorted(
            set(list(cfg_variant.forward.percentile_targets) + [0.8])
        )
        cfg_variant.forward.atr_targets.pct_targets = sorted(
            set(list(cfg_variant.forward.atr_targets.pct_targets) + [0.8])
        )
        if cfg_variant.forward.normalized_mode() != "time_only":
            raise SystemExit(f"Variant {variant} config must use forward.mode=time_only")
        if (args.parity or "").strip().lower() == "pine16" and variant == "baseline":
            _validate_parity_or_exit(cfg_variant, args.parity)

        for timeframe in timeframes:
            for session_mode in session_modes:
                cfg_run = copy.deepcopy(cfg_variant)
                cfg_run.timeframe = timeframe
                cfg_run.forward.horizons_hours = list(horizons)
                cfg_run.forward.horizons_hours_legacy = None
                cfg_run.forward.forward_horizons_hours = []
                cfg_run.labels.write_labeled = True
                _set_session_mode(
                    cfg_run,
                    session_mode,
                    session_on_regions=session_on_regions,
                    session_tz=session_tz,
                )
                _apply_gate_vol_entry_at_override(cfg_run, gate_vol_entry_at)
                scaled_params = _apply_time_scaling(
                    cfg_run=cfg_run,
                    cfg_ref=cfg_variant,
                    target_timeframe=timeframe,
                    time_scale=time_scale,
                )

                run_id = f"{run_stamp}_{variant}_{timeframe}_{session_mode}"
                cfg_run.output.run_id = run_id
                if (
                    (args.parity or "").strip().lower() == "pine16"
                    and variant == "baseline"
                    and session_mode == "session_off"
                ):
                    print("[parity] skipped for session_off mode.")

                t0 = perf_counter()
                outputs = run_dfd05(cfg_run, horizons_override=list(horizons), print_columns=False)
                elapsed = perf_counter() - t0

                labeled_path = outputs.get("labeled")
                if labeled_path is None:
                    raise RuntimeError("run() did not produce labeled output. Ensure labels.write_labeled=true.")
                events_path = outputs["events"]
                forward_path = outputs["forward"]

                labeled = pd.read_parquet(labeled_path)
                events_n = len(pd.read_parquet(events_path, columns=["symbol"]))
                forward_n = len(pd.read_parquet(forward_path, columns=["symbol"]))
                labeled_n = len(labeled)
                print(
                    f"[master_eval] variant={variant} timeframe={timeframe} session_mode={session_mode} "
                    f"seconds={elapsed:.2f} events={events_n} forward={forward_n} labeled={labeled_n}"
                )

                q_mfe = cfg_run.forward.normalized_quality_mfe_threshold()
                q_mae = cfg_run.forward.normalized_quality_mae_threshold()
                signal_overall, signal_per_sy = _evaluate_level(
                    labeled=labeled,
                    horizons=horizons,
                    quality_mfe_thr=q_mfe,
                    quality_mae_thr=q_mae,
                    selected_by_h=None,
                )
                signal_overall["level"] = "signal"
                signal_per_sy["level"] = "signal"

                selected = select_executed_trades(
                    events_df=labeled,
                    tf_minutes=timeframe_to_minutes(timeframe),
                    horizons_hours=list(horizons),
                    one_trade_at_a_time=bool(cfg_run.strategy.one_trade_at_a_time),
                    cooldown_bars=int(cfg_run.strategy.normalized_cooldown_bars()),
                )
                exec_overall, exec_per_sy = _evaluate_level(
                    labeled=labeled,
                    horizons=horizons,
                    quality_mfe_thr=q_mfe,
                    quality_mae_thr=q_mae,
                    selected_by_h=selected,
                )
                exec_overall["level"] = "executed"
                exec_per_sy["level"] = "executed"

                for frame in [signal_overall, signal_per_sy, exec_overall, exec_per_sy]:
                    frame["variant"] = variant
                    frame["timeframe"] = timeframe
                    frame["session_mode"] = session_mode

                overall_rows.extend([signal_overall, exec_overall])
                per_sy_rows.extend([signal_per_sy, exec_per_sy])
                run_logs.append(
                    {
                        "variant": variant,
                        "config_path": str(config_path),
                        "timeframe": timeframe,
                        "session_mode": session_mode,
                        "run_id": run_id,
                        "seconds": float(elapsed),
                        "events_n": int(events_n),
                        "forward_n": int(forward_n),
                        "labeled_n": int(labeled_n),
                        "strategy_mode": str(cfg_run.strategy.mode).upper(),
                        "use_bos_confirm": int(bool(cfg_run.strategy.use_bos_confirm)),
                        "bos_atr_buffer": float(cfg_run.strategy.bos_atr_buffer),
                        "max_wait_bars": int(cfg_run.strategy.max_wait_bars),
                        "trade_mode": str(cfg_run.strategy.trade_mode).upper(),
                        "enable_vol_ratio_entry_gate": int(bool(cfg_run.strategy.toggles.enable_vol_ratio_entry_gate)),
                        "entry_vol_ratio_min": float(cfg_run.strategy.toggles.entry_vol_ratio_min),
                        "one_trade_at_a_time": int(bool(cfg_run.strategy.one_trade_at_a_time)),
                        "cooldown_bars": int(cfg_run.strategy.normalized_cooldown_bars()),
                        "gate_vol_entry_at": cfg_run.strategy.normalized_gate_vol_entry_at(),
                        "session_gate_effective_json": json.dumps(effective_session_gate(cfg_run.strategy), sort_keys=True),
                        "session_on_regions_csv": ",".join(session_on_regions),
                        "session_tz": session_tz,
                        "scaled_params_json": json.dumps(scaled_params, sort_keys=True),
                        "events_path": str(events_path),
                        "forward_path": str(forward_path),
                        "labeled_path": str(labeled_path),
                        "dashboard_path": str(outputs["dashboard"]),
                        "truth_label": "UNVERIFIED_PYTHON_APPROXIMATION",
                    }
                )

    overall_raw = pd.concat(overall_rows, ignore_index=True, sort=False) if overall_rows else pd.DataFrame()
    per_sy_raw = pd.concat(per_sy_rows, ignore_index=True, sort=False) if per_sy_rows else pd.DataFrame()
    if overall_raw.empty or per_sy_raw.empty:
        raise SystemExit("No evaluation rows produced.")

    overall = _attach_worst_year_metrics(
        overall_raw,
        per_sy_raw,
        min_n_valid_per_symbol_year=min_n_valid_per_symbol_year,
    )
    overall = _attach_min_symbol_year_n_valid(overall, per_sy_raw)
    for col in ["horizon_h", "n_total", "n_valid"]:
        overall[col] = pd.to_numeric(overall[col], errors="coerce").fillna(0).astype("int64")
    overall["min_symbol_year_n_valid"] = pd.to_numeric(
        overall.get("min_symbol_year_n_valid"), errors="coerce"
    ).astype("float64")
    overall["selection_eligible"] = (
        (overall["n_valid"].astype(float) >= float(min_n_valid_global))
        & (
            pd.to_numeric(overall.get("eligible_year_count"), errors="coerce")
            .fillna(0.0)
            .astype(float)
            > 0.0
        )
    )

    per_sy = per_sy_raw.copy()
    for col in ["horizon_h", "n_total", "n_valid"]:
        per_sy[col] = pd.to_numeric(per_sy[col], errors="coerce").fillna(0).astype("int64")
    per_sy["year"] = pd.to_numeric(per_sy["year"], errors="coerce").astype("Int64")

    for col in [
        "trunc_rate",
        "up_rate",
        "ge_10bps_rate",
        "ge_25bps_rate",
        "ge_50bps_rate",
        "ge_100bps_rate",
        "worked_top_q80_rate",
        "worked_mfeatr_top_q80_rate",
        "good_rate",
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
        "eligible_year_count",
        "eligible_symbol_year_rows",
    ]:
        if col in overall.columns:
            overall[col] = pd.to_numeric(overall[col], errors="coerce").astype("float64")
        if col in per_sy.columns:
            per_sy[col] = pd.to_numeric(per_sy[col], errors="coerce").astype("float64")

    overall = overall.sort_values(
        ["variant", "timeframe", "session_mode", "level", "horizon_h"]
    ).reset_index(drop=True)
    per_sy = per_sy.sort_values(
        ["variant", "timeframe", "session_mode", "level", "horizon_h", "symbol", "year"]
    ).reset_index(drop=True)
    best = _best_horizon_selection(
        overall=overall,
        selection_levels=selection_levels,
        min_n_valid_global=min_n_valid_global,
        min_n_valid_per_symbol_year=min_n_valid_per_symbol_year,
        prefer_horizon=prefer_horizon,
    )
    run_logs_df = pd.DataFrame(run_logs).sort_values(
        ["variant", "timeframe", "session_mode"]
    ).reset_index(drop=True)

    return write_master_outputs(
        overall=overall,
        per_symbol_year=per_sy,
        best=best,
        run_logs=run_logs_df,
        outdir=Path(args.outdir),
        min_n_valid_global=min_n_valid_global,
        min_n_valid_per_symbol_year=min_n_valid_per_symbol_year,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Master DFD05 horizon evaluation across variants, timeframes, sessions, and executed-trade simulation."
    )
    ap.add_argument(
        "--configs",
        default="baseline,prod",
        help="CSV of variants (baseline,gate_only,bos_only,bos_plus_vol,prod) or config file paths.",
    )
    ap.add_argument(
        "--parity",
        required=False,
        choices=["pine16"],
        help="Optional parity check for baseline variant.",
    )
    ap.add_argument("--timeframes", required=True, help="CSV list, e.g. m15,h1,h4")
    ap.add_argument("--horizons", required=True, help="CSV list in hours, e.g. 4,24,72")
    ap.add_argument(
        "--compare_sessions",
        action="store_true",
        help="Run each variant/timeframe with session_on and session_off.",
    )
    ap.add_argument(
        "--time_scale_params",
        default=None,
        help="Optional params scaling, e.g. reference_tf=m15",
    )
    ap.add_argument(
        "--selection_levels",
        default="executed",
        help="CSV of levels eligible for best selection. Defaults to executed.",
    )
    ap.add_argument(
        "--min_n_valid_global",
        type=int,
        default=200,
        help="Minimum n_valid per row to be eligible for best selection.",
    )
    ap.add_argument(
        "--min_n_valid_per_symbol_year",
        type=int,
        default=10,
        help="Minimum n_valid across symbol-year rows to be eligible for best selection.",
    )
    ap.add_argument(
        "--prefer_horizon",
        type=int,
        default=None,
        help="Optional preferred horizon tie-breaker (applied only when metrics are extremely close).",
    )
    ap.add_argument(
        "--gate_vol_entry_at",
        default=None,
        help="Optional override for strategy.gate_vol_entry_at across runs: signal or trigger.",
    )
    ap.add_argument(
        "--session_on_regions",
        "--session_on_regions_csv",
        dest="session_on_regions",
        default="ny",
        help="CSV session regions to enable in session_on mode: ny,london,tokyo,sydney",
    )
    ap.add_argument(
        "--session_tz",
        default="Etc/GMT-3",
        help="Timezone string for session_gate when session_on is enabled.",
    )
    ap.add_argument(
        "--outdir",
        default="data/reports/dfd05_master_eval",
        help="Output directory for master report artifacts.",
    )
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    outputs = run_master_evaluation(args)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
