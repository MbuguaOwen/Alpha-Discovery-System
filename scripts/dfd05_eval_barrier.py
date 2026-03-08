from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfd05.config import RunConfig, load_config
from dfd05.data import load_bars_for_symbol, timeframe_to_minutes
from dfd05.forward import compute_forward_for_events
from dfd05.trade_sim import select_executed_trades


VARIANT_PRESETS: Dict[str, str] = {
    "baseline": "configs/dfd05_pine16_baseline.yaml",
    "gate_only": "configs/dfd05_ablate_gate_only.yaml",
    "bos_only": "configs/dfd05_ablate_bos_only.yaml",
    "bos_plus_vol": "configs/dfd05_ablate_bos_plus_vol.yaml",
    "prod": "configs/dfd05_pine16_prod_bos_vol.yaml",
}


def _horizon_suffix(hours: int) -> str:
    return f"{int(hours)}h"


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    if not mask.any():
        return np.nan
    return float(np.average(v[mask], weights=w[mask]))


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return bool(value)
    token = str(value).strip().lower()
    return token in {"1", "true", "yes", "y", "t"}


def _coerce_exec_selection(best: pd.DataFrame) -> pd.DataFrame:
    if best.empty:
        return pd.DataFrame(columns=["variant", "timeframe", "session_mode", "selected_horizon_h"])
    out = best.copy()
    if "level" in out.columns:
        out = out[out["level"].astype(str).str.lower() == "executed"].copy()
    if "selection_status" in out.columns:
        out = out[out["selection_status"].astype(str).str.lower() == "selected"].copy()
    out["selected_horizon_h"] = pd.to_numeric(out.get("selected_horizon_h"), errors="coerce").astype("Int64")
    out = out.dropna(subset=["selected_horizon_h"]).copy()
    out["selected_horizon_h"] = out["selected_horizon_h"].astype("int64")
    keys = ["variant", "timeframe", "session_mode", "selected_horizon_h"]
    for c in keys:
        if c not in out.columns:
            raise ValueError(f"best_selection is missing required column: {c}")
    out = out[keys].drop_duplicates().reset_index(drop=True)
    return out


def _resolve_config_path(variant: str, config_path_raw: object) -> str:
    p = str(config_path_raw or "").strip()
    if p.lower() == "nan":
        p = ""
    if p:
        return p
    preset = VARIANT_PRESETS.get(str(variant).strip().lower())
    if preset is None:
        raise ValueError(f"Could not resolve config_path for variant={variant}")
    return preset


def _prepare_execution_views(
    bars_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    bars = bars_df.copy()
    events = events_df.copy()
    pricing_mode = "mid_ohlc"

    has_bid_exit = {"bid_high", "bid_low"}.issubset(set(bars.columns))
    if has_bid_exit:
        bid_high = pd.to_numeric(bars["bid_high"], errors="coerce").to_numpy(dtype=float)
        bid_low = pd.to_numeric(bars["bid_low"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(bid_high).all() and np.isfinite(bid_low).all():
            bars["high"] = bid_high
            bars["low"] = bid_low
            pricing_mode = "bid_exit"

    has_ask_entry = "ask_close" in bars.columns
    if has_bid_exit and has_ask_entry:
        ask_close = pd.to_numeric(bars["ask_close"], errors="coerce").to_numpy(dtype=float)
        entry_idx = pd.to_numeric(events["entry_index"], errors="coerce").to_numpy(dtype=float)
        entry_price = pd.to_numeric(events["entry_price"], errors="coerce").to_numpy(dtype=float)
        valid_idx = (
            np.isfinite(entry_idx)
            & (entry_idx >= 0.0)
            & (entry_idx < float(len(ask_close)))
        )
        if valid_idx.any():
            pos = np.flatnonzero(valid_idx)
            idx_int = entry_idx[valid_idx].astype(np.int64)
            repl = ask_close[idx_int]
            repl_ok = np.isfinite(repl)
            if repl_ok.any():
                entry_price[pos[repl_ok]] = repl[repl_ok]
                events["entry_price"] = entry_price
                pricing_mode = "bid_ask"
    return bars, events, pricing_mode


def _summarize_barrier(
    barrier_df: pd.DataFrame,
    horizon_h: int,
    rr_mult: float,
) -> Dict[str, float]:
    suffix = _horizon_suffix(horizon_h)
    trunc_col = f"is_truncated_{suffix}"
    tp_col = f"tp_first_resolved_{suffix}" if f"tp_first_resolved_{suffix}" in barrier_df.columns else f"tp_first_{suffix}"
    sl_col = f"sl_first_resolved_{suffix}" if f"sl_first_resolved_{suffix}" in barrier_df.columns else f"sl_first_{suffix}"
    req = [trunc_col, tp_col, sl_col, "symbol", "event_time_ms"]
    missing = [c for c in req if c not in barrier_df.columns]
    if missing:
        raise ValueError(f"Barrier frame is missing required columns for {suffix}: {missing}")

    work = barrier_df.copy()
    work[trunc_col] = pd.to_numeric(work[trunc_col], errors="coerce").fillna(1).astype("int64")
    work = work[work[trunc_col] == 0].copy()
    if work.empty:
        return {
            "n_trades": 0.0,
            "win_rate_1to3": np.nan,
            "avg_R": np.nan,
            "worst_year_win_rate": np.nan,
        }

    tp = pd.to_numeric(work[tp_col], errors="coerce").to_numpy(dtype=float)
    sl = pd.to_numeric(work[sl_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(tp) & np.isfinite(sl)
    work = work.loc[valid].copy()
    tp = tp[valid]
    sl = sl[valid]
    n = int(len(work))
    if n <= 0:
        return {
            "n_trades": 0.0,
            "win_rate_1to3": np.nan,
            "avg_R": np.nan,
            "worst_year_win_rate": np.nan,
        }

    work["_tp"] = tp
    work["_year"] = pd.to_datetime(work["event_time_ms"], unit="ms", utc=True, errors="coerce").dt.year.astype("Int64")
    sy = (
        work.dropna(subset=["_year"])
        .groupby(["symbol", "_year"], dropna=False)["_tp"]
        .agg(n_trades="size", win_rate="mean")
        .reset_index()
    )
    if sy.empty:
        worst_year_win_rate = np.nan
    else:
        yearly = (
            sy.groupby("_year", dropna=False)
            .apply(lambda g: pd.Series({"year_win_rate": _weighted_mean(g["win_rate"], g["n_trades"])}))
            .reset_index()
        )
        worst_year_win_rate = float(pd.to_numeric(yearly["year_win_rate"], errors="coerce").min())

    return {
        "n_trades": float(n),
        "win_rate_1to3": float(np.mean(tp)),
        "avg_R": float(np.mean(float(rr_mult) * tp - sl)),
        "worst_year_win_rate": float(worst_year_win_rate) if np.isfinite(worst_year_win_rate) else np.nan,
    }


def run_barrier_evaluation(args: argparse.Namespace) -> Dict[str, Path]:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_logs_path = Path(args.run_logs)
    best_path = Path(args.best_selection)
    if not run_logs_path.exists():
        raise FileNotFoundError(f"Missing run_logs file: {run_logs_path}")
    if not best_path.exists():
        raise FileNotFoundError(f"Missing best_selection file: {best_path}")

    run_logs = pd.read_csv(run_logs_path)
    best = pd.read_csv(best_path)
    selected = _coerce_exec_selection(best)
    if selected.empty:
        out = pd.DataFrame(
            columns=[
                "variant",
                "timeframe",
                "session_mode",
                "n_trades",
                "win_rate_1to3",
                "avg_R",
                "worst_year_win_rate",
                "selected_horizon_h",
                "pricing_mode",
            ]
        )
        out_path = outdir / "barrier_results.csv"
        out.to_csv(out_path, index=False)
        return {"barrier_results": out_path}

    req_run_cols = [
        "variant",
        "timeframe",
        "session_mode",
        "labeled_path",
        "config_path",
        "one_trade_at_a_time",
        "cooldown_bars",
    ]
    missing_run = [c for c in req_run_cols if c not in run_logs.columns]
    if missing_run:
        raise ValueError(f"run_logs is missing required columns for barrier eval: {missing_run}")

    meta = selected.merge(
        run_logs[req_run_cols],
        on=["variant", "timeframe", "session_mode"],
        how="left",
    )

    rows: List[Dict[str, object]] = []
    warnings: List[str] = []
    cfg_cache: Dict[str, RunConfig] = {}
    bars_cache: Dict[Tuple[str, str, str], pd.DataFrame] = {}
    rr_mult = float(args.rr_mult)
    sl_atr_mult = float(args.sl_atr_mult)

    for _, r in meta.iterrows():
        variant = str(r["variant"])
        timeframe = str(r["timeframe"])
        session_mode = str(r["session_mode"])
        horizon_h = int(pd.to_numeric(r["selected_horizon_h"], errors="coerce"))
        labeled_path_raw = str(r.get("labeled_path", ""))
        if labeled_path_raw.strip().lower() == "nan":
            labeled_path_raw = ""
        labeled_path = Path(labeled_path_raw)
        if not labeled_path.exists():
            warnings.append(
                f"Missing labeled_path for variant={variant}, tf={timeframe}, session={session_mode}: {labeled_path}"
            )
            continue

        config_path = _resolve_config_path(variant=variant, config_path_raw=r.get("config_path", ""))
        if config_path not in cfg_cache:
            cfg_cache[config_path] = load_config(config_path)
        cfg_template = cfg_cache[config_path]
        cfg = copy.deepcopy(cfg_template)

        labeled = pd.read_parquet(labeled_path)
        if labeled.empty:
            rows.append(
                {
                    "variant": variant,
                    "timeframe": timeframe,
                    "session_mode": session_mode,
                    "n_trades": 0,
                    "win_rate_1to3": np.nan,
                    "avg_R": np.nan,
                    "worst_year_win_rate": np.nan,
                    "selected_horizon_h": int(horizon_h),
                    "pricing_mode": "n/a",
                }
            )
            continue

        tf_minutes = timeframe_to_minutes(timeframe)
        cooldown_raw = pd.to_numeric(r.get("cooldown_bars"), errors="coerce")
        cooldown_bars = int(cooldown_raw) if np.isfinite(cooldown_raw) else 0
        selected_map = select_executed_trades(
            events_df=labeled,
            tf_minutes=tf_minutes,
            horizons_hours=[int(horizon_h)],
            one_trade_at_a_time=_to_bool(r.get("one_trade_at_a_time", 0)),
            cooldown_bars=cooldown_bars,
        )
        exec_events = selected_map.get(int(horizon_h), labeled.iloc[0:0].copy())
        if exec_events.empty:
            rows.append(
                {
                    "variant": variant,
                    "timeframe": timeframe,
                    "session_mode": session_mode,
                    "n_trades": 0,
                    "win_rate_1to3": np.nan,
                    "avg_R": np.nan,
                    "worst_year_win_rate": np.nan,
                    "selected_horizon_h": int(horizon_h),
                    "pricing_mode": "n/a",
                }
            )
            continue

        barrier_chunks: List[pd.DataFrame] = []
        pricing_modes: List[str] = []
        for symbol in sorted(exec_events["symbol"].astype(str).unique().tolist()):
            key = (config_path, timeframe, symbol)
            if key not in bars_cache:
                bars_cache[key] = load_bars_for_symbol(cfg, symbol=symbol, timeframe=timeframe)
            bars = bars_cache[key]
            if bars.empty:
                warnings.append(f"No bars found for symbol={symbol}, tf={timeframe}, variant={variant}")
                continue
            sym_events = exec_events[exec_events["symbol"].astype(str) == symbol].copy()
            bars_exec, events_exec, pricing_mode = _prepare_execution_views(bars, sym_events)
            pricing_modes.append(pricing_mode)
            barrier = compute_forward_for_events(
                bars_df=bars_exec,
                events_df=events_exec,
                horizons_hours=[int(horizon_h)],
                tf_minutes=tf_minutes,
                tie_break="sl",
                emit_resolved=True,
                sl_atr_mult=sl_atr_mult,
                rr_mult=rr_mult,
                mode="barrier",
            )
            if not barrier.empty:
                barrier_chunks.append(barrier)

        if not barrier_chunks:
            rows.append(
                {
                    "variant": variant,
                    "timeframe": timeframe,
                    "session_mode": session_mode,
                    "n_trades": 0,
                    "win_rate_1to3": np.nan,
                    "avg_R": np.nan,
                    "worst_year_win_rate": np.nan,
                    "selected_horizon_h": int(horizon_h),
                    "pricing_mode": "n/a",
                }
            )
            continue

        barrier_all = pd.concat(barrier_chunks, ignore_index=True, sort=False)
        summary = _summarize_barrier(
            barrier_df=barrier_all,
            horizon_h=int(horizon_h),
            rr_mult=rr_mult,
        )
        rows.append(
            {
                "variant": variant,
                "timeframe": timeframe,
                "session_mode": session_mode,
                "n_trades": int(summary["n_trades"]),
                "win_rate_1to3": summary["win_rate_1to3"],
                "avg_R": summary["avg_R"],
                "worst_year_win_rate": summary["worst_year_win_rate"],
                "selected_horizon_h": int(horizon_h),
                "pricing_mode": ",".join(sorted(set(pricing_modes))) if pricing_modes else "n/a",
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["n_trades"] = pd.to_numeric(out["n_trades"], errors="coerce").fillna(0).astype("int64")
        for c in ["win_rate_1to3", "avg_R", "worst_year_win_rate"]:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
        out = out.sort_values(["variant", "timeframe", "session_mode"], kind="mergesort").reset_index(drop=True)

    out_path = outdir / "barrier_results.csv"
    out.to_csv(out_path, index=False)
    warnings_path = outdir / "barrier_warnings.txt"
    warnings_path.write_text("\n".join(warnings).rstrip() + ("\n" if warnings else ""), encoding="utf-8")
    return {"barrier_results": out_path, "barrier_warnings": warnings_path}


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate true 1:3 barrier outcomes for selected executed horizons.")
    ap.add_argument("--run_logs", required=True, help="Path to run_logs.csv from ablation/master evaluation.")
    ap.add_argument("--best_selection", required=True, help="Path to best_selection.csv (executed selection rows).")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--sl_atr_mult", type=float, default=1.0, help="SL ATR multiple (1R).")
    ap.add_argument("--rr_mult", type=float, default=3.0, help="TP multiple in R (3R).")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    outputs = run_barrier_evaluation(args)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
