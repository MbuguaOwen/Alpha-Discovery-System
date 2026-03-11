from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from dfd05.data import load_bars_for_symbol, timeframe_to_minutes
from dfd05.indicators import atr
from dfd05.pine16_config import load_pine16_exact_config, to_legacy_run_config


DEFAULT_SYMBOLS = [
    "XAUUSD",
    "XAGUSD",
    "LIGHTCMDUSD",
    "BRENTCMDUSD",
    "EURJPY",
    "GBPJPY",
    "USDJPY",
]


def _scope_filter_master(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if df.empty:
        return df
    if scope == "all_sessions":
        return df.copy()
    if scope == "london_only":
        return df[df["entry_session_bucket"].astype(str) == "london_only"].copy()
    if scope == "london_or_newyork":
        return df[pd.to_numeric(df["entry_in_london_or_newyork"], errors="coerce") == 1].copy()
    return df.iloc[0:0].copy()


def _enrich_r_stats_from_master(
    master_path: Path,
    config_path: Path,
    timeframe: str,
    horizons_h: Iterable[int],
) -> pd.DataFrame:
    if not master_path.exists():
        return pd.DataFrame(columns=["symbol", "analysis_session_scope", "horizon_h", "mean_actual_return_r", "median_actual_return_r"])
    master = pd.read_parquet(master_path)
    if master.empty:
        return pd.DataFrame(columns=["symbol", "analysis_session_scope", "horizon_h", "mean_actual_return_r", "median_actual_return_r"])

    cfg = load_pine16_exact_config(config_path)
    cfg.timeframe = str(timeframe).strip().lower()
    legacy_cfg = to_legacy_run_config(cfg)
    tf_minutes = timeframe_to_minutes(cfg.timeframe)

    work = master.copy()
    work["entry_time"] = pd.to_datetime(work["entry_time"], utc=True, errors="coerce")
    work["entry_price"] = pd.to_numeric(work["entry_price"], errors="coerce")
    work["stop_price"] = pd.to_numeric(work.get("stop_price"), errors="coerce")
    work["risk_distance"] = pd.to_numeric(work.get("risk_distance"), errors="coerce")
    work = work.dropna(subset=["symbol", "entry_time", "entry_price"])
    if work.empty:
        return pd.DataFrame(columns=["symbol", "analysis_session_scope", "horizon_h", "mean_actual_return_r", "median_actual_return_r"])

    parts: List[pd.DataFrame] = []
    use_h = [int(h) for h in horizons_h]
    for symbol, g in work.groupby("symbol", dropna=False, sort=True):
        try:
            bars = load_bars_for_symbol(legacy_cfg, symbol=str(symbol), timeframe=cfg.timeframe)
        except FileNotFoundError:
            continue
        if bars.empty:
            continue
        times = pd.to_datetime(bars["time"], utc=True)
        high = bars["high"].to_numpy(dtype=float)
        low = bars["low"].to_numpy(dtype=float)
        close = bars["close"].to_numpy(dtype=float)
        atr_arr = atr(high, low, close, int(cfg.risk.atrLen))
        idx = pd.Index(times).get_indexer(
            pd.to_datetime(g["entry_time"], utc=True, errors="coerce"),
            method="nearest",
            tolerance=pd.Timedelta(minutes=tf_minutes),
        )
        gg = g.reset_index(drop=True).copy()
        for j in range(len(gg)):
            ei = int(idx[j]) if j < len(idx) else -1
            if ei < 0 or ei >= len(close):
                continue
            ep = float(pd.to_numeric(gg.at[j, "entry_price"], errors="coerce"))
            stop_price = float(pd.to_numeric(gg.at[j, "stop_price"], errors="coerce")) if "stop_price" in gg.columns else np.nan
            risk_distance = float(pd.to_numeric(gg.at[j, "risk_distance"], errors="coerce")) if "risk_distance" in gg.columns else np.nan
            if np.isfinite(stop_price) and (not np.isfinite(risk_distance) or risk_distance <= 0.0):
                risk_distance = float(abs(ep - stop_price))
            if not np.isfinite(risk_distance) or risk_distance <= 0.0:
                atr_entry = float(atr_arr[ei]) if ei < len(atr_arr) else np.nan
                if np.isfinite(atr_entry) and atr_entry > 0.0:
                    risk_distance = float(cfg.risk.slAtrMult) * float(atr_entry)
            if not np.isfinite(risk_distance) or risk_distance <= 0.0:
                continue
            base = gg.iloc[[j]].copy()
            base["risk_distance"] = risk_distance
            for h in use_h:
                close_col = f"close_{int(h)}h"
                if close_col not in base.columns:
                    continue
                close_h = float(pd.to_numeric(base.iloc[0][close_col], errors="coerce"))
                if not np.isfinite(close_h):
                    continue
                tmp = base.copy()
                tmp["horizon_h"] = int(h)
                tmp["actual_return_r"] = float((close_h - ep) / risk_distance)
                parts.append(tmp[["symbol", "entry_session_bucket", "entry_in_london_or_newyork", "horizon_h", "actual_return_r"]])
    stacked = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    if stacked.empty:
        return pd.DataFrame(columns=["symbol", "analysis_session_scope", "horizon_h", "mean_actual_return_r", "median_actual_return_r"])

    rows: List[Dict[str, object]] = []
    for scope in ("all_sessions", "london_only", "london_or_newyork"):
        scoped = _scope_filter_master(stacked, scope)
        if scoped.empty:
            continue
        grp = (
            scoped.groupby(["symbol", "horizon_h"], dropna=False)["actual_return_r"]
            .agg(["mean", "median"])
            .reset_index()
            .rename(columns={"mean": "mean_actual_return_r", "median": "median_actual_return_r"})
        )
        grp["analysis_session_scope"] = scope
        rows.append(grp)
    out = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    if out.empty:
        return pd.DataFrame(columns=["symbol", "analysis_session_scope", "horizon_h", "mean_actual_return_r", "median_actual_return_r"])
    return out[["symbol", "analysis_session_scope", "horizon_h", "mean_actual_return_r", "median_actual_return_r"]]


def _action_24_336(w24: float, m24: float, w336: float, m336: float) -> str:
    keep = (
        (np.isfinite(w24) and w24 > 0.52 and np.isfinite(m24) and m24 > 0.0)
        or (np.isfinite(w336) and w336 > 0.52 and np.isfinite(m336) and m336 > 0.0)
    )
    cut = (
        (not np.isfinite(w24) or w24 < 0.50)
        and (not np.isfinite(w336) or w336 < 0.50)
        and (not np.isfinite(m24) or m24 <= 0.0)
        and (not np.isfinite(m336) or m336 <= 0.0)
    )
    if keep:
        return "KEEP"
    if cut:
        return "CUT"
    return "WATCH"


def _build_summary(
    by_symbol_session: pd.DataFrame,
    symbols: Iterable[str],
    timeframe: str,
    scope: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    use = by_symbol_session[
        (by_symbol_session["analysis_session_scope"].astype(str) == scope)
        & (by_symbol_session["symbol"].astype(str).isin(list(symbols)))
    ].copy()
    for symbol in symbols:
        sub = use[use["symbol"].astype(str) == str(symbol)]
        row: Dict[str, object] = {
            "timeframe": str(timeframe),
            "analysis_session_scope": str(scope),
            "symbol": str(symbol),
            "truth_label_block": (sub["truth_label_block"].iloc[0] if not sub.empty else "UNVERIFIED_PYTHON_APPROXIMATION"),
        }
        for h in (24, 168, 336):
            hs = sub[pd.to_numeric(sub["horizon_h"], errors="coerce") == int(h)]
            row[f"n_{h}h"] = int(pd.to_numeric(hs["n_signals"], errors="coerce").iloc[0]) if not hs.empty else 0
            row[f"win_rate_{h}h"] = float(pd.to_numeric(hs["win_rate"], errors="coerce").iloc[0]) if not hs.empty else np.nan
            row[f"mean_return_{h}h_pct"] = (
                float(pd.to_numeric(hs["mean_forward_return_pct"], errors="coerce").iloc[0]) if not hs.empty else np.nan
            )
            row[f"median_return_{h}h_pct"] = (
                float(pd.to_numeric(hs["median_forward_return_pct"], errors="coerce").iloc[0]) if not hs.empty else np.nan
            )
            row[f"mean_return_{h}h_r"] = (
                float(pd.to_numeric(hs["mean_actual_return_r"], errors="coerce").iloc[0])
                if (not hs.empty and "mean_actual_return_r" in hs.columns)
                else np.nan
            )
            row[f"median_return_{h}h_r"] = (
                float(pd.to_numeric(hs["median_actual_return_r"], errors="coerce").iloc[0])
                if (not hs.empty and "median_actual_return_r" in hs.columns)
                else np.nan
            )
        row["session_action_24h_336h"] = _action_24_336(
            float(row["win_rate_24h"]),
            float(row["mean_return_24h_pct"]),
            float(row["win_rate_336h"]),
            float(row["mean_return_336h_pct"]),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _build_changes(summary_all: pd.DataFrame, summary_lny: pd.DataFrame) -> pd.DataFrame:
    rank = {"CUT": 0, "WATCH": 1, "KEEP": 2}
    rows: List[Dict[str, object]] = []
    tfs = sorted(set(summary_all["timeframe"].astype(str)) & set(summary_lny["timeframe"].astype(str)))
    for tf in tfs:
        a_tf = summary_all[summary_all["timeframe"].astype(str) == str(tf)].copy()
        l_tf = summary_lny[summary_lny["timeframe"].astype(str) == str(tf)].copy()
        amap = {str(r["symbol"]): r for _, r in a_tf.iterrows()}
        lmap = {str(r["symbol"]): r for _, r in l_tf.iterrows()}
        for symbol in sorted(set(amap.keys()) & set(lmap.keys())):
            a = amap[symbol]
            l = lmap[symbol]
            a_act = str(a["session_action_24h_336h"])
            l_act = str(l["session_action_24h_336h"])
            delta = int(rank.get(l_act, 0) - rank.get(a_act, 0))
            change = "UNCHANGED"
            if delta > 0:
                change = "UPGRADE"
            elif delta < 0:
                change = "DOWNGRADE"
            rows.append(
                {
                    "timeframe": str(tf),
                    "symbol": symbol,
                    "action_all_sessions_24h_336h": a_act,
                    "action_london_or_newyork_24h_336h": l_act,
                    "change_type": change,
                    "delta_rank": delta,
                    "all_336h_win_rate": float(a["win_rate_336h"]),
                    "london_ny_336h_win_rate": float(l["win_rate_336h"]),
                    "all_336h_mean_return_pct": float(a["mean_return_336h_pct"]),
                    "london_ny_336h_mean_return_pct": float(l["mean_return_336h_pct"]),
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["timeframe", "delta_rank", "symbol"], kind="mergesort").reset_index(drop=True)


def _build_matrix(summary_lny: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    syms = sorted(summary_lny["symbol"].astype(str).unique().tolist())
    for symbol in syms:
        s15 = summary_lny[(summary_lny["timeframe"] == "m15") & (summary_lny["symbol"] == symbol)]
        s30 = summary_lny[(summary_lny["timeframe"] == "m30") & (summary_lny["symbol"] == symbol)]
        if s15.empty or s30.empty:
            continue
        r15 = s15.iloc[0]
        r30 = s30.iloc[0]
        a15 = str(r15["session_action_24h_336h"])
        a30 = str(r30["session_action_24h_336h"])
        allow15 = bool(a15 == "KEEP")
        allow30 = bool(a30 == "KEEP")
        tier = "C"
        if allow15 and allow30:
            tier = "A"
        elif allow15 or allow30:
            tier = "B"
        rows.append(
            {
                "symbol": symbol,
                "session_restriction": "london_or_newyork",
                "m15_action_24h_336h": a15,
                "m30_action_24h_336h": a30,
                "m15_allowed": allow15,
                "m30_allowed": allow30,
                "confidence_tier": tier,
                "m15_win_rate_336h": float(r15["win_rate_336h"]),
                "m30_win_rate_336h": float(r30["win_rate_336h"]),
                "m15_mean_return_336h_pct": float(r15["mean_return_336h_pct"]),
                "m30_mean_return_336h_pct": float(r30["mean_return_336h_pct"]),
                "m15_win_rate_168h": float(r15["win_rate_168h"]),
                "m30_win_rate_168h": float(r30["win_rate_168h"]),
                "m15_mean_return_168h_r": float(r15["mean_return_168h_r"]),
                "m30_mean_return_168h_r": float(r30["mean_return_168h_r"]),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["confidence_tier", "symbol"], kind="mergesort").reset_index(drop=True)


def _rank_desc(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").rank(method="min", ascending=False)


def _build_combined_rank(summary_lny: pd.DataFrame, horizon_h: int = 168) -> pd.DataFrame:
    n_col = f"n_{int(horizon_h)}h"
    wr_col = f"win_rate_{int(horizon_h)}h"
    perf_r_col = f"mean_return_{int(horizon_h)}h_r"
    perf_pct_col = f"mean_return_{int(horizon_h)}h_pct"
    cols = [
        "timeframe",
        "analysis_session_scope",
        "symbol",
        "truth_label_block",
        n_col,
        wr_col,
        perf_r_col,
        perf_pct_col,
    ]
    work = summary_lny[cols].copy()
    work[n_col] = pd.to_numeric(work[n_col], errors="coerce")
    work[wr_col] = pd.to_numeric(work[wr_col], errors="coerce")
    work[perf_r_col] = pd.to_numeric(work[perf_r_col], errors="coerce")
    work[perf_pct_col] = pd.to_numeric(work[perf_pct_col], errors="coerce")
    work = work[work[n_col] > 0].copy()
    if work.empty:
        return pd.DataFrame(
            columns=cols
            + [
                "cell_id",
                "win_rate_rank",
                "performance_rank",
                "combined_rank",
                "action_168h_combined_60pct",
            ]
        )
    work["cell_id"] = work["timeframe"].astype(str) + ":" + work["symbol"].astype(str)
    work["win_rate_rank"] = _rank_desc(work[wr_col])
    work["performance_rank"] = _rank_desc(work[perf_r_col])
    work["combined_rank"] = (work["win_rate_rank"] + work["performance_rank"]) / 2.0
    return work.sort_values(
        ["combined_rank", wr_col, perf_r_col, n_col, "timeframe", "symbol"],
        ascending=[True, False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_keep_drop_to_target(
    combined_rank: pd.DataFrame,
    horizon_h: int = 168,
    target_win_rate: float = 0.60,
) -> pd.DataFrame:
    n_col = f"n_{int(horizon_h)}h"
    wr_col = f"win_rate_{int(horizon_h)}h"
    perf_r_col = f"mean_return_{int(horizon_h)}h_r"
    if combined_rank.empty:
        return pd.DataFrame(
            columns=[
                "keep_top_n",
                "retained_signals",
                f"retained_weighted_win_rate_{int(horizon_h)}h",
                f"retained_weighted_mean_return_{int(horizon_h)}h_r",
                "target_hit",
                "keep_cells",
                "drop_cells",
            ]
        )
    rows: List[Dict[str, object]] = []
    work = combined_rank.reset_index(drop=True).copy()
    for keep_top_n in range(1, len(work) + 1):
        kept = work.head(keep_top_n).copy()
        dropped = work.iloc[keep_top_n:].copy()
        retained_n = float(pd.to_numeric(kept[n_col], errors="coerce").sum())
        weighted_wr = np.nan
        weighted_mean_r = np.nan
        if retained_n > 0:
            weighted_wr = float((pd.to_numeric(kept[n_col], errors="coerce") * pd.to_numeric(kept[wr_col], errors="coerce")).sum() / retained_n)
            weighted_mean_r = float((pd.to_numeric(kept[n_col], errors="coerce") * pd.to_numeric(kept[perf_r_col], errors="coerce")).sum() / retained_n)
        rows.append(
            {
                "keep_top_n": int(keep_top_n),
                "retained_signals": int(retained_n),
                f"retained_weighted_win_rate_{int(horizon_h)}h": weighted_wr,
                f"retained_weighted_mean_return_{int(horizon_h)}h_r": weighted_mean_r,
                "target_hit": bool(np.isfinite(weighted_wr) and weighted_wr >= float(target_win_rate)),
                "keep_cells": "; ".join(kept["cell_id"].astype(str).tolist()),
                "drop_cells": "; ".join(dropped["cell_id"].astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build London+NY validation tables (24h/168h/336h) from forward-sign outputs.")
    ap.add_argument(
        "--m15-dir",
        default="outputs/reports_watchlist_m15_allsessions_24h_168h_336h",
        help="Directory containing m15 forward-sign outputs.",
    )
    ap.add_argument(
        "--m30-dir",
        default="outputs/reports_watchlist_m30_allsessions_24h_168h_336h",
        help="Directory containing m30 forward-sign outputs.",
    )
    ap.add_argument(
        "--output-dir",
        default="outputs/reports",
        help="Directory for validation outputs.",
    )
    ap.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Watchlist symbols in reporting order.",
    )
    ap.add_argument(
        "--config-path",
        default="configs/pine16_forwardsign_m30_watchlist_all_sessions.yaml",
        help="Config used to derive ATR-based risk distance for R-normalized returns.",
    )
    ap.add_argument(
        "--m15-master-path",
        default="data/derived/pine16_exact/forward_sign_24h_168h_336h_master_watchlist_m15_allsessions.parquet",
        help="Forward-sign master parquet for m15.",
    )
    ap.add_argument(
        "--m30-master-path",
        default="data/derived/pine16_exact/forward_sign_24h_168h_336h_master_watchlist_m30_allsessions.parquet",
        help="Forward-sign master parquet for m30.",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = {
        "m15": Path(args.m15_dir),
        "m30": Path(args.m30_dir),
    }
    master_paths = {
        "m15": Path(args.m15_master_path),
        "m30": Path(args.m30_master_path),
    }
    config_path = Path(args.config_path)

    summaries_all: List[pd.DataFrame] = []
    summaries_lny: List[pd.DataFrame] = []
    yearly_parts: List[pd.DataFrame] = []

    for tf, run_dir in runs.items():
        by_symbol_session = pd.read_csv(run_dir / "pine16_forward_sign_24h_168h_336h_by_symbol_session.csv")
        by_symbol_year_session = pd.read_csv(run_dir / "pine16_forward_sign_24h_168h_336h_by_symbol_year_session.csv")
        r_stats = _enrich_r_stats_from_master(master_paths[tf], config_path=config_path, timeframe=tf, horizons_h=(24, 168, 336))
        if not r_stats.empty:
            by_symbol_session = by_symbol_session.drop(columns=[c for c in ("mean_actual_return_r", "median_actual_return_r") if c in by_symbol_session.columns])
            by_symbol_session = by_symbol_session.merge(
                r_stats,
                on=["symbol", "analysis_session_scope", "horizon_h"],
                how="left",
            )

        s_all = _build_summary(by_symbol_session, args.symbols, timeframe=tf, scope="all_sessions")
        s_lny = _build_summary(by_symbol_session, args.symbols, timeframe=tf, scope="london_or_newyork")
        summaries_all.append(s_all)
        summaries_lny.append(s_lny)

        y = by_symbol_year_session[
            (by_symbol_year_session["analysis_session_scope"].astype(str) == "london_or_newyork")
            & (pd.to_numeric(by_symbol_year_session["horizon_h"], errors="coerce") == 336)
            & (by_symbol_year_session["symbol"].astype(str).isin(list(args.symbols)))
        ].copy()
        y.insert(0, "timeframe", tf)
        yearly_parts.append(
            y[
                [
                    "timeframe",
                    "symbol",
                    "year",
                    "n_signals",
                    "win_rate",
                    "mean_forward_return_pct",
                    "median_forward_return_pct",
                    "truth_label_block",
                ]
            ]
        )

    summary_all = pd.concat(summaries_all, ignore_index=True)
    summary_lny = pd.concat(summaries_lny, ignore_index=True)
    summary = summary_lny.sort_values(["timeframe", "symbol"], kind="mergesort").reset_index(drop=True)
    changes = _build_changes(summary_all, summary_lny)
    yearly = pd.concat(yearly_parts, ignore_index=True).sort_values(["timeframe", "symbol", "year"], kind="mergesort")
    matrix = _build_matrix(summary_lny)
    combined_168 = _build_combined_rank(summary_lny, horizon_h=168)
    keep_drop_60 = _build_keep_drop_to_target(combined_168, horizon_h=168, target_win_rate=0.60)
    first_hit = keep_drop_60[keep_drop_60["target_hit"]].head(1) if not keep_drop_60.empty else pd.DataFrame()
    keep_cells_60 = (
        {cell for cell in str(first_hit.iloc[0]["keep_cells"]).split("; ") if cell}
        if not first_hit.empty
        else set()
    )
    if not combined_168.empty:
        combined_168["action_168h_combined_60pct"] = np.where(
            combined_168["cell_id"].isin(keep_cells_60),
            "KEEP",
            np.where(first_hit.empty, "NO_60_PATH", "DROP"),
        )

    summary_csv = out_dir / "pine16_watchlist_london_ny_summary_24h_168h_336h.csv"
    changes_csv = out_dir / "pine16_watchlist_london_ny_action_changes_vs_all_sessions_24h_168h_336h.csv"
    yearly_csv = out_dir / "pine16_watchlist_london_ny_336h_yearly_consistency.csv"
    matrix_csv = out_dir / "pine16_watchlist_execution_matrix_london_ny.csv"
    combined_csv = out_dir / "pine16_watchlist_london_ny_168h_combined_rank.csv"
    keep_drop_60_csv = out_dir / "pine16_watchlist_london_ny_168h_keep_drop_to_60.csv"
    note_md = out_dir / "pine16_watchlist_london_ny_validation_24h_168h_336h.md"

    summary.to_csv(summary_csv, index=False)
    changes.to_csv(changes_csv, index=False)
    yearly.to_csv(yearly_csv, index=False)
    matrix.to_csv(matrix_csv, index=False)
    combined_168.to_csv(combined_csv, index=False)
    keep_drop_60.to_csv(keep_drop_60_csv, index=False)

    m30 = summary[summary["timeframe"] == "m30"][
        [
            "symbol",
            "n_24h",
            "win_rate_24h",
            "mean_return_24h_pct",
            "n_168h",
            "win_rate_168h",
            "mean_return_168h_pct",
            "n_336h",
            "win_rate_336h",
            "mean_return_336h_pct",
            "session_action_24h_336h",
        ]
    ].copy()
    for c in ("win_rate_24h", "win_rate_168h", "win_rate_336h"):
        m30[c] = (pd.to_numeric(m30[c], errors="coerce") * 100.0).round(3)
    for c in ("mean_return_24h_pct", "mean_return_168h_pct", "mean_return_336h_pct"):
        m30[c] = pd.to_numeric(m30[c], errors="coerce").round(3)

    chg = changes[
        [
            "timeframe",
            "symbol",
            "action_all_sessions_24h_336h",
            "action_london_or_newyork_24h_336h",
            "change_type",
            "all_336h_win_rate",
            "london_ny_336h_win_rate",
            "all_336h_mean_return_pct",
            "london_ny_336h_mean_return_pct",
        ]
    ].copy()
    for c in ("all_336h_win_rate", "london_ny_336h_win_rate"):
        chg[c] = (pd.to_numeric(chg[c], errors="coerce") * 100.0).round(3)
    for c in ("all_336h_mean_return_pct", "london_ny_336h_mean_return_pct"):
        chg[c] = pd.to_numeric(chg[c], errors="coerce").round(3)

    rank168 = combined_168[
        [
            "cell_id",
            "n_168h",
            "win_rate_168h",
            "mean_return_168h_r",
            "mean_return_168h_pct",
            "win_rate_rank",
            "performance_rank",
            "combined_rank",
            "action_168h_combined_60pct",
        ]
    ].copy() if not combined_168.empty else pd.DataFrame()
    if not rank168.empty:
        rank168["win_rate_168h"] = (pd.to_numeric(rank168["win_rate_168h"], errors="coerce") * 100.0).round(3)
        rank168["mean_return_168h_r"] = pd.to_numeric(rank168["mean_return_168h_r"], errors="coerce").round(3)
        rank168["mean_return_168h_pct"] = pd.to_numeric(rank168["mean_return_168h_pct"], errors="coerce").round(3)
        rank168["win_rate_rank"] = pd.to_numeric(rank168["win_rate_rank"], errors="coerce").round(1)
        rank168["performance_rank"] = pd.to_numeric(rank168["performance_rank"], errors="coerce").round(1)
        rank168["combined_rank"] = pd.to_numeric(rank168["combined_rank"], errors="coerce").round(2)

    keep60 = keep_drop_60.copy()
    if not keep60.empty:
        keep60["retained_weighted_win_rate_168h"] = (pd.to_numeric(keep60["retained_weighted_win_rate_168h"], errors="coerce") * 100.0).round(3)
        keep60["retained_weighted_mean_return_168h_r"] = pd.to_numeric(keep60["retained_weighted_mean_return_168h_r"], errors="coerce").round(3)

    best_win = combined_168.sort_values(["win_rate_168h", "mean_return_168h_r", "n_168h"], ascending=[False, False, False], kind="mergesort").head(1) if not combined_168.empty else pd.DataFrame()
    best_perf = combined_168.sort_values(["mean_return_168h_r", "win_rate_168h", "n_168h"], ascending=[False, False, False], kind="mergesort").head(1) if not combined_168.empty else pd.DataFrame()
    best_combined = combined_168.head(1) if not combined_168.empty else pd.DataFrame()

    lines = [
        "# Pine16 London+NY Validation (24h/168h/336h)",
        "",
        "Truth stamp: `UNVERIFIED_PYTHON_APPROXIMATION`",
        "",
        "## Core outputs",
        f"- summary_csv: `{summary_csv.as_posix()}`",
        f"- changes_csv: `{changes_csv.as_posix()}`",
        f"- yearly_336_csv: `{yearly_csv.as_posix()}`",
        f"- execution_matrix_csv: `{matrix_csv.as_posix()}`",
        f"- combined_168h_rank_csv: `{combined_csv.as_posix()}`",
        f"- keep_drop_60_csv: `{keep_drop_60_csv.as_posix()}`",
        "",
        "## M30 London+New York quick table (24h / 168h / 336h)",
        m30.to_csv(index=False).strip(),
        "",
        "## Downgrades/Upgrades vs all_sessions (24h/336h action)",
        chg.to_csv(index=False).strip(),
        "",
        "## 168h best win rate / best performance / best combined",
        (
            f"- best_win_rate_cell: `{best_win.iloc[0]['cell_id']}` "
            f"(win_rate_168h={(float(best_win.iloc[0]['win_rate_168h']) * 100.0):.3f}%, "
            f"mean_return_168h_r={float(best_win.iloc[0]['mean_return_168h_r']):.3f}, n={int(best_win.iloc[0]['n_168h'])})"
            if not best_win.empty
            else "- best_win_rate_cell: `unknown`"
        ),
        (
            f"- best_performance_cell: `{best_perf.iloc[0]['cell_id']}` "
            f"(mean_return_168h_r={float(best_perf.iloc[0]['mean_return_168h_r']):.3f}, "
            f"win_rate_168h={(float(best_perf.iloc[0]['win_rate_168h']) * 100.0):.3f}%, n={int(best_perf.iloc[0]['n_168h'])})"
            if not best_perf.empty
            else "- best_performance_cell: `unknown`"
        ),
        (
            f"- best_combined_cell: `{best_combined.iloc[0]['cell_id']}` "
            f"(combined_rank={float(best_combined.iloc[0]['combined_rank']):.2f}, "
            f"win_rate_168h={(float(best_combined.iloc[0]['win_rate_168h']) * 100.0):.3f}%, "
            f"mean_return_168h_r={float(best_combined.iloc[0]['mean_return_168h_r']):.3f}, n={int(best_combined.iloc[0]['n_168h'])})"
            if not best_combined.empty
            else "- best_combined_cell: `unknown`"
        ),
        "",
        "## 168h combined rank (win rate + mean R)",
        rank168.to_csv(index=False).strip() if not rank168.empty else "_No rows._",
        "",
        "## 60% target plan using 168h combined rank",
        (
            f"- first_hit_keep_top_n: `{int(first_hit.iloc[0]['keep_top_n'])}`"
            if not first_hit.empty
            else "- first_hit_keep_top_n: `not reached`"
        ),
        (
            f"- retained_weighted_win_rate_168h: `{(float(first_hit.iloc[0]['retained_weighted_win_rate_168h']) * 100.0):.3f}%`"
            if not first_hit.empty
            else "- retained_weighted_win_rate_168h: `not reached`"
        ),
        (
            f"- retained_weighted_mean_return_168h_r: `{float(first_hit.iloc[0]['retained_weighted_mean_return_168h_r']):.3f}`"
            if not first_hit.empty
            else "- retained_weighted_mean_return_168h_r: `not reached`"
        ),
        (
            f"- keep_cells: `{first_hit.iloc[0]['keep_cells']}`"
            if not first_hit.empty
            else "- keep_cells: `not reached`"
        ),
        (
            f"- drop_cells: `{first_hit.iloc[0]['drop_cells']}`"
            if not first_hit.empty
            else "- drop_cells: `not reached`"
        ),
        keep60.to_csv(index=False).strip() if not keep60.empty else "_No rows._",
        "",
    ]
    note_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"summary_csv: {summary_csv}")
    print(f"changes_csv: {changes_csv}")
    print(f"yearly_336_csv: {yearly_csv}")
    print(f"execution_matrix_csv: {matrix_csv}")
    print(f"combined_168h_rank_csv: {combined_csv}")
    print(f"keep_drop_60_csv: {keep_drop_60_csv}")
    print(f"validation_md: {note_md}")


if __name__ == "__main__":
    main()
