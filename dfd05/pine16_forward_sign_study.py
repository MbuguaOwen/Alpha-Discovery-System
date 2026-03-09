from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .data import load_bars_for_symbol, timeframe_to_minutes
from .pine16_config import Pine16ExactConfig, load_pine16_exact_config, to_legacy_run_config
from .pine16_research import classify_signals, classify_trades, load_truth_datasets
from .pine16_truth import TruthLabel, TruthMode, normalize_truth_mode

DEFAULT_TARGET_HORIZONS_H: Tuple[int, int] = (24, 72)
TARGET_SYMBOLS = {"XAUUSD", "XAGUSD", "EURUSD"}
SESSION_SCOPES = ("all_sessions", "london_only", "london_or_newyork")


@dataclass
class ForwardSignArtifacts:
    audit_md: Path
    master_parquet: Path
    report_md: Path
    report_html: Path
    overall_csv: Path
    by_symbol_csv: Path
    by_session_csv: Path
    by_year_csv: Path
    by_symbol_year_csv: Path
    by_symbol_session_csv: Path
    by_year_session_csv: Path
    by_symbol_year_session_csv: Path
    keep_watch_cut_csv: Path


def _normalize_horizons_hours(horizons_hours: Sequence[int] | None) -> Tuple[int, ...]:
    raw = list(horizons_hours) if horizons_hours is not None else list(DEFAULT_TARGET_HORIZONS_H)
    vals = sorted({int(h) for h in raw if int(h) > 0})
    if not vals:
        vals = list(DEFAULT_TARGET_HORIZONS_H)
    return tuple(vals)


def _horizons_tag(horizons_hours: Sequence[int]) -> str:
    return "_".join(f"{int(h)}h" for h in horizons_hours)


def _empty_master_frame(horizons_hours: Sequence[int]) -> pd.DataFrame:
    cols: List[str] = [
        "signal_id",
        "trade_id",
        "symbol",
        "timeframe",
        "session_scope",
        "year",
        "month",
        "truth_label",
        "config_pack",
        "entry_time",
        "entry_price",
        "entry_session_bucket",
        "setup_session_bucket",
        "entry_in_london_or_newyork",
    ]
    for h in horizons_hours:
        hh = int(h)
        cols.extend(
            [
                f"close_{hh}h",
                f"forward_return_{hh}h_abs",
                f"forward_return_{hh}h_pct",
                f"outcome_{hh}h",
            ]
        )
    cols.extend(["classification_mode", "neutral_band_pct"])
    return pd.DataFrame(columns=cols)


def _md_table(df: pd.DataFrame, cols: Iterable[str]) -> str:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return "_No columns._"
    if df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                vals.append("" if not np.isfinite(float(v)) else f"{float(v):.6f}")
            elif pd.isna(v):
                vals.append("")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write_html(md: str, path: Path) -> None:
    import html

    page = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Pine16 Forward Sign Study</title>"
        "<style>body{font-family:ui-monospace,Consolas,monospace;padding:24px;}pre{white-space:pre-wrap;}</style>"
        f"</head><body><pre>{html.escape(md)}</pre></body></html>"
    )
    path.write_text(page, encoding="utf-8")


def _truth_label_block(vals: Sequence[str]) -> str:
    uniq = {str(v) for v in vals if pd.notna(v)}
    if TruthLabel.UNVERIFIED_PYTHON_APPROXIMATION.value in uniq:
        return TruthLabel.UNVERIFIED_PYTHON_APPROXIMATION.value
    if TruthLabel.VERIFIED_PYTHON_PARITY.value in uniq:
        return TruthLabel.VERIFIED_PYTHON_PARITY.value
    return TruthLabel.EXACT_PINE_EXPORTED.value


def _config_session_scope(cfg: Pine16ExactConfig) -> str:
    if not bool(cfg.session.useSessionGate):
        return "all_sessions"
    if bool(cfg.session.useLondon) and bool(cfg.session.useNY):
        return "london_or_newyork"
    if bool(cfg.session.useLondon) and not bool(cfg.session.useNY):
        return "london_only"
    if bool(cfg.session.useNY) and not bool(cfg.session.useLondon):
        return "newyork_only"
    return "all_sessions"


def _classify_outcome(ret_pct: float, mode: str, neutral_band_pct: float) -> str:
    if not np.isfinite(ret_pct):
        return "MISSING"
    if mode == "neutral_band":
        if abs(float(ret_pct)) <= float(neutral_band_pct):
            return "FLAT"
        return "WIN" if float(ret_pct) > float(neutral_band_pct) else "LOSS"
    if float(ret_pct) > 0.0:
        return "WIN"
    if float(ret_pct) < 0.0:
        return "LOSS"
    return "FLAT"


def _scope_filter(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if df.empty:
        return df
    if scope == "all_sessions":
        return df
    if scope == "london_only":
        return df[df["entry_session_bucket"].astype(str) == "london_only"]
    if scope == "london_or_newyork":
        return df[pd.to_numeric(df["entry_in_london_or_newyork"], errors="coerce") == 1]
    return df.iloc[0:0].copy()


def _expand_scopes(master: pd.DataFrame) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for scope in SESSION_SCOPES:
        tmp = _scope_filter(master, scope).copy()
        if tmp.empty:
            continue
        tmp["analysis_session_scope"] = scope
        parts.append(tmp)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = float(wins) / float(n)
    den = 1.0 + (z * z) / float(n)
    center = (p + (z * z) / (2.0 * float(n))) / den
    half = (z / den) * np.sqrt((p * (1.0 - p) / float(n)) + ((z * z) / (4.0 * float(n) * float(n))))
    return max(0.0, float(center - half)), min(1.0, float(center + half))


def _t_stat_mean(x: pd.Series) -> float:
    v = pd.to_numeric(x, errors="coerce").dropna()
    n = int(v.shape[0])
    if n < 2:
        return np.nan
    sd = float(v.std(ddof=1))
    if not np.isfinite(sd) or sd == 0.0:
        return np.nan
    return float(float(v.mean()) / (sd / np.sqrt(float(n))))


def _aggregate_group(g: pd.DataFrame) -> Dict[str, object]:
    ret = pd.to_numeric(g["forward_return_pct"], errors="coerce").dropna()
    n = int(ret.shape[0])
    if n == 0:
        return {
            "n_signals": 0,
            "win_rate": np.nan,
            "loss_rate": np.nan,
            "flat_rate": np.nan,
            "mean_forward_return_pct": np.nan,
            "median_forward_return_pct": np.nan,
            "std_forward_return_pct": np.nan,
            "p25_forward_return_pct": np.nan,
            "p50_forward_return_pct": np.nan,
            "p75_forward_return_pct": np.nan,
            "pct_positive_return": np.nan,
            "pct_negative_return": np.nan,
            "favorable_move_rate": np.nan,
            "t_stat_mean_return": np.nan,
            "win_rate_ci95_low": np.nan,
            "win_rate_ci95_high": np.nan,
        }
    gv = g.loc[ret.index]
    wins = int((gv["outcome"] == "WIN").sum())
    losses = int((gv["outcome"] == "LOSS").sum())
    flats = int((gv["outcome"] == "FLAT").sum())
    ci_lo, ci_hi = _wilson_ci(wins, n)
    return {
        "n_signals": n,
        "win_rate": float(wins / n),
        "loss_rate": float(losses / n),
        "flat_rate": float(flats / n),
        "mean_forward_return_pct": float(ret.mean()),
        "median_forward_return_pct": float(ret.median()),
        "std_forward_return_pct": float(ret.std(ddof=1)) if n > 1 else np.nan,
        "p25_forward_return_pct": float(ret.quantile(0.25)),
        "p50_forward_return_pct": float(ret.quantile(0.50)),
        "p75_forward_return_pct": float(ret.quantile(0.75)),
        "pct_positive_return": float((ret > 0).mean()),
        "pct_negative_return": float((ret < 0).mean()),
        "favorable_move_rate": float((ret > 0).mean()),
        "t_stat_mean_return": _t_stat_mean(ret),
        "win_rate_ci95_low": ci_lo,
        "win_rate_ci95_high": ci_hi,
    }


def _aggregate(df: pd.DataFrame, keys: Sequence[str], block_truth: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    if keys:
        for k, g in df.groupby(list(keys), dropna=False, sort=True):
            if not isinstance(k, tuple):
                k = (k,)
            row = {c: v for c, v in zip(keys, k)}
            row.update(_aggregate_group(g))
            rows.append(row)
    else:
        rows.append(_aggregate_group(df))
    out = pd.DataFrame(rows)
    out["truth_label_block"] = block_truth
    wr = pd.to_numeric(out["win_rate"], errors="coerce")
    out["win_rate_band"] = np.where(
        wr < 0.50,
        "no directional edge",
        np.where(wr < 0.52, "borderline", np.where(wr <= 0.55, "usable", "strong directional edge")),
    )
    return out


def _build_master_for_config(
    config_path: str,
    truth_mode: TruthMode,
    exact_dir: Path,
    classification_mode: str,
    neutral_band_pct: float,
    horizons_hours: Sequence[int],
    timeframe_override: str | None = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    cfg = load_pine16_exact_config(config_path)
    if timeframe_override is not None and str(timeframe_override).strip():
        cfg.timeframe = str(timeframe_override).strip().lower()
    else:
        cfg.timeframe = str(cfg.timeframe).strip().lower()
    _ = timeframe_to_minutes(cfg.timeframe)
    cfg.symbols = [s for s in cfg.symbols if str(s) in TARGET_SYMBOLS]

    trades, signals, truth_label, parity_metrics, exact_available = load_truth_datasets(cfg, truth_mode, exact_dir)
    if signals.empty and not trades.empty:
        signals = trades[["symbol", "timeframe", "bar_time_utc", "entry_time_utc", "entry_price"]].copy()
    _ = classify_signals(signals, truth_label=truth_label, config_pack=cfg.metadata.config_pack)
    tr = classify_trades(trades, truth_label=truth_label, config_pack=cfg.metadata.config_pack)
    tr = tr[tr["symbol"].astype(str).isin(TARGET_SYMBOLS)].copy() if not tr.empty else tr

    meta = {
        "config_pack": cfg.metadata.config_pack,
        "truth_label": truth_label.value,
        "parity_metrics": parity_metrics,
        "exact_available": bool(exact_available),
        "session_scope": _config_session_scope(cfg),
        "config_path": config_path,
    }
    if tr.empty:
        return _empty_master_frame(horizons_hours), meta

    legacy_cfg = to_legacy_run_config(cfg)
    tf_minutes = timeframe_to_minutes(cfg.timeframe)
    session_scope = _config_session_scope(cfg)
    horizons = sorted({int(h) for h in horizons_hours})
    rows: List[Dict[str, object]] = []

    for symbol in sorted(tr["symbol"].astype(str).unique().tolist()):
        sym_tr = tr[tr["symbol"].astype(str) == symbol].copy()
        if sym_tr.empty:
            continue
        try:
            bars = load_bars_for_symbol(legacy_cfg, symbol=symbol, timeframe=cfg.timeframe)
        except FileNotFoundError:
            continue
        if bars.empty:
            continue

        times = pd.to_datetime(bars["time"], utc=True)
        close = bars["close"].to_numpy(dtype=float)
        idx = pd.Index(times).get_indexer(
            pd.to_datetime(sym_tr["entry_time_utc"], utc=True, errors="coerce"),
            method="nearest",
            tolerance=pd.Timedelta(minutes=tf_minutes),
        )
        sym_tr = sym_tr.reset_index(drop=True)

        for j in range(len(sym_tr)):
            ei = int(idx[j]) if j < len(idx) else -1
            if ei < 0 or ei >= len(close):
                continue
            ep = float(pd.to_numeric(sym_tr.at[j, "entry_price"], errors="coerce"))
            if not np.isfinite(ep):
                ep = float(close[ei])
            if not np.isfinite(ep) or ep == 0:
                continue

            row: Dict[str, object] = {
                "signal_id": str(sym_tr.at[j, "signal_id"]),
                "trade_id": str(sym_tr.at[j, "trade_id"]),
                "symbol": str(symbol),
                "timeframe": str(cfg.timeframe),
                "session_scope": str(session_scope),
                "year": int(sym_tr.at[j, "year"]) if pd.notna(sym_tr.at[j, "year"]) else np.nan,
                "month": int(sym_tr.at[j, "month"]) if pd.notna(sym_tr.at[j, "month"]) else np.nan,
                "truth_label": truth_label.value,
                "config_pack": str(cfg.metadata.config_pack),
                "entry_time": pd.to_datetime(sym_tr.at[j, "entry_time_utc"], utc=True, errors="coerce"),
                "entry_price": float(ep),
                "entry_session_bucket": str(sym_tr.at[j, "entry_session_bucket"]),
                "setup_session_bucket": str(sym_tr.at[j, "setup_session_bucket"]),
                "entry_in_london_or_newyork": int(pd.to_numeric(sym_tr.at[j, "entry_in_london_or_newyork"], errors="coerce")),
            }
            for h in horizons:
                bars_fwd = int((h * 60) // tf_minutes)
                end = ei + bars_fwd
                if bars_fwd <= 0 or end >= len(close):
                    row[f"close_{h}h"] = np.nan
                    row[f"forward_return_{h}h_abs"] = np.nan
                    row[f"forward_return_{h}h_pct"] = np.nan
                    row[f"outcome_{h}h"] = "MISSING"
                    continue
                close_h = float(close[end])
                ret_abs = float(close_h - ep)
                ret_pct = float((ret_abs / ep) * 100.0)
                row[f"close_{h}h"] = close_h
                row[f"forward_return_{h}h_abs"] = ret_abs
                row[f"forward_return_{h}h_pct"] = ret_pct
                row[f"outcome_{h}h"] = _classify_outcome(ret_pct, mode=classification_mode, neutral_band_pct=neutral_band_pct)
            rows.append(row)

    master = pd.DataFrame(rows)
    if master.empty:
        return _empty_master_frame(horizons_hours), meta
    for c in ["year", "month"]:
        master[c] = pd.to_numeric(master[c], errors="coerce").astype("Int64")
    return master, meta


def _stack_horizons(expanded: pd.DataFrame, horizons_hours: Sequence[int]) -> pd.DataFrame:
    if expanded.empty:
        return pd.DataFrame()
    parts: List[pd.DataFrame] = []
    for h in horizons_hours:
        ret_col = f"forward_return_{int(h)}h_pct"
        abs_col = f"forward_return_{int(h)}h_abs"
        close_col = f"close_{int(h)}h"
        out_col = f"outcome_{int(h)}h"
        tmp = expanded[
            [
                "signal_id",
                "trade_id",
                "symbol",
                "timeframe",
                "session_scope",
                "year",
                "month",
                "truth_label",
                "config_pack",
                "entry_time",
                "entry_price",
                "entry_session_bucket",
                "setup_session_bucket",
                "entry_in_london_or_newyork",
                "analysis_session_scope",
                close_col,
                ret_col,
                abs_col,
                out_col,
            ]
        ].copy()
        tmp["horizon_h"] = int(h)
        tmp["close_at_horizon"] = pd.to_numeric(tmp[close_col], errors="coerce")
        tmp["forward_return_pct"] = pd.to_numeric(tmp[ret_col], errors="coerce")
        tmp["forward_return_abs"] = pd.to_numeric(tmp[abs_col], errors="coerce")
        tmp["outcome"] = tmp[out_col].astype(str)
        parts.append(tmp)
    return pd.concat(parts, ignore_index=True, sort=False)


def _keep_watch_cut(
    by_symbol_session: pd.DataFrame,
    min_n: int,
    block_truth: str,
    horizons_hours: Sequence[int],
) -> pd.DataFrame:
    if by_symbol_session.empty:
        return pd.DataFrame()
    work = by_symbol_session.rename(columns={"analysis_session_scope": "session_scope"}).copy()
    horizons = [int(h) for h in horizons_hours]
    h1 = int(horizons[0]) if horizons else 24
    h2 = int(horizons[1]) if len(horizons) >= 2 else h1
    rows = []
    for (symbol, session_scope), g in work.groupby(["symbol", "session_scope"], dropna=False, sort=True):
        r1 = g[g["horizon_h"] == h1].head(1)
        r2 = g[g["horizon_h"] == h2].head(1)
        n1 = int(pd.to_numeric(r1["n_signals"], errors="coerce").iloc[0]) if not r1.empty else 0
        n2 = int(pd.to_numeric(r2["n_signals"], errors="coerce").iloc[0]) if not r2.empty else 0
        w1 = float(pd.to_numeric(r1["win_rate"], errors="coerce").iloc[0]) if not r1.empty else np.nan
        w2 = float(pd.to_numeric(r2["win_rate"], errors="coerce").iloc[0]) if not r2.empty else np.nan
        m1 = float(pd.to_numeric(r1["mean_forward_return_pct"], errors="coerce").iloc[0]) if not r1.empty else np.nan
        m2 = float(pd.to_numeric(r2["mean_forward_return_pct"], errors="coerce").iloc[0]) if not r2.empty else np.nan
        med1 = float(pd.to_numeric(r1["median_forward_return_pct"], errors="coerce").iloc[0]) if not r1.empty else np.nan
        med2 = float(pd.to_numeric(r2["median_forward_return_pct"], errors="coerce").iloc[0]) if not r2.empty else np.nan
        n_ok = max(n1, n2) >= int(min_n)

        action = "WATCH"
        rationale = "mixed or borderline directional profile"
        keep_cond = n_ok and (
            (np.isfinite(w1) and w1 > 0.52 and np.isfinite(m1) and m1 > 0 and np.isfinite(med1) and med1 >= 0)
            or (np.isfinite(w2) and w2 > 0.52 and np.isfinite(m2) and m2 > 0 and np.isfinite(med2) and med2 >= 0)
        )
        cut_cond = n_ok and (
            (not np.isfinite(w1) or w1 < 0.50)
            and (not np.isfinite(w2) or w2 < 0.50)
            and (not np.isfinite(m1) or m1 <= 0)
            and (not np.isfinite(m2) or m2 <= 0)
        )
        if keep_cond:
            action = "KEEP"
            rationale = "win-rate >52% with positive forward returns"
        elif cut_cond:
            action = "CUT"
            rationale = "sub-50% directional profile and weak forward returns"

        rows.append(
            {
                "symbol": str(symbol),
                "session_scope": str(session_scope),
                f"n_{h1}h": n1,
                f"n_{h2}h": n2,
                f"win_rate_{h1}h": w1,
                f"win_rate_{h2}h": w2,
                f"mean_return_{h1}h": m1,
                f"mean_return_{h2}h": m2,
                f"median_return_{h1}h": med1,
                f"median_return_{h2}h": med2,
                "action": action,
                "rationale": rationale,
                "truth_label_block": block_truth,
            }
        )
    return pd.DataFrame(rows).sort_values(["symbol", "session_scope"], kind="mergesort").reset_index(drop=True)


def _pick_best(df: pd.DataFrame, keys: Sequence[str], min_n: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df[pd.to_numeric(df["n_signals"], errors="coerce") >= int(min_n)].copy()
    if work.empty:
        work = df.copy()
    work = work.sort_values(
        ["win_rate", "mean_forward_return_pct", "median_forward_return_pct", "n_signals"],
        ascending=[False, False, False, False],
        kind="mergesort",
    )
    return work[list(keys) + ["horizon_h", "n_signals", "win_rate", "mean_forward_return_pct", "median_forward_return_pct"]].head(1)


def _render_report(
    block_truth: str,
    mode: str,
    neutral_band_pct: float,
    timeframe_label: str,
    horizons_hours: Sequence[int],
    overall: pd.DataFrame,
    by_symbol: pd.DataFrame,
    by_session: pd.DataFrame,
    by_year: pd.DataFrame,
    by_symbol_year: pd.DataFrame,
    by_symbol_session: pd.DataFrame,
    by_year_session: pd.DataFrame,
    keep_watch_cut: pd.DataFrame,
    min_n: int,
) -> str:
    horizons = [int(h) for h in horizons_hours]
    h1 = int(horizons[0]) if horizons else 24
    h2 = int(horizons[1]) if len(horizons) >= 2 else h1
    o1 = overall[overall["horizon_h"] == h1].head(1)
    o2 = overall[overall["horizon_h"] == h2].head(1)
    wr1 = float(pd.to_numeric(o1["win_rate"], errors="coerce").iloc[0]) if not o1.empty else np.nan
    wr2 = float(pd.to_numeric(o2["win_rate"], errors="coerce").iloc[0]) if not o2.empty else np.nan
    best_symbol = _pick_best(by_symbol, ["symbol"], min_n=min_n)
    best_session = _pick_best(by_session, ["analysis_session_scope"], min_n=min_n)

    london_h2 = by_session[(by_session["analysis_session_scope"] == "london_only") & (by_session["horizon_h"] == h2)]
    lonny_h2 = by_session[(by_session["analysis_session_scope"] == "london_or_newyork") & (by_session["horizon_h"] == h2)]
    wl_h2 = float(pd.to_numeric(london_h2["win_rate"], errors="coerce").iloc[0]) if not london_h2.empty else np.nan
    wln_h2 = float(pd.to_numeric(lonny_h2["win_rate"], errors="coerce").iloc[0]) if not lonny_h2.empty else np.nan

    def _best_wr(symbol: str) -> float:
        d = by_symbol[by_symbol["symbol"] == symbol].copy()
        if d.empty:
            return np.nan
        d = d[pd.to_numeric(d["n_signals"], errors="coerce") >= int(min_n)]
        if d.empty:
            d = by_symbol[by_symbol["symbol"] == symbol].copy()
        return float(pd.to_numeric(d["win_rate"], errors="coerce").max())

    xau_best = _best_wr("XAUUSD")
    xag_best = _best_wr("XAGUSD")
    eur_best = _best_wr("EURUSD")
    directional_exists = bool((np.isfinite(wr1) and wr1 > 0.50) or (np.isfinite(wr2) and wr2 > 0.50))

    lines = [
        f"# Pine16 Forward Sign Study ({h1}h/{h2}h)",
        "",
        "## 1. Executive verdict",
        f"- truth_label_block: `{block_truth}`",
        f"- directional_edge_exists: `{directional_exists}`",
        f"- overall_win_rate_{h1}h: `{wr1:.6f}`" if np.isfinite(wr1) else f"- overall_win_rate_{h1}h: `nan`",
        f"- overall_win_rate_{h2}h: `{wr2:.6f}`" if np.isfinite(wr2) else f"- overall_win_rate_{h2}h: `nan`",
        "",
        "## 2. Truth source used",
        f"- truth_label_block: `{block_truth}`",
        "- allowed_truth_labels: `EXACT_PINE_EXPORTED | VERIFIED_PYTHON_PARITY | UNVERIFIED_PYTHON_APPROXIMATION`",
        "",
        "## 3. Study definition",
        f"- timeframe: `{timeframe_label}`",
        f"- horizons: `{h1}h,{h2}h`",
        "- no stop/target economics",
        f"- outcome_mode: `{mode}`",
        f"- neutral_band_pct: `{neutral_band_pct:.6f}`",
        "",
        f"## 4. {h1}h overall results",
        f"- truth_label_block: `{block_truth}`",
        _md_table(overall[overall["horizon_h"] == h1], ["horizon_h", "n_signals", "win_rate", "loss_rate", "flat_rate", "mean_forward_return_pct", "median_forward_return_pct", "win_rate_ci95_low", "win_rate_ci95_high", "win_rate_band"]),
        "",
        f"## 5. {h2}h overall results",
        f"- truth_label_block: `{block_truth}`",
        _md_table(overall[overall["horizon_h"] == h2], ["horizon_h", "n_signals", "win_rate", "loss_rate", "flat_rate", "mean_forward_return_pct", "median_forward_return_pct", "win_rate_ci95_low", "win_rate_ci95_high", "win_rate_band"]),
        "",
        "## 6. By symbol",
        f"- truth_label_block: `{block_truth}`",
        _md_table(by_symbol, ["symbol", "horizon_h", "n_signals", "win_rate", "mean_forward_return_pct", "median_forward_return_pct", "win_rate_band"]),
        "",
        "## 7. By session",
        f"- truth_label_block: `{block_truth}`",
        _md_table(by_session, ["analysis_session_scope", "horizon_h", "n_signals", "win_rate", "mean_forward_return_pct", "median_forward_return_pct", "win_rate_band"]),
        "",
        "## 8. By year",
        f"- truth_label_block: `{block_truth}`",
        _md_table(by_year, ["year", "horizon_h", "n_signals", "win_rate", "mean_forward_return_pct", "median_forward_return_pct"]),
        "",
        "## 9. By symbol x year",
        f"- truth_label_block: `{block_truth}`",
        _md_table(by_symbol_year, ["symbol", "year", "horizon_h", "n_signals", "win_rate", "mean_forward_return_pct"]),
        "",
        "## 10. By symbol x session",
        f"- truth_label_block: `{block_truth}`",
        _md_table(by_symbol_session, ["symbol", "analysis_session_scope", "horizon_h", "n_signals", "win_rate", "mean_forward_return_pct", "median_forward_return_pct"]),
        "",
        f"## 11. {h1}h vs {h2}h comparison",
        f"- truth_label_block: `{block_truth}`",
        f"- overall_win_rate_delta_{h2}h_minus_{h1}h: `{(wr2 - wr1):.6f}`" if np.isfinite(wr1) and np.isfinite(wr2) else f"- overall_win_rate_delta_{h2}h_minus_{h1}h: `nan`",
        f"- directional_improves_{h2}h_vs_{h1}h: `{bool(np.isfinite(wr1) and np.isfinite(wr2) and wr2 > wr1)}`",
        "",
        "## 12. Keep / watch / cut",
        f"- truth_label_block: `{block_truth}`",
        _md_table(
            keep_watch_cut,
            [
                "symbol",
                "session_scope",
                f"n_{h1}h",
                f"n_{h2}h",
                f"win_rate_{h1}h",
                f"win_rate_{h2}h",
                f"mean_return_{h1}h",
                f"mean_return_{h2}h",
                "action",
                "rationale",
            ],
        ),
        "",
        "## 13. Final answer: is there directional edge?",
        f"- truth_label_block: `{block_truth}`",
        f"- directional_edge_exists_over_4y: `{directional_exists}`",
        "",
        "## Final Questions",
        f"1. At {h1}h, is price above entry more than 50% of the time? `{bool(np.isfinite(wr1) and wr1 > 0.50)}`",
        f"2. At {h2}h, is price above entry more than 50% of the time? `{bool(np.isfinite(wr2) and wr2 > 0.50)}`",
        (
            f"3. Which symbol is strongest? `{best_symbol.iloc[0]['symbol']}` @ `{int(best_symbol.iloc[0]['horizon_h'])}h` (win_rate={float(best_symbol.iloc[0]['win_rate']):.6f}, n={int(best_symbol.iloc[0]['n_signals'])})"
            if not best_symbol.empty
            else "3. Which symbol is strongest? `unknown`"
        ),
        (
            f"4. Which session is strongest? `{best_session.iloc[0]['analysis_session_scope']}` @ `{int(best_session.iloc[0]['horizon_h'])}h` (win_rate={float(best_session.iloc[0]['win_rate']):.6f}, n={int(best_session.iloc[0]['n_signals'])})"
            if not best_session.empty
            else "4. Which session is strongest? `unknown`"
        ),
        (
            f"5. Is London-only better than London+NY? `{bool(np.isfinite(wl_h2) and np.isfinite(wln_h2) and wl_h2 > wln_h2)}` ({h2}h london={wl_h2:.6f}, london_or_newyork={wln_h2:.6f})"
            if np.isfinite(wl_h2) and np.isfinite(wln_h2)
            else "5. Is London-only better than London+NY? `unknown`"
        ),
        f"6. Does XAUUSD have directional edge? `{bool(np.isfinite(xau_best) and xau_best > 0.50)}`",
        f"7. Does XAGUSD have directional edge? `{bool(np.isfinite(xag_best) and xag_best > 0.50)}`",
        f"8. Does EURUSD deserve inclusion? `{bool(np.isfinite(eur_best) and eur_best > 0.50)}`",
        f"9. Is there a real 4-year directional edge anywhere? `{directional_exists}`",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _load_forward_sign_overrides(config_path: str) -> Dict[str, object]:
    try:
        from .config import _load_yaml as _legacy_load_yaml  # type: ignore

        raw = _legacy_load_yaml(Path(config_path)) or {}
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    node = raw.get("forward_sign")
    return node if isinstance(node, dict) else {}


def run_forward_sign_study(
    config_paths: Sequence[str],
    truth_mode_raw: str,
    exact_dir: Path,
    output_dir: Path,
    master_path: Path,
    classification_mode: str = "strict_zero",
    neutral_band_pct: float = 0.0,
    horizons_hours: Sequence[int] | None = None,
    timeframe_override: str | None = None,
    min_n: int = 20,
    export_html: bool = True,
    audit_path: Path = Path("outputs/audit_pine16_forward_sign_24h_72h.md"),
) -> ForwardSignArtifacts:
    if not config_paths:
        raise SystemExit("At least one --config path is required.")
    truth_mode = normalize_truth_mode(truth_mode_raw)
    mode = (classification_mode or "strict_zero").strip().lower()
    if mode not in {"strict_zero", "neutral_band"}:
        raise SystemExit("classification_mode must be strict_zero or neutral_band")
    horizons = _normalize_horizons_hours(horizons_hours)

    parts: List[pd.DataFrame] = []
    metas: List[Dict[str, object]] = []
    for cp in config_paths:
        overrides = _load_forward_sign_overrides(cp)
        cp_mode = str(overrides.get("classification_mode", mode)).strip().lower()
        cp_band = float(overrides.get("neutral_band_pct", neutral_band_pct))
        if cp_mode not in {"strict_zero", "neutral_band"}:
            cp_mode = mode
        m, meta = _build_master_for_config(
            cp,
            truth_mode,
            exact_dir,
            cp_mode,
            cp_band,
            horizons,
            timeframe_override=timeframe_override,
        )
        if not m.empty:
            m["classification_mode"] = cp_mode
            m["neutral_band_pct"] = float(cp_band)
            parts.append(m)
        metas.append(meta)

    master = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    block_truth = _truth_label_block(master["truth_label"].astype(str).tolist()) if not master.empty else _truth_label_block([m.get("truth_label", "") for m in metas])
    if master.empty and len(master.columns) == 0:
        master = _empty_master_frame(horizons)

    output_dir.mkdir(parents=True, exist_ok=True)
    master_path.parent.mkdir(parents=True, exist_ok=True)
    if not master.empty:
        master["entry_time"] = pd.to_datetime(master["entry_time"], utc=True, errors="coerce")
        master["entry_price"] = pd.to_numeric(master["entry_price"], errors="coerce")
        # When multiple overlapping configs are passed, keep first-seen signal instances.
        master["_dedup_key"] = (
            master["symbol"].astype(str)
            + "|"
            + master["entry_time"].astype(str)
            + "|"
            + pd.to_numeric(master["entry_price"], errors="coerce").round(10).astype(str)
            + "|"
            + master["truth_label"].astype(str)
        )
        master = master.drop_duplicates(subset=["_dedup_key"], keep="first").drop(columns=["_dedup_key"])
    master.to_parquet(master_path, index=False)

    if not master.empty and "timeframe" in master.columns:
        tf_vals = sorted({str(v) for v in master["timeframe"].dropna().astype(str).tolist() if str(v).strip()})
        timeframe_label = ",".join(tf_vals) if tf_vals else (str(timeframe_override).strip().lower() if timeframe_override else "unknown")
    else:
        timeframe_label = str(timeframe_override).strip().lower() if timeframe_override else "unknown"

    base = master.copy()
    if not base.empty:
        base["analysis_session_scope"] = "all_sessions"
    expanded = _expand_scopes(master)
    base_long = _stack_horizons(base, horizons)
    scoped_long = _stack_horizons(expanded, horizons)

    overall = _aggregate(base_long, ["horizon_h"], block_truth)
    by_symbol = _aggregate(base_long, ["symbol", "horizon_h"], block_truth)
    by_year = _aggregate(base_long, ["year", "horizon_h"], block_truth)
    by_symbol_year = _aggregate(base_long, ["symbol", "year", "horizon_h"], block_truth)
    by_session = _aggregate(scoped_long, ["analysis_session_scope", "horizon_h"], block_truth)
    by_symbol_session = _aggregate(scoped_long, ["symbol", "analysis_session_scope", "horizon_h"], block_truth)
    by_year_session = _aggregate(scoped_long, ["year", "analysis_session_scope", "horizon_h"], block_truth)
    by_symbol_year_session = _aggregate(scoped_long, ["symbol", "year", "analysis_session_scope", "horizon_h"], block_truth)
    keep_watch_cut = _keep_watch_cut(by_symbol_session, min_n=min_n, block_truth=block_truth, horizons_hours=horizons)

    stem = f"pine16_forward_sign_{_horizons_tag(horizons)}"

    paths = {
        "report_md": output_dir / f"{stem}_report.md",
        "report_html": output_dir / f"{stem}_report.html",
        "overall": output_dir / f"{stem}_overall.csv",
        "by_symbol": output_dir / f"{stem}_by_symbol.csv",
        "by_session": output_dir / f"{stem}_by_session.csv",
        "by_year": output_dir / f"{stem}_by_year.csv",
        "by_symbol_year": output_dir / f"{stem}_by_symbol_year.csv",
        "by_symbol_session": output_dir / f"{stem}_by_symbol_session.csv",
        "by_year_session": output_dir / f"{stem}_by_year_session.csv",
        "by_symbol_year_session": output_dir / f"{stem}_by_symbol_year_session.csv",
        "keep_watch_cut": output_dir / f"{stem}_keep_watch_cut.csv",
    }

    overall.to_csv(paths["overall"], index=False)
    by_symbol.to_csv(paths["by_symbol"], index=False)
    by_session.to_csv(paths["by_session"], index=False)
    by_year.to_csv(paths["by_year"], index=False)
    by_symbol_year.to_csv(paths["by_symbol_year"], index=False)
    by_symbol_session.to_csv(paths["by_symbol_session"], index=False)
    by_year_session.to_csv(paths["by_year_session"], index=False)
    by_symbol_year_session.to_csv(paths["by_symbol_year_session"], index=False)
    keep_watch_cut.to_csv(paths["keep_watch_cut"], index=False)

    report_md = _render_report(
        block_truth=block_truth,
        mode=mode,
        neutral_band_pct=neutral_band_pct,
        timeframe_label=timeframe_label,
        horizons_hours=horizons,
        overall=overall,
        by_symbol=by_symbol,
        by_session=by_session,
        by_year=by_year,
        by_symbol_year=by_symbol_year,
        by_symbol_session=by_symbol_session,
        by_year_session=by_year_session,
        keep_watch_cut=keep_watch_cut,
        min_n=min_n,
    )
    paths["report_md"].write_text(report_md, encoding="utf-8")
    if export_html:
        _write_html(report_md, paths["report_html"])
    else:
        paths["report_html"].write_text("", encoding="utf-8")

    return ForwardSignArtifacts(
        audit_md=audit_path,
        master_parquet=master_path,
        report_md=paths["report_md"],
        report_html=paths["report_html"],
        overall_csv=paths["overall"],
        by_symbol_csv=paths["by_symbol"],
        by_session_csv=paths["by_session"],
        by_year_csv=paths["by_year"],
        by_symbol_year_csv=paths["by_symbol_year"],
        by_symbol_session_csv=paths["by_symbol_session"],
        by_year_session_csv=paths["by_year_session"],
        by_symbol_year_session_csv=paths["by_symbol_year_session"],
        keep_watch_cut_csv=paths["keep_watch_cut"],
    )


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Pine16 forward sign study for configurable horizons.")
    ap.add_argument("--config", action="append", required=True, help="Config path; pass multiple times to combine runs.")
    ap.add_argument("--truth-mode", default=TruthMode.VERIFIED_PYTHON_PARITY.value, choices=[m.value for m in TruthMode], help="Truth mode.")
    ap.add_argument("--exact-dir", default="data/derived/pine16_exact", help="Exact/parity cache directory.")
    ap.add_argument("--output-dir", default="outputs/reports", help="Output report directory.")
    ap.add_argument("--master-path", default="data/derived/pine16_exact/forward_sign_24h_72h_master.parquet", help="Master parquet output path.")
    ap.add_argument("--classification-mode", default="strict_zero", choices=["strict_zero", "neutral_band"], help="Outcome mode.")
    ap.add_argument("--neutral-band-pct", type=float, default=0.0, help="Neutral band in percent.")
    ap.add_argument("--horizons-hours", nargs="+", type=int, default=None, help="Forward horizons in hours, e.g. 24 168.")
    ap.add_argument("--timeframe", default=None, help="Optional timeframe override (e.g., m15, m30).")
    ap.add_argument("--min-n", type=int, default=20, help="Minimum n for keep/watch/cut.")
    ap.add_argument("--export-html", action="store_true", help="Write html report.")
    ap.add_argument("--audit-path", default="outputs/audit_pine16_forward_sign_24h_72h.md", help="Audit markdown path.")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    artifacts = run_forward_sign_study(
        config_paths=args.config,
        truth_mode_raw=args.truth_mode,
        exact_dir=Path(args.exact_dir),
        output_dir=Path(args.output_dir),
        master_path=Path(args.master_path),
        classification_mode=str(args.classification_mode),
        neutral_band_pct=float(args.neutral_band_pct),
        horizons_hours=args.horizons_hours,
        timeframe_override=(str(args.timeframe).strip().lower() if args.timeframe else None),
        min_n=int(args.min_n),
        export_html=bool(args.export_html),
        audit_path=Path(args.audit_path),
    )
    print(f"audit_md: {artifacts.audit_md}")
    print(f"master_parquet: {artifacts.master_parquet}")
    print(f"report_md: {artifacts.report_md}")
    print(f"report_html: {artifacts.report_html}")
    print(f"overall_csv: {artifacts.overall_csv}")
    print(f"by_symbol_csv: {artifacts.by_symbol_csv}")
    print(f"by_session_csv: {artifacts.by_session_csv}")
    print(f"by_year_csv: {artifacts.by_year_csv}")
    print(f"by_symbol_year_csv: {artifacts.by_symbol_year_csv}")
    print(f"by_symbol_session_csv: {artifacts.by_symbol_session_csv}")
    print(f"by_year_session_csv: {artifacts.by_year_session_csv}")
    print(f"by_symbol_year_session_csv: {artifacts.by_symbol_year_session_csv}")
    print(f"keep_watch_cut_csv: {artifacts.keep_watch_cut_csv}")


if __name__ == "__main__":
    main()
