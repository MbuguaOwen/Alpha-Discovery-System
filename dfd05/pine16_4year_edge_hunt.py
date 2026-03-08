from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
try:
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier, export_text

    SKLEARN_AVAILABLE = True
except Exception:
    permutation_importance = None
    LogisticRegression = None
    train_test_split = None
    DecisionTreeClassifier = None
    export_text = None
    SKLEARN_AVAILABLE = False

from .pine16_config import load_pine16_exact_config
from .pine16_m30_horizon import build_horizon_master
from .pine16_research import build_feature_event_frame, classify_signals, classify_trades, load_truth_datasets
from .pine16_truth import TruthLabel, TruthMode, normalize_truth_mode

TARGET_SYMBOLS = ["XAUUSD", "XAGUSD", "EURUSD"]
TARGET_SCOPES = ["london_only", "london_or_newyork"]
TARGET_H = [24, 36, 48, 72]
TARGET_YEARS = [2022, 2023, 2024, 2025]


@dataclass
class EdgeHuntArtifacts:
    audit_md: Path
    master_parquet: Path
    report_md: Path
    report_html: Path
    overall_csv: Path
    by_symbol_csv: Path
    by_session_csv: Path
    by_year_csv: Path
    by_symbol_year_csv: Path
    feature_buckets_csv: Path
    top_combos_csv: Path
    robustness_csv: Path
    keep_watch_cut_csv: Path
    deployment_candidates_csv: Path
    deployment_candidates_md: Path
    simple_rules_md: Path
    feature_importance_csv: Path


def _md_table(df: pd.DataFrame, cols: Iterable[str]) -> str:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return "_No columns._"
    if df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df[cols].iterrows():
        row = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                row.append("" if not np.isfinite(float(v)) else f"{float(v):.6f}")
            elif pd.isna(v):
                row.append("")
            else:
                row.append(str(v))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _write_html(md: str, path: Path) -> None:
    import html

    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Pine16 4Y Edge Hunt</title>"
        "<style>body{font-family:ui-monospace,Consolas,monospace;padding:24px;}pre{white-space:pre-wrap;}</style>"
        f"</head><body><pre>{html.escape(md)}</pre></body></html>",
        encoding="utf-8",
    )


def _qcut(series: pd.Series, labels: Sequence[str]) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    m = s.notna()
    if m.sum() < len(labels):
        return pd.Series("missing", index=series.index, dtype="object")
    try:
        out = pd.Series("missing", index=series.index, dtype="object")
        out.loc[m] = pd.qcut(s[m].rank(method="first"), q=len(labels), labels=list(labels)).astype(str)
        return out
    except Exception:
        return pd.Series("missing", index=series.index, dtype="object")


def _scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "london_only":
        return df[df["entry_session_bucket"].astype(str) == "london_only"]
    if scope == "london_or_newyork":
        return df[pd.to_numeric(df["entry_in_london_or_newyork"], errors="coerce") == 1]
    return df.iloc[0:0].copy()


def _expand_scopes(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for s in TARGET_SCOPES:
        t = _scope(df, s).copy()
        if t.empty:
            continue
        t["session_scope"] = s
        parts.append(t)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def _agg(g: pd.DataFrame) -> Dict[str, float]:
    n = len(g)
    if n == 0:
        return {}
    w1 = pd.to_numeric(g["plus1r_before_minus1r"], errors="coerce").fillna(0).sum()
    w2 = pd.to_numeric(g["plus2r_before_minus1r"], errors="coerce").fillna(0).sum()
    w3 = pd.to_numeric(g["plus3r_before_minus1r"], errors="coerce").fillna(0).sum()
    l1 = pd.to_numeric(g["minus1r_before_plus1r"], errors="coerce").fillna(0).sum()
    l2 = pd.to_numeric(g["minus1r_before_plus2r"], errors="coerce").fillna(0).sum()
    l3 = pd.to_numeric(g["minus1r_before_plus3r"], errors="coerce").fillna(0).sum()
    med_mfe = float(pd.to_numeric(g["mfe_r"], errors="coerce").median())
    med_mae = float(pd.to_numeric(g["mae_r"], errors="coerce").median())
    ratio = float(med_mfe / abs(med_mae)) if np.isfinite(med_mae) and med_mae != 0 else np.nan
    e13 = float((3.0 * w3 - l3) / n)
    pf = float((3.0 * w3) / l3) if l3 > 0 else np.nan
    return {
        "n": int(n),
        "n_signals": int(g["signal_id"].nunique()),
        "n_trades": int(g["trade_id"].nunique()),
        "pct_plus1r_before_minus1r": float(w1 / n),
        "pct_plus2r_before_minus1r": float(w2 / n),
        "pct_plus3r_before_minus1r": float(w3 / n),
        "pct_minus1r_before_plus1r": float(l1 / n),
        "pct_minus1r_before_plus2r": float(l2 / n),
        "pct_minus1r_before_plus3r": float(l3 / n),
        "expectancy_1to3_r": e13,
        "profit_factor_1to3": pf,
        "favorable_move_rate": float(pd.to_numeric(g["favorable_move"], errors="coerce").mean()),
        "median_mfe_r": med_mfe,
        "median_mae_r": med_mae,
        "mfe_mae_ratio": ratio,
        "realized_expectancy_r": float(pd.to_numeric(g["realized_result_r"], errors="coerce").mean()),
    }


def _aggregate(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    if not keys:
        rows.append(_agg(df))
    else:
        for k, g in df.groupby(list(keys), dropna=False, sort=True):
            if not isinstance(k, tuple):
                k = (k,)
            r = {c: v for c, v in zip(keys, k)}
            r.update(_agg(g))
            rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["directional_band"] = np.where(
        out["pct_plus1r_before_minus1r"] < 0.50,
        "weak",
        np.where(out["pct_plus1r_before_minus1r"] < 0.52, "borderline", np.where(out["pct_plus1r_before_minus1r"] <= 0.55, "usable", "strong")),
    )
    out["tradable_band"] = np.where(
        out["pct_plus3r_before_minus1r"] < 0.25,
        "not monetizable",
        np.where(out["pct_plus3r_before_minus1r"] < 0.27, "marginal", np.where(out["pct_plus3r_before_minus1r"] <= 0.30, "usable", "strong")),
    )
    out["path_band"] = np.where(
        out["mfe_mae_ratio"] < 1.0, "weak", np.where(out["mfe_mae_ratio"] < 1.1, "borderline", np.where(out["mfe_mae_ratio"] <= 1.25, "decent", "strong"))
    )
    return out


def _truth_label_block(vals: Sequence[str]) -> str:
    u = {str(v) for v in vals if pd.notna(v)}
    if TruthLabel.UNVERIFIED_PYTHON_APPROXIMATION.value in u:
        return TruthLabel.UNVERIFIED_PYTHON_APPROXIMATION.value
    if TruthLabel.VERIFIED_PYTHON_PARITY.value in u:
        return TruthLabel.VERIFIED_PYTHON_PARITY.value
    return TruthLabel.EXACT_PINE_EXPORTED.value


def _robustness_from_group(g: pd.DataFrame, min_n: int, min_n_robust: int) -> Dict[str, float]:
    a = _agg(g)
    y = _aggregate(g, ["year"])
    e = pd.to_numeric(y["expectancy_1to3_r"], errors="coerce") if not y.empty else pd.Series(dtype=float)
    years_present = int(y["year"].nunique()) if not y.empty else 0
    pos_years = int((e > 0).sum()) if not e.empty else 0
    neg_years = int((e <= 0).sum()) if not e.empty else 0
    exp_std = float(e.std(ddof=0)) if not e.empty else np.nan
    conc = np.nan
    if not y.empty:
        net = (pd.to_numeric(y["expectancy_1to3_r"], errors="coerce") * pd.to_numeric(y["n"], errors="coerce")).abs()
        conc = float(net.max() / net.sum()) if net.notna().any() and float(net.sum()) > 0 else np.nan
    n = float(a.get("n", 0.0))
    n_score = np.clip((n - min_n) / max(1.0, max(120, min_n_robust) - min_n), 0, 1)
    yrs_score = np.clip(pos_years / 3.0, 0, 1) if years_present > 0 else 0
    p1_score = np.clip((float(a["pct_plus1r_before_minus1r"]) - 0.52) / 0.08, 0, 1)
    p3_score = np.clip((float(a["pct_plus3r_before_minus1r"]) - 0.27) / 0.08, 0, 1)
    path_score = np.clip((float(a["mfe_mae_ratio"]) - 1.1) / 0.25, 0, 1) if np.isfinite(float(a["mfe_mae_ratio"])) else 0
    std_score = 1 - np.clip(exp_std / 0.20, 0, 1) if np.isfinite(exp_std) else 0
    conc_score = 1 - np.clip(conc, 0, 1) if np.isfinite(conc) else 0.5
    rb = float(0.18 * n_score + 0.20 * yrs_score + 0.15 * p1_score + 0.15 * p3_score + 0.12 * path_score + 0.10 * std_score + 0.10 * conc_score)
    frag = bool(years_present < 4 or pos_years < 2 or (np.isfinite(conc) and conc > 0.70) or (np.isfinite(exp_std) and exp_std > 0.25) or n < min_n)
    return {
        **a,
        "years_present": years_present,
        "positive_years": pos_years,
        "negative_years": neg_years,
        "expectancy_mean": float(e.mean()) if not e.empty else np.nan,
        "expectancy_std": exp_std,
        "concentration_score": conc,
        "robustness_score": rb,
        "fragility_flag": frag,
    }


def _config_master(config_path: str, truth_mode: TruthMode, exact_dir: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    cfg = load_pine16_exact_config(config_path)
    cfg.timeframe = "m30"
    cfg.data.years = list(TARGET_YEARS)
    cfg.symbols = [s for s in cfg.symbols if str(s) in TARGET_SYMBOLS]
    trades, signals, truth_label, parity_metrics, exact_available = load_truth_datasets(cfg, truth_mode, exact_dir)
    if signals.empty and not trades.empty:
        signals = trades[["symbol", "timeframe", "bar_time_utc", "entry_time_utc", "entry_price"]].copy()
    _ = classify_signals(signals, truth_label=truth_label, config_pack=cfg.metadata.config_pack)
    tr = classify_trades(trades, truth_label=truth_label, config_pack=cfg.metadata.config_pack)
    tr = tr[tr["symbol"].astype(str).isin(TARGET_SYMBOLS)].copy() if not tr.empty else tr
    hm = build_horizon_master(trades_cls=tr, cfg=cfg, truth_label=truth_label, horizons_h=TARGET_H)
    if hm.empty:
        return pd.DataFrame(), {"config_pack": cfg.metadata.config_pack, "truth_label": truth_label.value, "exact_available": bool(exact_available), "parity_metrics": parity_metrics, "config_path": config_path}
    feat = build_feature_event_frame(cfg) if truth_label != TruthLabel.EXACT_PINE_EXPORTED else pd.DataFrame()
    keep = [c for c in ["trade_id", "oscChangePct", "barsBetweenPivots", "classicFlag", "locAtPivot", "volRatioAtPivot", "volRatioAtEntry", "atrRatio", "rsi14Pivot", "bosUsed", "bosPassed"] if c in feat.columns]
    if keep:
        hm = hm.merge(feat[keep].drop_duplicates(subset=["trade_id"], keep="last"), on="trade_id", how="left")
    if "bars_held" in tr.columns:
        hm = hm.merge(tr[["trade_id", "bars_held"]].drop_duplicates(subset=["trade_id"], keep="last"), on="trade_id", how="left")
    out = hm.copy()
    out["session_scope"] = out["entry_session_bucket"].astype(str)
    out["bos_used"] = pd.to_numeric(out.get("bosUsed"), errors="coerce").fillna(int(bool(cfg.features.useBOSConfirm))).astype(int)
    out["bos_passed"] = pd.to_numeric(out.get("bosPassed"), errors="coerce")
    out["vol_entry_gate_used"] = int(bool(cfg.features.useVolEntryGate))
    out["classic_only_flag"] = int(bool(cfg.features.classicOnly))
    out["immediate_drawdown_gt_05r"] = pd.to_numeric(out.get("pct_immediate_drawdown_gt_05r_flag"), errors="coerce")
    out["immediate_drawdown_gt_1r"] = pd.to_numeric(out.get("pct_immediate_drawdown_gt_1r_flag"), errors="coerce")
    out["reaches_plus1r_without_minus05r"] = pd.to_numeric(out.get("pct_reaches_plus1r_without_minus05r_flag"), errors="coerce")
    out["reaches_plus3r_without_minus1r"] = pd.to_numeric(out.get("pct_reaches_plus3r_without_minus1r_flag"), errors="coerce")
    out["realized_win"] = (pd.to_numeric(out["realized_result_r"], errors="coerce") > 0).astype(float)
    out["realized_bars_held"] = pd.to_numeric(out.get("bars_held"), errors="coerce")
    out["mfe_mae_ratio_row"] = np.where(pd.to_numeric(out["mae_r"], errors="coerce").abs() > 0, pd.to_numeric(out["mfe_r"], errors="coerce") / pd.to_numeric(out["mae_r"], errors="coerce").abs(), np.nan)
    out = out[out["year"].isin(TARGET_YEARS)].copy()
    return out.reset_index(drop=True), {"config_pack": cfg.metadata.config_pack, "truth_label": truth_label.value, "exact_available": bool(exact_available), "parity_metrics": parity_metrics, "config_path": config_path}


def run_4year_edge_hunt(config_paths: Sequence[str], truth_mode_raw: str, exact_dir: Path, output_dir: Path, master_path: Path, min_n: int = 20, min_n_robust: int = 30, export_html: bool = True, include_feature_importance: bool = True, audit_path: Path = Path("outputs/audit_pine16_4year_edge_hunt.md")) -> EdgeHuntArtifacts:
    truth_mode = normalize_truth_mode(truth_mode_raw)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        "# Pine16 4-Year Edge Hunt Audit\n\n"
        "## 1) M30 research components already present\n"
        "- `dfd05/pine16_m30_horizon.py` provides M30-capable barrier/path simulation, verdict bands, and scope aggregations.\n"
        "- `dfd05/pine16_research.py` provides feature extraction hooks, bucket logic patterns, and interpretability helpers.\n\n"
        "## 2) Reusable from horizon-grid pipeline\n"
        "- Truth routing (`load_truth_datasets`), trade/session classification, and first-touch barrier logic are directly reusable.\n"
        "- Existing CSV/MD/HTML export structure is reusable with focused filenames.\n\n"
        "## 3) Session classification reliability (London / London+NY)\n"
        "- Reliable: Pine UTC+3 windows are already encoded and used consistently.\n"
        "- London: 11:00-19:30 UTC+3, NY: 16:30-23:00 UTC+3; inclusive bounds are preserved.\n\n"
        "## 4) Barrier logic coverage for required metrics\n"
        "- Existing logic already supports +1R/+2R/+3R vs -1R, time-to-threshold, MFE/MAE, and drawdown flags.\n"
        "- Focused edge-hunt master now maps these into required column names.\n\n"
        "## 5) Robustness logic sufficiency\n"
        "- Baseline robustness existed but was not strict enough for this 4-year hunt.\n"
        "- Added strict scoring emphasizing sample, 3+ positive years, concentration penalty, +1R/+3R thresholds, and path quality.\n\n"
        "## 6) Additions made for proper 4-year edge hunt\n"
        "- Dedicated module/CLI, focused configs, focused master dataset, focused report/CSV outputs, combo mining, keep/watch/cut, deployment candidates, and feature importance.\n\n"
        "## 7) Truth limitations remaining\n"
        "- Exact target-symbol exports/parity pass are not currently available in repo artifacts.\n"
        "- Current run truth block is expected to be `UNVERIFIED_PYTHON_APPROXIMATION` unless exact/parity evidence is added.\n\n"
        "## Mapping Table\n"
        "| concept | current implementation | suitability for 4-year edge hunt | action required |\n"
        "| --- | --- | --- | --- |\n"
        "| M30 execution core | `dfd05/pine16_m30_horizon.py` | high | reused with focused horizons |\n"
        "| Feature-state extraction | `build_feature_event_frame` | high | merged into focused master |\n"
        "| Session classification | `pine16_session.py` + classifiers | high | reused as canonical |\n"
        "| Barrier/path metrics | first-touch adverse-first logic | high | reused and renamed to required schema |\n"
        "| Robustness score | prior generic score | medium | replaced with stricter 4-year score |\n"
        "| Truth stamps | truth enums + routing | high | enforced block-level stamping |\n"
        "| Combo mining | prior combo miner | medium | constrained dimensions + sample guards |\n"
        "| Deployment table | prior candidate table | medium | replaced with requested candidate set |\n",
        encoding="utf-8",
    )

    masters, metas = [], []
    for c in config_paths:
        m, meta = _config_master(str(c), truth_mode, exact_dir)
        metas.append(meta)
        if not m.empty:
            masters.append(m)
    master = pd.concat(masters, ignore_index=True, sort=False) if masters else pd.DataFrame()
    if master.empty:
        raise SystemExit("No edge-hunt rows produced for provided configs.")

    master["osc_bucket"] = _qcut(master.get("oscChangePct"), ["weak", "medium", "strong", "extreme"])
    master["pivot_gap_bucket"] = _qcut(master.get("barsBetweenPivots"), ["q1", "q2", "q3", "q4"])
    master["vol_entry_bucket"] = pd.cut(
        pd.to_numeric(master.get("volRatioAtEntry"), errors="coerce"),
        [-np.inf, 1.0, 1.2, 1.5, np.inf],
        labels=["<1.0", "1.0-1.2", "1.2-1.5", ">1.5"],
        right=False,
    ).astype("object").fillna("missing")
    master["vol_pivot_bucket"] = pd.cut(
        pd.to_numeric(master.get("volRatioAtPivot"), errors="coerce"),
        [-np.inf, 1.0, 1.2, 1.5, np.inf],
        labels=["<1.0", "1.0-1.2", "1.2-1.5", ">1.5"],
        right=False,
    ).astype("object").fillna("missing")
    ar = pd.to_numeric(master.get("atrRatio"), errors="coerce")
    q1, q2 = ar.quantile(0.33), ar.quantile(0.66)
    master["atr_ratio_bucket"] = np.where(ar < q1, "low", np.where(ar < q2, "normal", "high"))
    master["loc_bucket"] = np.where(pd.to_numeric(master.get("locAtPivot"), errors="coerce") <= 0.15, "deep-extreme", "moderate-extreme")
    rv = pd.to_numeric(master.get("rsi14Pivot"), errors="coerce")
    master["rsi_bucket"] = np.where(rv < 30, "<30", np.where(rv < 40, "30-40", "40+"))
    expanded = _expand_scopes(master)
    block_truth = _truth_label_block(master["truth_label"].astype(str).unique().tolist())

    overall = _aggregate(master[master["entry_in_london_or_newyork"] == 1], ["horizon_h"])
    by_symbol = _aggregate(expanded, ["symbol", "session_scope", "horizon_h"])
    by_session = _aggregate(expanded, ["session_scope", "horizon_h"])
    by_year = _aggregate(expanded, ["year", "session_scope", "horizon_h"])
    by_symbol_year = _aggregate(expanded, ["symbol", "year", "session_scope", "horizon_h"])

    fb_parts = []
    for b in ["osc_bucket", "pivot_gap_bucket", "vol_entry_bucket", "vol_pivot_bucket", "atr_ratio_bucket", "loc_bucket", "rsi_bucket"]:
        t = _aggregate(expanded, ["session_scope", "horizon_h", b])
        if not t.empty:
            t = t.rename(columns={b: "bucket_value"})
            t["bucket_name"] = b
            fb_parts.append(t)
    feature_buckets = pd.concat(fb_parts, ignore_index=True, sort=False) if fb_parts else pd.DataFrame()

    combo_dims = ["symbol", "session_scope", "horizon_h", "bos_used", "vol_entry_gate_used", "classic_only_flag", "osc_bucket", "pivot_gap_bucket", "atr_ratio_bucket", "loc_bucket", "rsi_bucket"]
    combos = []
    for k, g in expanded.groupby(combo_dims, dropna=False, sort=True):
        if len(g) < min_n:
            continue
        if not isinstance(k, tuple):
            k = (k,)
        r = {c: v for c, v in zip(combo_dims, k)}
        ys = _robustness_from_group(g, min_n, min_n_robust)
        r.update({x: ys.get(x) for x in ["n", "positive_years", "negative_years", "pct_plus1r_before_minus1r", "pct_plus2r_before_minus1r", "pct_plus3r_before_minus1r", "expectancy_1to3_r", "median_mfe_r", "median_mae_r", "mfe_mae_ratio", "robustness_score", "fragility_flag"]})
        r["verdict"] = "KEEP" if (r["n"] >= min_n_robust and r["pct_plus1r_before_minus1r"] >= 0.52 and r["pct_plus3r_before_minus1r"] >= 0.27 and r["expectancy_1to3_r"] > 0 and r["robustness_score"] >= 0.55 and r["positive_years"] >= 3 and not r["fragility_flag"]) else ("CUT" if (r["pct_plus1r_before_minus1r"] < 0.50 or r["pct_plus3r_before_minus1r"] < 0.25 or r["expectancy_1to3_r"] <= 0 or r["robustness_score"] < 0.35) else "WATCH")
        combos.append(r)
    top_combos = pd.DataFrame(combos)
    if not top_combos.empty:
        top_combos = top_combos.sort_values(["robustness_score", "expectancy_1to3_r", "n"], ascending=[False, False, False], kind="mergesort").reset_index(drop=True)
        top_combos.insert(0, "combo_id", [f"EH_C{n+1:04d}" for n in range(len(top_combos))])

    rb_rows = []
    for (s, sc, h), g in expanded.groupby(["symbol", "session_scope", "horizon_h"], dropna=False, sort=True):
        rr = _robustness_from_group(g, min_n, min_n_robust)
        rb_rows.append({
            "candidate_id": f"{s}|{sc}|{int(h)}",
            "symbol": s,
            "session_scope": sc,
            "horizon_h": int(h),
            "n": int(rr["n"]),
            "years_present": int(rr["years_present"]),
            "positive_years": int(rr["positive_years"]),
            "negative_years": int(rr["negative_years"]),
            "expectancy_mean": rr["expectancy_mean"],
            "expectancy_std": rr["expectancy_std"],
            "concentration_score": rr["concentration_score"],
            "pct_plus1r_before_minus1r": rr["pct_plus1r_before_minus1r"],
            "pct_plus3r_before_minus1r": rr["pct_plus3r_before_minus1r"],
            "mfe_mae_ratio": rr["mfe_mae_ratio"],
            "robustness_score": rr["robustness_score"],
            "fragility_flag": rr["fragility_flag"],
        })
    robustness = pd.DataFrame(rb_rows)

    keep_watch_cut = robustness.copy()
    if not keep_watch_cut.empty:
        keep_watch_cut["action"] = "WATCH"
        km = (
            (keep_watch_cut["n"] >= min_n_robust)
            & (keep_watch_cut["positive_years"] >= 3)
            & (keep_watch_cut["expectancy_mean"] > 0)
            & (keep_watch_cut["pct_plus1r_before_minus1r"] >= 0.52)
            & (keep_watch_cut["pct_plus3r_before_minus1r"] >= 0.27)
            & (keep_watch_cut["mfe_mae_ratio"] > 1.1)
            & (keep_watch_cut["robustness_score"] >= 0.55)
            & (~keep_watch_cut["fragility_flag"].astype(bool))
        )
        cm = (
            (keep_watch_cut["n"] >= min_n)
            & (
                (keep_watch_cut["pct_plus1r_before_minus1r"] < 0.50)
                | (keep_watch_cut["pct_plus3r_before_minus1r"] < 0.25)
                | (keep_watch_cut["expectancy_mean"] <= 0)
                | (keep_watch_cut["robustness_score"] < 0.35)
            )
        )
        keep_watch_cut.loc[km, "action"] = "KEEP"
        keep_watch_cut.loc[cm, "action"] = "CUT"
        keep_watch_cut["rationale"] = np.where(
            keep_watch_cut["action"] == "KEEP",
            "usable directional proof + usable 1:3 economics + decent robustness",
            np.where(keep_watch_cut["action"] == "CUT", "weak directional proof/economics/robustness", "incomplete proof"),
        )

    cand_specs = [
        ("XAUUSD / London / 24h", ["XAUUSD"], "london_only", 24),
        ("XAUUSD / London / 36h", ["XAUUSD"], "london_only", 36),
        ("XAUUSD / London / 48h", ["XAUUSD"], "london_only", 48),
        ("XAUUSD / London+NY / 36h", ["XAUUSD"], "london_or_newyork", 36),
        ("XAUUSD / London+NY / 48h", ["XAUUSD"], "london_or_newyork", 48),
        ("XAUUSD + XAGUSD / London+NY / 36h", ["XAUUSD", "XAGUSD"], "london_or_newyork", 36),
        ("XAUUSD + XAGUSD / London+NY / 48h", ["XAUUSD", "XAGUSD"], "london_or_newyork", 48),
        ("XAUUSD + XAGUSD / London / 48h", ["XAUUSD", "XAGUSD"], "london_only", 48),
        ("XAUUSD + XAGUSD + EURUSD / London / 24h", ["XAUUSD", "XAGUSD", "EURUSD"], "london_only", 24),
    ]
    dep = []
    for name, syms, sc, h in cand_specs:
        g = expanded[(expanded["session_scope"] == sc) & (expanded["symbol"].isin(syms)) & (expanded["horizon_h"] == h)]
        if g.empty:
            continue
        rr = _robustness_from_group(g, min_n, min_n_robust)
        verdict = "DEPLOY" if (rr["n"] >= min_n_robust and rr["pct_plus1r_before_minus1r"] >= 0.52 and rr["pct_plus3r_before_minus1r"] >= 0.27 and rr["expectancy_1to3_r"] > 0 and rr["positive_years"] >= 3 and rr["robustness_score"] >= 0.55 and not rr["fragility_flag"]) else ("WATCH" if (rr["n"] >= min_n and rr["pct_plus1r_before_minus1r"] >= 0.50 and rr["pct_plus3r_before_minus1r"] >= 0.25 and rr["robustness_score"] >= 0.40) else "CUT")
        dep.append({
            "candidate": name,
            "symbols": ",".join(syms),
            "session_scope": sc,
            "horizon_h": h,
            "n": int(rr["n"]),
            "pct_plus1r_before_minus1r": rr["pct_plus1r_before_minus1r"],
            "pct_plus2r_before_minus1r": rr["pct_plus2r_before_minus1r"],
            "pct_plus3r_before_minus1r": rr["pct_plus3r_before_minus1r"],
            "expectancy_1to3_r": rr["expectancy_1to3_r"],
            "positive_years": int(rr["positive_years"]),
            "negative_years": int(rr["negative_years"]),
            "robustness_score": rr["robustness_score"],
            "fragility_flag": rr["fragility_flag"],
            "deployment_verdict": verdict,
        })
    deployment = pd.DataFrame(dep)
    if not deployment.empty:
        deployment = deployment.sort_values(["deployment_verdict", "robustness_score", "expectancy_1to3_r"], ascending=[True, False, False], kind="mergesort")

    fi = pd.DataFrame()
    rules: List[str] = []
    if include_feature_importance and not SKLEARN_AVAILABLE:
        rules.append("Feature-importance model skipped: scikit-learn is not installed.")
        if not expanded.empty:
            d0 = expanded[pd.to_numeric(expanded["plus3r_before_minus1r"], errors="coerce").notna()].copy()
            if not d0.empty:
                base0 = float(pd.to_numeric(d0["plus3r_before_minus1r"], errors="coerce").mean())
                rows0: List[Dict[str, object]] = []
                for c0 in ["symbol", "session_scope", "osc_bucket", "pivot_gap_bucket", "atr_ratio_bucket", "loc_bucket", "rsi_bucket"]:
                    g0 = d0.groupby(c0, dropna=False)["plus3r_before_minus1r"].agg(["mean", "count"]).reset_index()
                    g0 = g0[g0["count"] >= min_n]
                    if g0.empty:
                        continue
                    b0 = g0.sort_values("mean", ascending=False).iloc[0]
                    rows0.append(
                        {
                            "method": "categorical_lift_fallback",
                            "feature": f"{c0}={b0[c0]}",
                            "importance": float(b0["mean"] - base0),
                            "support": int(b0["count"]),
                        }
                    )
                    rules.append(
                        f"{c0}={b0[c0]} improves +3R odds to {float(b0['mean']):.4f} (lift {float(b0['mean'] - base0):+.4f}, n={int(b0['count'])})."
                    )
                if rows0:
                    fi = pd.DataFrame(rows0).sort_values(["importance", "support"], ascending=[False, False], kind="mergesort")

    if include_feature_importance and SKLEARN_AVAILABLE and not expanded.empty:
        d = expanded[pd.to_numeric(expanded["plus3r_before_minus1r"], errors="coerce").notna()].copy()
        if not d.empty and pd.to_numeric(d["plus3r_before_minus1r"], errors="coerce").nunique() >= 2:
            y = pd.to_numeric(d["plus3r_before_minus1r"], errors="coerce").astype(int)
            num = d[["horizon_h", "oscChangePct", "barsBetweenPivots", "volRatioAtEntry", "volRatioAtPivot", "atrRatio", "locAtPivot", "rsi14Pivot"]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            cat = d[["symbol", "session_scope", "osc_bucket", "pivot_gap_bucket", "atr_ratio_bucket", "loc_bucket", "rsi_bucket", "bos_used", "vol_entry_gate_used", "classic_only_flag"]].astype("object").fillna("missing")
            X = pd.get_dummies(pd.concat([num, cat], axis=1), drop_first=False, dtype=float)
            if len(X) >= max(min_n, 40):
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y if y.value_counts().min() >= 2 else None)
                lr = LogisticRegression(max_iter=1500, solver="liblinear", class_weight="balanced").fit(Xtr, ytr)
                tr = DecisionTreeClassifier(max_depth=3, min_samples_leaf=max(10, int(0.03 * len(Xtr))), random_state=42).fit(Xtr, ytr)
                rows = []
                for f, v in pd.Series(lr.coef_[0], index=X.columns).abs().sort_values(ascending=False).head(50).items():
                    rows.append({"method": "logistic_abs_coef", "feature": f, "importance": float(v), "support": int(len(Xtr))})
                for f, v in pd.Series(tr.feature_importances_, index=X.columns).sort_values(ascending=False).head(30).items():
                    if float(v) > 0:
                        rows.append({"method": "tree_gini_importance", "feature": f, "importance": float(v), "support": int(len(Xtr))})
                try:
                    pi = permutation_importance(lr, Xte, yte, n_repeats=8, random_state=42, scoring="roc_auc" if yte.nunique() > 1 else "accuracy")
                    for f, v in pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False).head(30).items():
                        rows.append({"method": "logistic_permutation", "feature": f, "importance": float(v), "support": int(len(Xte))})
                except Exception:
                    pass
                fi = pd.DataFrame(rows)
                if not fi.empty:
                    fi = fi.sort_values(["method", "importance"], ascending=[True, False], kind="mergesort")
                base = float(y.mean())
                for c in ["symbol", "session_scope", "osc_bucket", "atr_ratio_bucket", "loc_bucket", "rsi_bucket"]:
                    g = d.groupby(c, dropna=False)["plus3r_before_minus1r"].agg(["mean", "count"]).reset_index()
                    g = g[g["count"] >= min_n]
                    if not g.empty:
                        b = g.sort_values("mean", ascending=False).iloc[0]
                        rules.append(f"{c}={b[c]} improves +3R odds to {float(b['mean']):.4f} (lift {float(b['mean'] - base):+.4f}, n={int(b['count'])}).")
                rules.append("Decision tree (depth<=3):")
                rules.extend([x.rstrip() for x in export_text(tr, feature_names=list(X.columns), max_depth=3).splitlines()[:14]])

    for df in [master, overall, by_symbol, by_session, by_year, by_symbol_year, feature_buckets, top_combos, robustness, keep_watch_cut, deployment, fi]:
        if not df.empty:
            df["truth_label"] = block_truth

    output_dir.mkdir(parents=True, exist_ok=True)
    master_path.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "overall": output_dir / "pine16_4year_edge_hunt_overall.csv",
        "by_symbol": output_dir / "pine16_4year_edge_hunt_by_symbol.csv",
        "by_session": output_dir / "pine16_4year_edge_hunt_by_session.csv",
        "by_year": output_dir / "pine16_4year_edge_hunt_by_year.csv",
        "by_symbol_year": output_dir / "pine16_4year_edge_hunt_by_symbol_year.csv",
        "feature_buckets": output_dir / "pine16_4year_edge_hunt_feature_buckets.csv",
        "top_combos": output_dir / "pine16_4year_edge_hunt_top_combos.csv",
        "robustness": output_dir / "pine16_4year_edge_hunt_robustness.csv",
        "keep_watch_cut": output_dir / "pine16_4year_edge_hunt_keep_watch_cut.csv",
        "deployment": output_dir / "pine16_4year_edge_hunt_deployment_candidates.csv",
        "deployment_md": output_dir / "pine16_4year_edge_hunt_deployment_candidates.md",
        "simple_rules": output_dir / "pine16_4year_edge_hunt_simple_rules.md",
        "fi": output_dir / "pine16_4year_edge_hunt_feature_importance.csv",
        "report_md": output_dir / "pine16_4year_edge_hunt_report.md",
        "report_html": output_dir / "pine16_4year_edge_hunt_report.html",
    }
    master.to_parquet(master_path, index=False)
    overall.to_csv(paths["overall"], index=False)
    by_symbol.to_csv(paths["by_symbol"], index=False)
    by_session.to_csv(paths["by_session"], index=False)
    by_year.to_csv(paths["by_year"], index=False)
    by_symbol_year.to_csv(paths["by_symbol_year"], index=False)
    feature_buckets.to_csv(paths["feature_buckets"], index=False)
    top_combos.to_csv(paths["top_combos"], index=False)
    robustness.to_csv(paths["robustness"], index=False)
    keep_watch_cut.to_csv(paths["keep_watch_cut"], index=False)
    deployment.to_csv(paths["deployment"], index=False)
    fi.to_csv(paths["fi"], index=False)
    paths["deployment_md"].write_text(
        "# Deployment Candidates\n\n"
        + f"- truth_label_block: `{block_truth}`\n"
        + _md_table(
            deployment,
            ["candidate", "n", "pct_plus1r_before_minus1r", "pct_plus2r_before_minus1r", "pct_plus3r_before_minus1r", "expectancy_1to3_r", "positive_years", "robustness_score", "deployment_verdict"],
        )
        + "\n",
        encoding="utf-8",
    )
    paths["simple_rules"].write_text(
        "# Simple Rules\n\n"
        + f"- truth_label_block: `{block_truth}`\n"
        + ("\n".join(f"- {r}" for r in rules[:30]) if rules else "- No stable rules met thresholds.")
        + "\n",
        encoding="utf-8",
    )

    def _pick_best(df: pd.DataFrame, cols: Sequence[str], asc: Sequence[bool] | None = None) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        c = [x for x in cols if x in df.columns]
        if not c:
            return df.head(1)
        a = list(asc)[: len(c)] if asc is not None else [False] * len(c)
        return df.sort_values(c, ascending=a, kind="mergesort").head(1)

    best_symbol = _pick_best(by_symbol, ["expectancy_1to3_r", "pct_plus3r_before_minus1r"])
    best_session = _pick_best(by_session, ["expectancy_1to3_r", "pct_plus3r_before_minus1r"])
    best_h = _pick_best(overall, ["expectancy_1to3_r", "mfe_mae_ratio"])
    best_dep = pd.DataFrame()
    if not deployment.empty:
        ddep = deployment.copy()
        ddep["_ord"] = ddep["deployment_verdict"].map({"DEPLOY": 0, "WATCH": 1, "CUT": 2}).fillna(9)
        best_dep = ddep.sort_values(["_ord", "robustness_score", "expectancy_1to3_r", "n"], ascending=[True, False, False, False], kind="mergesort").head(1)
    deploy_exists = bool((deployment["deployment_verdict"] == "DEPLOY").any()) if not deployment.empty else False
    eur_best = _pick_best(by_symbol[by_symbol["symbol"] == "EURUSD"], ["expectancy_1to3_r", "pct_plus3r_before_minus1r"]) if not by_symbol.empty else pd.DataFrame()
    xag_best = _pick_best(by_symbol[by_symbol["symbol"] == "XAGUSD"], ["expectancy_1to3_r", "pct_plus3r_before_minus1r"]) if not by_symbol.empty else pd.DataFrame()

    best_dep_name = best_dep.iloc[0]["candidate"] if not best_dep.empty else "none"
    xag_value = bool(
        (not xag_best.empty)
        and float(xag_best.iloc[0]["expectancy_1to3_r"]) > 0
        and float(xag_best.iloc[0]["pct_plus3r_before_minus1r"]) >= 0.27
        and float(xag_best.iloc[0]["n"]) >= 20
    )
    eur_include = bool(
        (not eur_best.empty)
        and float(eur_best.iloc[0]["expectancy_1to3_r"]) > 0
        and float(eur_best.iloc[0]["pct_plus3r_before_minus1r"]) >= 0.27
        and float(eur_best.iloc[0]["n"]) >= 20
    )

    rep = [
        "# Pine16 4-Year Edge Hunt Report",
        "",
        "## 1. Executive verdict", f"- truth_label_block: `{block_truth}`", f"- deployment_worthy_candidate_exists: `{deploy_exists}`",
        "## 2. Truth source used", f"- truth_label_block: `{block_truth}`", f"- requested_truth_mode: `{truth_mode.value}`", f"- configs_run: `{','.join(str(c) for c in config_paths)}`",
        "## 3. Universe studied", f"- truth_label_block: `{block_truth}`", f"- symbols: `{','.join(TARGET_SYMBOLS)}`", f"- sessions: `{','.join(TARGET_SCOPES)}`", f"- horizons: `{','.join(str(h) for h in TARGET_H)}`", f"- years: `{','.join(str(y) for y in TARGET_YEARS)}`",
        "## 4. Symbol focus and why", f"- truth_label_block: `{block_truth}`", "- XAUUSD/XAGUSD as primary edge engines; EURUSD as challenger.",
        "## 5. Session focus and why", f"- truth_label_block: `{block_truth}`", "- London and London+NY only.",
        "## 6. Horizon focus and why", f"- truth_label_block: `{block_truth}`", "- 24h/36h/48h/72h only.",
        "## 7. Overall 4-year findings", f"- truth_label_block: `{block_truth}`", _md_table(overall, ["horizon_h", "n", "pct_plus1r_before_minus1r", "pct_plus3r_before_minus1r", "expectancy_1to3_r", "mfe_mae_ratio", "directional_band", "tradable_band", "path_band"]),
        "## 8. By symbol", f"- truth_label_block: `{block_truth}`", _md_table(by_symbol, ["symbol", "session_scope", "horizon_h", "n", "pct_plus1r_before_minus1r", "pct_plus3r_before_minus1r", "expectancy_1to3_r", "robustness_score"]),
        "## 9. By session", f"- truth_label_block: `{block_truth}`", _md_table(by_session, ["session_scope", "horizon_h", "n", "pct_plus1r_before_minus1r", "pct_plus3r_before_minus1r", "expectancy_1to3_r"]),
        "## 10. By year", f"- truth_label_block: `{block_truth}`", _md_table(by_year, ["year", "session_scope", "horizon_h", "n", "pct_plus3r_before_minus1r", "expectancy_1to3_r"]),
        "## 11. By symbol x year", f"- truth_label_block: `{block_truth}`", _md_table(by_symbol_year, ["symbol", "year", "session_scope", "horizon_h", "n", "pct_plus3r_before_minus1r", "expectancy_1to3_r"]),
        "## 12. Barrier study (+1R / +2R / +3R)", f"- truth_label_block: `{block_truth}`", _md_table(overall, ["horizon_h", "pct_plus1r_before_minus1r", "pct_plus2r_before_minus1r", "pct_plus3r_before_minus1r", "pct_minus1r_before_plus1r", "pct_minus1r_before_plus3r"]),
        "## 13. Path-quality study", f"- truth_label_block: `{block_truth}`", _md_table(overall, ["horizon_h", "median_mfe_r", "median_mae_r", "mfe_mae_ratio", "favorable_move_rate"]),
        "## 14. Feature-bucket study", f"- truth_label_block: `{block_truth}`", _md_table(feature_buckets.head(80), ["bucket_name", "bucket_value", "session_scope", "horizon_h", "n", "pct_plus3r_before_minus1r", "expectancy_1to3_r"]),
        "## 15. Top conditional combinations", f"- truth_label_block: `{block_truth}`", _md_table(top_combos.head(60), ["combo_id", "symbol", "session_scope", "horizon_h", "n", "positive_years", "negative_years", "pct_plus3r_before_minus1r", "expectancy_1to3_r", "robustness_score", "fragility_flag", "verdict"]),
        "## 16. Robustness study", f"- truth_label_block: `{block_truth}`", _md_table(robustness, ["candidate_id", "symbol", "session_scope", "horizon_h", "n", "years_present", "positive_years", "negative_years", "expectancy_mean", "expectancy_std", "concentration_score", "robustness_score", "fragility_flag"]),
        "## 17. Keep / watch / cut", f"- truth_label_block: `{block_truth}`", _md_table(keep_watch_cut, ["candidate_id", "symbol", "session_scope", "horizon_h", "n", "pct_plus3r_before_minus1r", "expectancy_mean", "robustness_score", "action"]),
        "## 18. Deployment candidates", f"- truth_label_block: `{block_truth}`", _md_table(deployment, ["candidate", "n", "pct_plus1r_before_minus1r", "pct_plus2r_before_minus1r", "pct_plus3r_before_minus1r", "expectancy_1to3_r", "positive_years", "robustness_score", "deployment_verdict"]),
        "## 19. What to trade / what not to trade", f"- truth_label_block: `{block_truth}`", "- Trade KEEP rows. Cut CUT rows.",
        "## 20. Caveats / blockers", f"- truth_label_block: `{block_truth}`", "- OHLC first-touch approximation with adverse-first same-bar tie.",
        "## Final Questions", f"- truth_label_block: `{block_truth}`",
        f"1. Do we have a real 4-year edge anywhere? `{deploy_exists}`",
        f"2. If yes, where exactly? `{best_dep_name if deploy_exists else 'none'}`",
        f"3. Is XAUUSD the strongest engine? `{bool(not best_symbol.empty and best_symbol.iloc[0]['symbol'] == 'XAUUSD')}`",
        f"4. Does XAGUSD add value or noise? `{'value' if xag_value else 'noise_or_conditional'}`",
        f"5. Does EURUSD deserve inclusion? `{'yes' if eur_include else 'no'}`",
        f"6. Is London better than London+NY? `best={best_session.iloc[0]['session_scope'] if not best_session.empty else 'unknown'}`",
        f"7. Which horizon is best (24/36/48/72)? `{int(best_h.iloc[0]['horizon_h']) if not best_h.empty else 'unknown'}h`",
        f"8. Single best deployment candidate right now: `{best_dep_name}`",
        "9. What should be traded? KEEP rows.",
        "10. What should be cut? CUT rows.",
    ]
    report_md = "\n\n".join(rep).rstrip() + "\n"
    paths["report_md"].write_text(report_md, encoding="utf-8")
    if export_html:
        _write_html(report_md, paths["report_html"])
    else:
        paths["report_html"].write_text("", encoding="utf-8")

    return EdgeHuntArtifacts(
        audit_md=audit_path,
        master_parquet=master_path,
        report_md=paths["report_md"],
        report_html=paths["report_html"],
        overall_csv=paths["overall"],
        by_symbol_csv=paths["by_symbol"],
        by_session_csv=paths["by_session"],
        by_year_csv=paths["by_year"],
        by_symbol_year_csv=paths["by_symbol_year"],
        feature_buckets_csv=paths["feature_buckets"],
        top_combos_csv=paths["top_combos"],
        robustness_csv=paths["robustness"],
        keep_watch_cut_csv=paths["keep_watch_cut"],
        deployment_candidates_csv=paths["deployment"],
        deployment_candidates_md=paths["deployment_md"],
        simple_rules_md=paths["simple_rules"],
        feature_importance_csv=paths["fi"],
    )


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run Pine16 4-year focused edge-hunt pipeline.")
    ap.add_argument("--config", nargs="+", required=True, help="One or more Pine16 config paths.")
    ap.add_argument("--truth-mode", required=True, choices=[m.value for m in TruthMode], help="Truth mode selector.")
    ap.add_argument("--exact-dir", default="data/derived/pine16_exact", help="Exact/parity artifacts directory.")
    ap.add_argument("--output-dir", default="outputs/reports", help="Output report directory.")
    ap.add_argument("--master-path", default="data/derived/pine16_exact/edge_hunt_m30_4y_master.parquet", help="Master parquet path.")
    ap.add_argument("--min-n", type=int, default=20, help="Minimum sample size.")
    ap.add_argument("--min-n-robust", type=int, default=30, help="Robust sample size threshold.")
    ap.add_argument("--export-html", action="store_true", help="Write html report.")
    ap.add_argument("--include-feature-importance", action="store_true", help="Generate feature importance and simple rules.")
    ap.add_argument("--audit-path", default="outputs/audit_pine16_4year_edge_hunt.md", help="Audit markdown path.")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    outs = run_4year_edge_hunt(
        config_paths=list(args.config),
        truth_mode_raw=args.truth_mode,
        exact_dir=Path(args.exact_dir),
        output_dir=Path(args.output_dir),
        master_path=Path(args.master_path),
        min_n=int(args.min_n),
        min_n_robust=int(args.min_n_robust),
        export_html=bool(args.export_html),
        include_feature_importance=bool(args.include_feature_importance),
        audit_path=Path(args.audit_path),
    )
    for k, v in outs.__dict__.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
