from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data import load_bars_for_symbol, timeframe_to_minutes
from .indicators import atr
from .pine16_config import Pine16ExactConfig, load_pine16_exact_config, to_legacy_run_config
from .pine16_research import classify_signals, classify_trades, load_truth_datasets
from .pine16_truth import TruthLabel, TruthMode, normalize_truth_mode


HORIZON_GRID_H: List[int] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48, 72, 96, 120, 168, 240]
DEPLOYMENT_FOCUS_H: List[int] = [4, 8, 12, 24, 48, 72, 120]
SESSION_SCOPES: List[str] = ["all_sessions", "london_only", "newyork_only", "london_or_newyork", "other"]
COMPARISON_KEY_H: List[int] = [24, 48, 72, 120]
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)


@dataclass
class HorizonStudyResult:
    cfg: Pine16ExactConfig
    truth_mode: TruthMode
    truth_label: TruthLabel
    parity_metrics: Dict[str, object] | None
    exact_available: bool
    signals_cls: pd.DataFrame
    trades_cls: pd.DataFrame
    master: pd.DataFrame
    overall: pd.DataFrame
    by_symbol: pd.DataFrame
    by_session: pd.DataFrame
    by_year: pd.DataFrame
    by_symbol_year: pd.DataFrame
    by_symbol_session: pd.DataFrame
    by_year_session: pd.DataFrame
    by_symbol_year_session: pd.DataFrame


@dataclass
class M30HorizonArtifacts:
    audit_md: Path | None
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
    leaderboard_csv: Path
    keep_watch_cut_csv: Path
    deployment_candidates_csv: Path
    deployment_candidates_md: Path
    best_horizons_csv: Path
    m15_vs_m30_comparison_csv: Path
    by_symbol_year_session_csv: Path
    horizon_leaderboard_csv: Path


def _md_table(df: pd.DataFrame, cols: Iterable[str]) -> str:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return "_No columns._"
    if df.empty:
        return "_No rows._"
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df[cols].iterrows():
        vals: List[str] = []
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


def _write_html_from_markdown(md: str, path: Path) -> None:
    import html

    body = html.escape(md)
    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Pine16 M30 Horizon Report</title>"
        "<style>body{font-family:ui-monospace,Consolas,monospace;padding:24px;}pre{white-space:pre-wrap;}</style>"
        "</head><body><pre>"
        f"{body}"
        "</pre></body></html>"
    )
    path.write_text(page, encoding="utf-8")


def directional_proof_verdict(p: float) -> str:
    if not np.isfinite(p):
        return "weak"
    if p < 0.50:
        return "weak"
    if p < 0.55:
        return "borderline"
    if p <= 0.60:
        return "clear"
    return "strong"


def tradable_1to3_verdict(p: float) -> str:
    if not np.isfinite(p):
        return "not monetizable"
    if p < 0.25:
        return "not monetizable"
    if p < 0.30:
        return "marginal"
    if p <= 0.40:
        return "monetizable"
    return "strong monetizable edge"


def path_quality_verdict(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "weak"
    if ratio < 1.0:
        return "weak"
    if ratio < 1.1:
        return "borderline"
    if ratio <= 1.25:
        return "decent"
    return "strong"


def _scan_first_touch(
    high: np.ndarray,
    low: np.ndarray,
    start: int,
    end: int,
    target_up: float,
    target_down: float,
) -> Tuple[str, Optional[int]]:
    if start > end:
        return "OPEN", None
    for i in range(start, end + 1):
        hit_up = bool(high[i] >= target_up)
        hit_down = bool(low[i] <= target_down)
        if hit_up and hit_down:
            return "LOSS", i
        if hit_up:
            return "WIN", i
        if hit_down:
            return "LOSS", i
    return "OPEN", None


def _first_touch_up(high: np.ndarray, start: int, end: int, target_up: float) -> Optional[int]:
    if start > end:
        return None
    for i in range(start, end + 1):
        if high[i] >= target_up:
            return i
    return None


def _first_touch_down(low: np.ndarray, start: int, end: int, target_down: float) -> Optional[int]:
    if start > end:
        return None
    for i in range(start, end + 1):
        if low[i] <= target_down:
            return i
    return None


def _hours_from_bars(n_bars: Optional[int], tf_minutes: int) -> float:
    if n_bars is None:
        return np.nan
    return float((float(n_bars) * float(tf_minutes)) / 60.0)


def _filter_scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if df.empty:
        return df
    if scope == "all_sessions":
        return df
    if scope == "london_only":
        return df[df["entry_session_bucket"].astype(str) == "london_only"]
    if scope == "newyork_only":
        return df[df["entry_session_bucket"].astype(str) == "newyork_only"]
    if scope == "london_or_newyork":
        return df[pd.to_numeric(df["entry_in_london_or_newyork"], errors="coerce") == 1]
    if scope == "other":
        return df[df["entry_session_bucket"].astype(str) == "other"]
    return df.iloc[0:0].copy()


def build_horizon_master(
    trades_cls: pd.DataFrame,
    cfg: Pine16ExactConfig,
    truth_label: TruthLabel,
    horizons_h: Sequence[int] = HORIZON_GRID_H,
) -> pd.DataFrame:
    if trades_cls.empty:
        return pd.DataFrame()

    legacy_cfg = to_legacy_run_config(cfg)
    tf_minutes = timeframe_to_minutes(cfg.timeframe)
    max_h = max(int(h) for h in horizons_h)
    max_bars_fwd = int((max_h * 60) // tf_minutes)
    rows: List[Dict[str, object]] = []

    for symbol in sorted(trades_cls["symbol"].astype(str).unique().tolist()):
        sym_tr = trades_cls[trades_cls["symbol"].astype(str) == symbol].copy()
        if sym_tr.empty:
            continue

        try:
            bars = load_bars_for_symbol(legacy_cfg, symbol=symbol, timeframe=cfg.timeframe)
        except FileNotFoundError:
            continue
        if bars.empty:
            continue

        t = pd.to_datetime(bars["time"], utc=True)
        high = bars["high"].to_numpy(dtype=float)
        low = bars["low"].to_numpy(dtype=float)
        close = bars["close"].to_numpy(dtype=float)
        atr_arr = atr(high, low, close, int(cfg.risk.atrLen))

        idx = pd.Index(t).get_indexer(
            pd.to_datetime(sym_tr["entry_time_utc"], utc=True, errors="coerce"),
            method="nearest",
            tolerance=pd.Timedelta(minutes=tf_minutes),
        )
        sym_tr = sym_tr.reset_index(drop=True)

        for j in range(len(sym_tr)):
            ei = int(idx[j]) if j < len(idx) else -1
            if ei < 0 or ei >= len(bars):
                continue
            start = ei + 1
            end_max = min(len(high) - 1, ei + max_bars_fwd)
            if start > end_max:
                continue

            ep = float(pd.to_numeric(sym_tr.at[j, "entry_price"], errors="coerce"))
            if not np.isfinite(ep):
                ep = float(close[ei])

            atr_entry = float(pd.to_numeric(sym_tr.at[j, "atr_entry"], errors="coerce"))
            if not np.isfinite(atr_entry) or atr_entry <= 0:
                atr_entry = float(atr_arr[ei])

            risk_unit = float(cfg.risk.slAtrMult) * float(atr_entry)
            if not np.isfinite(risk_unit) or risk_unit <= 0:
                continue

            sl1 = float(ep - risk_unit)
            sl05 = float(ep - 0.5 * risk_unit)
            tp05 = float(ep + 0.5 * risk_unit)
            tp1 = float(ep + 1.0 * risk_unit)
            tp2 = float(ep + 2.0 * risk_unit)
            tp3 = float(ep + 3.0 * risk_unit)

            immediate_low = low[start] if start < len(low) else np.nan
            immediate_drawdown_gt_05r = int(np.isfinite(immediate_low) and immediate_low <= sl05)
            immediate_drawdown_gt_1r = int(np.isfinite(immediate_low) and immediate_low <= sl1)

            base = {
                "signal_id": str(sym_tr.at[j, "signal_id"]),
                "trade_id": str(sym_tr.at[j, "trade_id"]),
                "symbol": str(symbol),
                "timeframe": str(cfg.timeframe),
                "year": int(sym_tr.at[j, "year"]) if pd.notna(sym_tr.at[j, "year"]) else np.nan,
                "month": int(sym_tr.at[j, "month"]) if pd.notna(sym_tr.at[j, "month"]) else np.nan,
                "truth_label": truth_label.value,
                "config_pack": str(cfg.metadata.config_pack),
                "entry_time": pd.to_datetime(sym_tr.at[j, "entry_time_utc"], utc=True, errors="coerce"),
                "entry_price": float(ep),
                "setup_session_bucket": str(sym_tr.at[j, "setup_session_bucket"]),
                "entry_session_bucket": str(sym_tr.at[j, "entry_session_bucket"]),
                "entry_in_london_or_newyork": int(pd.to_numeric(sym_tr.at[j, "entry_in_london_or_newyork"], errors="coerce")),
                "realized_result_r": float(pd.to_numeric(sym_tr.at[j, "result_r"], errors="coerce")),
            }

            for h in sorted(set(int(x) for x in horizons_h if int(x) > 0)):
                bars_fwd = int((h * 60) // tf_minutes)
                end = min(len(high) - 1, ei + bars_fwd)
                if start > end:
                    continue

                out05, _ = _scan_first_touch(high, low, start, end, tp05, sl1)
                out1, _ = _scan_first_touch(high, low, start, end, tp1, sl1)
                out2, _ = _scan_first_touch(high, low, start, end, tp2, sl1)
                out3, _ = _scan_first_touch(high, low, start, end, tp3, sl1)

                plus1_i = _first_touch_up(high, start, end, tp1)
                plus2_i = _first_touch_up(high, start, end, tp2)
                plus3_i = _first_touch_up(high, start, end, tp3)
                minus1_i = _first_touch_down(low, start, end, sl1)
                minus05_i = _first_touch_down(low, start, end, sl05)

                win_high = high[start : end + 1]
                win_low = low[start : end + 1]
                mfe_r = float((float(np.max(win_high)) - ep) / risk_unit) if len(win_high) else np.nan
                mae_r = float((float(np.min(win_low)) - ep) / risk_unit) if len(win_low) else np.nan
                favorable = int(np.isfinite(ep) and close[end] > ep)
                mfe_gt_mae = int(np.isfinite(mfe_r) and np.isfinite(mae_r) and (mfe_r > abs(mae_r)))

                row = dict(base)
                row.update(
                    {
                        "horizon_h": int(h),
                        "mfe_r": mfe_r,
                        "mae_r": mae_r,
                        "favorable_move": float(favorable),
                        "mfe_gt_mae": float(mfe_gt_mae),
                        "plus05r_before_minus1r": 1.0 if out05 == "WIN" else (0.0 if out05 == "LOSS" else np.nan),
                        "plus1r_before_minus1r": 1.0 if out1 == "WIN" else (0.0 if out1 == "LOSS" else np.nan),
                        "plus2r_before_minus1r": 1.0 if out2 == "WIN" else (0.0 if out2 == "LOSS" else np.nan),
                        "plus3r_before_minus1r": 1.0 if out3 == "WIN" else (0.0 if out3 == "LOSS" else np.nan),
                        "minus1r_before_plus1r": 1.0 if out1 == "LOSS" else (0.0 if out1 == "WIN" else np.nan),
                        "minus1r_before_plus2r": 1.0 if out2 == "LOSS" else (0.0 if out2 == "WIN" else np.nan),
                        "minus1r_before_plus3r": 1.0 if out3 == "LOSS" else (0.0 if out3 == "WIN" else np.nan),
                        "open_plus1r_test": 1.0 if out1 == "OPEN" else 0.0,
                        "open_plus2r_test": 1.0 if out2 == "OPEN" else 0.0,
                        "open_plus3r_test": 1.0 if out3 == "OPEN" else 0.0,
                        "pct_reaches_plus1r_without_minus05r_flag": 1.0
                        if (plus1_i is not None and (minus05_i is None or plus1_i < minus05_i))
                        else 0.0,
                        "pct_reaches_plus3r_without_minus1r_flag": 1.0
                        if (plus3_i is not None and (minus1_i is None or plus3_i < minus1_i))
                        else 0.0,
                        "pct_immediate_drawdown_gt_05r_flag": float(immediate_drawdown_gt_05r),
                        "pct_immediate_drawdown_gt_1r_flag": float(immediate_drawdown_gt_1r),
                        "time_to_plus1r": _hours_from_bars((plus1_i - ei) if plus1_i is not None else None, tf_minutes),
                        "time_to_plus2r": _hours_from_bars((plus2_i - ei) if plus2_i is not None else None, tf_minutes),
                        "time_to_plus3r": _hours_from_bars((plus3_i - ei) if plus3_i is not None else None, tf_minutes),
                        "time_to_minus1r": _hours_from_bars((minus1_i - ei) if minus1_i is not None else None, tf_minutes),
                    }
                )
                rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["session_scope_bucket"] = out["entry_session_bucket"].astype(str)
    return out


def _empty_agg_row() -> Dict[str, object]:
    return {
        "n_signals": 0,
        "n_trades": 0,
        "realized_win_rate": np.nan,
        "pct_plus05r_before_minus1r": np.nan,
        "pct_plus1r_before_minus1r": np.nan,
        "pct_plus2r_before_minus1r": np.nan,
        "pct_plus3r_before_minus1r": np.nan,
        "pct_minus1r_before_plus1r": np.nan,
        "pct_minus1r_before_plus2r": np.nan,
        "pct_minus1r_before_plus3r": np.nan,
        "open_rate_plus1r_test": np.nan,
        "open_rate_plus2r_test": np.nan,
        "open_rate_plus3r_test": np.nan,
        "favorable_move_rate": np.nan,
        "median_mfe_r": np.nan,
        "median_mae_r": np.nan,
        "mfe_mae_ratio": np.nan,
        "pct_mfe_gt_mae": np.nan,
        "pct_reaches_plus1r_without_minus05r": np.nan,
        "pct_reaches_plus3r_without_minus1r": np.nan,
        "pct_immediate_drawdown_gt_05r": np.nan,
        "pct_immediate_drawdown_gt_1r": np.nan,
        "median_time_to_plus1r": np.nan,
        "median_time_to_plus2r": np.nan,
        "median_time_to_plus3r": np.nan,
        "median_time_to_minus1r": np.nan,
        "expectancy_1r": np.nan,
        "expectancy_1to2_r": np.nan,
        "expectancy_1to3_r": np.nan,
        "profit_factor_1to3": np.nan,
        "net_r_1to3": np.nan,
    }

def _agg_metrics(g: pd.DataFrame) -> Dict[str, object]:
    n = len(g)
    if n == 0:
        return _empty_agg_row()

    def _median(series: pd.Series) -> float:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return np.nan
        return float(s.median())

    win05 = float(pd.to_numeric(g["plus05r_before_minus1r"], errors="coerce").fillna(0.0).sum())
    win1 = float(pd.to_numeric(g["plus1r_before_minus1r"], errors="coerce").fillna(0.0).sum())
    win2 = float(pd.to_numeric(g["plus2r_before_minus1r"], errors="coerce").fillna(0.0).sum())
    win3 = float(pd.to_numeric(g["plus3r_before_minus1r"], errors="coerce").fillna(0.0).sum())
    loss1 = float(pd.to_numeric(g["minus1r_before_plus1r"], errors="coerce").fillna(0.0).sum())
    loss2 = float(pd.to_numeric(g["minus1r_before_plus2r"], errors="coerce").fillna(0.0).sum())
    loss3 = float(pd.to_numeric(g["minus1r_before_plus3r"], errors="coerce").fillna(0.0).sum())

    open1 = float(pd.to_numeric(g["open_plus1r_test"], errors="coerce").fillna(0.0).sum())
    open2 = float(pd.to_numeric(g["open_plus2r_test"], errors="coerce").fillna(0.0).sum())
    open3 = float(pd.to_numeric(g["open_plus3r_test"], errors="coerce").fillna(0.0).sum())

    med_mfe = _median(g["mfe_r"])
    med_mae = _median(g["mae_r"])
    ratio = float(med_mfe / abs(med_mae)) if np.isfinite(med_mae) and med_mae != 0 else np.nan

    expectancy_1r = float((win1 - loss1) / n)
    expectancy_1to2 = float((2.0 * win2 - loss2) / n)
    expectancy_1to3 = float((3.0 * win3 - loss3) / n)
    gross_win_1to3 = float(3.0 * win3)
    gross_loss_1to3 = float(loss3)
    pf_1to3 = float(gross_win_1to3 / gross_loss_1to3) if gross_loss_1to3 > 0 else np.nan
    net_r_1to3 = float(gross_win_1to3 - gross_loss_1to3)

    realized_r = pd.to_numeric(g["realized_result_r"], errors="coerce")
    realized_win_rate = float((realized_r > 0).mean()) if realized_r.notna().any() else np.nan

    return {
        "n_signals": int(g["signal_id"].nunique()),
        "n_trades": int(g["trade_id"].nunique()),
        "realized_win_rate": realized_win_rate,
        "pct_plus05r_before_minus1r": float(win05 / n),
        "pct_plus1r_before_minus1r": float(win1 / n),
        "pct_plus2r_before_minus1r": float(win2 / n),
        "pct_plus3r_before_minus1r": float(win3 / n),
        "pct_minus1r_before_plus1r": float(loss1 / n),
        "pct_minus1r_before_plus2r": float(loss2 / n),
        "pct_minus1r_before_plus3r": float(loss3 / n),
        "open_rate_plus1r_test": float(open1 / n),
        "open_rate_plus2r_test": float(open2 / n),
        "open_rate_plus3r_test": float(open3 / n),
        "favorable_move_rate": float(pd.to_numeric(g["favorable_move"], errors="coerce").mean()),
        "median_mfe_r": med_mfe,
        "median_mae_r": med_mae,
        "mfe_mae_ratio": ratio,
        "pct_mfe_gt_mae": float(pd.to_numeric(g["mfe_gt_mae"], errors="coerce").mean()),
        "pct_reaches_plus1r_without_minus05r": float(
            pd.to_numeric(g["pct_reaches_plus1r_without_minus05r_flag"], errors="coerce").mean()
        ),
        "pct_reaches_plus3r_without_minus1r": float(
            pd.to_numeric(g["pct_reaches_plus3r_without_minus1r_flag"], errors="coerce").mean()
        ),
        "pct_immediate_drawdown_gt_05r": float(
            pd.to_numeric(g["pct_immediate_drawdown_gt_05r_flag"], errors="coerce").mean()
        ),
        "pct_immediate_drawdown_gt_1r": float(
            pd.to_numeric(g["pct_immediate_drawdown_gt_1r_flag"], errors="coerce").mean()
        ),
        "median_time_to_plus1r": _median(g["time_to_plus1r"]),
        "median_time_to_plus2r": _median(g["time_to_plus2r"]),
        "median_time_to_plus3r": _median(g["time_to_plus3r"]),
        "median_time_to_minus1r": _median(g["time_to_minus1r"]),
        "expectancy_1r": expectancy_1r,
        "expectancy_1to2_r": expectancy_1to2,
        "expectancy_1to3_r": expectancy_1to3,
        "profit_factor_1to3": pf_1to3,
        "net_r_1to3": net_r_1to3,
    }


def _robustness_score(row: pd.Series) -> float:
    n = float(pd.to_numeric(row.get("n_trades"), errors="coerce"))
    p1 = float(pd.to_numeric(row.get("pct_plus1r_before_minus1r"), errors="coerce"))
    p3 = float(pd.to_numeric(row.get("pct_plus3r_before_minus1r"), errors="coerce"))
    exp = float(pd.to_numeric(row.get("expectancy_1to3_r"), errors="coerce"))
    ratio = float(pd.to_numeric(row.get("mfe_mae_ratio"), errors="coerce"))
    fav = float(pd.to_numeric(row.get("favorable_move_rate"), errors="coerce"))

    n_score = np.clip(n / 100.0, 0.0, 1.0) if np.isfinite(n) else 0.0
    p1_score = np.clip((p1 - 0.50) / 0.15, 0.0, 1.0) if np.isfinite(p1) else 0.0
    p3_score = np.clip((p3 - 0.25) / 0.20, 0.0, 1.0) if np.isfinite(p3) else 0.0
    exp_score = np.clip((exp + 0.05) / 0.20, 0.0, 1.0) if np.isfinite(exp) else 0.0
    path_score = np.clip((ratio - 1.0) / 0.25, 0.0, 1.0) if np.isfinite(ratio) else 0.0
    fav_score = np.clip((fav - 0.50) / 0.15, 0.0, 1.0) if np.isfinite(fav) else 0.0
    return float(0.20 * n_score + 0.20 * p1_score + 0.20 * p3_score + 0.20 * exp_score + 0.10 * path_score + 0.10 * fav_score)


def _apply_verdicts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["directional_proof_verdict"] = out["pct_plus1r_before_minus1r"].apply(directional_proof_verdict)
    out["tradable_1to3_verdict"] = out["pct_plus3r_before_minus1r"].apply(tradable_1to3_verdict)
    out["path_quality_verdict"] = out["mfe_mae_ratio"].apply(path_quality_verdict)
    out["robustness_score"] = out.apply(_robustness_score, axis=1)
    out["verdict"] = out["directional_proof_verdict"] + " | " + out["tradable_1to3_verdict"] + " | path:" + out["path_quality_verdict"]
    return out


def aggregate_metrics(master: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame()
    gb_cols = list(group_cols)
    rows: List[Dict[str, object]] = []
    if gb_cols:
        grouped = master.groupby(gb_cols, dropna=False, sort=True)
        for keys, g in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {k: v for k, v in zip(gb_cols, keys)}
            row.update(_agg_metrics(g))
            rows.append(row)
    else:
        row = _agg_metrics(master)
        rows.append(row)
    out = pd.DataFrame(rows)
    return _apply_verdicts(out)


def aggregate_by_scopes(master: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    tables: List[pd.DataFrame] = []
    for scope in SESSION_SCOPES:
        scoped = _filter_scope(master, scope)
        if scoped.empty:
            continue
        t = aggregate_metrics(scoped, group_cols=list(group_cols) + ["horizon_h"])
        if t.empty:
            continue
        t["session_scope"] = scope
        tables.append(t)
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True, sort=False)


def run_horizon_study(
    config_path: str,
    truth_mode_raw: str,
    exact_dir: Path,
    horizons_h: Sequence[int] = HORIZON_GRID_H,
) -> HorizonStudyResult:
    cfg = load_pine16_exact_config(config_path)
    truth_mode = normalize_truth_mode(truth_mode_raw)
    trades, signals, truth_label, parity_metrics, exact_available = load_truth_datasets(cfg, truth_mode, exact_dir)

    if signals.empty and not trades.empty:
        signals = trades[["symbol", "timeframe", "bar_time_utc", "entry_time_utc", "entry_price"]].copy()

    signals_cls = classify_signals(signals, truth_label=truth_label, config_pack=cfg.metadata.config_pack)
    trades_cls = classify_trades(trades, truth_label=truth_label, config_pack=cfg.metadata.config_pack)
    master = build_horizon_master(trades_cls=trades_cls, cfg=cfg, truth_label=truth_label, horizons_h=horizons_h)

    by_session = aggregate_by_scopes(master, group_cols=[])
    overall = by_session[by_session["session_scope"] == "all_sessions"].copy() if not by_session.empty else pd.DataFrame()
    by_symbol = aggregate_metrics(master, group_cols=["symbol", "horizon_h"]) if not master.empty else pd.DataFrame()
    by_year = aggregate_metrics(master, group_cols=["year", "horizon_h"]) if not master.empty else pd.DataFrame()
    by_symbol_year = (
        aggregate_metrics(master, group_cols=["symbol", "year", "horizon_h"]) if not master.empty else pd.DataFrame()
    )
    by_symbol_session = aggregate_by_scopes(master, group_cols=["symbol"])
    by_year_session = aggregate_by_scopes(master, group_cols=["year"])
    by_symbol_year_session = aggregate_by_scopes(master, group_cols=["symbol", "year"])

    for df in [
        signals_cls,
        trades_cls,
        master,
        overall,
        by_symbol,
        by_session,
        by_year,
        by_symbol_year,
        by_symbol_session,
        by_year_session,
        by_symbol_year_session,
    ]:
        if not df.empty:
            df["truth_label"] = truth_label.value
            df["config_pack"] = cfg.metadata.config_pack

    return HorizonStudyResult(
        cfg=cfg,
        truth_mode=truth_mode,
        truth_label=truth_label,
        parity_metrics=parity_metrics,
        exact_available=exact_available,
        signals_cls=signals_cls,
        trades_cls=trades_cls,
        master=master,
        overall=overall,
        by_symbol=by_symbol,
        by_session=by_session,
        by_year=by_year,
        by_symbol_year=by_symbol_year,
        by_symbol_session=by_symbol_session,
        by_year_session=by_year_session,
        by_symbol_year_session=by_symbol_year_session,
    )


def _sample_tiers(min_n: int, min_n_robust: int) -> List[int]:
    tiers = [int(min_n), int(max(min_n_robust, 30)), 50]
    return sorted(set(x for x in tiers if x > 0))


def _leaderboard_universe(res: HorizonStudyResult) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    if not res.overall.empty:
        t = res.overall.copy()
        t["symbol"] = "ALL"
        t["entity_level"] = "overall"
        parts.append(t)
    if not res.by_session.empty:
        t = res.by_session.copy()
        t["symbol"] = "ALL"
        t["entity_level"] = "session"
        parts.append(t)
    if not res.by_symbol.empty:
        t = res.by_symbol.copy()
        t["session_scope"] = "all_sessions"
        t["entity_level"] = "symbol"
        parts.append(t)
    if not res.by_symbol_session.empty:
        t = res.by_symbol_session.copy()
        t["entity_level"] = "symbol_session"
        parts.append(t)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True, sort=False)
    out["n"] = pd.to_numeric(out["n_trades"], errors="coerce")
    return out


def build_horizon_leaderboard(res: HorizonStudyResult, min_n: int, min_n_robust: int) -> pd.DataFrame:
    base = _leaderboard_universe(res)
    if base.empty:
        return base

    rank_metrics = [
        "pct_plus1r_before_minus1r",
        "pct_plus2r_before_minus1r",
        "pct_plus3r_before_minus1r",
        "expectancy_1to3_r",
        "mfe_mae_ratio",
        "favorable_move_rate",
    ]
    rows: List[pd.DataFrame] = []
    for thr in _sample_tiers(min_n, min_n_robust):
        sub = base[pd.to_numeric(base["n"], errors="coerce") >= float(thr)].copy()
        if sub.empty:
            continue
        for metric in rank_metrics:
            ranked = sub.sort_values(
                [metric, "robustness_score", "n_trades"],
                ascending=[False, False, False],
                kind="mergesort",
            ).copy()
            ranked["rank_metric"] = metric
            ranked["sample_tier"] = f"n>={thr}"
            ranked["rank"] = np.arange(1, len(ranked) + 1)
            rows.append(ranked)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)
    return out.reset_index(drop=True)


def build_best_horizons(res: HorizonStudyResult, min_n: int, min_n_robust: int) -> pd.DataFrame:
    base = _leaderboard_universe(res)
    if base.empty:
        return base

    rows: List[pd.DataFrame] = []
    for thr in _sample_tiers(min_n, min_n_robust):
        sub = base[pd.to_numeric(base["n"], errors="coerce") >= float(thr)].copy()
        if sub.empty:
            continue
        ranked = sub.sort_values(
            ["symbol", "session_scope", "robustness_score", "expectancy_1to3_r", "pct_plus3r_before_minus1r", "n_trades"],
            ascending=[True, True, False, False, False, False],
            kind="mergesort",
        )
        best = ranked.groupby(["symbol", "session_scope"], dropna=False, sort=True).head(1).copy()
        best["sample_tier"] = f"n>={thr}"
        best["n"] = pd.to_numeric(best["n_trades"], errors="coerce")
        rows.append(best)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True, sort=False)
    keep_cols = [
        "symbol",
        "session_scope",
        "horizon_h",
        "n",
        "pct_plus1r_before_minus1r",
        "pct_plus2r_before_minus1r",
        "pct_plus3r_before_minus1r",
        "expectancy_1to3_r",
        "favorable_move_rate",
        "median_mfe_r",
        "median_mae_r",
        "mfe_mae_ratio",
        "robustness_score",
        "verdict",
        "sample_tier",
        "entity_level",
        "truth_label",
        "config_pack",
    ]
    for c in keep_cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[keep_cols].reset_index(drop=True)


def build_keep_watch_cut(res: HorizonStudyResult, min_n: int, min_n_robust: int) -> pd.DataFrame:
    src = res.by_symbol_session.copy()
    if src.empty:
        return src
    src = src[src["session_scope"].astype(str).isin(["all_sessions", "london_only", "newyork_only", "london_or_newyork"])].copy()
    src = src[src["horizon_h"].isin(DEPLOYMENT_FOCUS_H)].copy()
    if src.empty:
        return src

    ranked = src.sort_values(
        ["symbol", "session_scope", "robustness_score", "expectancy_1to3_r", "pct_plus3r_before_minus1r", "n_trades"],
        ascending=[True, True, False, False, False, False],
        kind="mergesort",
    )
    best = ranked.groupby(["symbol", "session_scope"], dropna=False, sort=True).head(1).copy()

    best["action"] = "WATCH"
    keep_mask = (
        (pd.to_numeric(best["n_trades"], errors="coerce") >= float(min_n_robust))
        & (pd.to_numeric(best["expectancy_1to3_r"], errors="coerce") > 0)
        & (pd.to_numeric(best["pct_plus1r_before_minus1r"], errors="coerce") >= 0.55)
        & (pd.to_numeric(best["pct_plus3r_before_minus1r"], errors="coerce") >= 0.30)
        & (pd.to_numeric(best["robustness_score"], errors="coerce") >= 0.55)
    )
    cut_mask = (
        (pd.to_numeric(best["n_trades"], errors="coerce") >= float(min_n))
        & (pd.to_numeric(best["pct_plus1r_before_minus1r"], errors="coerce") < 0.50)
        & (pd.to_numeric(best["pct_plus3r_before_minus1r"], errors="coerce") < 0.25)
        & (pd.to_numeric(best["expectancy_1to3_r"], errors="coerce") <= 0)
    )
    best.loc[keep_mask, "action"] = "KEEP"
    best.loc[cut_mask, "action"] = "CUT"
    best["rationale"] = np.where(
        best["action"] == "KEEP",
        "decent sample + positive 1:3 expectancy + directional proof + repeatability",
        np.where(
            best["action"] == "CUT",
            "weak directional proof and weak 1:3 monetization",
            "mixed / conditional profile",
        ),
    )

    cols = [
        "symbol",
        "session_scope",
        "horizon_h",
        "n_trades",
        "pct_plus1r_before_minus1r",
        "pct_plus2r_before_minus1r",
        "pct_plus3r_before_minus1r",
        "expectancy_1to3_r",
        "favorable_move_rate",
        "mfe_mae_ratio",
        "robustness_score",
        "action",
        "rationale",
        "truth_label",
        "config_pack",
    ]
    for c in cols:
        if c not in best.columns:
            best[c] = np.nan
    return best[cols].sort_values(["action", "expectancy_1to3_r"], ascending=[True, False], kind="mergesort").reset_index(drop=True)


def _pick_best_row(
    df: pd.DataFrame,
    *,
    metric: str,
    min_n: int,
    allowed_horizons: Sequence[int] | None = None,
) -> pd.Series | None:
    if df.empty:
        return None
    sub = df.copy()
    sub = sub[pd.to_numeric(sub["n_trades"], errors="coerce") >= float(min_n)]
    if allowed_horizons is not None:
        sub = sub[sub["horizon_h"].isin([int(h) for h in allowed_horizons])]
    if sub.empty:
        return None
    sub = sub.sort_values([metric, "robustness_score", "n_trades"], ascending=[False, False, False], kind="mergesort")
    return sub.iloc[0]


def build_deployment_candidates(res: HorizonStudyResult, min_n: int, min_n_robust: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    overall = res.overall.copy()
    if not overall.empty:
        fast = _pick_best_row(overall, metric="pct_plus1r_before_minus1r", min_n=min_n, allowed_horizons=DEPLOYMENT_FOCUS_H)
        medium = _pick_best_row(overall, metric="expectancy_1to2_r", min_n=min_n, allowed_horizons=[8, 12, 24, 48, 72])
        full = _pick_best_row(overall, metric="expectancy_1to3_r", min_n=min_n, allowed_horizons=[24, 48, 72, 120, 168, 240])
        for name, row in [("fast_proof_of_direction", fast), ("medium_swing_capture", medium), ("full_3r_monetization", full)]:
            if row is None:
                continue
            n = float(pd.to_numeric(row.get("n_trades"), errors="coerce"))
            p1 = float(pd.to_numeric(row.get("pct_plus1r_before_minus1r"), errors="coerce"))
            p3 = float(pd.to_numeric(row.get("pct_plus3r_before_minus1r"), errors="coerce"))
            e13 = float(pd.to_numeric(row.get("expectancy_1to3_r"), errors="coerce"))
            rb = float(pd.to_numeric(row.get("robustness_score"), errors="coerce"))
            action = "WATCH"
            if n >= float(min_n_robust) and p1 >= 0.55 and p3 >= 0.30 and e13 > 0 and rb >= 0.55:
                action = "DEPLOY"
            elif n >= float(min_n) and (p1 < 0.50 and p3 < 0.25 and e13 <= 0):
                action = "REJECT"
            rows.append(
                {
                    "candidate_type": "overall_horizon",
                    "candidate_name": name,
                    "symbol": "ALL",
                    "session_scope": "all_sessions",
                    "horizon_h": int(row["horizon_h"]),
                    "n_trades": row.get("n_trades"),
                    "pct_plus1r_before_minus1r": row.get("pct_plus1r_before_minus1r"),
                    "pct_plus2r_before_minus1r": row.get("pct_plus2r_before_minus1r"),
                    "pct_plus3r_before_minus1r": row.get("pct_plus3r_before_minus1r"),
                    "expectancy_1to2_r": row.get("expectancy_1to2_r"),
                    "expectancy_1to3_r": row.get("expectancy_1to3_r"),
                    "favorable_move_rate": row.get("favorable_move_rate"),
                    "mfe_mae_ratio": row.get("mfe_mae_ratio"),
                    "robustness_score": row.get("robustness_score"),
                    "deployment_action": action,
                    "truth_label": row.get("truth_label"),
                    "config_pack": row.get("config_pack"),
                }
            )

    by_symbol = res.by_symbol.copy()
    if not by_symbol.empty:
        for symbol in sorted(by_symbol["symbol"].astype(str).unique().tolist()):
            s = by_symbol[by_symbol["symbol"].astype(str) == symbol].copy()
            short = _pick_best_row(s, metric="expectancy_1to2_r", min_n=min_n, allowed_horizons=[1, 2, 3, 4, 6, 8])
            medium = _pick_best_row(s, metric="expectancy_1to2_r", min_n=min_n, allowed_horizons=[12, 16, 24, 36, 48])
            long = _pick_best_row(s, metric="expectancy_1to3_r", min_n=min_n, allowed_horizons=[72, 96, 120, 168, 240])
            for bucket, row in [("short_horizons", short), ("medium_horizons", medium), ("long_horizons", long)]:
                if row is None:
                    continue
                rows.append(
                    {
                        "candidate_type": "symbol_horizon_bucket",
                        "candidate_name": bucket,
                        "symbol": symbol,
                        "session_scope": "all_sessions",
                        "horizon_h": int(row["horizon_h"]),
                        "n_trades": row.get("n_trades"),
                        "pct_plus1r_before_minus1r": row.get("pct_plus1r_before_minus1r"),
                        "pct_plus2r_before_minus1r": row.get("pct_plus2r_before_minus1r"),
                        "pct_plus3r_before_minus1r": row.get("pct_plus3r_before_minus1r"),
                        "expectancy_1to2_r": row.get("expectancy_1to2_r"),
                        "expectancy_1to3_r": row.get("expectancy_1to3_r"),
                        "favorable_move_rate": row.get("favorable_move_rate"),
                        "mfe_mae_ratio": row.get("mfe_mae_ratio"),
                        "robustness_score": row.get("robustness_score"),
                        "deployment_action": "DEPLOY"
                        if float(pd.to_numeric(row.get("n_trades"), errors="coerce")) >= float(min_n_robust)
                        and float(pd.to_numeric(row.get("expectancy_1to3_r"), errors="coerce")) > 0
                        else "WATCH",
                        "truth_label": row.get("truth_label"),
                        "config_pack": row.get("config_pack"),
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["deployment_action", "robustness_score", "expectancy_1to3_r"],
        ascending=[True, False, False],
        kind="mergesort",
    ).reset_index(drop=True)

def _extract_row(df: pd.DataFrame, key: Dict[str, object]) -> pd.Series | None:
    if df.empty:
        return None
    sub = df
    for k, v in key.items():
        if k not in sub.columns:
            return None
        sub = sub[sub[k] == v]
    if sub.empty:
        return None
    return sub.iloc[0]


def _comparison_row(
    *,
    level: str,
    symbol: str,
    session_scope: str,
    horizon_h: Optional[int],
    r15: pd.Series | None,
    r30: pd.Series | None,
    horizon_h_m15: Optional[int] = None,
    horizon_h_m30: Optional[int] = None,
) -> Dict[str, object]:
    def g(row: pd.Series | None, col: str) -> float:
        if row is None or col not in row.index:
            return np.nan
        return float(pd.to_numeric(row[col], errors="coerce"))

    out = {
        "comparison_level": level,
        "symbol": symbol,
        "session_scope": session_scope,
        "horizon_h": horizon_h if horizon_h is not None else np.nan,
        "horizon_h_m15": horizon_h_m15 if horizon_h_m15 is not None else np.nan,
        "horizon_h_m30": horizon_h_m30 if horizon_h_m30 is not None else np.nan,
        "n_signals_m15": g(r15, "n_signals"),
        "n_signals_m30": g(r30, "n_signals"),
        "n_trades_m15": g(r15, "n_trades"),
        "n_trades_m30": g(r30, "n_trades"),
        "pct_plus1r_before_minus1r_m15": g(r15, "pct_plus1r_before_minus1r"),
        "pct_plus1r_before_minus1r_m30": g(r30, "pct_plus1r_before_minus1r"),
        "pct_plus2r_before_minus1r_m15": g(r15, "pct_plus2r_before_minus1r"),
        "pct_plus2r_before_minus1r_m30": g(r30, "pct_plus2r_before_minus1r"),
        "pct_plus3r_before_minus1r_m15": g(r15, "pct_plus3r_before_minus1r"),
        "pct_plus3r_before_minus1r_m30": g(r30, "pct_plus3r_before_minus1r"),
        "expectancy_1to3_r_m15": g(r15, "expectancy_1to3_r"),
        "expectancy_1to3_r_m30": g(r30, "expectancy_1to3_r"),
        "favorable_move_rate_m15": g(r15, "favorable_move_rate"),
        "favorable_move_rate_m30": g(r30, "favorable_move_rate"),
        "mfe_mae_ratio_m15": g(r15, "mfe_mae_ratio"),
        "mfe_mae_ratio_m30": g(r30, "mfe_mae_ratio"),
    }
    for metric in [
        "n_signals",
        "n_trades",
        "pct_plus1r_before_minus1r",
        "pct_plus2r_before_minus1r",
        "pct_plus3r_before_minus1r",
        "expectancy_1to3_r",
        "favorable_move_rate",
        "mfe_mae_ratio",
    ]:
        out[f"delta_{metric}_m30_minus_m15"] = out[f"{metric}_m30"] - out[f"{metric}_m15"]
    return out


def _best_by(df: pd.DataFrame, key_cols: Sequence[str], min_n: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df[pd.to_numeric(df["n_trades"], errors="coerce") >= float(min_n)].copy()
    if work.empty:
        return work
    work = work.sort_values(
        list(key_cols) + ["expectancy_1to3_r", "pct_plus3r_before_minus1r", "robustness_score", "n_trades"],
        ascending=[True] * len(key_cols) + [False, False, False, False],
        kind="mergesort",
    )
    if not key_cols:
        return work.head(1).reset_index(drop=True)
    return work.groupby(list(key_cols), dropna=False, sort=True).head(1).reset_index(drop=True)


def build_m15_vs_m30_comparison(
    m15: HorizonStudyResult,
    m30: HorizonStudyResult,
    min_n: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for h in COMPARISON_KEY_H:
        r15 = _extract_row(m15.overall, {"horizon_h": int(h)})
        r30 = _extract_row(m30.overall, {"horizon_h": int(h)})
        rows.append(
            _comparison_row(
                level="overall_horizon",
                symbol="ALL",
                session_scope="all_sessions",
                horizon_h=int(h),
                r15=r15,
                r30=r30,
            )
        )

    symbols = sorted(set(m15.by_symbol.get("symbol", pd.Series(dtype=str)).astype(str)).union(set(m30.by_symbol.get("symbol", pd.Series(dtype=str)).astype(str))))
    for symbol in symbols:
        for h in COMPARISON_KEY_H:
            r15 = _extract_row(m15.by_symbol, {"symbol": symbol, "horizon_h": int(h)})
            r30 = _extract_row(m30.by_symbol, {"symbol": symbol, "horizon_h": int(h)})
            rows.append(
                _comparison_row(
                    level="by_symbol_horizon",
                    symbol=symbol,
                    session_scope="all_sessions",
                    horizon_h=int(h),
                    r15=r15,
                    r30=r30,
                )
            )

    scopes = sorted(set(m15.by_session.get("session_scope", pd.Series(dtype=str)).astype(str)).union(set(m30.by_session.get("session_scope", pd.Series(dtype=str)).astype(str))))
    for scope in scopes:
        for h in COMPARISON_KEY_H:
            r15 = _extract_row(m15.by_session, {"session_scope": scope, "horizon_h": int(h)})
            r30 = _extract_row(m30.by_session, {"session_scope": scope, "horizon_h": int(h)})
            rows.append(
                _comparison_row(
                    level="by_session_horizon",
                    symbol="ALL",
                    session_scope=scope,
                    horizon_h=int(h),
                    r15=r15,
                    r30=r30,
                )
            )

    b15_overall = _best_by(m15.overall, key_cols=[], min_n=min_n)
    b30_overall = _best_by(m30.overall, key_cols=[], min_n=min_n)
    r15_o = b15_overall.iloc[0] if not b15_overall.empty else None
    r30_o = b30_overall.iloc[0] if not b30_overall.empty else None
    rows.append(
        _comparison_row(
            level="overall_best_horizon",
            symbol="ALL",
            session_scope="all_sessions",
            horizon_h=None,
            r15=r15_o,
            r30=r30_o,
            horizon_h_m15=int(r15_o["horizon_h"]) if r15_o is not None else None,
            horizon_h_m30=int(r30_o["horizon_h"]) if r30_o is not None else None,
        )
    )

    b15_symbol = _best_by(m15.by_symbol, key_cols=["symbol"], min_n=min_n)
    b30_symbol = _best_by(m30.by_symbol, key_cols=["symbol"], min_n=min_n)
    for symbol in symbols:
        r15 = _extract_row(b15_symbol, {"symbol": symbol})
        r30 = _extract_row(b30_symbol, {"symbol": symbol})
        rows.append(
            _comparison_row(
                level="by_symbol_best_horizon",
                symbol=symbol,
                session_scope="all_sessions",
                horizon_h=None,
                r15=r15,
                r30=r30,
                horizon_h_m15=int(r15["horizon_h"]) if r15 is not None else None,
                horizon_h_m30=int(r30["horizon_h"]) if r30 is not None else None,
            )
        )

    b15_session = _best_by(m15.by_session, key_cols=["session_scope"], min_n=min_n)
    b30_session = _best_by(m30.by_session, key_cols=["session_scope"], min_n=min_n)
    for scope in scopes:
        r15 = _extract_row(b15_session, {"session_scope": scope})
        r30 = _extract_row(b30_session, {"session_scope": scope})
        rows.append(
            _comparison_row(
                level="by_session_best_horizon",
                symbol="ALL",
                session_scope=scope,
                horizon_h=None,
                r15=r15,
                r30=r30,
                horizon_h_m15=int(r15["horizon_h"]) if r15 is not None else None,
                horizon_h_m30=int(r30["horizon_h"]) if r30 is not None else None,
            )
        )

    return pd.DataFrame(rows)


def _truth_line(res: HorizonStudyResult) -> str:
    return f"- truth_label: `{res.truth_label.value}` | requested_truth_mode: `{res.truth_mode.value}` | exact_export_available: `{bool(res.exact_available)}`"


def _best_symbol_horizon(by_symbol: pd.DataFrame, symbol: str, min_n: int) -> pd.Series | None:
    if by_symbol.empty:
        return None
    sub = by_symbol[by_symbol["symbol"].astype(str) == symbol].copy()
    if sub.empty:
        return None
    sub = sub[pd.to_numeric(sub["n_trades"], errors="coerce") >= float(min_n)]
    if sub.empty:
        return None
    sub = sub.sort_values(
        ["expectancy_1to3_r", "pct_plus3r_before_minus1r", "robustness_score", "n_trades"],
        ascending=[False, False, False, False],
        kind="mergesort",
    )
    return sub.iloc[0]


def build_report_markdown(
    *,
    m30: HorizonStudyResult,
    keep_watch_cut: pd.DataFrame,
    deployment: pd.DataFrame,
    best_horizons: pd.DataFrame,
    comparison: pd.DataFrame,
    min_n: int,
) -> str:
    lines: List[str] = []
    lines.append("# Pine16 M30 Horizon Grid Report")
    lines.append("")
    lines.append("## 1. Truth stamp")
    lines.append(_truth_line(m30))
    if m30.parity_metrics is not None:
        lines.append(f"- parity_context: `{json.dumps(m30.parity_metrics, sort_keys=True)}`")
    lines.append("")
    lines.append("## 2. Study scope")
    lines.append(_truth_line(m30))
    lines.append(f"- config_pack: `{m30.cfg.metadata.config_pack}`")
    lines.append(f"- timeframe_signal_generation: `{m30.cfg.timeframe}`")
    lines.append(f"- horizon_grid_h: `{','.join(str(h) for h in HORIZON_GRID_H)}`")
    lines.append(f"- deployment_focus_h: `{','.join(str(h) for h in DEPLOYMENT_FOCUS_H)}`")
    lines.append(f"- n_master_rows: `{len(m30.master)}`")
    lines.append("")

    overall = m30.overall.copy()
    overall_focus = overall[overall["horizon_h"].isin(DEPLOYMENT_FOCUS_H)].copy() if not overall.empty else pd.DataFrame()
    fast = _pick_best_row(overall_focus, metric="pct_plus1r_before_minus1r", min_n=min_n)
    medium = _pick_best_row(overall_focus, metric="expectancy_1to2_r", min_n=min_n)
    full = _pick_best_row(overall_focus, metric="expectancy_1to3_r", min_n=min_n)

    lines.append("## 3. Horizon answers (M30)")
    lines.append(_truth_line(m30))
    if fast is not None:
        lines.append(
            f"- fast proof of direction: horizon `{int(fast['horizon_h'])}h` (plus1={float(fast['pct_plus1r_before_minus1r']):.4f}, n={int(fast['n_trades'])})"
        )
    if medium is not None:
        lines.append(
            f"- medium swing capture: horizon `{int(medium['horizon_h'])}h` (E[1:2]={float(medium['expectancy_1to2_r']):.4f}, n={int(medium['n_trades'])})"
        )
    if full is not None:
        lines.append(
            f"- full 3R monetization: horizon `{int(full['horizon_h'])}h` (E[1:3]={float(full['expectancy_1to3_r']):.4f}, plus3={float(full['pct_plus3r_before_minus1r']):.4f})"
        )
    lines.append("")

    best_xau = _best_symbol_horizon(m30.by_symbol, "XAUUSD", min_n=min_n)
    best_xag = _best_symbol_horizon(m30.by_symbol, "XAGUSD", min_n=min_n)
    best_session = _pick_best_row(m30.by_session, metric="expectancy_1to3_r", min_n=min_n)
    london_best = _pick_best_row(
        m30.by_session[m30.by_session["session_scope"].astype(str) == "london_only"].copy(),
        metric="expectancy_1to3_r",
        min_n=min_n,
    )
    ny_best = _pick_best_row(
        m30.by_session[m30.by_session["session_scope"].astype(str) == "newyork_only"].copy(),
        metric="expectancy_1to3_r",
        min_n=min_n,
    )
    lon_ny_best = _pick_best_row(
        m30.by_session[m30.by_session["session_scope"].astype(str) == "london_or_newyork"].copy(),
        metric="expectancy_1to3_r",
        min_n=min_n,
    )
    all_best = _pick_best_row(
        m30.by_session[m30.by_session["session_scope"].astype(str) == "all_sessions"].copy(),
        metric="expectancy_1to3_r",
        min_n=min_n,
    )

    lines.append("## 4. Symbol/session conclusions")
    lines.append(_truth_line(m30))
    if best_xau is not None:
        lines.append(
            f"- XAUUSD best horizon: `{int(best_xau['horizon_h'])}h` | plus1={float(best_xau['pct_plus1r_before_minus1r']):.4f}, plus3={float(best_xau['pct_plus3r_before_minus1r']):.4f}, E[1:3]={float(best_xau['expectancy_1to3_r']):.4f}"
        )
    if best_xag is not None:
        lines.append(
            f"- XAGUSD best horizon: `{int(best_xag['horizon_h'])}h` | plus1={float(best_xag['pct_plus1r_before_minus1r']):.4f}, plus3={float(best_xag['pct_plus3r_before_minus1r']):.4f}, E[1:3]={float(best_xag['expectancy_1to3_r']):.4f}"
        )
    if best_session is not None:
        lines.append(
            f"- best session scope on M30: `{best_session['session_scope']}` at `{int(best_session['horizon_h'])}h` | E[1:3]={float(best_session['expectancy_1to3_r']):.4f}"
        )
    if london_best is not None and ny_best is not None:
        lines.append(
            f"- London vs New York (best-horizon expectancy): London={float(london_best['expectancy_1to3_r']):.4f}, NewYork={float(ny_best['expectancy_1to3_r']):.4f}"
        )
    if lon_ny_best is not None and all_best is not None:
        lines.append(
            f"- London+NY vs All sessions (best-horizon expectancy): London+NY={float(lon_ny_best['expectancy_1to3_r']):.4f}, All={float(all_best['expectancy_1to3_r']):.4f}"
        )
    lines.append("")

    cmp_overall_best = comparison[comparison["comparison_level"] == "overall_best_horizon"].head(1)
    lines.append("## 5. M15 vs M30")
    lines.append(_truth_line(m30))
    if not cmp_overall_best.empty:
        r = cmp_overall_best.iloc[0]
        m30_better = float(pd.to_numeric(r["delta_expectancy_1to3_r_m30_minus_m15"], errors="coerce")) > 0
        cleaner_lower_freq = (
            float(pd.to_numeric(r["delta_n_trades_m30_minus_m15"], errors="coerce")) < 0
            and float(pd.to_numeric(r["delta_pct_plus1r_before_minus1r_m30_minus_m15"], errors="coerce")) > 0
        )
        lines.append(
            f"- best-horizon comparison: M15 h={int(r['horizon_h_m15']) if pd.notna(r['horizon_h_m15']) else 'na'} vs M30 h={int(r['horizon_h_m30']) if pd.notna(r['horizon_h_m30']) else 'na'}"
        )
        lines.append(
            f"- M30 better than M15 on expectancy_1to3: `{bool(m30_better)}` (delta={float(pd.to_numeric(r['delta_expectancy_1to3_r_m30_minus_m15'], errors='coerce')):.4f})"
        )
        lines.append(f"- M30 cleaner but lower frequency: `{bool(cleaner_lower_freq)}`")
    lines.append("")

    lines.append("## 6. Overall horizon table")
    lines.append(_truth_line(m30))
    lines.append(
        _md_table(
            m30.overall.sort_values("horizon_h"),
            [
                "horizon_h",
                "n_trades",
                "pct_plus1r_before_minus1r",
                "pct_plus2r_before_minus1r",
                "pct_plus3r_before_minus1r",
                "expectancy_1to3_r",
                "favorable_move_rate",
                "mfe_mae_ratio",
                "directional_proof_verdict",
                "tradable_1to3_verdict",
                "path_quality_verdict",
            ],
        )
    )
    lines.append("")

    lines.append("## 7. Keep/Watch/Cut")
    lines.append(_truth_line(m30))
    lines.append(
        _md_table(
            keep_watch_cut,
            [
                "symbol",
                "session_scope",
                "horizon_h",
                "n_trades",
                "pct_plus1r_before_minus1r",
                "pct_plus3r_before_minus1r",
                "expectancy_1to3_r",
                "robustness_score",
                "action",
            ],
        )
    )
    lines.append("")

    lines.append("## 8. Deployment candidates")
    lines.append(_truth_line(m30))
    lines.append(
        _md_table(
            deployment,
            [
                "candidate_type",
                "candidate_name",
                "symbol",
                "session_scope",
                "horizon_h",
                "n_trades",
                "pct_plus1r_before_minus1r",
                "pct_plus3r_before_minus1r",
                "expectancy_1to3_r",
                "deployment_action",
            ],
        )
    )
    lines.append("")

    lines.append("## 9. Best horizons")
    lines.append(_truth_line(m30))
    lines.append(
        _md_table(
            best_horizons.head(40),
            [
                "symbol",
                "session_scope",
                "horizon_h",
                "n",
                "pct_plus1r_before_minus1r",
                "pct_plus2r_before_minus1r",
                "pct_plus3r_before_minus1r",
                "expectancy_1to3_r",
                "mfe_mae_ratio",
                "robustness_score",
                "verdict",
                "sample_tier",
            ],
        )
    )
    lines.append("")

    lines.append("## 10. Caveats")
    lines.append(_truth_line(m30))
    lines.append("- Same-bar TP/SL conflicts are resolved conservatively as adverse-first (LOSS).")
    lines.append("- Barrier/path outcomes are bar-based (OHLC path), not tick reconstruction.")
    if m30.truth_label != TruthLabel.EXACT_PINE_EXPORTED:
        lines.append("- Findings are not exact Pine export truth for this run.")
    if m30.truth_label == TruthLabel.UNVERIFIED_PYTHON_APPROXIMATION:
        lines.append("- Findings are UNVERIFIED_PYTHON_APPROXIMATION and must be treated as provisional.")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_m30_horizon_audit(audit_path: Path, exact_dir: Path) -> Path:
    def _rows(path: Path) -> int:
        if not path.exists():
            return -1
        try:
            return int(len(pd.read_parquet(path)))
        except Exception:
            return -1

    trades_exact_rows = _rows(exact_dir / "trades_exact_pine.parquet")
    signals_exact_rows = _rows(exact_dir / "signals_exact_pine.parquet")
    parity_trades_rows = _rows(exact_dir / "parity_trades.parquet")
    parity_signals_rows = _rows(exact_dir / "parity_signals.parquet")

    parity_v = None
    parity_path = exact_dir / "parity_verification.json"
    if parity_path.exists():
        try:
            parity_v = json.loads(parity_path.read_text(encoding="utf-8"))
        except Exception:
            parity_v = None

    lines = [
        "# Pine16 M30 Horizon Grid Audit",
        "",
        "## Current artifact state",
        f"- trades_exact_pine rows: `{trades_exact_rows}`",
        f"- signals_exact_pine rows: `{signals_exact_rows}`",
        f"- parity_trades rows: `{parity_trades_rows}`",
        f"- parity_signals rows: `{parity_signals_rows}`",
        f"- parity_verification: `{json.dumps(parity_v, sort_keys=True) if parity_v is not None else 'missing'}`",
        "",
        "## Readiness summary",
        "1. M30 signal generation support: present via `Pine16ExactConfig.timeframe` + timeframe-aware data loading/resampling.",
        "2. Forward simulation style: timeframe-agnostic in core path; existing deep research has fixed horizon list and must be extended.",
        "3. Horizon grid support: partial; dedicated module needed for full requested grid.",
        "4. Session classification on M30: valid (UTC+3 named sessions, inclusive bounds).",
        "5. Barrier reuse: +1R/+2R/+3R reusable; +0.5R currently not generalized across all horizons.",
        "6. Required changes: dedicated M30 horizon master + full grouped outputs + leaderboard/deployment/comparison artifacts.",
        "7. Truth limits: exact and verified parity claims remain blocked without M30 exact exports/parity pass evidence.",
        "",
        "## Mapping table",
        "| concept | current implementation | M30 readiness | action required |",
        "| --- | --- | --- | --- |",
        "| Timeframe handling | `data.py` timeframe map + resample fallback | ready | none |",
        "| Session logic | `pine16_session.py` and `session_gate.py` UTC+3 named sessions | ready | none |",
        "| Horizon list | deep research constant `[2,4,8,24,48,72,120,168]` | partial | implement full grid |",
        "| +0.5R barrier | available as 24h-only field | partial | compute across all horizons |",
        "| Truth labels | exact/verified/unverified enums + routing | ready | stamp every new output |",
        "| M15 vs M30 comparison | no dedicated artifact | missing | add explicit comparison CSV/report section |",
        "",
    ]
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return audit_path


def run_m30_horizon_research(
    *,
    config_path: str,
    truth_mode_raw: str,
    exact_dir: Path,
    output_dir: Path,
    min_n: int = 20,
    min_n_robust: int = 30,
    export_html: bool = True,
    compare_m15_config: str = "configs/pine16_exact_prod_all_sessions.yaml",
    run_audit: bool = False,
    audit_path: Path | None = None,
) -> M30HorizonArtifacts:
    m30 = run_horizon_study(config_path=config_path, truth_mode_raw=truth_mode_raw, exact_dir=exact_dir, horizons_h=HORIZON_GRID_H)

    m15_cfg_path = Path(compare_m15_config)
    if m15_cfg_path.exists():
        m15 = run_horizon_study(
            config_path=str(m15_cfg_path),
            truth_mode_raw=truth_mode_raw,
            exact_dir=exact_dir,
            horizons_h=HORIZON_GRID_H,
        )
    else:
        m15 = HorizonStudyResult(
            cfg=m30.cfg,
            truth_mode=m30.truth_mode,
            truth_label=m30.truth_label,
            parity_metrics=None,
            exact_available=m30.exact_available,
            signals_cls=pd.DataFrame(),
            trades_cls=pd.DataFrame(),
            master=pd.DataFrame(),
            overall=pd.DataFrame(),
            by_symbol=pd.DataFrame(),
            by_session=pd.DataFrame(),
            by_year=pd.DataFrame(),
            by_symbol_year=pd.DataFrame(),
            by_symbol_session=pd.DataFrame(),
            by_year_session=pd.DataFrame(),
            by_symbol_year_session=pd.DataFrame(),
        )

    best_horizons = build_best_horizons(m30, min_n=min_n, min_n_robust=min_n_robust)
    leaderboard = build_horizon_leaderboard(m30, min_n=min_n, min_n_robust=min_n_robust)
    keep_watch_cut = build_keep_watch_cut(m30, min_n=min_n, min_n_robust=min_n_robust)
    deployment = build_deployment_candidates(m30, min_n=min_n, min_n_robust=min_n_robust)
    comparison = build_m15_vs_m30_comparison(m15=m15, m30=m30, min_n=min_n)

    report_md = build_report_markdown(
        m30=m30,
        keep_watch_cut=keep_watch_cut,
        deployment=deployment,
        best_horizons=best_horizons,
        comparison=comparison,
        min_n=min_n,
    )

    exact_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    master_path = exact_dir / "m30_horizon_master.parquet"
    report_md_path = output_dir / "pine16_m30_horizon_report.md"
    report_html_path = output_dir / "pine16_m30_horizon_report.html"
    overall_path = output_dir / "pine16_m30_horizon_overall.csv"
    by_symbol_path = output_dir / "pine16_m30_horizon_by_symbol.csv"
    by_session_path = output_dir / "pine16_m30_horizon_by_session.csv"
    by_year_path = output_dir / "pine16_m30_horizon_by_year.csv"
    by_symbol_year_path = output_dir / "pine16_m30_horizon_by_symbol_year.csv"
    by_symbol_session_path = output_dir / "pine16_m30_horizon_by_symbol_session.csv"
    by_year_session_path = output_dir / "pine16_m30_horizon_by_year_session.csv"
    by_symbol_year_session_path = output_dir / "pine16_m30_horizon_by_symbol_year_session.csv"
    horizon_leaderboard_path = output_dir / "pine16_m30_horizon_leaderboard.csv"
    best_horizons_path = output_dir / "pine16_m30_best_horizons.csv"
    keep_watch_cut_path = output_dir / "pine16_m30_horizon_keep_watch_cut.csv"
    deployment_csv_path = output_dir / "pine16_m30_horizon_deployment_candidates.csv"
    deployment_md_path = output_dir / "pine16_m30_horizon_deployment_candidates.md"
    comparison_path = output_dir / "pine16_m15_vs_m30_horizon_comparison.csv"

    m30.master.to_parquet(master_path, index=False)
    m30.overall.to_csv(overall_path, index=False)
    m30.by_symbol.to_csv(by_symbol_path, index=False)
    m30.by_session.to_csv(by_session_path, index=False)
    m30.by_year.to_csv(by_year_path, index=False)
    m30.by_symbol_year.to_csv(by_symbol_year_path, index=False)
    m30.by_symbol_session.to_csv(by_symbol_session_path, index=False)
    m30.by_year_session.to_csv(by_year_session_path, index=False)
    m30.by_symbol_year_session.to_csv(by_symbol_year_session_path, index=False)
    leaderboard.to_csv(horizon_leaderboard_path, index=False)
    best_horizons.to_csv(best_horizons_path, index=False)
    keep_watch_cut.to_csv(keep_watch_cut_path, index=False)
    deployment.to_csv(deployment_csv_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    report_md_path.write_text(report_md, encoding="utf-8")
    if export_html:
        _write_html_from_markdown(report_md, report_html_path)
    else:
        report_html_path.write_text("", encoding="utf-8")

    dep_lines = [
        "# Pine16 M30 Deployment Candidates",
        "",
        f"- truth_label: `{m30.truth_label.value}`",
        _md_table(
            deployment,
            [
                "candidate_type",
                "candidate_name",
                "symbol",
                "session_scope",
                "horizon_h",
                "n_trades",
                "pct_plus1r_before_minus1r",
                "pct_plus3r_before_minus1r",
                "expectancy_1to3_r",
                "deployment_action",
            ],
        ),
    ]
    deployment_md_path.write_text("\n".join(dep_lines).rstrip() + "\n", encoding="utf-8")

    out_audit: Path | None = None
    if run_audit:
        out_audit = write_m30_horizon_audit(
            audit_path=audit_path or Path("outputs/audit_pine16_m30_horizon_grid.md"),
            exact_dir=exact_dir,
        )

    return M30HorizonArtifacts(
        audit_md=out_audit,
        master_parquet=master_path,
        report_md=report_md_path,
        report_html=report_html_path,
        overall_csv=overall_path,
        by_symbol_csv=by_symbol_path,
        by_session_csv=by_session_path,
        by_year_csv=by_year_path,
        by_symbol_year_csv=by_symbol_year_path,
        by_symbol_session_csv=by_symbol_session_path,
        by_year_session_csv=by_year_session_path,
        leaderboard_csv=horizon_leaderboard_path,
        keep_watch_cut_csv=keep_watch_cut_path,
        deployment_candidates_csv=deployment_csv_path,
        deployment_candidates_md=deployment_md_path,
        best_horizons_csv=best_horizons_path,
        m15_vs_m30_comparison_csv=comparison_path,
        by_symbol_year_session_csv=by_symbol_year_session_path,
        horizon_leaderboard_csv=horizon_leaderboard_path,
    )
