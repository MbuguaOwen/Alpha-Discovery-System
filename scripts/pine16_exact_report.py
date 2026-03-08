from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from dfd05.data import load_bars_for_symbol, timeframe_to_minutes
from dfd05.indicators import atr
from dfd05.pine16_config import load_pine16_exact_config, to_legacy_run_config
from dfd05.pine16_parity_engine import compare_parity_to_exact, run_python_parity
from dfd05.pine16_truth import TruthLabel, TruthMode, normalize_truth_mode


HORIZONS_H = [2, 4, 8, 24, 48, 72, 120, 168]


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


def _load_exact_exports(output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades_path = output_dir / "trades_exact_pine.parquet"
    signals_path = output_dir / "signals_exact_pine.parquet"
    summary_path = output_dir / "summary_exact_pine.parquet"

    trades = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()
    signals = pd.read_parquet(signals_path) if signals_path.exists() else pd.DataFrame()
    summary = pd.read_parquet(summary_path) if summary_path.exists() else pd.DataFrame()
    return trades, signals, summary


def _load_parity_cache(exact_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    trades_path = exact_dir / "parity_trades.parquet"
    signals_path = exact_dir / "parity_signals.parquet"
    trades = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()
    signals = pd.read_parquet(signals_path) if signals_path.exists() else pd.DataFrame()
    return trades, signals


def _save_parity_cache(exact_dir: Path, trades: pd.DataFrame, signals: pd.DataFrame) -> None:
    exact_dir.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(exact_dir / "parity_trades.parquet", index=False)
    signals.to_parquet(exact_dir / "parity_signals.parquet", index=False)


def _load_parity_verification(exact_dir: Path) -> Dict[str, object] | None:
    p = exact_dir / "parity_verification.json"
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_trade_frame(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    out = trades.copy()
    out["entry_time_utc"] = pd.to_datetime(out["entry_time_utc"], utc=True, errors="coerce")
    out["exit_time_utc"] = pd.to_datetime(out["exit_time_utc"], utc=True, errors="coerce")
    out["entry_price"] = pd.to_numeric(out["entry_price"], errors="coerce")
    out["exit_price"] = pd.to_numeric(out["exit_price"], errors="coerce")
    out["result_r"] = pd.to_numeric(out["result_r"], errors="coerce")
    out["result_pct"] = pd.to_numeric(out["result_pct"], errors="coerce")
    out["year"] = pd.to_numeric(out.get("year"), errors="coerce").astype("Int64")
    out["month"] = pd.to_numeric(out.get("month"), errors="coerce").astype("Int64")
    return out


def _normalize_signal_frame(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals
    out = signals.copy()
    out["bar_time_utc"] = pd.to_datetime(out["bar_time_utc"], utc=True, errors="coerce")
    out["entry_time_utc"] = pd.to_datetime(out["entry_time_utc"], utc=True, errors="coerce")
    out["entry_price"] = pd.to_numeric(out["entry_price"], errors="coerce")
    return out


def _trade_stats(trades: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    tr = trades.copy()
    tr = tr.dropna(subset=["result_r"])
    if tr.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    def agg(g: pd.DataFrame) -> pd.Series:
        r = pd.to_numeric(g["result_r"], errors="coerce")
        wins = r[r > 0]
        losses = r[r < 0]
        pf = float(wins.sum() / (-losses.sum())) if len(losses) > 0 and (-losses.sum()) > 0 else np.nan
        return pd.Series(
            {
                "n_trades": int(len(g)),
                "win_rate": float((r > 0).mean()),
                "expectancy_r": float(r.mean()),
                "net_r": float(r.sum()),
                "profit_factor": pf,
            }
        )

    overall = agg(tr).to_frame().T
    by_symbol = tr.groupby("symbol", dropna=False).apply(agg).reset_index()
    by_year = tr.groupby("year", dropna=False).apply(agg).reset_index()
    by_symbol_year = tr.groupby(["symbol", "year"], dropna=False).apply(agg).reset_index()
    return overall, by_symbol, by_year, by_symbol_year


def _scan_first_touch(high: np.ndarray, low: np.ndarray, entry_i: int, tp: float, sl: float, bars: int) -> str:
    start = entry_i + 1
    end = entry_i + bars
    if start < 0 or end >= len(high) or bars <= 0:
        return "TRUNCATED"
    for i in range(start, end + 1):
        hit_tp = high[i] >= tp
        hit_sl = low[i] <= sl
        if hit_tp and hit_sl:
            return "SL"  # conservative tie-break
        if hit_tp:
            return "TP"
        if hit_sl:
            return "SL"
    return "OPEN"


def _directional_rows(signals: pd.DataFrame, cfg) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()

    legacy_cfg = to_legacy_run_config(cfg)
    tf_minutes = timeframe_to_minutes(cfg.timeframe)
    per_symbol_bars: Dict[str, pd.DataFrame] = {}
    rows: List[Dict[str, object]] = []

    for symbol in sorted(signals["symbol"].astype(str).unique().tolist()):
        try:
            bars = load_bars_for_symbol(legacy_cfg, symbol=symbol, timeframe=cfg.timeframe)
        except FileNotFoundError:
            continue
        if bars.empty:
            continue
        per_symbol_bars[symbol] = bars

    for _, s in signals.iterrows():
        symbol = str(s["symbol"])
        if symbol not in per_symbol_bars:
            continue
        bars = per_symbol_bars[symbol]
        times = pd.to_datetime(bars["time"], utc=True)
        entry_t = pd.to_datetime(s["entry_time_utc"], utc=True, errors="coerce")
        if pd.isna(entry_t):
            continue

        idx = pd.Index(times).get_indexer([entry_t], method="nearest")
        if len(idx) == 0 or idx[0] < 0:
            continue
        ei = int(idx[0])
        entry_px = float(pd.to_numeric(s.get("entry_price"), errors="coerce"))
        if not np.isfinite(entry_px):
            entry_px = float(pd.to_numeric(bars.iloc[ei]["close"], errors="coerce"))
        if not np.isfinite(entry_px) or entry_px == 0.0:
            continue

        high = bars["high"].to_numpy(dtype=float)
        low = bars["low"].to_numpy(dtype=float)
        close = bars["close"].to_numpy(dtype=float)
        atr_arr = atr(high, low, close, int(cfg.risk.atrLen))
        atr_e = float(atr_arr[ei]) if ei < len(atr_arr) else np.nan
        if not np.isfinite(atr_e) or atr_e <= 0:
            continue

        year = int(pd.Timestamp(entry_t).year)

        for h in HORIZONS_H:
            bars_fwd = int((h * 60) // tf_minutes)
            start = ei + 1
            end = ei + bars_fwd
            if bars_fwd <= 0 or end >= len(bars):
                continue
            win_high = high[start : end + 1]
            win_low = low[start : end + 1]
            c_end = close[end]
            mfe_r = float((float(np.max(win_high)) - entry_px) / (float(cfg.risk.slAtrMult) * atr_e))
            mae_r = float((float(np.min(win_low)) - entry_px) / (float(cfg.risk.slAtrMult) * atr_e))
            favorable = float(c_end > entry_px)
            mfe_gt_mae = float(mfe_r > abs(mae_r))

            sl = float(entry_px - float(cfg.risk.slAtrMult) * atr_e)
            tp1 = float(entry_px + float(cfg.risk.slAtrMult) * 1.0 * atr_e)
            tp3 = float(entry_px + float(cfg.risk.slAtrMult) * 3.0 * atr_e)
            hit1 = _scan_first_touch(high, low, ei, tp1, sl, bars_fwd)
            hit3 = _scan_first_touch(high, low, ei, tp3, sl, bars_fwd)

            rows.append(
                {
                    "symbol": symbol,
                    "year": year,
                    "horizon_h": int(h),
                    "favorable": favorable,
                    "mfe_r": mfe_r,
                    "mae_r": mae_r,
                    "mfe_gt_mae": mfe_gt_mae,
                    "reach_1r_before_minus1r": float(hit1 == "TP"),
                    "reach_3r_before_minus1r": float(hit3 == "TP"),
                    "sl_before_3r": float(hit3 == "SL"),
                }
            )

    return pd.DataFrame(rows)


def _classify_directional(favorable: float, ratio: float, mfe_gt_mae: float) -> str:
    if np.isfinite(favorable) and np.isfinite(ratio) and np.isfinite(mfe_gt_mae):
        if favorable >= 0.60 and ratio >= 1.30 and mfe_gt_mae >= 0.60:
            return "Strong directional edge"
        if favorable >= 0.55 and ratio >= 1.15 and mfe_gt_mae >= 0.55:
            return "Clear directional edge"
        if favorable >= 0.52 and ratio >= 1.02:
            return "Weak directional edge"
    return "No directional edge"


def _classify_monetizable(expectancy_r: float, pf: float, net_r: float, n_trades: int) -> str:
    if n_trades < 30:
        return "Marginal at 1:3"
    if np.isfinite(expectancy_r) and np.isfinite(pf) and np.isfinite(net_r):
        if expectancy_r >= 0.20 and pf >= 1.30 and net_r > 0:
            return "Strongly monetizable at 1:3"
        if expectancy_r >= 0.05 and pf >= 1.10 and net_r > 0:
            return "Monetizable at 1:3"
        if -0.02 <= expectancy_r < 0.05:
            return "Marginal at 1:3"
    return "Not monetizable at 1:3"


def _aggregate_directional(edge_rows: pd.DataFrame, *, group_cols: List[str]) -> pd.DataFrame:
    if edge_rows.empty:
        return pd.DataFrame()

    def agg(g: pd.DataFrame) -> pd.Series:
        med_mfe = float(g["mfe_r"].median())
        med_mae = float(g["mae_r"].median())
        ratio = float(med_mfe / abs(med_mae)) if np.isfinite(med_mae) and med_mae != 0 else np.nan
        win3 = float(g["reach_3r_before_minus1r"].mean())
        loss3 = float(g["sl_before_3r"].mean())
        expectancy_1to3 = float(3.0 * win3 - loss3)
        pf = float((3.0 * win3) / loss3) if loss3 > 0 else np.nan
        return pd.Series(
            {
                "n_signals": int(len(g)),
                "favorable_move_rate": float(g["favorable"].mean()),
                "median_mfe_r": med_mfe,
                "median_mae_r": med_mae,
                "mfe_mae_ratio": ratio,
                "pct_mfe_gt_mae": float(g["mfe_gt_mae"].mean()),
                "pct_reach_1r_before_minus1r": float(g["reach_1r_before_minus1r"].mean()),
                "pct_reach_3r_before_minus1r": win3,
                "pct_sl_before_3r": loss3,
                "expectancy_1to3_r": expectancy_1to3,
                "pf_1to3": pf,
            }
        )

    out = edge_rows.groupby(group_cols, dropna=False).apply(agg).reset_index()
    out["directional_verdict"] = [
        _classify_directional(r.favorable_move_rate, r.mfe_mae_ratio, r.pct_mfe_gt_mae)
        for r in out.itertuples(index=False)
    ]
    out["monetizable_verdict"] = [
        _classify_monetizable(r.expectancy_1to3_r, r.pf_1to3, r.expectancy_1to3_r * r.n_signals, int(r.n_signals))
        for r in out.itertuples(index=False)
    ]
    return out


def _truth_for_block(mode: TruthMode, exact_available: bool, parity_verified: bool) -> TruthLabel:
    if mode == TruthMode.EXACT_PINE_EXPORTED:
        if exact_available:
            return TruthLabel.EXACT_PINE_EXPORTED
        raise SystemExit("Exact Pine exported mode requested, but normalized TradingView exports are missing.")
    if mode == TruthMode.VERIFIED_PYTHON_PARITY:
        if exact_available:
            return TruthLabel.EXACT_PINE_EXPORTED
        return TruthLabel.VERIFIED_PYTHON_PARITY if parity_verified else TruthLabel.UNVERIFIED_PYTHON_APPROXIMATION
    return TruthLabel.UNVERIFIED_PYTHON_APPROXIMATION


def _write_html_from_markdown(md: str, path: Path) -> None:
    body = html.escape(md)
    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Pine16 Exact Report</title>"
        "<style>body{font-family:ui-monospace,Consolas,monospace;padding:24px;}pre{white-space:pre-wrap;}</style>"
        "</head><body><pre>"
        f"{body}"
        "</pre></body></html>"
    )
    path.write_text(page, encoding="utf-8")


def run_report(config_path: str, truth_mode_raw: str, output_dir: Path, exact_dir: Path) -> Dict[str, Path]:
    cfg = load_pine16_exact_config(config_path)
    truth_mode = normalize_truth_mode(truth_mode_raw)

    exact_trades, exact_signals, _exact_summary = _load_exact_exports(exact_dir)
    exact_trades = _normalize_trade_frame(exact_trades)
    exact_signals = _normalize_signal_frame(exact_signals)
    exact_available = not exact_trades.empty
    if truth_mode == TruthMode.EXACT_PINE_EXPORTED and not exact_available:
        raise SystemExit("Exact Pine exported mode requested, but normalized TradingView exports are missing.")

    parity_verification = _load_parity_verification(exact_dir)
    parity_metrics: Dict[str, object] | None = None

    need_parity_data = truth_mode != TruthMode.EXACT_PINE_EXPORTED
    if need_parity_data:
        parity_trades, parity_signals = _load_parity_cache(exact_dir)
        parity_trades = _normalize_trade_frame(parity_trades)
        parity_signals = _normalize_signal_frame(parity_signals)
        if parity_trades.empty and parity_signals.empty:
            parity_run = run_python_parity(cfg)
            parity_trades = _normalize_trade_frame(parity_run.trades)
            parity_signals = _normalize_signal_frame(parity_run.signals)
            _save_parity_cache(exact_dir, parity_trades, parity_signals)
    else:
        parity_trades = pd.DataFrame()
        parity_signals = pd.DataFrame()

    parity_verified = False
    if exact_available and not parity_trades.empty:
        parity_cmp = compare_parity_to_exact(
            exact_trades=exact_trades,
            parity_trades=parity_trades,
            timeframe=cfg.timeframe,
        )
        parity_verified = bool(parity_cmp.pass_thresholds)
        parity_metrics = {
            "signal_count_mismatch_pct": float(parity_cmp.signal_count_mismatch_pct),
            "entry_timestamp_max_bar_diff": float(parity_cmp.entry_timestamp_max_bar_diff),
            "exit_timestamp_max_bar_diff": float(parity_cmp.exit_timestamp_max_bar_diff),
            "aggregate_net_r_mismatch_pct": float(parity_cmp.aggregate_net_r_mismatch_pct),
            "pass_thresholds": bool(parity_cmp.pass_thresholds),
            "source": "exact_exports_vs_python_parity",
        }
    elif parity_verification is not None:
        parity_verified = bool(parity_verification.get("pass_thresholds", False))
        parity_metrics = parity_verification

    truth_label = _truth_for_block(truth_mode, exact_available=exact_available, parity_verified=parity_verified)

    if truth_label == TruthLabel.EXACT_PINE_EXPORTED:
        trades = exact_trades.copy()
        if not exact_signals.empty:
            signals = exact_signals.copy()
        elif not exact_trades.empty:
            signals = exact_trades[
                [
                    "symbol",
                    "timeframe",
                    "tv_strategy_name",
                    "config_pack",
                    "bar_time_utc",
                    "entry_time_utc",
                    "side",
                    "entry_price",
                    "source_file",
                    "truth_label",
                ]
            ].copy()
        else:
            signals = pd.DataFrame()
    elif truth_label == TruthLabel.VERIFIED_PYTHON_PARITY:
        trades = parity_trades.copy()
        signals = parity_signals.copy()
    else:
        trades = parity_trades.copy()
        signals = parity_signals.copy()

    if not trades.empty:
        trades["truth_label"] = truth_label.value
    if not signals.empty:
        signals["truth_label"] = truth_label.value

    overall_stats, by_symbol, by_year, by_symbol_year = _trade_stats(trades)
    edge_rows = _directional_rows(signals, cfg)
    edge_overall = _aggregate_directional(edge_rows, group_cols=["horizon_h"])
    edge_by_symbol = _aggregate_directional(edge_rows, group_cols=["symbol", "horizon_h"])
    edge_by_year = _aggregate_directional(edge_rows, group_cols=["year", "horizon_h"])
    edge_by_symbol_year = _aggregate_directional(edge_rows, group_cols=["symbol", "year", "horizon_h"])

    final_directional = "No directional edge"
    final_monetizable = "Not monetizable at 1:3"
    if not edge_overall.empty:
        row24 = edge_overall[edge_overall["horizon_h"] == 24]
        pick = row24.iloc[0] if not row24.empty else edge_overall.iloc[0]
        final_directional = str(pick["directional_verdict"])
        final_monetizable = str(pick["monetizable_verdict"])

    if not overall_stats.empty and not edge_overall.empty:
        o = overall_stats.iloc[0]
        n_trades = int(o["n_trades"])
        final_monetizable = _classify_monetizable(
            float(o["expectancy_r"]),
            float(o["profit_factor"]) if "profit_factor" in o else np.nan,
            float(o["net_r"]),
            n_trades,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "pine16_exact_report.md"
    html_path = output_dir / "pine16_exact_report.html"
    summary_csv = output_dir / "pine16_exact_summary.csv"
    by_sy_csv = output_dir / "pine16_by_symbol_year.csv"
    edge_csv = output_dir / "pine16_directional_edge.csv"
    trade_log_path = output_dir / "pine16_trade_log.parquet"

    if not overall_stats.empty:
        overall_out = overall_stats.copy()
        overall_out["truth_label"] = truth_label.value
        overall_out["config_pack"] = cfg.metadata.config_pack
        overall_out.to_csv(summary_csv, index=False)
    else:
        pd.DataFrame(columns=["truth_label", "config_pack"]).to_csv(summary_csv, index=False)

    by_symbol_year_out = by_symbol_year.copy()
    if not by_symbol_year_out.empty:
        by_symbol_year_out["truth_label"] = truth_label.value
        by_symbol_year_out["config_pack"] = cfg.metadata.config_pack
    by_symbol_year_out.to_csv(by_sy_csv, index=False)

    edge_parts = []
    if not edge_overall.empty:
        a = edge_overall.copy()
        a["scope"] = "overall_by_horizon"
        edge_parts.append(a)
    if not edge_by_symbol.empty:
        b = edge_by_symbol.copy()
        b["scope"] = "by_symbol_horizon"
        edge_parts.append(b)
    if not edge_by_year.empty:
        c = edge_by_year.copy()
        c["scope"] = "by_year_horizon"
        edge_parts.append(c)
    if not edge_by_symbol_year.empty:
        d = edge_by_symbol_year.copy()
        d["scope"] = "by_symbol_year_horizon"
        edge_parts.append(d)
    edge_out = pd.concat(edge_parts, ignore_index=True, sort=False) if edge_parts else pd.DataFrame()
    if not edge_out.empty:
        edge_out["truth_label"] = truth_label.value
        edge_out["config_pack"] = cfg.metadata.config_pack
    edge_out.to_csv(edge_csv, index=False)
    trades.to_parquet(trade_log_path, index=False)

    lines: List[str] = []
    lines.append("# Pine16 Exact Report")
    lines.append("")
    lines.append("## 1. Executive verdict")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(f"- Directional edge verdict: **{final_directional}**")
    lines.append(f"- Monetizable 1:3 verdict: **{final_monetizable}**")
    lines.append("")
    lines.append("## 2. Truth source used")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(f"- Requested truth mode: `{truth_mode.value}`")
    lines.append(f"- Effective truth label: `{truth_label.value}`")
    lines.append(f"- Exact export available: `{bool(exact_available)}`")
    if parity_metrics is not None:
        lines.append(
            "- Parity comparison: "
            f"count_mismatch={parity_metrics.get('signal_count_mismatch_pct')}% "
            f"entry_max_bar_diff={parity_metrics.get('entry_timestamp_max_bar_diff')} "
            f"exit_max_bar_diff={parity_metrics.get('exit_timestamp_max_bar_diff')} "
            f"net_r_mismatch={parity_metrics.get('aggregate_net_r_mismatch_pct')}% "
            f"pass={parity_metrics.get('pass_thresholds')}"
        )
    elif truth_mode != TruthMode.UNVERIFIED_PYTHON_APPROX:
        lines.append("- Parity comparison: unavailable (exact exports missing or no parity trades).")
    lines.append("")
    lines.append("## 3. Config pack definitions")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(f"- config_pack: `{cfg.metadata.config_pack}`")
    lines.append(f"- timeframe: `{cfg.timeframe}`")
    lines.append(f"- symbols: `{','.join(cfg.symbols)}`")
    lines.append(f"- years: `{','.join(str(y) for y in cfg.data.years)}`")
    lines.append(f"- risk: atrLen={cfg.risk.atrLen}, slAtrMult={cfg.risk.slAtrMult}, rrMult={cfg.risk.rrMult}")
    lines.append("")
    lines.append("## 4. Strategy-level performance")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(overall_stats, ["n_trades", "win_rate", "expectancy_r", "profit_factor", "net_r"]))
    lines.append("")
    lines.append("## 5. By symbol")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(by_symbol, ["symbol", "n_trades", "win_rate", "expectancy_r", "profit_factor", "net_r"]))
    lines.append("")
    lines.append("## 6. By year")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(by_year, ["year", "n_trades", "win_rate", "expectancy_r", "profit_factor", "net_r"]))
    lines.append("")
    lines.append("## 7. By symbol x year")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(by_symbol_year, ["symbol", "year", "n_trades", "win_rate", "expectancy_r", "profit_factor", "net_r"]))
    lines.append("")
    lines.append("## 8. Trade count / win rate / expectancy / PF / net R")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(overall_stats, ["n_trades", "win_rate", "expectancy_r", "profit_factor", "net_r"]))
    lines.append("")
    lines.append("## 9. 1R stop / 3R target results")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(edge_overall, ["horizon_h", "n_signals", "pct_reach_3r_before_minus1r", "pct_sl_before_3r", "expectancy_1to3_r", "pf_1to3"]))
    lines.append("")
    lines.append("## 10. Directional edge analysis")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(edge_overall, ["horizon_h", "favorable_move_rate", "median_mfe_r", "median_mae_r", "mfe_mae_ratio", "pct_mfe_gt_mae", "directional_verdict"]))
    lines.append("")
    lines.append("Directional verdict by symbol (24h):")
    lines.append(_md_table(edge_by_symbol[edge_by_symbol["horizon_h"] == 24], ["symbol", "horizon_h", "n_signals", "directional_verdict", "monetizable_verdict"]))
    lines.append("")
    lines.append("Directional verdict by year (24h):")
    lines.append(_md_table(edge_by_year[edge_by_year["horizon_h"] == 24], ["year", "horizon_h", "n_signals", "directional_verdict", "monetizable_verdict"]))
    lines.append("")
    lines.append("## 11. MAE/MFE horizon analysis")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(edge_overall, ["horizon_h", "median_mfe_r", "median_mae_r", "mfe_mae_ratio"]))
    lines.append("")
    lines.append("## 12. Difference between directional edge and tradable edge")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append("- Directional edge uses excursion and favorable move behavior after signal.")
    lines.append("- Tradable edge uses realized strategy outcomes (expectancy/PF/netR) under 1R stop and 3R target mechanics.")
    lines.append("")
    lines.append("## 13. Exact Pine vs Python parity comparison")
    lines.append(f"- truth_label: `{truth_label.value}`")
    if parity_metrics is None:
        lines.append("- Parity comparison unavailable.")
    else:
        lines.append(
            "- "
            f"signal_count_mismatch_pct={parity_metrics.get('signal_count_mismatch_pct')}, "
            f"entry_timestamp_max_bar_diff={parity_metrics.get('entry_timestamp_max_bar_diff')}, "
            f"exit_timestamp_max_bar_diff={parity_metrics.get('exit_timestamp_max_bar_diff')}, "
            f"aggregate_net_r_mismatch_pct={parity_metrics.get('aggregate_net_r_mismatch_pct')}, "
            f"pass_thresholds={parity_metrics.get('pass_thresholds')}"
        )
    lines.append("")
    lines.append("## 14. Failure points / caveats")
    lines.append(f"- truth_label: `{truth_label.value}`")
    if not exact_available:
        lines.append("- Exact Pine TradingView exports are missing; exact exported truth cannot be established.")
    if truth_label == TruthLabel.UNVERIFIED_PYTHON_APPROXIMATION:
        lines.append("- Output is explicitly unverified approximation and must not be presented as exact Pine results.")
    if truth_label == TruthLabel.VERIFIED_PYTHON_PARITY:
        lines.append("- Output is verified parity against available Pine artifacts and not direct Pine export truth.")
    lines.append("")
    lines.append("## Final Hard Verdict")
    lines.append(f"- Directional edge: **{final_directional}**")
    lines.append(f"- Monetizable at 1:3: **{final_monetizable}**")

    md = "\n".join(lines).rstrip() + "\n"
    md_path.write_text(md, encoding="utf-8")
    _write_html_from_markdown(md, html_path)

    return {
        "report_md": md_path,
        "report_html": html_path,
        "summary_csv": summary_csv,
        "by_symbol_year_csv": by_sy_csv,
        "directional_edge_csv": edge_csv,
        "trade_log": trade_log_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate Pine16 exact report with explicit truth labels.")
    ap.add_argument("--config", required=True, help="Pine16 exact config path.")
    ap.add_argument("--truth-mode", required=True, choices=[m.value for m in TruthMode], help="Truth mode selector.")
    ap.add_argument("--exact-dir", default="data/derived/pine16_exact", help="Normalized exact export directory.")
    ap.add_argument("--output-dir", default="outputs/reports", help="Output report directory.")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    outs = run_report(
        config_path=args.config,
        truth_mode_raw=args.truth_mode,
        output_dir=Path(args.output_dir),
        exact_dir=Path(args.exact_dir),
    )
    for k, v in outs.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

