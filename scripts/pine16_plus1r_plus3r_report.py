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
from dfd05.pine16_session import SESSION_HM, hm_from_utc, in_hm_range_inclusive
from dfd05.pine16_truth import TruthLabel, TruthMode, normalize_truth_mode


HORIZONS_H = [2, 4, 8, 24, 48, 72, 120, 168]
SESSION_SCOPES = ["all_sessions", "london_only", "newyork_only", "london_or_newyork", "other"]


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
    body = html.escape(md)
    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Pine16 +1R/+3R Report</title>"
        "<style>body{font-family:ui-monospace,Consolas,monospace;padding:24px;}pre{white-space:pre-wrap;}</style>"
        "</head><body><pre>"
        f"{body}"
        "</pre></body></html>"
    )
    path.write_text(page, encoding="utf-8")


def _load_exact_exports(exact_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    trades_path = exact_dir / "trades_exact_pine.parquet"
    signals_path = exact_dir / "signals_exact_pine.parquet"
    trades = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()
    signals = pd.read_parquet(signals_path) if signals_path.exists() else pd.DataFrame()
    return trades, signals


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
    for c in ["entry_time_utc", "exit_time_utc", "bar_time_utc", "signal_time_utc", "setup_time_utc"]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], utc=True, errors="coerce")
    for c in ["entry_price", "exit_price", "atr_entry", "sl_price", "tp_price", "result_r", "result_pct"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "entry_time_utc" in out.columns:
        out["year"] = pd.to_datetime(out["entry_time_utc"], utc=True, errors="coerce").dt.year.astype("Int64")
        out["month"] = pd.to_datetime(out["entry_time_utc"], utc=True, errors="coerce").dt.month.astype("Int64")
    if "trade_status" not in out.columns:
        out["trade_status"] = "CLOSED"
    return out


def _normalize_signal_frame(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals
    out = signals.copy()
    for c in ["bar_time_utc", "entry_time_utc", "signal_time_utc", "setup_time_utc"]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], utc=True, errors="coerce")
    if "entry_price" in out.columns:
        out["entry_price"] = pd.to_numeric(out["entry_price"], errors="coerce")
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


def _session_flags(ts_utc: pd.Timestamp | None) -> Dict[str, object]:
    if ts_utc is None or pd.isna(ts_utc):
        return {
            "in_london": 0,
            "in_newyork": 0,
            "in_london_or_newyork": 0,
            "bucket": "unknown",
        }
    hm = hm_from_utc(pd.Timestamp(ts_utc), "Etc/GMT-3")
    in_london = bool(in_hm_range_inclusive(hm, *SESSION_HM["london"]))
    in_newyork = bool(in_hm_range_inclusive(hm, *SESSION_HM["ny"]))
    in_lon_ny = bool(in_london or in_newyork)
    if in_london and not in_newyork:
        bucket = "london_only"
    elif in_newyork and not in_london:
        bucket = "newyork_only"
    elif in_lon_ny:
        bucket = "london_or_newyork"
    else:
        bucket = "other"
    return {
        "in_london": int(in_london),
        "in_newyork": int(in_newyork),
        "in_london_or_newyork": int(in_lon_ny),
        "bucket": bucket,
    }


def _choose_first_time(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            ser = pd.to_datetime(df[c], utc=True, errors="coerce")
            if ser.notna().any():
                return ser
    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")


def _classify_signals(signals: pd.DataFrame, truth_label: TruthLabel, config_pack: str) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "timeframe",
                "config_pack",
                "signal_time_utc",
                "setup_time_utc",
                "entry_time_utc",
                "entry_price",
                "setup_session_bucket",
                "entry_session_bucket",
                "setup_in_london",
                "setup_in_newyork",
                "entry_in_london",
                "entry_in_newyork",
                "entry_in_london_or_newyork",
                "truth_label",
            ]
        )

    out = signals.copy()
    out["signal_time_utc"] = _choose_first_time(out, ["signal_time_utc", "bar_time_utc"])
    out["setup_time_utc"] = _choose_first_time(out, ["setup_time_utc", "pivot_time_utc"])
    if "pivot_time_ms" in out.columns and out["setup_time_utc"].isna().all():
        out["setup_time_utc"] = pd.to_datetime(pd.to_numeric(out["pivot_time_ms"], errors="coerce"), unit="ms", utc=True, errors="coerce")
    out["entry_time_utc"] = _choose_first_time(out, ["entry_time_utc", "signal_time_utc", "bar_time_utc"])
    out["entry_price"] = pd.to_numeric(out.get("entry_price"), errors="coerce")
    out["config_pack"] = str(config_pack)
    out["truth_label"] = truth_label.value

    setup_flags = out["setup_time_utc"].apply(lambda x: _session_flags(x))
    entry_flags = out["entry_time_utc"].apply(lambda x: _session_flags(x))
    out["setup_session_bucket"] = setup_flags.apply(lambda d: d["bucket"])
    out["entry_session_bucket"] = entry_flags.apply(lambda d: d["bucket"])
    out["setup_in_london"] = setup_flags.apply(lambda d: d["in_london"]).astype(int)
    out["setup_in_newyork"] = setup_flags.apply(lambda d: d["in_newyork"]).astype(int)
    out["entry_in_london"] = entry_flags.apply(lambda d: d["in_london"]).astype(int)
    out["entry_in_newyork"] = entry_flags.apply(lambda d: d["in_newyork"]).astype(int)
    out["entry_in_london_or_newyork"] = entry_flags.apply(lambda d: d["in_london_or_newyork"]).astype(int)

    keep_cols = [
        "symbol",
        "timeframe",
        "config_pack",
        "signal_time_utc",
        "setup_time_utc",
        "entry_time_utc",
        "entry_price",
        "setup_session_bucket",
        "entry_session_bucket",
        "setup_in_london",
        "setup_in_newyork",
        "entry_in_london",
        "entry_in_newyork",
        "entry_in_london_or_newyork",
        "truth_label",
    ]
    for c in keep_cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[keep_cols].reset_index(drop=True)


def _classify_trades(trades: pd.DataFrame, truth_label: TruthLabel, config_pack: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    out = trades.copy()
    out["signal_time_utc"] = _choose_first_time(out, ["signal_time_utc", "bar_time_utc"])
    out["setup_time_utc"] = _choose_first_time(out, ["setup_time_utc", "pivot_time_utc"])
    if "pivot_time_ms" in out.columns and out["setup_time_utc"].isna().all():
        out["setup_time_utc"] = pd.to_datetime(pd.to_numeric(out["pivot_time_ms"], errors="coerce"), unit="ms", utc=True, errors="coerce")
    out["entry_time_utc"] = _choose_first_time(out, ["entry_time_utc", "signal_time_utc", "bar_time_utc"])

    setup_flags = out["setup_time_utc"].apply(lambda x: _session_flags(x))
    entry_flags = out["entry_time_utc"].apply(lambda x: _session_flags(x))
    out["setup_session_bucket"] = setup_flags.apply(lambda d: d["bucket"])
    out["entry_session_bucket"] = entry_flags.apply(lambda d: d["bucket"])
    out["setup_in_london"] = setup_flags.apply(lambda d: d["in_london"]).astype(int)
    out["setup_in_newyork"] = setup_flags.apply(lambda d: d["in_newyork"]).astype(int)
    out["entry_in_london"] = entry_flags.apply(lambda d: d["in_london"]).astype(int)
    out["entry_in_newyork"] = entry_flags.apply(lambda d: d["in_newyork"]).astype(int)
    out["entry_in_london_or_newyork"] = entry_flags.apply(lambda d: d["in_london_or_newyork"]).astype(int)
    out["truth_label"] = truth_label.value
    out["config_pack"] = str(config_pack)
    out["year"] = pd.to_datetime(out["entry_time_utc"], utc=True, errors="coerce").dt.year.astype("Int64")
    out["month"] = pd.to_datetime(out["entry_time_utc"], utc=True, errors="coerce").dt.month.astype("Int64")

    key_base = (
        out["symbol"].astype(str)
        + "|"
        + out["entry_time_utc"].astype(str)
        + "|"
        + pd.to_numeric(out["entry_price"], errors="coerce").round(10).astype(str)
    )
    out["trade_id"] = out.get("trade_id", key_base)
    out["signal_id"] = out.get("signal_id", key_base)

    return out.reset_index(drop=True)


def _scan_first_touch(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    entry_i: int,
    tp: float,
    sl: float,
    bars_fwd: int,
    ep: float,
    risk_unit: float,
) -> Tuple[str, float, float, int, int]:
    start = int(entry_i) + 1
    end_target = int(entry_i) + int(bars_fwd)
    if bars_fwd <= 0 or start >= len(high):
        return "OPEN", np.nan, np.nan, 0, 1

    truncated = 0
    end = end_target
    if end >= len(high):
        end = len(high) - 1
        truncated = 1
    if end < start:
        return "OPEN", np.nan, np.nan, 0, 1

    win_high = high[start : end + 1]
    win_low = low[start : end + 1]
    if risk_unit > 0:
        mfe_r = float((float(np.max(win_high)) - ep) / risk_unit)
        mae_r = float((float(np.min(win_low)) - ep) / risk_unit)
    else:
        mfe_r = np.nan
        mae_r = np.nan
    favorable = int(close[end] > ep) if np.isfinite(ep) else 0

    for i in range(start, end + 1):
        hit_tp = high[i] >= tp
        hit_sl = low[i] <= sl
        if hit_tp and hit_sl:
            return "LOSS", mfe_r, mae_r, favorable, truncated
        if hit_tp:
            return "WIN", mfe_r, mae_r, favorable, truncated
        if hit_sl:
            return "LOSS", mfe_r, mae_r, favorable, truncated
    return "OPEN", mfe_r, mae_r, favorable, truncated


def _build_barrier_outcomes(trades: pd.DataFrame, cfg) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    legacy_cfg = to_legacy_run_config(cfg)
    tf_minutes = timeframe_to_minutes(cfg.timeframe)
    rows: List[Dict[str, object]] = []

    for symbol in sorted(trades["symbol"].astype(str).unique().tolist()):
        sym_tr = trades[trades["symbol"].astype(str) == symbol].copy()
        if sym_tr.empty:
            continue
        try:
            bars = load_bars_for_symbol(legacy_cfg, symbol=symbol, timeframe=cfg.timeframe)
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
            pd.to_datetime(sym_tr["entry_time_utc"], utc=True, errors="coerce"),
            method="nearest",
            tolerance=pd.Timedelta(minutes=tf_minutes),
        )
        sym_tr = sym_tr.reset_index(drop=True)

        for j in range(len(sym_tr)):
            ei = int(idx[j]) if j < len(idx) else -1
            if ei < 0 or ei >= len(bars):
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

            sl = float(ep - risk_unit)
            tp1 = float(ep + risk_unit)
            tp3 = float(ep + 3.0 * risk_unit)

            for h in HORIZONS_H:
                bars_fwd = int((int(h) * 60) // tf_minutes)
                out1, mfe_r, mae_r, favorable, truncated = _scan_first_touch(
                    high=high,
                    low=low,
                    close=close,
                    entry_i=ei,
                    tp=tp1,
                    sl=sl,
                    bars_fwd=bars_fwd,
                    ep=ep,
                    risk_unit=risk_unit,
                )
                out3, _mfe3, _mae3, _fav3, _tr3 = _scan_first_touch(
                    high=high,
                    low=low,
                    close=close,
                    entry_i=ei,
                    tp=tp3,
                    sl=sl,
                    bars_fwd=bars_fwd,
                    ep=ep,
                    risk_unit=risk_unit,
                )
                rows.append(
                    {
                        "trade_id": str(sym_tr.at[j, "trade_id"]),
                        "signal_id": str(sym_tr.at[j, "signal_id"]),
                        "symbol": symbol,
                        "year": int(sym_tr.at[j, "year"]) if pd.notna(sym_tr.at[j, "year"]) else np.nan,
                        "month": int(sym_tr.at[j, "month"]) if pd.notna(sym_tr.at[j, "month"]) else np.nan,
                        "horizon_h": int(h),
                        "entry_time_utc": pd.to_datetime(sym_tr.at[j, "entry_time_utc"], utc=True, errors="coerce"),
                        "setup_time_utc": pd.to_datetime(sym_tr.at[j, "setup_time_utc"], utc=True, errors="coerce"),
                        "entry_price": ep,
                        "atr_entry": atr_entry,
                        "sl_price": sl,
                        "tp1_price": tp1,
                        "tp3_price": tp3,
                        "outcome_1r": out1,
                        "outcome_3r": out3,
                        "win_1r": float(out1 == "WIN"),
                        "loss_1r": float(out1 == "LOSS"),
                        "open_1r": float(out1 == "OPEN"),
                        "win_3r": float(out3 == "WIN"),
                        "loss_3r": float(out3 == "LOSS"),
                        "open_3r": float(out3 == "OPEN"),
                        "mfe_r": float(mfe_r),
                        "mae_r": float(mae_r),
                        "favorable_move": float(favorable),
                        "mfe_gt_mae": float(np.isfinite(mfe_r) and np.isfinite(mae_r) and (mfe_r > abs(mae_r))),
                        "truncated": int(truncated),
                        "entry_session_bucket": sym_tr.at[j, "entry_session_bucket"],
                        "setup_session_bucket": sym_tr.at[j, "setup_session_bucket"],
                        "entry_in_london": int(sym_tr.at[j, "entry_in_london"]),
                        "entry_in_newyork": int(sym_tr.at[j, "entry_in_newyork"]),
                        "entry_in_london_or_newyork": int(sym_tr.at[j, "entry_in_london_or_newyork"]),
                        "setup_in_london": int(sym_tr.at[j, "setup_in_london"]),
                        "setup_in_newyork": int(sym_tr.at[j, "setup_in_newyork"]),
                        "truth_label": sym_tr.at[j, "truth_label"],
                        "config_pack": sym_tr.at[j, "config_pack"],
                    }
                )

    return pd.DataFrame(rows)


def _filter_scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "all_sessions":
        return df
    if scope == "london_only":
        return df[df["entry_session_bucket"] == "london_only"]
    if scope == "newyork_only":
        return df[df["entry_session_bucket"] == "newyork_only"]
    if scope == "london_or_newyork":
        return df[df["entry_in_london_or_newyork"] == 1]
    if scope == "other":
        return df[df["entry_session_bucket"] == "other"]
    return df.iloc[0:0].copy()


def _plus1r_verdict(p: float) -> str:
    if not np.isfinite(p):
        return "weak"
    if p < 0.50:
        return "weak"
    if p < 0.55:
        return "borderline"
    if p <= 0.60:
        return "clear"
    return "strong"


def _plus3r_verdict(p: float) -> str:
    if not np.isfinite(p):
        return "not monetizable at 1:3"
    if p < 0.25:
        return "not monetizable at 1:3"
    if p < 0.30:
        return "marginal"
    if p <= 0.40:
        return "monetizable"
    return "strong monetizable edge"


def _directional_verdict(p1: float, favorable: float, ratio: float, pct_mfe_gt_mae: float) -> str:
    if not (np.isfinite(p1) and np.isfinite(favorable) and np.isfinite(ratio) and np.isfinite(pct_mfe_gt_mae)):
        return "No directional edge"
    if p1 > 0.60 and favorable >= 0.60 and ratio >= 1.20 and pct_mfe_gt_mae >= 0.60:
        return "Strong directional edge"
    if p1 >= 0.55 and favorable >= 0.55 and ratio >= 1.10 and pct_mfe_gt_mae >= 0.55:
        return "Clear directional edge"
    if p1 >= 0.50 and ratio >= 1.00:
        return "Weak directional edge"
    return "No directional edge"


def _tradable_verdict(p3: float, expectancy_1to3: float, pf_1to3: float, net_r_1to3: float) -> str:
    if not np.isfinite(p3):
        return "Not monetizable at 1:3"
    if p3 > 0.40 and np.isfinite(expectancy_1to3) and expectancy_1to3 >= 0.20 and np.isfinite(pf_1to3) and pf_1to3 >= 1.30 and net_r_1to3 > 0:
        return "Strongly monetizable at 1:3"
    if p3 >= 0.30 and np.isfinite(expectancy_1to3) and expectancy_1to3 >= 0.05 and np.isfinite(pf_1to3) and pf_1to3 >= 1.10 and net_r_1to3 > 0:
        return "Monetizable at 1:3"
    if p3 >= 0.25:
        return "Marginal at 1:3"
    return "Not monetizable at 1:3"


def _agg_metrics(g: pd.DataFrame) -> Dict[str, object]:
    n = len(g)
    if n == 0:
        return {
            "n_signals": 0,
            "n_trades": 0,
            "pct_reach_plus1r_before_minus1r": np.nan,
            "pct_reach_plus3r_before_minus1r": np.nan,
            "pct_minus1r_before_plus1r_for_1r_test": np.nan,
            "pct_minus1r_before_plus3r_for_3r_test": np.nan,
            "open_rate_1r": np.nan,
            "open_rate_3r": np.nan,
            "expectancy_1r_style": np.nan,
            "expectancy_1to3_r": np.nan,
            "profit_factor_1to3": np.nan,
            "net_r_1to3": np.nan,
            "median_mfe_r": np.nan,
            "median_mae_r": np.nan,
            "favorable_move_rate": np.nan,
            "mfe_mae_ratio": np.nan,
            "pct_mfe_gt_mae": np.nan,
        }

    wins1 = float(pd.to_numeric(g["win_1r"], errors="coerce").sum())
    loss1 = float(pd.to_numeric(g["loss_1r"], errors="coerce").sum())
    open1 = float(pd.to_numeric(g["open_1r"], errors="coerce").sum())
    wins3 = float(pd.to_numeric(g["win_3r"], errors="coerce").sum())
    loss3 = float(pd.to_numeric(g["loss_3r"], errors="coerce").sum())
    open3 = float(pd.to_numeric(g["open_3r"], errors="coerce").sum())

    med_mfe = float(pd.to_numeric(g["mfe_r"], errors="coerce").median())
    med_mae = float(pd.to_numeric(g["mae_r"], errors="coerce").median())
    ratio = float(med_mfe / abs(med_mae)) if np.isfinite(med_mae) and med_mae != 0 else np.nan

    expectancy_1r = float((wins1 - loss1) / n)
    expectancy_1to3 = float((3.0 * wins3 - loss3) / n)
    gross_win_1to3 = float(3.0 * wins3)
    gross_loss_1to3 = float(loss3)
    pf_1to3 = float(gross_win_1to3 / gross_loss_1to3) if gross_loss_1to3 > 0 else np.nan
    net_r_1to3 = float(gross_win_1to3 - gross_loss_1to3)

    return {
        "n_signals": int(g["signal_id"].nunique()),
        "n_trades": int(g["trade_id"].nunique()),
        "pct_reach_plus1r_before_minus1r": float(wins1 / n),
        "pct_reach_plus3r_before_minus1r": float(wins3 / n),
        "pct_minus1r_before_plus1r_for_1r_test": float(loss1 / n),
        "pct_minus1r_before_plus3r_for_3r_test": float(loss3 / n),
        "open_rate_1r": float(open1 / n),
        "open_rate_3r": float(open3 / n),
        "expectancy_1r_style": expectancy_1r,
        "expectancy_1to3_r": expectancy_1to3,
        "profit_factor_1to3": pf_1to3,
        "net_r_1to3": net_r_1to3,
        "median_mfe_r": med_mfe,
        "median_mae_r": med_mae,
        "favorable_move_rate": float(pd.to_numeric(g["favorable_move"], errors="coerce").mean()),
        "mfe_mae_ratio": ratio,
        "pct_mfe_gt_mae": float(pd.to_numeric(g["mfe_gt_mae"], errors="coerce").mean()),
    }


def _aggregate(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    if not group_cols:
        row = _agg_metrics(df)
        rows.append(row)
    else:
        gb = df.groupby(group_cols, dropna=False, sort=True)
        for keys, g in gb:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {c: v for c, v in zip(group_cols, keys)}
            row.update(_agg_metrics(g))
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["plus1r_verdict"] = out["pct_reach_plus1r_before_minus1r"].apply(_plus1r_verdict)
    out["plus3r_verdict"] = out["pct_reach_plus3r_before_minus1r"].apply(_plus3r_verdict)
    out["directional_edge_verdict"] = out.apply(
        lambda r: _directional_verdict(
            float(r["pct_reach_plus1r_before_minus1r"]),
            float(r["favorable_move_rate"]),
            float(r["mfe_mae_ratio"]),
            float(r["pct_mfe_gt_mae"]),
        ),
        axis=1,
    )
    out["tradable_edge_verdict"] = out.apply(
        lambda r: _tradable_verdict(
            float(r["pct_reach_plus3r_before_minus1r"]),
            float(r["expectancy_1to3_r"]),
            float(r["profit_factor_1to3"]),
            float(r["net_r_1to3"]),
        ),
        axis=1,
    )
    return out


def _with_scope_tables(outcomes: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    tables: List[pd.DataFrame] = []
    for scope in SESSION_SCOPES:
        s = _filter_scope(outcomes, scope)
        t = _aggregate(s, group_cols=group_cols + ["horizon_h"])
        if t.empty:
            continue
        t["session_scope"] = scope
        tables.append(t)
    return pd.concat(tables, ignore_index=True, sort=False) if tables else pd.DataFrame()


def _load_truth_datasets(cfg, truth_mode: TruthMode, exact_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, TruthLabel, Dict[str, object] | None, bool]:
    exact_trades, exact_signals = _load_exact_exports(exact_dir)
    exact_trades = _normalize_trade_frame(exact_trades)
    exact_signals = _normalize_signal_frame(exact_signals)
    exact_available = not exact_trades.empty
    if truth_mode == TruthMode.EXACT_PINE_EXPORTED and not exact_available:
        raise SystemExit("Exact Pine exported mode requested, but normalized TradingView exports are missing.")

    parity_verification = _load_parity_verification(exact_dir)
    parity_metrics: Dict[str, object] | None = None

    parity_trades = pd.DataFrame()
    parity_signals = pd.DataFrame()
    if truth_mode != TruthMode.EXACT_PINE_EXPORTED:
        parity_trades, parity_signals = _load_parity_cache(exact_dir)
        parity_trades = _normalize_trade_frame(parity_trades)
        parity_signals = _normalize_signal_frame(parity_signals)
        cache_ok = True
        if not parity_trades.empty and "config_pack" in parity_trades.columns:
            cache_ok = bool(parity_trades["config_pack"].astype(str).eq(str(cfg.metadata.config_pack)).all())
        if cache_ok and not parity_signals.empty and "config_pack" in parity_signals.columns:
            cache_ok = bool(parity_signals["config_pack"].astype(str).eq(str(cfg.metadata.config_pack)).all())
        if not cache_ok:
            parity_trades = pd.DataFrame()
            parity_signals = pd.DataFrame()
        if parity_trades.empty and parity_signals.empty:
            parity = run_python_parity(cfg)
            parity_trades = _normalize_trade_frame(parity.trades)
            parity_signals = _normalize_signal_frame(parity.signals)
            _save_parity_cache(exact_dir, parity_trades, parity_signals)

    parity_verified = False
    if exact_available and not parity_trades.empty:
        cmp = compare_parity_to_exact(
            exact_trades=exact_trades,
            parity_trades=parity_trades,
            timeframe=cfg.timeframe,
        )
        parity_verified = bool(cmp.pass_thresholds)
        parity_metrics = {
            "signal_count_mismatch_pct": float(cmp.signal_count_mismatch_pct),
            "entry_timestamp_max_bar_diff": float(cmp.entry_timestamp_max_bar_diff),
            "exit_timestamp_max_bar_diff": float(cmp.exit_timestamp_max_bar_diff),
            "aggregate_net_r_mismatch_pct": float(cmp.aggregate_net_r_mismatch_pct),
            "pass_thresholds": bool(cmp.pass_thresholds),
            "source": "exact_exports_vs_python_parity",
        }
    elif parity_verification is not None:
        pv_cfg = str(parity_verification.get("config", ""))
        if str(cfg.metadata.config_pack) in pv_cfg:
            parity_verified = bool(parity_verification.get("pass_thresholds", False))
            parity_metrics = parity_verification
        else:
            parity_verified = False
            parity_metrics = {
                "config": pv_cfg,
                "reason": "verification_config_mismatch",
                "pass_thresholds": False,
                "truth_label": TruthLabel.UNVERIFIED_PYTHON_APPROXIMATION.value,
            }

    truth_label = _truth_for_block(truth_mode, exact_available=exact_available, parity_verified=parity_verified)
    if truth_label == TruthLabel.EXACT_PINE_EXPORTED:
        return exact_trades, exact_signals, truth_label, parity_metrics, exact_available
    return parity_trades, parity_signals, truth_label, parity_metrics, exact_available


def run_plus1r_plus3r_report(config_path: str, truth_mode_raw: str, exact_dir: Path, output_dir: Path) -> Dict[str, Path]:
    cfg = load_pine16_exact_config(config_path)
    truth_mode = normalize_truth_mode(truth_mode_raw)
    trades, signals, truth_label, parity_metrics, exact_available = _load_truth_datasets(cfg, truth_mode, exact_dir)

    if signals.empty and not trades.empty:
        signals = trades[["symbol", "timeframe", "bar_time_utc", "entry_time_utc", "entry_price"]].copy()

    signals_cls = _classify_signals(signals, truth_label=truth_label, config_pack=cfg.metadata.config_pack)
    trades_cls = _classify_trades(trades, truth_label=truth_label, config_pack=cfg.metadata.config_pack)

    outcomes = _build_barrier_outcomes(trades_cls, cfg)
    if not outcomes.empty:
        outcomes["truth_label"] = truth_label.value
        outcomes["config_pack"] = cfg.metadata.config_pack

    by_session = _with_scope_tables(outcomes, group_cols=[])
    overall = by_session[by_session["session_scope"] == "all_sessions"].copy() if not by_session.empty else pd.DataFrame()
    by_symbol = _aggregate(_filter_scope(outcomes, "all_sessions"), group_cols=["symbol", "horizon_h"]) if not outcomes.empty else pd.DataFrame()
    by_year = _aggregate(_filter_scope(outcomes, "all_sessions"), group_cols=["year", "horizon_h"]) if not outcomes.empty else pd.DataFrame()
    by_symbol_year = _aggregate(_filter_scope(outcomes, "all_sessions"), group_cols=["symbol", "year", "horizon_h"]) if not outcomes.empty else pd.DataFrame()
    by_symbol_session = _with_scope_tables(outcomes, group_cols=["symbol"])
    by_year_session = _with_scope_tables(outcomes, group_cols=["year"])
    by_symbol_year_session = _with_scope_tables(outcomes, group_cols=["symbol", "year"])

    for df in [overall, by_session, by_symbol, by_year, by_symbol_year, by_symbol_session, by_year_session, by_symbol_year_session]:
        if not df.empty:
            df["truth_label"] = truth_label.value
            df["config_pack"] = cfg.metadata.config_pack

    plus1 = outcomes[
        [
            "trade_id",
            "signal_id",
            "symbol",
            "year",
            "horizon_h",
            "entry_time_utc",
            "setup_time_utc",
            "entry_price",
            "atr_entry",
            "sl_price",
            "tp1_price",
            "outcome_1r",
            "win_1r",
            "loss_1r",
            "open_1r",
            "entry_session_bucket",
            "setup_session_bucket",
            "entry_in_london",
            "entry_in_newyork",
            "entry_in_london_or_newyork",
            "truth_label",
            "config_pack",
        ]
    ].copy() if not outcomes.empty else pd.DataFrame()

    plus3 = outcomes[
        [
            "trade_id",
            "signal_id",
            "symbol",
            "year",
            "horizon_h",
            "entry_time_utc",
            "setup_time_utc",
            "entry_price",
            "atr_entry",
            "sl_price",
            "tp3_price",
            "outcome_3r",
            "win_3r",
            "loss_3r",
            "open_3r",
            "entry_session_bucket",
            "setup_session_bucket",
            "entry_in_london",
            "entry_in_newyork",
            "entry_in_london_or_newyork",
            "truth_label",
            "config_pack",
        ]
    ].copy() if not outcomes.empty else pd.DataFrame()

    exact_dir.mkdir(parents=True, exist_ok=True)
    plus1_path = exact_dir / "barrier_outcomes_plus1r.parquet"
    plus3_path = exact_dir / "barrier_outcomes_plus3r.parquet"
    sig_cls_path = exact_dir / "session_classified_signals.parquet"
    tr_cls_path = exact_dir / "session_classified_trades.parquet"
    plus1.to_parquet(plus1_path, index=False)
    plus3.to_parquet(plus3_path, index=False)
    signals_cls.to_parquet(sig_cls_path, index=False)
    trades_cls.to_parquet(tr_cls_path, index=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "pine16_plus1r_plus3r_report.md"
    html_path = output_dir / "pine16_plus1r_plus3r_report.html"
    summary_csv = output_dir / "pine16_plus1r_plus3r_summary.csv"
    by_symbol_csv = output_dir / "pine16_plus1r_plus3r_by_symbol.csv"
    by_year_csv = output_dir / "pine16_plus1r_plus3r_by_year.csv"
    by_symbol_year_csv = output_dir / "pine16_plus1r_plus3r_by_symbol_year.csv"
    by_session_csv = output_dir / "pine16_plus1r_plus3r_by_session.csv"
    by_year_session_csv = output_dir / "pine16_plus1r_plus3r_by_year_session.csv"
    by_symbol_year_session_csv = output_dir / "pine16_plus1r_plus3r_by_symbol_year_session.csv"
    trade_log_path = output_dir / "pine16_plus1r_plus3r_trade_log.parquet"

    overall.to_csv(summary_csv, index=False)
    by_symbol.to_csv(by_symbol_csv, index=False)
    by_year.to_csv(by_year_csv, index=False)
    by_symbol_year.to_csv(by_symbol_year_csv, index=False)
    by_session.to_csv(by_session_csv, index=False)
    by_year_session.to_csv(by_year_session_csv, index=False)
    by_symbol_year_session.to_csv(by_symbol_year_session_csv, index=False)
    outcomes.to_parquet(trade_log_path, index=False)

    p24 = overall[overall["horizon_h"] == 24].iloc[0] if not overall.empty and (overall["horizon_h"] == 24).any() else None
    final_directional = str(p24["directional_edge_verdict"]) if p24 is not None else "No directional edge"
    final_tradable = str(p24["tradable_edge_verdict"]) if p24 is not None else "Not monetizable at 1:3"

    by_session_24 = by_session[by_session["horizon_h"] == 24].copy() if not by_session.empty else pd.DataFrame()
    london24 = by_session_24[by_session_24["session_scope"] == "london_only"]
    ny24 = by_session_24[by_session_24["session_scope"] == "newyork_only"]
    lny24 = by_session_24[by_session_24["session_scope"] == "london_or_newyork"]
    all24 = by_session_24[by_session_24["session_scope"] == "all_sessions"]

    symbol_action = by_symbol[(by_symbol["horizon_h"] == 24)].copy() if not by_symbol.empty else pd.DataFrame()
    if not symbol_action.empty:
        symbol_action["action"] = "WATCH"
        keep_mask = (
            (symbol_action["n_trades"] >= 20)
            & (symbol_action["pct_reach_plus1r_before_minus1r"] >= 0.55)
            & (symbol_action["pct_reach_plus3r_before_minus1r"] >= 0.30)
            & (symbol_action["expectancy_1to3_r"] > 0)
        )
        cut_mask = (
            (symbol_action["n_trades"] >= 10)
            & (symbol_action["pct_reach_plus1r_before_minus1r"] < 0.50)
            & (symbol_action["pct_reach_plus3r_before_minus1r"] < 0.25)
            & (symbol_action["expectancy_1to3_r"] <= 0)
        )
        symbol_action.loc[keep_mask, "action"] = "KEEP"
        symbol_action.loc[cut_mask, "action"] = "CUT"

    keep_syms = symbol_action[symbol_action["action"] == "KEEP"]["symbol"].astype(str).tolist() if not symbol_action.empty else []
    cut_syms = symbol_action[symbol_action["action"] == "CUT"]["symbol"].astype(str).tolist() if not symbol_action.empty else []

    london_better_than_ny = "insufficient_data"
    if not london24.empty and not ny24.empty:
        l = london24.iloc[0]
        n = ny24.iloc[0]
        london_better_than_ny = "yes" if float(l["pct_reach_plus3r_before_minus1r"]) > float(n["pct_reach_plus3r_before_minus1r"]) else "no"

    filtering_improves = "insufficient_data"
    if not lny24.empty and not all24.empty:
        l = lny24.iloc[0]
        a = all24.iloc[0]
        quality_up = float(l["pct_reach_plus3r_before_minus1r"]) > float(a["pct_reach_plus3r_before_minus1r"])
        sample_down = float(l["n_trades"]) < float(a["n_trades"])
        filtering_improves = "yes_quality_up_sample_down" if quality_up and sample_down else ("yes" if quality_up else "no")

    lines: List[str] = []
    lines.append("# Pine16 +1R/+3R Session-Scoped Report")
    lines.append("")
    lines.append("## 1. Executive verdict")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(f"- Directional edge verdict (24h): **{final_directional}**")
    lines.append(f"- Tradable 1:3 verdict (24h): **{final_tradable}**")
    if p24 is not None:
        lines.append(f"- +1R before -1R (24h): **{float(p24['pct_reach_plus1r_before_minus1r']):.4f}**")
        lines.append(f"- +3R before -1R (24h): **{float(p24['pct_reach_plus3r_before_minus1r']):.4f}**")
    lines.append("")
    lines.append("## 2. Truth source used")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(f"- requested_truth_mode: `{truth_mode.value}`")
    lines.append(f"- exact_export_available: `{bool(exact_available)}`")
    if parity_metrics is not None:
        lines.append(f"- parity_metrics: `{json.dumps(parity_metrics, sort_keys=True)}`")
    lines.append("")
    lines.append("## 3. Config pack definitions")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(f"- config_pack: `{cfg.metadata.config_pack}`")
    lines.append(f"- timeframe: `{cfg.timeframe}`")
    lines.append(f"- symbols: `{','.join(cfg.symbols)}`")
    lines.append(f"- years: `{','.join(str(y) for y in cfg.data.years)}`")
    lines.append(f"- risk: atrLen={cfg.risk.atrLen}, slAtrMult={cfg.risk.slAtrMult}, rrMult={cfg.risk.rrMult}")
    lines.append("")
    lines.append("## 4. Session scope definitions")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append("- Classification uses Pine-native UTC+3 named session windows.")
    lines.append("- Setup/session columns are setup/pivot-time based when available.")
    lines.append("- Scope filtering for production confirm analysis uses entry/trigger-time buckets.")
    lines.append("- `london_only`: entry in London and not New York.")
    lines.append("- `newyork_only`: entry in New York and not London.")
    lines.append("- `london_or_newyork`: entry in either London or New York (including overlap).")
    lines.append("- `other`: entry outside both London and New York.")
    lines.append("- `all_sessions`: no entry session restriction.")
    lines.append("")
    lines.append("## 5. Overall +1R before -1R results")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(overall, ["horizon_h", "n_trades", "pct_reach_plus1r_before_minus1r", "pct_minus1r_before_plus1r_for_1r_test", "open_rate_1r", "plus1r_verdict"]))
    lines.append("")
    lines.append("## 6. Overall +3R before -1R results")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(overall, ["horizon_h", "n_trades", "pct_reach_plus3r_before_minus1r", "pct_minus1r_before_plus3r_for_3r_test", "open_rate_3r", "expectancy_1to3_r", "plus3r_verdict"]))
    lines.append("")
    lines.append("## 7. By session scope")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(by_session[by_session["horizon_h"].isin([24, 48, 72])] if not by_session.empty else by_session, ["session_scope", "horizon_h", "n_trades", "pct_reach_plus1r_before_minus1r", "pct_reach_plus3r_before_minus1r", "expectancy_1to3_r", "directional_edge_verdict", "tradable_edge_verdict"]))
    lines.append("")
    lines.append("## 8. By symbol")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(by_symbol[by_symbol["horizon_h"].isin([24, 48, 72])] if not by_symbol.empty else by_symbol, ["symbol", "horizon_h", "n_trades", "pct_reach_plus1r_before_minus1r", "pct_reach_plus3r_before_minus1r", "expectancy_1to3_r", "directional_edge_verdict", "tradable_edge_verdict"]))
    lines.append("")
    lines.append("## 9. By year")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(by_year[by_year["horizon_h"].isin([24, 48, 72])] if not by_year.empty else by_year, ["year", "horizon_h", "n_trades", "pct_reach_plus1r_before_minus1r", "pct_reach_plus3r_before_minus1r", "expectancy_1to3_r", "directional_edge_verdict", "tradable_edge_verdict"]))
    lines.append("")
    lines.append("## 10. By symbol x year")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(by_symbol_year[by_symbol_year["horizon_h"].isin([24, 48, 72])] if not by_symbol_year.empty else by_symbol_year, ["symbol", "year", "horizon_h", "n_trades", "pct_reach_plus1r_before_minus1r", "pct_reach_plus3r_before_minus1r", "expectancy_1to3_r"]))
    lines.append("")
    lines.append("## 11. By symbol x session scope")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(by_symbol_session[by_symbol_session["horizon_h"] == 24] if not by_symbol_session.empty else by_symbol_session, ["symbol", "session_scope", "horizon_h", "n_trades", "pct_reach_plus1r_before_minus1r", "pct_reach_plus3r_before_minus1r", "expectancy_1to3_r", "directional_edge_verdict", "tradable_edge_verdict"]))
    lines.append("")
    lines.append("By year x session scope (24h):")
    lines.append(_md_table(by_year_session[by_year_session["horizon_h"] == 24] if not by_year_session.empty else by_year_session, ["year", "session_scope", "horizon_h", "n_trades", "pct_reach_plus1r_before_minus1r", "pct_reach_plus3r_before_minus1r", "expectancy_1to3_r"]))
    lines.append("")
    lines.append("By symbol x year x session scope (24h):")
    lines.append(_md_table(by_symbol_year_session[by_symbol_year_session["horizon_h"] == 24] if not by_symbol_year_session.empty else by_symbol_year_session, ["symbol", "year", "session_scope", "horizon_h", "n_trades", "pct_reach_plus1r_before_minus1r", "pct_reach_plus3r_before_minus1r", "expectancy_1to3_r"]))
    lines.append("")
    lines.append("## 12. London-only verdict")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(london24, ["session_scope", "horizon_h", "n_trades", "pct_reach_plus1r_before_minus1r", "pct_reach_plus3r_before_minus1r", "directional_edge_verdict", "tradable_edge_verdict"]))
    lines.append("")
    lines.append("## 13. New-York-only verdict")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(ny24, ["session_scope", "horizon_h", "n_trades", "pct_reach_plus1r_before_minus1r", "pct_reach_plus3r_before_minus1r", "directional_edge_verdict", "tradable_edge_verdict"]))
    lines.append("")
    lines.append("## 14. London-or-New-York verdict")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(lny24, ["session_scope", "horizon_h", "n_trades", "pct_reach_plus1r_before_minus1r", "pct_reach_plus3r_before_minus1r", "directional_edge_verdict", "tradable_edge_verdict"]))
    lines.append("")
    lines.append("## 15. Symbols to keep")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(f"- keep: `{', '.join(keep_syms) if keep_syms else 'none'}`")
    lines.append("")
    lines.append("## 16. Symbols to cut")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(f"- cut: `{', '.join(cut_syms) if cut_syms else 'none'}`")
    lines.append("")
    lines.append("## 17. Whether London/NY filtering improves quality")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(f"- london_vs_newyork_better: `{london_better_than_ny}`")
    lines.append(f"- london_or_newyork_vs_all_sessions: `{filtering_improves}`")
    lines.append("")
    lines.append("## 18. Directional edge verdict")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(f"- overall_directional_24h: **{final_directional}**")
    lines.append("")
    lines.append("## 19. Monetizable 1:3 verdict")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(f"- overall_tradable_24h: **{final_tradable}**")
    lines.append("")
    lines.append("## 20. Caveats / blockers")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append("- Same-bar target/stop touches use conservative adverse-first resolution (LOSS).")
    if truth_label != TruthLabel.EXACT_PINE_EXPORTED:
        lines.append("- Exact Pine export artifacts are missing or parity thresholds are not passed; this is not exact Pine truth.")
    if truth_label == TruthLabel.UNVERIFIED_PYTHON_APPROXIMATION:
        lines.append("- Outputs are unverified approximation and must not be presented as exact Pine or verified parity.")
    lines.append("")
    lines.append("## Symbol Action Table (24h)")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(
        _md_table(
            symbol_action,
            [
                "symbol",
                "n_trades",
                "pct_reach_plus1r_before_minus1r",
                "pct_reach_plus3r_before_minus1r",
                "expectancy_1to3_r",
                "directional_edge_verdict",
                "tradable_edge_verdict",
                "action",
            ],
        )
    )

    md = "\n".join(lines).rstrip() + "\n"
    md_path.write_text(md, encoding="utf-8")
    _write_html_from_markdown(md, html_path)

    return {
        "report_md": md_path,
        "report_html": html_path,
        "summary_csv": summary_csv,
        "by_symbol_csv": by_symbol_csv,
        "by_year_csv": by_year_csv,
        "by_symbol_year_csv": by_symbol_year_csv,
        "by_session_csv": by_session_csv,
        "by_year_session_csv": by_year_session_csv,
        "by_symbol_year_session_csv": by_symbol_year_session_csv,
        "trade_log": trade_log_path,
        "barrier_plus1r": plus1_path,
        "barrier_plus3r": plus3_path,
        "session_classified_signals": sig_cls_path,
        "session_classified_trades": tr_cls_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Pine16 session-scoped +1R/+3R report generator.")
    ap.add_argument("--config", required=True, help="Pine16 exact config path.")
    ap.add_argument("--truth-mode", required=True, choices=[m.value for m in TruthMode], help="Truth mode selector.")
    ap.add_argument("--exact-dir", default="data/derived/pine16_exact", help="Normalized exact/parity artifact directory.")
    ap.add_argument("--output-dir", default="outputs/reports", help="Report output directory.")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    outs = run_plus1r_plus3r_report(
        config_path=args.config,
        truth_mode_raw=args.truth_mode,
        exact_dir=Path(args.exact_dir),
        output_dir=Path(args.output_dir),
    )
    for k, v in outs.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
