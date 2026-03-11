from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data import load_bars_for_symbol, timeframe_to_minutes
from .indicators import atr
from .pine16_config import Pine16ExactConfig, load_pine16_exact_config, to_legacy_run_config
from .pine16_parity_engine import _build_cvd_proxy, compare_parity_to_exact, run_python_parity
from .pine16_session import SESSION_HM, hm_from_utc, in_hm_range_inclusive
from .pine16_truth import TruthLabel, TruthMode, normalize_truth_mode
from .strategy import extract_dfd05_events


HORIZONS_H: List[int] = [2, 4, 8, 24, 48, 72, 120, 168]
SESSION_SCOPES: List[str] = ["all_sessions", "london_only", "newyork_only", "london_or_newyork", "other"]


@dataclass
class ResearchArtifacts:
    report_md: Path
    report_html: Path
    research_master: Path
    research_overall: Path
    research_by_symbol: Path
    research_by_session: Path
    research_by_year: Path
    research_by_symbol_year: Path
    research_by_symbol_session: Path
    research_by_year_session: Path
    research_by_feature_bucket: Path
    research_leaderboard: Path
    research_keep_watch_cut: Path
    research_top_combinations: Path
    research_robustness: Path
    deployment_candidates_md: Path
    deployment_candidates_csv: Path
    feature_importance_csv: Path | None = None
    simple_rules_md: Path | None = None


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
        "<title>Pine16 Conditional Research</title>"
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


def load_truth_datasets(
    cfg: Pine16ExactConfig,
    truth_mode: TruthMode,
    exact_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, TruthLabel, Dict[str, object] | None, bool]:
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
        if cache_ok and not parity_trades.empty and "timeframe" in parity_trades.columns:
            cache_ok = bool(parity_trades["timeframe"].astype(str).str.lower().eq(str(cfg.timeframe).lower()).all())
        if cache_ok and not parity_signals.empty and "config_pack" in parity_signals.columns:
            cache_ok = bool(parity_signals["config_pack"].astype(str).eq(str(cfg.metadata.config_pack)).all())
        if cache_ok and not parity_signals.empty and "timeframe" in parity_signals.columns:
            cache_ok = bool(parity_signals["timeframe"].astype(str).str.lower().eq(str(cfg.timeframe).lower()).all())
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


def _choose_first_time(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            ser = pd.to_datetime(df[c], utc=True, errors="coerce")
            if ser.notna().any():
                return ser
    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")


def classify_signals(signals: pd.DataFrame, truth_label: TruthLabel, config_pack: str) -> pd.DataFrame:
    if signals.empty:
        cols = [
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
            "signal_id",
        ]
        return pd.DataFrame(columns=cols)

    out = signals.copy()
    out["signal_time_utc"] = _choose_first_time(out, ["signal_time_utc", "bar_time_utc"])
    out["setup_time_utc"] = _choose_first_time(out, ["setup_time_utc", "pivot_time_utc"])
    if "pivot_time_ms" in out.columns and out["setup_time_utc"].isna().all():
        out["setup_time_utc"] = pd.to_datetime(
            pd.to_numeric(out["pivot_time_ms"], errors="coerce"), unit="ms", utc=True, errors="coerce"
        )
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

    key_base = (
        out["symbol"].astype(str)
        + "|"
        + out["entry_time_utc"].astype(str)
        + "|"
        + pd.to_numeric(out["entry_price"], errors="coerce").round(10).astype(str)
    )
    out["signal_id"] = out.get("signal_id", key_base)

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
        "signal_id",
    ]
    for c in keep_cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[keep_cols].reset_index(drop=True)


def classify_trades(trades: pd.DataFrame, truth_label: TruthLabel, config_pack: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    out = trades.copy()
    out["signal_time_utc"] = _choose_first_time(out, ["signal_time_utc", "bar_time_utc"])
    out["setup_time_utc"] = _choose_first_time(out, ["setup_time_utc", "pivot_time_utc"])
    if "pivot_time_ms" in out.columns and out["setup_time_utc"].isna().all():
        out["setup_time_utc"] = pd.to_datetime(
            pd.to_numeric(out["pivot_time_ms"], errors="coerce"), unit="ms", utc=True, errors="coerce"
        )
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
    out["entry_in_other"] = (out["entry_in_london_or_newyork"] == 0).astype(int)
    out["truth_label"] = truth_label.value
    out["config_pack"] = str(config_pack)

    out["year"] = pd.to_datetime(out["entry_time_utc"], utc=True, errors="coerce").dt.year.astype("Int64")
    out["month"] = pd.to_datetime(out["entry_time_utc"], utc=True, errors="coerce").dt.month.astype("Int64")
    out["quarter"] = pd.to_datetime(out["entry_time_utc"], utc=True, errors="coerce").dt.quarter.astype("Int64")
    out["week"] = pd.to_datetime(out["entry_time_utc"], utc=True, errors="coerce").dt.isocalendar().week.astype("Int64")
    out["date"] = pd.to_datetime(out["entry_time_utc"], utc=True, errors="coerce").dt.date.astype(str)

    key_base = (
        out["symbol"].astype(str)
        + "|"
        + out["entry_time_utc"].astype(str)
        + "|"
        + pd.to_numeric(out["entry_price"], errors="coerce").round(10).astype(str)
    )
    out["trade_id"] = out.get("trade_id", key_base)
    out["signal_id"] = out.get("signal_id", key_base)
    out["side"] = out.get("side", "long")
    out["session_scope_bucket"] = out["entry_session_bucket"].astype(str)

    return out.reset_index(drop=True)


def _signal_key(symbol: pd.Series, entry_time: pd.Series, entry_price: pd.Series) -> pd.Series:
    return (
        symbol.astype(str)
        + "|"
        + pd.to_datetime(entry_time, utc=True, errors="coerce").astype(str)
        + "|"
        + pd.to_numeric(entry_price, errors="coerce").round(10).astype(str)
    )


def build_feature_event_frame(cfg: Pine16ExactConfig) -> pd.DataFrame:
    legacy_cfg = to_legacy_run_config(cfg)
    rows: List[pd.DataFrame] = []

    for symbol in cfg.symbols:
        try:
            bars = load_bars_for_symbol(legacy_cfg, symbol=symbol, timeframe=cfg.timeframe)
        except FileNotFoundError:
            continue
        if bars.empty:
            continue

        cvd_proxy = _build_cvd_proxy(legacy_cfg, symbol=str(symbol), timeframe=str(cfg.timeframe), bars=bars)
        events = extract_dfd05_events(
            bars=bars,
            symbol=str(symbol),
            timeframe=str(cfg.timeframe),
            config=legacy_cfg,
            cvd_proxy=cvd_proxy,
        )
        if events.empty:
            continue

        ev = events.copy()
        ev["entry_time_utc"] = pd.to_datetime(ev["entry_time_ms"], unit="ms", utc=True, errors="coerce")
        ev["signal_time_utc"] = pd.to_datetime(ev["event_time_ms"], unit="ms", utc=True, errors="coerce")
        ev["setup_time_utc"] = pd.to_datetime(ev["pivot_time_ms"], unit="ms", utc=True, errors="coerce")
        ev["entry_price"] = pd.to_numeric(ev["entry_price"], errors="coerce")
        ev["pivot_price"] = pd.to_numeric(ev["pivot_price"], errors="coerce")

        ev = ev.sort_values("signal_time_utc", kind="mergesort").reset_index(drop=True)
        ev["prev_pivot_price"] = ev["pivot_price"].shift(1)
        prev = ev["prev_pivot_price"].abs()
        ev["priceChangePct"] = np.where(prev > 0, 100.0 * (ev["pivot_price"] - ev["prev_pivot_price"]) / prev, np.nan)

        ev["trade_id"] = _signal_key(ev["symbol"], ev["entry_time_utc"], ev["entry_price"])
        ev["signal_id"] = ev.get("signal_id", ev["trade_id"])

        ev["oscChangePct"] = pd.to_numeric(ev.get("osc_change_pct"), errors="coerce")
        ev["barsBetweenPivots"] = pd.to_numeric(ev.get("bars_gap"), errors="coerce")
        ev["divType"] = np.where(pd.to_numeric(ev.get("div_type"), errors="coerce") == 1, "classic", "equal")
        ev["classicFlag"] = (ev["divType"] == "classic").astype(int)
        ev["locAtPivot"] = pd.to_numeric(ev.get("loc_pivot"), errors="coerce")
        ev["volRatioAtPivot"] = pd.to_numeric(ev.get("vol_ratio_pivot"), errors="coerce")
        ev["volRatioAtEntry"] = pd.to_numeric(ev.get("vol_ratio_entry"), errors="coerce")
        ev["atrRatio"] = pd.to_numeric(ev.get("atr_ratio_entry"), errors="coerce")
        ev["atrEntry"] = pd.to_numeric(ev.get("atr_entry"), errors="coerce")
        ev["rsi14Pivot"] = pd.to_numeric(ev.get("rsi_pivot"), errors="coerce")
        ev["rsi14Entry"] = np.nan
        ev["macdHistEntry"] = np.nan
        ev["recentMomentum"] = ev["priceChangePct"]
        ev["dailyEmaOk"] = pd.to_numeric(ev.get("daily_ema_ok"), errors="coerce")
        ev["dailySlopeOk"] = pd.to_numeric(ev.get("daily_slope_ok"), errors="coerce")
        ev["dailyAdxOk"] = pd.to_numeric(ev.get("daily_adx_gate_ok"), errors="coerce")
        ev["dailyDiOk"] = pd.to_numeric(ev.get("daily_di_ok"), errors="coerce")
        ev["cvdOk"] = pd.to_numeric(ev.get("cvd_proxy_gate_ok"), errors="coerce")
        ev["volBehaviorOk"] = pd.to_numeric(ev.get("volume_behavior_gate_ok"), errors="coerce")
        ev["volSpikeOk"] = pd.to_numeric(ev.get("vol_spike_gate_ok"), errors="coerce")
        ev["wv70Ok"] = pd.to_numeric(ev.get("wv70_gate_ok"), errors="coerce")
        ev["bosUsed"] = int(bool(cfg.features.useBOSConfirm))
        ev["bosPassed"] = pd.to_numeric(ev.get("triggered_ok"), errors="coerce")
        ev["bosWaitBars"] = pd.to_numeric(ev.get("entry_index"), errors="coerce") - pd.to_numeric(ev.get("event_index"), errors="coerce")
        ev["warmupSatisfied"] = pd.to_numeric(ev.get("signal_allowed_ok"), errors="coerce")
        ev["cooldownSatisfied"] = 1
        ev["dailyAdxValue"] = pd.to_numeric(ev.get("daily_adx"), errors="coerce")

        ev["trendState"] = np.where(
            (ev["dailyEmaOk"] == 1) & (ev["dailyDiOk"] == 1),
            "trend_up",
            np.where(ev["dailyEmaOk"] == 1, "ema_only", "neutral_or_down"),
        )

        keep = [
            "trade_id",
            "signal_id",
            "symbol",
            "entry_time_utc",
            "signal_time_utc",
            "setup_time_utc",
            "oscChangePct",
            "priceChangePct",
            "barsBetweenPivots",
            "divType",
            "classicFlag",
            "locAtPivot",
            "volRatioAtPivot",
            "volRatioAtEntry",
            "atrRatio",
            "atrEntry",
            "rsi14Pivot",
            "rsi14Entry",
            "macdHistEntry",
            "recentMomentum",
            "trendState",
            "dailyEmaOk",
            "dailySlopeOk",
            "dailyAdxOk",
            "dailyDiOk",
            "cvdOk",
            "volBehaviorOk",
            "volSpikeOk",
            "wv70Ok",
            "bosUsed",
            "bosPassed",
            "bosWaitBars",
            "warmupSatisfied",
            "cooldownSatisfied",
            "dailyAdxValue",
        ]
        for c in keep:
            if c not in ev.columns:
                ev[c] = np.nan
        rows.append(ev[keep].copy())

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True, sort=False)
    out = out.sort_values(["symbol", "entry_time_utc", "signal_time_utc"], kind="mergesort")
    out = out.drop_duplicates(subset=["trade_id"], keep="last").reset_index(drop=True)
    out["oscPercentile"] = out.groupby("symbol")["oscChangePct"].rank(pct=True)
    return out


def _scan_barrier_outcome(
    high: np.ndarray,
    low: np.ndarray,
    start: int,
    end: int,
    tp: float,
    sl: float,
) -> Tuple[str, Optional[int]]:
    if start > end:
        return "OPEN", None
    for i in range(start, end + 1):
        hit_tp = bool(high[i] >= tp)
        hit_sl = bool(low[i] <= sl)
        if hit_tp and hit_sl:
            return "LOSS", i
        if hit_tp:
            return "WIN", i
        if hit_sl:
            return "LOSS", i
    return "OPEN", None


def _first_touch_index(
    high: np.ndarray,
    low: np.ndarray,
    start: int,
    end: int,
    target_up: float | None = None,
    target_down: float | None = None,
) -> Optional[int]:
    if start > end:
        return None
    for i in range(start, end + 1):
        if target_up is not None and high[i] >= target_up:
            return i
        if target_down is not None and low[i] <= target_down:
            return i
    return None


def build_research_master(
    trades_cls: pd.DataFrame,
    cfg: Pine16ExactConfig,
    truth_label: TruthLabel,
    feature_events: pd.DataFrame,
) -> pd.DataFrame:
    if trades_cls.empty:
        return pd.DataFrame()

    legacy_cfg = to_legacy_run_config(cfg)
    tf_minutes = timeframe_to_minutes(cfg.timeframe)
    max_h = max(HORIZONS_H)
    max_bars_fwd = int((max_h * 60) // tf_minutes)

    feat = feature_events.copy() if not feature_events.empty else pd.DataFrame()
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
        sym_feat = feat[feat["symbol"].astype(str) == symbol] if not feat.empty else pd.DataFrame()
        feat_map = sym_feat.set_index("trade_id", drop=False) if not sym_feat.empty and "trade_id" in sym_feat.columns else None

        for j in range(len(sym_tr)):
            ei = int(idx[j]) if j < len(idx) else -1
            if ei < 0 or ei >= len(bars):
                continue

            trade_id = str(sym_tr.at[j, "trade_id"])
            signal_id = str(sym_tr.at[j, "signal_id"])
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

            start = ei + 1
            end_max = min(len(high) - 1, ei + max_bars_fwd)
            if start > end_max:
                continue

            first_plus1 = _first_touch_index(high, low, start, end_max, target_up=tp1)
            first_plus2 = _first_touch_index(high, low, start, end_max, target_up=tp2)
            first_plus3 = _first_touch_index(high, low, start, end_max, target_up=tp3)
            first_minus1 = _first_touch_index(high, low, start, end_max, target_down=sl1)
            first_minus05 = _first_touch_index(high, low, start, end_max, target_down=sl05)

            immediate_low = low[start] if start < len(low) else np.nan
            immediate_drawdown_gt_05r = int(np.isfinite(immediate_low) and immediate_low <= sl05)
            immediate_drawdown_gt_1r = int(np.isfinite(immediate_low) and immediate_low <= sl1)

            base: Dict[str, object] = {
                "signal_id": signal_id,
                "trade_id": trade_id,
                "symbol": symbol,
                "timeframe": str(cfg.timeframe),
                "year": int(sym_tr.at[j, "year"]) if pd.notna(sym_tr.at[j, "year"]) else np.nan,
                "month": int(sym_tr.at[j, "month"]) if pd.notna(sym_tr.at[j, "month"]) else np.nan,
                "quarter": int(sym_tr.at[j, "quarter"]) if pd.notna(sym_tr.at[j, "quarter"]) else np.nan,
                "week": int(sym_tr.at[j, "week"]) if pd.notna(sym_tr.at[j, "week"]) else np.nan,
                "date": sym_tr.at[j, "date"],
                "truth_label": truth_label.value,
                "config_pack": str(cfg.metadata.config_pack),
                "setup_session_bucket": sym_tr.at[j, "setup_session_bucket"],
                "entry_session_bucket": sym_tr.at[j, "entry_session_bucket"],
                "setup_in_london": int(sym_tr.at[j, "setup_in_london"]),
                "setup_in_newyork": int(sym_tr.at[j, "setup_in_newyork"]),
                "entry_in_london": int(sym_tr.at[j, "entry_in_london"]),
                "entry_in_newyork": int(sym_tr.at[j, "entry_in_newyork"]),
                "entry_in_london_or_newyork": int(sym_tr.at[j, "entry_in_london_or_newyork"]),
                "entry_in_other": int(sym_tr.at[j, "entry_in_other"]),
                "entry_time": pd.to_datetime(sym_tr.at[j, "entry_time_utc"], utc=True, errors="coerce"),
                "entry_price": ep,
                "sl_price_1r": sl1,
                "tp_price_1r": tp1,
                "tp_price_2r": tp2,
                "tp_price_3r": tp3,
                "atrEntry": atr_entry,
                "time_to_plus1r_bars": float(first_plus1 - ei) if first_plus1 is not None else np.nan,
                "time_to_plus2r_bars": float(first_plus2 - ei) if first_plus2 is not None else np.nan,
                "time_to_plus3r_bars": float(first_plus3 - ei) if first_plus3 is not None else np.nan,
                "time_to_minus1r_bars": float(first_minus1 - ei) if first_minus1 is not None else np.nan,
                "immediate_drawdown_gt_05r": immediate_drawdown_gt_05r,
                "immediate_drawdown_gt_1r": immediate_drawdown_gt_1r,
                "realized_win": int(float(pd.to_numeric(sym_tr.at[j, "result_r"], errors="coerce")) > 0) if pd.notna(sym_tr.at[j, "result_r"]) else np.nan,
                "realized_result_r": pd.to_numeric(sym_tr.at[j, "result_r"], errors="coerce"),
                "realized_bars_held": pd.to_numeric(sym_tr.at[j, "bars_held"], errors="coerce"),
                "donLen": int(cfg.core.donLen),
                "pivotLen": int(cfg.core.pivotLen),
                "oscLen": int(cfg.core.oscLen),
                "extBandPct": float(cfg.core.extBandPct),
            }

            feature_cols = [
                "oscChangePct",
                "priceChangePct",
                "barsBetweenPivots",
                "divType",
                "classicFlag",
                "locAtPivot",
                "volRatioAtPivot",
                "volRatioAtEntry",
                "atrRatio",
                "rsi14Pivot",
                "rsi14Entry",
                "macdHistEntry",
                "oscPercentile",
                "recentMomentum",
                "trendState",
                "dailyEmaOk",
                "dailySlopeOk",
                "dailyAdxOk",
                "dailyDiOk",
                "cvdOk",
                "volBehaviorOk",
                "volSpikeOk",
                "wv70Ok",
                "bosUsed",
                "bosPassed",
                "bosWaitBars",
                "warmupSatisfied",
                "cooldownSatisfied",
                "dailyAdxValue",
            ]
            if feat_map is not None and trade_id in feat_map.index:
                frow = feat_map.loc[trade_id]
                if isinstance(frow, pd.DataFrame):
                    frow = frow.iloc[-1]
                for c in feature_cols:
                    base[c] = frow[c] if c in frow.index else np.nan
            else:
                for c in feature_cols:
                    base[c] = np.nan

            for h in HORIZONS_H:
                bars_fwd = int((h * 60) // tf_minutes)
                end = min(len(high) - 1, ei + bars_fwd)
                out1, _ = _scan_barrier_outcome(high, low, start, end, tp1, sl1)
                out2, _ = _scan_barrier_outcome(high, low, start, end, tp2, sl1)
                out3, _ = _scan_barrier_outcome(high, low, start, end, tp3, sl1)

                win_high = high[start : end + 1]
                win_low = low[start : end + 1]
                mfe_r = float((float(np.max(win_high)) - ep) / risk_unit) if len(win_high) else np.nan
                mae_r = float((float(np.min(win_low)) - ep) / risk_unit) if len(win_low) else np.nan
                favorable = int(close[end] > ep) if np.isfinite(ep) else 0
                mfe_gt_mae = int(np.isfinite(mfe_r) and np.isfinite(mae_r) and (mfe_r > abs(mae_r)))

                suffix = f"_{h}h"
                base[f"outcome_plus1r_before_minus1r{suffix}"] = out1
                base[f"plus1r_before_minus1r{suffix}"] = 1.0 if out1 == "WIN" else (0.0 if out1 == "LOSS" else np.nan)
                base[f"minus1r_before_plus1r{suffix}"] = 1.0 if out1 == "LOSS" else (0.0 if out1 == "WIN" else np.nan)
                base[f"open_1r{suffix}"] = 1.0 if out1 == "OPEN" else 0.0

                base[f"outcome_plus2r_before_minus1r{suffix}"] = out2
                base[f"plus2r_before_minus1r{suffix}"] = 1.0 if out2 == "WIN" else (0.0 if out2 == "LOSS" else np.nan)
                base[f"minus1r_before_plus2r{suffix}"] = 1.0 if out2 == "LOSS" else (0.0 if out2 == "WIN" else np.nan)
                base[f"open_2r{suffix}"] = 1.0 if out2 == "OPEN" else 0.0

                base[f"outcome_plus3r_before_minus1r{suffix}"] = out3
                base[f"plus3r_before_minus1r{suffix}"] = 1.0 if out3 == "WIN" else (0.0 if out3 == "LOSS" else np.nan)
                base[f"minus1r_before_plus3r{suffix}"] = 1.0 if out3 == "LOSS" else (0.0 if out3 == "WIN" else np.nan)
                base[f"open_3r{suffix}"] = 1.0 if out3 == "OPEN" else 0.0

                base[f"mfe_{h}h_r"] = mfe_r
                base[f"mae_{h}h_r"] = mae_r
                base[f"favorable_move_{h}h"] = float(favorable)
                base[f"mfe_gt_mae_{h}h"] = float(mfe_gt_mae)

            out05_24, _ = _scan_barrier_outcome(
                high,
                low,
                start,
                min(len(high) - 1, ei + int((24 * 60) // tf_minutes)),
                tp05,
                sl1,
            )
            base["plus05r_before_minus1r_24h"] = 1.0 if out05_24 == "WIN" else (0.0 if out05_24 == "LOSS" else np.nan)
            base["reaches_plus1r_without_minus05r"] = 1 if (first_plus1 is not None and (first_minus05 is None or first_plus1 < first_minus05)) else 0
            base["reaches_plus3r_without_minus1r"] = 1 if (first_plus3 is not None and (first_minus1 is None or first_plus3 < first_minus1)) else 0

            rows.append(base)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["entry_session_bucket"] = out["entry_session_bucket"].fillna("unknown").astype(str)
    out["session_scope_bucket"] = out["entry_session_bucket"]
    out["daily_regime_state"] = np.where(
        (pd.to_numeric(out["dailyEmaOk"], errors="coerce") == 1)
        & (pd.to_numeric(out["dailyDiOk"], errors="coerce") == 1),
        "ema_di_aligned",
        np.where(pd.to_numeric(out["dailyEmaOk"], errors="coerce") == 1, "ema_only", "other"),
    )
    out["volRatioEntryGateOn"] = int(bool(cfg.features.useVolEntryGate))
    out["volRatioPivotGateOn"] = int(bool(cfg.features.useVolPivotGate))
    out["sessionGateOn"] = int(bool(cfg.session.useSessionGate))

    return out


def _qcut_or_nan(series: pd.Series, labels: Sequence[str]) -> pd.Series:
    ser = pd.to_numeric(series, errors="coerce")
    mask = ser.notna()
    if mask.sum() < len(labels):
        return pd.Series(np.nan, index=series.index, dtype="object")
    ranked = ser[mask].rank(method="first")
    try:
        buckets = pd.qcut(ranked, q=len(labels), labels=list(labels))
    except Exception:
        return pd.Series(np.nan, index=series.index, dtype="object")
    out = pd.Series(np.nan, index=series.index, dtype="object")
    out.loc[mask] = buckets.astype(str)
    return out


def add_research_buckets(master: pd.DataFrame, cfg: Pine16ExactConfig) -> pd.DataFrame:
    if master.empty:
        return master
    out = master.copy()

    out["osc_bucket"] = _qcut_or_nan(out["oscChangePct"], ["weak", "medium", "strong", "extreme"])
    out["pivot_gap_bucket"] = _qcut_or_nan(out["barsBetweenPivots"], ["q1", "q2", "q3", "q4"])

    v_entry = pd.to_numeric(out["volRatioAtEntry"], errors="coerce")
    out["vol_entry_bucket"] = pd.cut(
        v_entry,
        bins=[-np.inf, 1.0, 1.2, 1.5, np.inf],
        labels=["<1.0", "1.0-1.2", "1.2-1.5", ">1.5"],
        right=False,
    ).astype("object")

    v_pivot = pd.to_numeric(out["volRatioAtPivot"], errors="coerce")
    out["vol_pivot_bucket"] = pd.cut(
        v_pivot,
        bins=[-np.inf, 1.0, 1.2, 1.5, np.inf],
        labels=["<1.0", "1.0-1.2", "1.2-1.5", ">1.5"],
        right=False,
    ).astype("object")

    atr_ratio = pd.to_numeric(out["atrRatio"], errors="coerce")
    q1 = atr_ratio.quantile(0.33)
    q2 = atr_ratio.quantile(0.66)
    out["atr_ratio_bucket"] = np.where(
        atr_ratio < q1,
        "low",
        np.where(atr_ratio < q2, "normal", "high"),
    )

    loc = pd.to_numeric(out["locAtPivot"], errors="coerce")
    out["loc_bucket"] = np.where(loc <= float(cfg.core.extBandPct), "deep-extreme", "moderate-extreme")

    rsi = pd.to_numeric(out["rsi14Pivot"], errors="coerce")
    out["rsi_bucket"] = np.where(rsi < 30.0, "<30", np.where(rsi < 40.0, "30-40", "40+"))
    out["year_bucket"] = out["year"].astype("Int64").astype(str)

    p1_24 = pd.to_numeric(out.get("plus1r_before_minus1r_24h"), errors="coerce")
    p3_24 = pd.to_numeric(out.get("plus3r_before_minus1r_24h"), errors="coerce")
    out["trade_quality_bucket"] = np.where(
        (p3_24 == 1.0) & (p1_24 == 1.0),
        "A_high",
        np.where((p1_24 == 1.0), "B_directional", np.where((p1_24.isna()), "C_open", "D_weak")),
    )
    return out


def build_long_horizon_frame(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame()
    rows: List[pd.DataFrame] = []
    base_cols = [
        "signal_id",
        "trade_id",
        "symbol",
        "timeframe",
        "year",
        "month",
        "quarter",
        "week",
        "date",
        "truth_label",
        "config_pack",
        "setup_session_bucket",
        "entry_session_bucket",
        "entry_in_london",
        "entry_in_newyork",
        "entry_in_london_or_newyork",
        "entry_in_other",
        "session_scope_bucket",
        "osc_bucket",
        "pivot_gap_bucket",
        "vol_entry_bucket",
        "vol_pivot_bucket",
        "atr_ratio_bucket",
        "loc_bucket",
        "rsi_bucket",
        "year_bucket",
        "trade_quality_bucket",
        "divType",
        "bosUsed",
        "volRatioEntryGateOn",
        "daily_regime_state",
        "time_to_plus1r_bars",
        "time_to_plus2r_bars",
        "time_to_plus3r_bars",
        "time_to_minus1r_bars",
        "reaches_plus1r_without_minus05r",
        "reaches_plus3r_without_minus1r",
        "immediate_drawdown_gt_05r",
        "immediate_drawdown_gt_1r",
    ]

    for h in HORIZONS_H:
        suffix = f"_{h}h"
        tmp = master[base_cols].copy()
        tmp["horizon_h"] = int(h)
        tmp["win_1r"] = pd.to_numeric(master.get(f"plus1r_before_minus1r{suffix}"), errors="coerce")
        tmp["loss_1r"] = pd.to_numeric(master.get(f"minus1r_before_plus1r{suffix}"), errors="coerce")
        tmp["open_1r"] = pd.to_numeric(master.get(f"open_1r{suffix}"), errors="coerce")
        tmp["win_2r"] = pd.to_numeric(master.get(f"plus2r_before_minus1r{suffix}"), errors="coerce")
        tmp["loss_2r"] = pd.to_numeric(master.get(f"minus1r_before_plus2r{suffix}"), errors="coerce")
        tmp["open_2r"] = pd.to_numeric(master.get(f"open_2r{suffix}"), errors="coerce")
        tmp["win_3r"] = pd.to_numeric(master.get(f"plus3r_before_minus1r{suffix}"), errors="coerce")
        tmp["loss_3r"] = pd.to_numeric(master.get(f"minus1r_before_plus3r{suffix}"), errors="coerce")
        tmp["open_3r"] = pd.to_numeric(master.get(f"open_3r{suffix}"), errors="coerce")
        tmp["mfe_r"] = pd.to_numeric(master.get(f"mfe_{h}h_r"), errors="coerce")
        tmp["mae_r"] = pd.to_numeric(master.get(f"mae_{h}h_r"), errors="coerce")
        tmp["favorable_move"] = pd.to_numeric(master.get(f"favorable_move_{h}h"), errors="coerce")
        tmp["mfe_gt_mae"] = pd.to_numeric(master.get(f"mfe_gt_mae_{h}h"), errors="coerce")
        rows.append(tmp)

    return pd.concat(rows, ignore_index=True, sort=False)


def _filter_scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
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


def plus1r_verdict(p: float) -> str:
    if not np.isfinite(p):
        return "weak"
    if p < 0.50:
        return "weak"
    if p < 0.55:
        return "borderline"
    if p <= 0.60:
        return "clear"
    return "strong"


def plus3r_verdict(p: float) -> str:
    if not np.isfinite(p):
        return "not monetizable at 1:3"
    if p < 0.25:
        return "not monetizable at 1:3"
    if p < 0.30:
        return "marginal"
    if p <= 0.40:
        return "monetizable"
    return "strong monetizable edge"


def directional_verdict(p1: float, favorable: float, ratio: float, pct_mfe_gt_mae: float) -> str:
    if not (np.isfinite(p1) and np.isfinite(favorable) and np.isfinite(ratio) and np.isfinite(pct_mfe_gt_mae)):
        return "No directional edge"
    if p1 > 0.60 and favorable >= 0.60 and ratio >= 1.30 and pct_mfe_gt_mae >= 0.60:
        return "Strong directional edge"
    if p1 >= 0.55 and favorable >= 0.55 and ratio >= 1.15 and pct_mfe_gt_mae >= 0.55:
        return "Clear directional edge"
    if p1 >= 0.52 and ratio >= 1.02:
        return "Weak directional edge"
    return "No directional edge"


def tradable_verdict(p3: float, expectancy_1to3: float, pf_1to3: float, net_r_1to3: float) -> str:
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
            "pct_reach_plus2r_before_minus1r": np.nan,
            "pct_reach_plus3r_before_minus1r": np.nan,
            "pct_minus1r_before_plus1r_for_1r_test": np.nan,
            "pct_minus1r_before_plus3r_for_3r_test": np.nan,
            "open_rate_1r": np.nan,
            "open_rate_2r": np.nan,
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
            "pct_reaches_plus1r_without_minus05r": np.nan,
            "pct_reaches_plus3r_without_minus1r": np.nan,
            "pct_immediate_drawdown_gt_05r": np.nan,
            "pct_immediate_drawdown_gt_1r": np.nan,
            "median_time_to_plus1r_bars": np.nan,
            "median_time_to_plus2r_bars": np.nan,
            "median_time_to_plus3r_bars": np.nan,
            "median_time_to_minus1r_bars": np.nan,
        }

    win1 = float(pd.to_numeric(g["win_1r"], errors="coerce").fillna(0.0).sum())
    loss1 = float(pd.to_numeric(g["loss_1r"], errors="coerce").fillna(0.0).sum())
    open1 = float(pd.to_numeric(g["open_1r"], errors="coerce").fillna(0.0).sum())

    win2 = float(pd.to_numeric(g["win_2r"], errors="coerce").fillna(0.0).sum())
    open2 = float(pd.to_numeric(g["open_2r"], errors="coerce").fillna(0.0).sum())

    win3 = float(pd.to_numeric(g["win_3r"], errors="coerce").fillna(0.0).sum())
    loss3 = float(pd.to_numeric(g["loss_3r"], errors="coerce").fillna(0.0).sum())
    open3 = float(pd.to_numeric(g["open_3r"], errors="coerce").fillna(0.0).sum())

    med_mfe = float(pd.to_numeric(g["mfe_r"], errors="coerce").median())
    med_mae = float(pd.to_numeric(g["mae_r"], errors="coerce").median())
    ratio = float(med_mfe / abs(med_mae)) if np.isfinite(med_mae) and med_mae != 0 else np.nan

    expectancy_1r = float((win1 - loss1) / n)
    expectancy_1to3 = float((3.0 * win3 - loss3) / n)
    gross_win_1to3 = float(3.0 * win3)
    gross_loss_1to3 = float(loss3)
    pf_1to3 = float(gross_win_1to3 / gross_loss_1to3) if gross_loss_1to3 > 0 else np.nan
    net_r_1to3 = float(gross_win_1to3 - gross_loss_1to3)

    return {
        "n_signals": int(g["signal_id"].nunique()),
        "n_trades": int(g["trade_id"].nunique()),
        "pct_reach_plus1r_before_minus1r": float(win1 / n),
        "pct_reach_plus2r_before_minus1r": float(win2 / n),
        "pct_reach_plus3r_before_minus1r": float(win3 / n),
        "pct_minus1r_before_plus1r_for_1r_test": float(loss1 / n),
        "pct_minus1r_before_plus3r_for_3r_test": float(loss3 / n),
        "open_rate_1r": float(open1 / n),
        "open_rate_2r": float(open2 / n),
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
        "pct_reaches_plus1r_without_minus05r": float(pd.to_numeric(g["reaches_plus1r_without_minus05r"], errors="coerce").mean()),
        "pct_reaches_plus3r_without_minus1r": float(pd.to_numeric(g["reaches_plus3r_without_minus1r"], errors="coerce").mean()),
        "pct_immediate_drawdown_gt_05r": float(pd.to_numeric(g["immediate_drawdown_gt_05r"], errors="coerce").mean()),
        "pct_immediate_drawdown_gt_1r": float(pd.to_numeric(g["immediate_drawdown_gt_1r"], errors="coerce").mean()),
        "median_time_to_plus1r_bars": float(pd.to_numeric(g["time_to_plus1r_bars"], errors="coerce").median()),
        "median_time_to_plus2r_bars": float(pd.to_numeric(g["time_to_plus2r_bars"], errors="coerce").median()),
        "median_time_to_plus3r_bars": float(pd.to_numeric(g["time_to_plus3r_bars"], errors="coerce").median()),
        "median_time_to_minus1r_bars": float(pd.to_numeric(g["time_to_minus1r_bars"], errors="coerce").median()),
    }


def aggregate_metrics(long_df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    gb_cols = list(group_cols)
    if not gb_cols:
        row = _agg_metrics(long_df)
        row["horizon_h"] = int(long_df["horizon_h"].iloc[0]) if "horizon_h" in long_df.columns and not long_df.empty else np.nan
        rows.append(row)
    else:
        gb = long_df.groupby(gb_cols, dropna=False, sort=True)
        for keys, g in gb:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {c: v for c, v in zip(gb_cols, keys)}
            row.update(_agg_metrics(g))
            if "horizon_h" in g.columns:
                row["horizon_h"] = int(g["horizon_h"].iloc[0])
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["plus1r_verdict"] = out["pct_reach_plus1r_before_minus1r"].apply(plus1r_verdict)
    out["plus3r_verdict"] = out["pct_reach_plus3r_before_minus1r"].apply(plus3r_verdict)
    out["directional_edge_verdict"] = out.apply(
        lambda r: directional_verdict(
            float(r["pct_reach_plus1r_before_minus1r"]),
            float(r["favorable_move_rate"]),
            float(r["mfe_mae_ratio"]),
            float(r["pct_mfe_gt_mae"]),
        ),
        axis=1,
    )
    out["tradable_edge_verdict"] = out.apply(
        lambda r: tradable_verdict(
            float(r["pct_reach_plus3r_before_minus1r"]),
            float(r["expectancy_1to3_r"]),
            float(r["profit_factor_1to3"]),
            float(r["net_r_1to3"]),
        ),
        axis=1,
    )
    return out


def aggregate_by_scopes(long_df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    tables: List[pd.DataFrame] = []
    for scope in SESSION_SCOPES:
        scoped = _filter_scope(long_df, scope)
        if scoped.empty:
            continue
        gb_cols = list(group_cols) + ["horizon_h"]
        t = aggregate_metrics(scoped, group_cols=gb_cols)
        if t.empty:
            continue
        t["session_scope"] = scope
        tables.append(t)
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True, sort=False)


def _robustness_from_year_table(df: pd.DataFrame, key_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    if key_cols:
        group_iter = df.groupby(list(key_cols), dropna=False, sort=True)
    else:
        group_iter = [((), df)]
    for keys, g in group_iter:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: v for c, v in zip(key_cols, keys)} if key_cols else {}
        n_total = float(pd.to_numeric(g["n_trades"], errors="coerce").sum())
        years_present = int(g["year"].nunique()) if "year" in g.columns else 0
        pos_years = int((pd.to_numeric(g["expectancy_1to3_r"], errors="coerce") > 0).sum())
        neg_years = int((pd.to_numeric(g["expectancy_1to3_r"], errors="coerce") <= 0).sum())
        std_expect = float(pd.to_numeric(g["expectancy_1to3_r"], errors="coerce").std(ddof=0))
        nabs = pd.to_numeric(g["net_r_1to3"], errors="coerce").abs()
        concentration = float(nabs.max() / nabs.sum()) if np.isfinite(float(nabs.sum())) and float(nabs.sum()) > 0 else np.nan

        p1 = float(pd.to_numeric(g["pct_reach_plus1r_before_minus1r"], errors="coerce").mean())
        p3 = float(pd.to_numeric(g["pct_reach_plus3r_before_minus1r"], errors="coerce").mean())
        exp = float(pd.to_numeric(g["expectancy_1to3_r"], errors="coerce").mean())

        n_score = min(1.0, n_total / 100.0)
        year_consistency = (pos_years / years_present) if years_present > 0 else 0.0
        p1_score = np.clip((p1 - 0.50) / 0.15, 0.0, 1.0)
        p3_score = np.clip((p3 - 0.25) / 0.20, 0.0, 1.0)
        exp_score = np.clip((exp + 0.05) / 0.20, 0.0, 1.0)
        conc_penalty = 1.0 - (concentration if np.isfinite(concentration) else 0.5)
        std_penalty = 1.0 - np.clip(std_expect / 0.25 if np.isfinite(std_expect) else 0.5, 0.0, 1.0)

        robustness = float(
            0.20 * n_score
            + 0.20 * year_consistency
            + 0.15 * p1_score
            + 0.15 * p3_score
            + 0.15 * exp_score
            + 0.10 * conc_penalty
            + 0.05 * std_penalty
        )
        fragility = bool(years_present < 2 or (np.isfinite(concentration) and concentration > 0.70) or (np.isfinite(std_expect) and std_expect > 0.20))

        row.update(
            {
                "n": n_total,
                "years_present": years_present,
                "positive_years": pos_years,
                "negative_years": neg_years,
                "std_expectancy_by_year": std_expect,
                "concentration_score": concentration,
                "robustness_score": robustness,
                "fragility_flag": fragility,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_top_combinations(
    h24: pd.DataFrame,
    min_n: int,
    min_n_robust: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if h24.empty:
        return pd.DataFrame(), pd.DataFrame()

    work = h24.copy()
    for col in ["session_scope_bucket", "osc_bucket", "pivot_gap_bucket", "vol_entry_bucket", "atr_ratio_bucket", "divType", "daily_regime_state"]:
        if col in work.columns:
            work[col] = work[col].fillna("missing").astype(str)

    combo_specs: List[Tuple[str, List[str]]] = [
        ("L1_symbol_session", ["symbol", "session_scope_bucket"]),
        ("L2_symbol_session_osc", ["symbol", "session_scope_bucket", "osc_bucket"]),
        ("L2_symbol_session_volentry", ["symbol", "session_scope_bucket", "vol_entry_bucket"]),
        ("L2_symbol_session_atr", ["symbol", "session_scope_bucket", "atr_ratio_bucket"]),
        ("L3_symbol_session_osc_pivotgap", ["symbol", "session_scope_bucket", "osc_bucket", "pivot_gap_bucket"]),
        ("L3_symbol_session_vol_atr_div", ["symbol", "session_scope_bucket", "vol_entry_bucket", "atr_ratio_bucket", "divType"]),
    ]

    combo_tables: List[pd.DataFrame] = []
    robustness_tables: List[pd.DataFrame] = []
    for level_name, spec_cols in combo_specs:
        base_cols = spec_cols + ["bosUsed", "volRatioEntryGateOn"]
        agg = aggregate_metrics(work, group_cols=base_cols)
        if agg.empty:
            continue
        agg["combo_level"] = level_name
        agg["year_scope"] = "all_years"
        combo_tables.append(agg)

        yr = aggregate_metrics(work, group_cols=base_cols + ["year"])
        if not yr.empty:
            rb = _robustness_from_year_table(yr, key_cols=base_cols)
            if not rb.empty:
                rb["combo_level"] = level_name
                robustness_tables.append(rb)

    if not combo_tables:
        return pd.DataFrame(), pd.DataFrame()

    combos = pd.concat(combo_tables, ignore_index=True, sort=False)
    if robustness_tables:
        rb_all = pd.concat(robustness_tables, ignore_index=True, sort=False)
        key_merge = [c for c in ["combo_level", "symbol", "session_scope_bucket", "osc_bucket", "pivot_gap_bucket", "vol_entry_bucket", "atr_ratio_bucket", "divType", "daily_regime_state", "bosUsed", "volRatioEntryGateOn"] if c in combos.columns and c in rb_all.columns]
        combos = combos.merge(rb_all, on=key_merge, how="left")

    combos = combos[pd.to_numeric(combos["n_trades"], errors="coerce") >= float(min_n)].copy()
    if combos.empty:
        return combos, combos

    combos["feature_conditions"] = combos.apply(
        lambda r: (
            f"level={r.get('combo_level', 'na')};"
            f"bosUsed={int(pd.to_numeric(r.get('bosUsed'), errors='coerce')) if pd.notna(r.get('bosUsed')) else 'na'};"
            f"volEntryGate={int(pd.to_numeric(r.get('volRatioEntryGateOn'), errors='coerce')) if pd.notna(r.get('volRatioEntryGateOn')) else 'na'};"
            f"osc={r.get('osc_bucket', 'any')};pivot_gap={r.get('pivot_gap_bucket', 'any')};"
            f"vol_entry={r.get('vol_entry_bucket', 'any')};atr={r.get('atr_ratio_bucket', 'any')};"
            f"div={r.get('divType', 'any')};daily={r.get('daily_regime_state', 'any')}"
        ),
        axis=1,
    )

    combos["action"] = "WATCH"
    keep_mask = (
        (pd.to_numeric(combos["n_trades"], errors="coerce") >= float(min_n_robust))
        & (pd.to_numeric(combos["pct_reach_plus1r_before_minus1r"], errors="coerce") >= 0.55)
        & (pd.to_numeric(combos["pct_reach_plus3r_before_minus1r"], errors="coerce") >= 0.30)
        & (pd.to_numeric(combos["expectancy_1to3_r"], errors="coerce") > 0)
        & (pd.to_numeric(combos["robustness_score"], errors="coerce") >= 0.55)
    )
    cut_mask = (
        (pd.to_numeric(combos["n_trades"], errors="coerce") >= float(min_n))
        & (pd.to_numeric(combos["pct_reach_plus1r_before_minus1r"], errors="coerce") < 0.50)
        & (pd.to_numeric(combos["pct_reach_plus3r_before_minus1r"], errors="coerce") < 0.25)
        & (pd.to_numeric(combos["expectancy_1to3_r"], errors="coerce") <= 0)
    )
    combos.loc[keep_mask, "action"] = "KEEP"
    combos.loc[cut_mask, "action"] = "CUT"

    combos = combos.sort_values(
        ["robustness_score", "expectancy_1to3_r", "pct_reach_plus3r_before_minus1r", "n_trades"],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    combos.insert(0, "combo_id", [f"C{idx+1:04d}" for idx in range(len(combos))])

    leaderboard_parts: List[pd.DataFrame] = []
    for thr in [min_n, max(min_n_robust, 30), 50]:
        part = combos[pd.to_numeric(combos["n_trades"], errors="coerce") >= float(thr)].copy()
        if part.empty:
            continue
        part["sample_tier"] = f"n>={thr}"
        leaderboard_parts.append(part)
    leaderboard = pd.concat(leaderboard_parts, ignore_index=True, sort=False) if leaderboard_parts else combos.copy()
    return combos, leaderboard


def build_keep_watch_cut(
    by_symbol_24: pd.DataFrame,
    by_symbol_year_24: pd.DataFrame,
    by_symbol_session_24: pd.DataFrame,
    by_symbol_year_session_24: pd.DataFrame,
    min_n: int,
    min_n_robust: int,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    if not by_symbol_24.empty:
        rb_sym = _robustness_from_year_table(by_symbol_year_24, key_cols=["symbol"]) if not by_symbol_year_24.empty else pd.DataFrame()
        sym = by_symbol_24.merge(rb_sym[["symbol", "robustness_score", "fragility_flag"]], on="symbol", how="left") if not rb_sym.empty else by_symbol_24.copy()
        sym["entity_type"] = "symbol"
        sym["session_scope"] = "all_sessions"
        frames.append(sym)

    if not by_symbol_session_24.empty:
        key_cols = ["symbol", "session_scope"]
        rb_ss = _robustness_from_year_table(by_symbol_year_session_24, key_cols=key_cols) if not by_symbol_year_session_24.empty else pd.DataFrame()
        ss = by_symbol_session_24.merge(rb_ss[key_cols + ["robustness_score", "fragility_flag"]], on=key_cols, how="left") if not rb_ss.empty else by_symbol_session_24.copy()
        ss["entity_type"] = "symbol_session"
        frames.append(ss)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    out["action"] = "WATCH"

    keep_mask = (
        (pd.to_numeric(out["n_trades"], errors="coerce") >= float(min_n_robust))
        & (pd.to_numeric(out["expectancy_1to3_r"], errors="coerce") > 0)
        & (pd.to_numeric(out["pct_reach_plus1r_before_minus1r"], errors="coerce") >= 0.55)
        & (pd.to_numeric(out["pct_reach_plus3r_before_minus1r"], errors="coerce") >= 0.30)
        & (pd.to_numeric(out.get("robustness_score"), errors="coerce") >= 0.50)
    )
    cut_mask = (
        (pd.to_numeric(out["n_trades"], errors="coerce") >= float(min_n))
        & (pd.to_numeric(out["pct_reach_plus1r_before_minus1r"], errors="coerce") < 0.50)
        & (pd.to_numeric(out["pct_reach_plus3r_before_minus1r"], errors="coerce") < 0.25)
        & (pd.to_numeric(out["expectancy_1to3_r"], errors="coerce") <= 0)
    )
    out.loc[keep_mask, "action"] = "KEEP"
    out.loc[cut_mask, "action"] = "CUT"

    cols = [
        "entity_type",
        "symbol",
        "session_scope",
        "n_trades",
        "pct_reach_plus1r_before_minus1r",
        "pct_reach_plus2r_before_minus1r",
        "pct_reach_plus3r_before_minus1r",
        "expectancy_1to3_r",
        "directional_edge_verdict",
        "tradable_edge_verdict",
        "robustness_score",
        "fragility_flag",
        "action",
    ]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[cols].sort_values(["entity_type", "action", "expectancy_1to3_r"], ascending=[True, True, False], kind="mergesort")
    return out.reset_index(drop=True)


def build_deployment_candidates(h24: pd.DataFrame, min_n: int, min_n_robust: int) -> pd.DataFrame:
    if h24.empty:
        return pd.DataFrame()

    candidates = [
        {"candidate": "XAUUSD_london_only", "symbols": ["XAUUSD"], "scope": "london_only"},
        {"candidate": "XAUUSD_london_or_newyork", "symbols": ["XAUUSD"], "scope": "london_or_newyork"},
        {"candidate": "metals_london_or_newyork", "symbols": ["XAUUSD", "XAGUSD"], "scope": "london_or_newyork"},
        {"candidate": "metals_all_sessions", "symbols": ["XAUUSD", "XAGUSD"], "scope": "all_sessions"},
        {"candidate": "metals_plus_eurusd_london_only", "symbols": ["XAUUSD", "XAGUSD", "EURUSD"], "scope": "london_only"},
        {"candidate": "full_basket_all_sessions", "symbols": ["XAUUSD", "XAGUSD", "EURUSD", "GBPUSD", "USDCHF"], "scope": "all_sessions"},
    ]

    rows: List[Dict[str, object]] = []
    for cand in candidates:
        sub = h24[h24["symbol"].astype(str).isin(cand["symbols"])].copy()
        sub = _filter_scope(sub, cand["scope"])
        if sub.empty:
            continue
        agg = aggregate_metrics(sub, group_cols=[])
        if agg.empty:
            continue
        r = agg.iloc[0].to_dict()
        by_year = aggregate_metrics(sub, group_cols=["year"])
        rb = _robustness_from_year_table(by_year, key_cols=[])
        robustness = float(rb["robustness_score"].iloc[0]) if not rb.empty else np.nan

        deploy = "REJECT"
        if (
            float(r.get("n_trades", 0)) >= float(min_n_robust)
            and float(r.get("pct_reach_plus3r_before_minus1r", 0.0)) >= 0.30
            and float(r.get("expectancy_1to3_r", -1.0)) > 0
            and (np.isfinite(robustness) and robustness >= 0.50)
        ):
            deploy = "DEPLOY"
        elif float(r.get("n_trades", 0)) >= float(min_n) and float(r.get("expectancy_1to3_r", -1.0)) > -0.02:
            deploy = "WATCH"

        rows.append(
            {
                "candidate": cand["candidate"],
                "symbols": ",".join(cand["symbols"]),
                "session_scope": cand["scope"],
                "n_trades": r.get("n_trades"),
                "pct_plus1r_before_minus1r_24h": r.get("pct_reach_plus1r_before_minus1r"),
                "pct_plus2r_before_minus1r_24h": r.get("pct_reach_plus2r_before_minus1r"),
                "pct_plus3r_before_minus1r_24h": r.get("pct_reach_plus3r_before_minus1r"),
                "expectancy_1to3_r_24h": r.get("expectancy_1to3_r"),
                "robustness_score": robustness,
                "directional_edge_verdict": r.get("directional_edge_verdict"),
                "tradable_edge_verdict": r.get("tradable_edge_verdict"),
                "deployment_action": deploy,
                "why": (
                    "quality+expectancy+robustness acceptable"
                    if deploy == "DEPLOY"
                    else ("mixed quality or limited robustness" if deploy == "WATCH" else "negative or fragile profile")
                ),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["deployment_action", "expectancy_1to3_r_24h", "robustness_score"], ascending=[True, False, False], kind="mergesort")
    return out.reset_index(drop=True)


def compute_feature_importance(h24: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    if h24.empty:
        return pd.DataFrame(), []

    target = pd.to_numeric(h24["win_3r"], errors="coerce").fillna(0.0)
    base_rate = float(target.mean())
    rows: List[Dict[str, object]] = []
    rules: List[str] = []

    numeric_features = [
        "oscChangePct",
        "barsBetweenPivots",
        "volRatioAtEntry",
        "volRatioAtPivot",
        "atrRatio",
        "locAtPivot",
        "rsi14Pivot",
        "dailyAdxValue",
        "bosWaitBars",
    ]
    for f in numeric_features:
        if f not in h24.columns:
            continue
        x = pd.to_numeric(h24[f], errors="coerce")
        valid = x.notna() & target.notna()
        if valid.sum() < 20:
            continue
        corr = float(x[valid].corr(target[valid]))
        rows.append({"feature": f, "kind": "numeric_corr", "score": corr, "support": int(valid.sum())})

    cat_features = ["symbol", "session_scope_bucket", "osc_bucket", "pivot_gap_bucket", "vol_entry_bucket", "atr_ratio_bucket", "divType", "daily_regime_state"]
    for f in cat_features:
        if f not in h24.columns:
            continue
        grp = h24.groupby(f, dropna=False)["win_3r"].agg(["mean", "count"]).reset_index()
        grp = grp[pd.to_numeric(grp["count"], errors="coerce") >= 20]
        if grp.empty:
            continue
        grp["lift"] = pd.to_numeric(grp["mean"], errors="coerce") - base_rate
        best = grp.sort_values("lift", ascending=False).iloc[0]
        rows.append(
            {
                "feature": f,
                "kind": "categorical_lift",
                "score": float(best["lift"]),
                "support": int(best["count"]),
                "best_bucket": best[f],
                "bucket_rate": float(best["mean"]),
            }
        )
        rules.append(f"IF {f} == {best[f]} THEN win_3r_rate ~ {float(best['mean']):.4f} (lift {float(best['lift']):+.4f}, n={int(best['count'])})")

    out = pd.DataFrame(rows).sort_values("score", ascending=False, kind="mergesort") if rows else pd.DataFrame()
    return out, rules


def run_conditional_research(
    config_path: str,
    truth_mode_raw: str,
    exact_dir: Path,
    output_dir: Path,
    min_n: int = 20,
    min_n_robust: int = 30,
    export_html: bool = True,
    include_feature_importance: bool = False,
) -> ResearchArtifacts:
    cfg = load_pine16_exact_config(config_path)
    truth_mode = normalize_truth_mode(truth_mode_raw)
    trades, signals, truth_label, parity_metrics, exact_available = load_truth_datasets(cfg, truth_mode, exact_dir)

    if signals.empty and not trades.empty:
        signals = trades[["symbol", "timeframe", "bar_time_utc", "entry_time_utc", "entry_price"]].copy()

    _ = classify_signals(signals, truth_label=truth_label, config_pack=cfg.metadata.config_pack)
    tr_cls = classify_trades(trades, truth_label=truth_label, config_pack=cfg.metadata.config_pack)

    feature_events = build_feature_event_frame(cfg) if truth_label != TruthLabel.EXACT_PINE_EXPORTED else pd.DataFrame()
    master = build_research_master(tr_cls, cfg=cfg, truth_label=truth_label, feature_events=feature_events)
    master = add_research_buckets(master, cfg=cfg)
    long_df = build_long_horizon_frame(master)

    by_session = aggregate_by_scopes(long_df, group_cols=[])
    overall = by_session[by_session["session_scope"] == "all_sessions"].copy() if not by_session.empty else pd.DataFrame()
    by_symbol = aggregate_metrics(long_df, group_cols=["symbol", "horizon_h"]) if not long_df.empty else pd.DataFrame()
    by_year = aggregate_metrics(long_df, group_cols=["year", "horizon_h"]) if not long_df.empty else pd.DataFrame()
    by_symbol_year = aggregate_metrics(long_df, group_cols=["symbol", "year", "horizon_h"]) if not long_df.empty else pd.DataFrame()
    by_symbol_session = aggregate_by_scopes(long_df, group_cols=["symbol"])
    by_year_session = aggregate_by_scopes(long_df, group_cols=["year"])
    by_symbol_year_session = aggregate_by_scopes(long_df, group_cols=["symbol", "year"])
    h24 = long_df[long_df["horizon_h"] == 24].copy() if not long_df.empty else pd.DataFrame()

    feature_tables: List[pd.DataFrame] = []
    for bucket in ["osc_bucket", "pivot_gap_bucket", "vol_entry_bucket", "vol_pivot_bucket", "atr_ratio_bucket", "loc_bucket", "rsi_bucket", "trade_quality_bucket"]:
        if h24.empty or bucket not in h24.columns:
            continue
        t = aggregate_metrics(h24, group_cols=[bucket])
        if t.empty:
            continue
        t["bucket_name"] = bucket
        t = t.rename(columns={bucket: "bucket_value"})
        feature_tables.append(t)
    by_feature_bucket = pd.concat(feature_tables, ignore_index=True, sort=False) if feature_tables else pd.DataFrame()

    combos, leaderboard = build_top_combinations(h24, min_n=min_n, min_n_robust=min_n_robust)
    by_symbol_24 = by_symbol[by_symbol["horizon_h"] == 24].copy() if not by_symbol.empty else pd.DataFrame()
    by_symbol_year_24 = by_symbol_year[by_symbol_year["horizon_h"] == 24].copy() if not by_symbol_year.empty else pd.DataFrame()
    by_symbol_session_24 = by_symbol_session[by_symbol_session["horizon_h"] == 24].copy() if not by_symbol_session.empty else pd.DataFrame()
    by_symbol_year_session_24 = by_symbol_year_session[by_symbol_year_session["horizon_h"] == 24].copy() if not by_symbol_year_session.empty else pd.DataFrame()

    keep_watch_cut = build_keep_watch_cut(
        by_symbol_24=by_symbol_24,
        by_symbol_year_24=by_symbol_year_24,
        by_symbol_session_24=by_symbol_session_24,
        by_symbol_year_session_24=by_symbol_year_session_24,
        min_n=min_n,
        min_n_robust=min_n_robust,
    )

    robustness = _robustness_from_year_table(by_symbol_year_24, key_cols=["symbol"]) if not by_symbol_year_24.empty else pd.DataFrame()
    if not combos.empty:
        combo_key_cols = ["symbol", "session_scope_bucket", "bosUsed", "volRatioEntryGateOn", "osc_bucket", "pivot_gap_bucket", "atr_ratio_bucket", "divType", "daily_regime_state"]
        combo_year = aggregate_metrics(h24, group_cols=combo_key_cols + ["year"])
        rb_combo = _robustness_from_year_table(combo_year, key_cols=combo_key_cols)
        if not rb_combo.empty:
            rb_combo["combo_level"] = "conditional_combo"
            if robustness.empty:
                robustness = rb_combo
            else:
                robustness = pd.concat([robustness.assign(combo_level="symbol"), rb_combo], ignore_index=True, sort=False)
    elif not robustness.empty:
        robustness["combo_level"] = "symbol"

    deployment = build_deployment_candidates(h24, min_n=min_n, min_n_robust=min_n_robust)

    feature_importance = pd.DataFrame()
    rules: List[str] = []
    if include_feature_importance:
        feature_importance, rules = compute_feature_importance(h24)

    for df in [master, overall, by_symbol, by_session, by_year, by_symbol_year, by_symbol_session, by_year_session, by_feature_bucket, leaderboard, keep_watch_cut, combos, robustness, deployment]:
        if not df.empty:
            df["truth_label"] = truth_label.value
            df["config_pack"] = cfg.metadata.config_pack

    exact_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    research_master_path = exact_dir / "research_master.parquet"
    master.to_parquet(research_master_path, index=False)

    overall_path = output_dir / "pine16_research_overall.csv"
    by_symbol_path = output_dir / "pine16_research_by_symbol.csv"
    by_session_path = output_dir / "pine16_research_by_session.csv"
    by_year_path = output_dir / "pine16_research_by_year.csv"
    by_symbol_year_path = output_dir / "pine16_research_by_symbol_year.csv"
    by_symbol_session_path = output_dir / "pine16_research_by_symbol_session.csv"
    by_year_session_path = output_dir / "pine16_research_by_year_session.csv"
    by_feature_bucket_path = output_dir / "pine16_research_by_feature_bucket.csv"
    leaderboard_path = output_dir / "pine16_research_leaderboard.csv"
    keep_watch_cut_path = output_dir / "pine16_research_keep_watch_cut.csv"
    combos_path = output_dir / "pine16_research_top_combinations.csv"
    robustness_path = output_dir / "pine16_research_robustness.csv"
    dep_csv_path = output_dir / "pine16_deployment_candidates.csv"
    dep_md_path = output_dir / "pine16_deployment_candidates.md"

    overall.to_csv(overall_path, index=False)
    by_symbol.to_csv(by_symbol_path, index=False)
    by_session.to_csv(by_session_path, index=False)
    by_year.to_csv(by_year_path, index=False)
    by_symbol_year.to_csv(by_symbol_year_path, index=False)
    by_symbol_session.to_csv(by_symbol_session_path, index=False)
    by_year_session.to_csv(by_year_session_path, index=False)
    by_feature_bucket.to_csv(by_feature_bucket_path, index=False)
    leaderboard.to_csv(leaderboard_path, index=False)
    keep_watch_cut.to_csv(keep_watch_cut_path, index=False)
    combos.to_csv(combos_path, index=False)
    robustness.to_csv(robustness_path, index=False)
    deployment.to_csv(dep_csv_path, index=False)

    h24_overall = overall[overall["horizon_h"] == 24].iloc[0] if not overall.empty and (overall["horizon_h"] == 24).any() else None
    final_directional = str(h24_overall["directional_edge_verdict"]) if h24_overall is not None else "No directional edge"
    final_tradable = str(h24_overall["tradable_edge_verdict"]) if h24_overall is not None else "Not monetizable at 1:3"

    strongest_symbol = by_symbol_24.sort_values("expectancy_1to3_r", ascending=False).head(1) if not by_symbol_24.empty else pd.DataFrame()
    strongest_session = by_session[by_session["horizon_h"] == 24].sort_values("expectancy_1to3_r", ascending=False).head(1) if not by_session.empty else pd.DataFrame()
    strongest_symbol_session = by_symbol_session_24.sort_values("expectancy_1to3_r", ascending=False).head(1) if not by_symbol_session_24.empty else pd.DataFrame()
    strongest_year = by_year[by_year["horizon_h"] == 24].sort_values("expectancy_1to3_r", ascending=False).head(1) if not by_year.empty else pd.DataFrame()
    strongest_combo = combos.head(1) if not combos.empty else pd.DataFrame()
    cut_entities = keep_watch_cut[keep_watch_cut["action"] == "CUT"] if not keep_watch_cut.empty else pd.DataFrame()
    full_basket_row = deployment[deployment["candidate"] == "full_basket_all_sessions"] if not deployment.empty else pd.DataFrame()
    reduced_best = deployment[deployment["candidate"] != "full_basket_all_sessions"].sort_values("expectancy_1to3_r_24h", ascending=False).head(1) if not deployment.empty else pd.DataFrame()

    lines: List[str] = []
    lines.append("# Pine16 Conditional Deep Research Report")
    lines.append("")
    lines.append("## 1. Executive verdict")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(f"- directional_edge_verdict_24h: **{final_directional}**")
    lines.append(f"- tradable_edge_verdict_24h: **{final_tradable}**")
    if h24_overall is not None:
        lines.append(f"- pct_plus1r_before_minus1r_24h: **{float(h24_overall['pct_reach_plus1r_before_minus1r']):.4f}**")
        lines.append(f"- pct_plus2r_before_minus1r_24h: **{float(h24_overall['pct_reach_plus2r_before_minus1r']):.4f}**")
        lines.append(f"- pct_plus3r_before_minus1r_24h: **{float(h24_overall['pct_reach_plus3r_before_minus1r']):.4f}**")
        lines.append(f"- expectancy_1to3_r_24h: **{float(h24_overall['expectancy_1to3_r']):.4f}**")
    lines.append("")
    lines.append("## 2. Truth source used")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(f"- requested_truth_mode: `{truth_mode.value}`")
    lines.append(f"- exact_export_available: `{bool(exact_available)}`")
    if parity_metrics is not None:
        lines.append(f"- parity_metrics: `{json.dumps(parity_metrics, sort_keys=True)}`")
    lines.append("")
    lines.append("## 3. What was analyzed")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(f"- config_pack: `{cfg.metadata.config_pack}`")
    lines.append(f"- symbols: `{','.join(cfg.symbols)}`")
    lines.append(f"- years: `{','.join(str(y) for y in cfg.data.years)}`")
    lines.append(f"- timeframe: `{cfg.timeframe}`")
    lines.append(f"- rows_research_master: `{len(master)}`")
    lines.append("")
    lines.append("## 4. Overall edge summary")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(overall, ["horizon_h", "n_trades", "pct_reach_plus1r_before_minus1r", "pct_reach_plus2r_before_minus1r", "pct_reach_plus3r_before_minus1r", "expectancy_1to3_r", "directional_edge_verdict", "tradable_edge_verdict"]))
    lines.append("")
    lines.append("## 5. Symbol breakdown")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(by_symbol[by_symbol["horizon_h"] == 24] if not by_symbol.empty else by_symbol, ["symbol", "n_trades", "pct_reach_plus1r_before_minus1r", "pct_reach_plus2r_before_minus1r", "pct_reach_plus3r_before_minus1r", "expectancy_1to3_r", "directional_edge_verdict", "tradable_edge_verdict"]))
    lines.append("")
    lines.append("## 6. Session breakdown")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(by_session[by_session["horizon_h"] == 24] if not by_session.empty else by_session, ["session_scope", "n_trades", "pct_reach_plus1r_before_minus1r", "pct_reach_plus2r_before_minus1r", "pct_reach_plus3r_before_minus1r", "expectancy_1to3_r", "directional_edge_verdict", "tradable_edge_verdict"]))
    lines.append("")
    lines.append("## 7. Year breakdown")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(by_year[by_year["horizon_h"] == 24] if not by_year.empty else by_year, ["year", "n_trades", "pct_reach_plus1r_before_minus1r", "pct_reach_plus2r_before_minus1r", "pct_reach_plus3r_before_minus1r", "expectancy_1to3_r", "directional_edge_verdict", "tradable_edge_verdict"]))
    lines.append("")
    lines.append("## 8. Symbol x session")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(by_symbol_session_24, ["symbol", "session_scope", "n_trades", "pct_reach_plus1r_before_minus1r", "pct_reach_plus2r_before_minus1r", "pct_reach_plus3r_before_minus1r", "expectancy_1to3_r", "directional_edge_verdict", "tradable_edge_verdict"]))
    lines.append("")
    lines.append("## 9. Symbol x year")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(by_symbol_year_24, ["symbol", "year", "n_trades", "pct_reach_plus1r_before_minus1r", "pct_reach_plus2r_before_minus1r", "pct_reach_plus3r_before_minus1r", "expectancy_1to3_r"]))
    lines.append("")
    lines.append("## 10. Year x session")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(by_year_session[by_year_session["horizon_h"] == 24] if not by_year_session.empty else by_year_session, ["year", "session_scope", "n_trades", "pct_reach_plus1r_before_minus1r", "pct_reach_plus2r_before_minus1r", "pct_reach_plus3r_before_minus1r", "expectancy_1to3_r"]))
    lines.append("")
    lines.append("## 11. Barrier outcome study (+1R,+2R,+3R)")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(overall, ["horizon_h", "n_trades", "pct_reach_plus1r_before_minus1r", "pct_reach_plus2r_before_minus1r", "pct_reach_plus3r_before_minus1r", "open_rate_1r", "open_rate_2r", "open_rate_3r"]))
    lines.append("")
    lines.append("## 12. Path-quality study")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(overall, ["horizon_h", "pct_reaches_plus1r_without_minus05r", "pct_reaches_plus3r_without_minus1r", "pct_immediate_drawdown_gt_05r", "pct_immediate_drawdown_gt_1r", "median_time_to_plus1r_bars", "median_time_to_plus2r_bars", "median_time_to_plus3r_bars", "median_time_to_minus1r_bars"]))
    lines.append("")
    lines.append("## 13. Feature bucket study")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(by_feature_bucket.sort_values(["bucket_name", "expectancy_1to3_r"], ascending=[True, False]).head(60) if not by_feature_bucket.empty else by_feature_bucket, ["bucket_name", "bucket_value", "n_trades", "pct_reach_plus1r_before_minus1r", "pct_reach_plus2r_before_minus1r", "pct_reach_plus3r_before_minus1r", "expectancy_1to3_r"]))
    lines.append("")
    lines.append("## 14. Top conditional combinations")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(combos.head(30), ["combo_id", "symbol", "session_scope_bucket", "year_scope", "n_trades", "pct_reach_plus1r_before_minus1r", "pct_reach_plus2r_before_minus1r", "pct_reach_plus3r_before_minus1r", "expectancy_1to3_r", "robustness_score", "action"]))
    lines.append("")
    lines.append("## 15. Robustness study")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(robustness.head(40), ["combo_level", "symbol", "n", "years_present", "positive_years", "negative_years", "std_expectancy_by_year", "concentration_score", "robustness_score", "fragility_flag"]))
    lines.append("")
    lines.append("## 16. Keep / watch / cut table")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(keep_watch_cut, ["entity_type", "symbol", "session_scope", "n_trades", "pct_reach_plus1r_before_minus1r", "pct_reach_plus2r_before_minus1r", "pct_reach_plus3r_before_minus1r", "expectancy_1to3_r", "robustness_score", "action"]))
    lines.append("")
    lines.append("## 17. Deployment candidates")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append(_md_table(deployment, ["candidate", "symbols", "session_scope", "n_trades", "pct_plus1r_before_minus1r_24h", "pct_plus2r_before_minus1r_24h", "pct_plus3r_before_minus1r_24h", "expectancy_1to3_r_24h", "robustness_score", "deployment_action", "why"]))
    lines.append("")
    lines.append("## 18. Caveats / blockers")
    lines.append(f"- truth_label: `{truth_label.value}`")
    lines.append("- Same-bar TP/SL conflicts use conservative adverse-first rule.")
    lines.append("- +2R and path-quality metrics are computed from local OHLC path, not tick-level reconstruction.")
    if truth_label != TruthLabel.EXACT_PINE_EXPORTED:
        lines.append("- Exact Pine exports are missing or parity is not verified for this config; this is not exact Pine truth.")
    if truth_label == TruthLabel.UNVERIFIED_PYTHON_APPROXIMATION:
        lines.append("- All findings are unverified approximation and must not be presented as exact Pine or verified parity.")
    lines.append("")
    lines.append("## Research insights")
    lines.append(f"- truth_label: `{truth_label.value}`")
    if not strongest_symbol.empty:
        r = strongest_symbol.iloc[0]
        lines.append(f"- Strongest symbol (24h expectancy): `{r['symbol']}` | +1R={float(r['pct_reach_plus1r_before_minus1r']):.4f}, +3R={float(r['pct_reach_plus3r_before_minus1r']):.4f}, E[1:3]={float(r['expectancy_1to3_r']):.4f}")
    if not strongest_session.empty:
        r = strongest_session.iloc[0]
        lines.append(f"- Strongest session scope (24h expectancy): `{r['session_scope']}` | +1R={float(r['pct_reach_plus1r_before_minus1r']):.4f}, +3R={float(r['pct_reach_plus3r_before_minus1r']):.4f}, E[1:3]={float(r['expectancy_1to3_r']):.4f}")
    if not strongest_symbol_session.empty:
        r = strongest_symbol_session.iloc[0]
        lines.append(f"- Strongest symbol-session pocket: `{r['symbol']} @ {r['session_scope']}` | +3R={float(r['pct_reach_plus3r_before_minus1r']):.4f}, E[1:3]={float(r['expectancy_1to3_r']):.4f}")
    if not strongest_year.empty:
        r = strongest_year.iloc[0]
        lines.append(f"- Strongest year: `{int(r['year'])}` | +3R={float(r['pct_reach_plus3r_before_minus1r']):.4f}, E[1:3]={float(r['expectancy_1to3_r']):.4f}")
    if not strongest_combo.empty:
        r = strongest_combo.iloc[0]
        lines.append(f"- Strongest conditional combo: `{r['combo_id']}` {r['symbol']} / {r['session_scope_bucket']} / {r['feature_conditions']} | n={int(r['n_trades'])}, +3R={float(r['pct_reach_plus3r_before_minus1r']):.4f}, E[1:3]={float(r['expectancy_1to3_r']):.4f}, robustness={float(pd.to_numeric(r['robustness_score'], errors='coerce')):.4f}")
    if not cut_entities.empty:
        cut_txt = ", ".join(cut_entities[cut_entities["entity_type"] == "symbol"]["symbol"].astype(str).unique().tolist()) or "none"
        lines.append(f"- Weakest symbols to cut: `{cut_txt}`")
    if not full_basket_row.empty:
        r = full_basket_row.iloc[0]
        lines.append(f"- Full basket (all sessions): action={r['deployment_action']} | +3R={float(r['pct_plus3r_before_minus1r_24h']):.4f}, E[1:3]={float(r['expectancy_1to3_r_24h']):.4f}")
    if not reduced_best.empty:
        r = reduced_best.iloc[0]
        lines.append(f"- Best reduced candidate: `{r['candidate']}` | action={r['deployment_action']} | +3R={float(r['pct_plus3r_before_minus1r_24h']):.4f}, E[1:3]={float(r['expectancy_1to3_r_24h']):.4f}, robustness={float(r['robustness_score']):.4f}")

    report_md_path = output_dir / "pine16_research_master_report.md"
    report_html_path = output_dir / "pine16_research_master_report.html"
    report_md = "\n".join(lines).rstrip() + "\n"
    report_md_path.write_text(report_md, encoding="utf-8")
    if export_html:
        _write_html_from_markdown(report_md, report_html_path)
    else:
        report_html_path.write_text("", encoding="utf-8")

    dep_lines = [
        "# Pine16 Deployment Candidates",
        "",
        f"- truth_label: `{truth_label.value}`",
        _md_table(deployment, ["candidate", "symbols", "session_scope", "n_trades", "pct_plus1r_before_minus1r_24h", "pct_plus2r_before_minus1r_24h", "pct_plus3r_before_minus1r_24h", "expectancy_1to3_r_24h", "robustness_score", "deployment_action", "why"]),
    ]
    dep_md_path.write_text("\n".join(dep_lines).rstrip() + "\n", encoding="utf-8")

    fi_csv_path: Path | None = None
    rules_md_path: Path | None = None
    if include_feature_importance:
        fi_csv_path = output_dir / "pine16_feature_importance.csv"
        feature_importance.to_csv(fi_csv_path, index=False)
        rules_md_path = output_dir / "pine16_simple_rules.md"
        text = "# Pine16 Simple Rules\n\n"
        text += f"- truth_label: `{truth_label.value}`\n"
        if rules:
            text += "\n".join(f"- {r}" for r in rules) + "\n"
        else:
            text += "- No stable simple rules met support thresholds.\n"
        rules_md_path.write_text(text, encoding="utf-8")

    return ResearchArtifacts(
        report_md=report_md_path,
        report_html=report_html_path,
        research_master=research_master_path,
        research_overall=overall_path,
        research_by_symbol=by_symbol_path,
        research_by_session=by_session_path,
        research_by_year=by_year_path,
        research_by_symbol_year=by_symbol_year_path,
        research_by_symbol_session=by_symbol_session_path,
        research_by_year_session=by_year_session_path,
        research_by_feature_bucket=by_feature_bucket_path,
        research_leaderboard=leaderboard_path,
        research_keep_watch_cut=keep_watch_cut_path,
        research_top_combinations=combos_path,
        research_robustness=robustness_path,
        deployment_candidates_md=dep_md_path,
        deployment_candidates_csv=dep_csv_path,
        feature_importance_csv=fi_csv_path,
        simple_rules_md=rules_md_path,
    )


def write_deep_research_audit(audit_path: Path, exact_dir: Path) -> Path:
    trades = pd.read_parquet(exact_dir / "trades_exact_pine.parquet") if (exact_dir / "trades_exact_pine.parquet").exists() else pd.DataFrame()
    signals = pd.read_parquet(exact_dir / "signals_exact_pine.parquet") if (exact_dir / "signals_exact_pine.parquet").exists() else pd.DataFrame()
    parity_v = _load_parity_verification(exact_dir)
    lines = [
        "# Pine16 Deep Research Audit",
        "",
        f"- exact_trades_rows: `{len(trades)}`",
        f"- exact_signals_rows: `{len(signals)}`",
        f"- parity_verification: `{json.dumps(parity_v, sort_keys=True) if parity_v else 'missing'}`",
        "- Available truth modes: EXACT_PINE_EXPORTED, VERIFIED_PYTHON_PARITY, UNVERIFIED_PYTHON_APPROXIMATION",
        "",
        "## Mapping",
        "| concept | source field / behavior | current repo implementation | truth sensitivity | action required |",
        "| --- | --- | --- | --- | --- |",
        "| session classification | Pine UTC+3 named sessions | `dfd05/pine16_session.py` + session-classified parquet outputs | high | keep as canonical |",
        "| +1R/+3R barrier outcomes | first-touch with adverse-first ties | `scripts/pine16_plus1r_plus3r_report.py` | high | reuse in deep research |",
        "| +2R outcomes | not in existing outputs | missing | medium | add in research mart |",
        "| feature-state mining | strategy event columns (`extract_dfd05_events`) | available but not fully surfaced in reports | medium | merge into research_master |",
        "| truth stamping | `truth_label` fields | present in recent reports | high | stamp every section/table |",
    ]
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return audit_path
