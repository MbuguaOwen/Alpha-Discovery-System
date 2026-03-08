from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .data import load_bars_for_symbol, timeframe_to_minutes
from .indicators import atr, rolling_sum_via_sma
from .pine16_config import Pine16ExactConfig, to_legacy_run_config
from .pine16_truth import TruthLabel
from .strategy import extract_dfd05_events


PARITY_TRADE_COLUMNS = [
    "symbol",
    "timeframe",
    "tv_strategy_name",
    "config_pack",
    "bar_time_utc",
    "signal_time_utc",
    "setup_time_utc",
    "entry_time_utc",
    "exit_time_utc",
    "entry_price",
    "exit_price",
    "side",
    "atr_entry",
    "sl_price",
    "tp_price",
    "rr_multiple",
    "bars_held",
    "result_r",
    "result_pct",
    "year",
    "month",
    "source_file",
    "truth_label",
    "trade_status",
    "session_ok_pivot",
    "session_ok_entry",
]


@dataclass
class ParityComparison:
    signal_count_mismatch_pct: float
    entry_timestamp_max_bar_diff: float
    exit_timestamp_max_bar_diff: float
    aggregate_net_r_mismatch_pct: float
    pass_thresholds: bool


@dataclass
class ParityRunResult:
    signals: pd.DataFrame
    trades: pd.DataFrame


def _empty_signals() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "symbol",
            "timeframe",
            "tv_strategy_name",
            "config_pack",
            "bar_time_utc",
            "signal_time_utc",
            "setup_time_utc",
            "entry_time_utc",
            "side",
            "entry_price",
            "source_file",
            "truth_label",
            "session_ok_pivot",
            "session_ok_entry",
        ]
    )


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=PARITY_TRADE_COLUMNS)


def _build_cvd_proxy(cfg_legacy, symbol: str, timeframe: str, bars: pd.DataFrame) -> np.ndarray:
    toggles = cfg_legacy.strategy.toggles
    src = bars
    if timeframe.lower() != "m1":
        try:
            src = load_bars_for_symbol(cfg_legacy, symbol=symbol, timeframe="m1")
            src = src[(src["time"] >= bars["time"].iloc[0]) & (src["time"] <= bars["time"].iloc[-1])]
            if src.empty:
                src = bars
        except FileNotFoundError:
            src = bars

    signed = np.sign(src["close"].to_numpy(dtype=float) - src["open"].to_numpy(dtype=float)) * src[
        "volume"
    ].to_numpy(dtype=float)
    cvd = rolling_sum_via_sma(signed, max(1, int(toggles.cvd_len)))
    c = pd.Series(cvd, index=src["time"]).sort_index()
    c = c[~c.index.duplicated(keep="last")]
    return c.reindex(bars["time"], method="ffill").to_numpy(dtype=float)


def _simulate_symbol_trades(
    *,
    bars: pd.DataFrame,
    events: pd.DataFrame,
    cfg: Pine16ExactConfig,
    symbol: str,
) -> pd.DataFrame:
    if events.empty:
        return _empty_trades()

    mode_trade = str(cfg.trading.mode).strip().upper() == "TRADE"
    if not mode_trade:
        return _empty_trades()

    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    t = pd.to_datetime(bars["time"], utc=True)
    atr_arr = atr(high, low, close, int(cfg.risk.atrLen))

    ev = events.copy().sort_values(["entry_index", "event_time_ms", "signal_id"], kind="mergesort").reset_index(drop=True)
    ev["entry_index"] = pd.to_numeric(ev["entry_index"], errors="coerce").astype("Int64")

    rows: List[Dict[str, object]] = []
    i = 0
    n = len(bars)
    ev_ptr = 0

    in_pos = False
    entry_i = -1
    entry_px = np.nan
    atr_entry = np.nan
    sl_px = np.nan
    tp_px = np.nan
    last_entry_bar: int | None = None
    signal_t_utc = pd.NaT
    setup_t_utc = pd.NaT
    setup_session_ok = np.nan
    entry_session_ok = np.nan

    cooldown = max(0, int(cfg.safety.cooldownBars))

    while i < n:
        if in_pos and i > entry_i:
            hit_tp = bool(high[i] >= tp_px)
            hit_sl = bool(low[i] <= sl_px)
            exit_now = False
            if hit_tp and hit_sl:
                # Conservative tie-break. Reconciliation gate decides verified/unverified label.
                exit_px = float(sl_px)
                result_r = -1.0
                exit_now = True
            elif hit_tp:
                exit_px = float(tp_px)
                result_r = float(cfg.risk.rrMult)
                exit_now = True
            elif hit_sl:
                exit_px = float(sl_px)
                result_r = -1.0
                exit_now = True
            if exit_now:
                ep = float(entry_px)
                result_pct = ((float(exit_px) - ep) / ep) * 100.0 if np.isfinite(ep) and ep != 0 else np.nan
                ts_entry = pd.Timestamp(t.iloc[entry_i])
                ts_exit = pd.Timestamp(t.iloc[i])
                rows.append(
                    {
                        "symbol": symbol,
                        "timeframe": str(cfg.timeframe),
                        "tv_strategy_name": str(cfg.metadata.tv_strategy_name),
                        "config_pack": str(cfg.metadata.config_pack),
                        "bar_time_utc": ts_entry,
                        "signal_time_utc": signal_t_utc,
                        "setup_time_utc": setup_t_utc,
                        "entry_time_utc": ts_entry,
                        "exit_time_utc": ts_exit,
                        "entry_price": float(entry_px),
                        "exit_price": float(exit_px),
                        "side": "long",
                        "atr_entry": float(atr_entry),
                        "sl_price": float(sl_px),
                        "tp_price": float(tp_px),
                        "rr_multiple": float(cfg.risk.rrMult),
                        "bars_held": int(i - entry_i),
                        "result_r": float(result_r),
                        "result_pct": float(result_pct),
                        "year": int(ts_entry.year),
                        "month": int(ts_entry.month),
                        "source_file": "python_parity_engine",
                        "truth_label": TruthLabel.VERIFIED_PYTHON_PARITY.value,
                        "trade_status": "CLOSED",
                        "session_ok_pivot": setup_session_ok,
                        "session_ok_entry": entry_session_ok,
                    }
                )
                in_pos = False

        while ev_ptr < len(ev):
            nxt = ev.iloc[ev_ptr]
            ei = int(nxt["entry_index"]) if pd.notna(nxt["entry_index"]) else -1
            if ei < i:
                ev_ptr += 1
                continue
            break

        if not in_pos:
            same_bar: List[pd.Series] = []
            k = ev_ptr
            while k < len(ev):
                nxt = ev.iloc[k]
                ei = int(nxt["entry_index"]) if pd.notna(nxt["entry_index"]) else -1
                if ei != i:
                    break
                same_bar.append(nxt)
                k += 1

            if same_bar:
                cooldown_ok = (
                    cooldown <= 0
                    or last_entry_bar is None
                    or (i - int(last_entry_bar)) >= cooldown
                )
                if cooldown_ok:
                    chosen = same_bar[0]
                    entry_i = i
                    entry_px = float(close[i])
                    ev_signal_ms = pd.to_numeric(chosen.get("event_time_ms"), errors="coerce")
                    ev_setup_ms = pd.to_numeric(chosen.get("pivot_time_ms"), errors="coerce")
                    signal_t_utc = pd.to_datetime(ev_signal_ms, unit="ms", utc=True, errors="coerce")
                    setup_t_utc = pd.to_datetime(ev_setup_ms, unit="ms", utc=True, errors="coerce")
                    setup_session_ok = pd.to_numeric(chosen.get("session_ok_pivot"), errors="coerce")
                    entry_session_ok = pd.to_numeric(chosen.get("session_ok_entry"), errors="coerce")
                    atr_from_event = pd.to_numeric(chosen.get("atr_entry"), errors="coerce")
                    atr_entry = float(atr_from_event) if pd.notna(atr_from_event) else float(atr_arr[i])
                    if not np.isfinite(atr_entry) or atr_entry <= 0.0:
                        atr_entry = float(atr_arr[i])
                    sl_px = float(entry_px - float(cfg.risk.slAtrMult) * atr_entry)
                    tp_px = float(entry_px + float(cfg.risk.slAtrMult) * float(cfg.risk.rrMult) * atr_entry)
                    in_pos = True
                    last_entry_bar = i
                ev_ptr = k

        i += 1

    if in_pos:
        ts_entry = pd.Timestamp(t.iloc[entry_i])
        rows.append(
            {
                "symbol": symbol,
                "timeframe": str(cfg.timeframe),
                "tv_strategy_name": str(cfg.metadata.tv_strategy_name),
                "config_pack": str(cfg.metadata.config_pack),
                "bar_time_utc": ts_entry,
                "signal_time_utc": signal_t_utc,
                "setup_time_utc": setup_t_utc,
                "entry_time_utc": ts_entry,
                "exit_time_utc": pd.NaT,
                "entry_price": float(entry_px),
                "exit_price": np.nan,
                "side": "long",
                "atr_entry": float(atr_entry),
                "sl_price": float(sl_px),
                "tp_price": float(tp_px),
                "rr_multiple": float(cfg.risk.rrMult),
                "bars_held": np.nan,
                "result_r": np.nan,
                "result_pct": np.nan,
                "year": int(ts_entry.year),
                "month": int(ts_entry.month),
                "source_file": "python_parity_engine",
                "truth_label": TruthLabel.VERIFIED_PYTHON_PARITY.value,
                "trade_status": "OPEN",
                "session_ok_pivot": setup_session_ok,
                "session_ok_entry": entry_session_ok,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return _empty_trades()
    for c in PARITY_TRADE_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    return out[PARITY_TRADE_COLUMNS]


def run_python_parity(cfg: Pine16ExactConfig) -> ParityRunResult:
    legacy_cfg = to_legacy_run_config(cfg)

    signal_rows: List[pd.DataFrame] = []
    trade_rows: List[pd.DataFrame] = []

    for symbol in cfg.symbols:
        bars = load_bars_for_symbol(legacy_cfg, symbol=str(symbol), timeframe=str(cfg.timeframe))
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

        sig = pd.DataFrame(
            {
                "symbol": events["symbol"].astype(str),
                "timeframe": events["timeframe"].astype(str),
                "tv_strategy_name": str(cfg.metadata.tv_strategy_name),
                "config_pack": str(cfg.metadata.config_pack),
                "bar_time_utc": pd.to_datetime(events["event_time_ms"], unit="ms", utc=True),
                "signal_time_utc": pd.to_datetime(events["event_time_ms"], unit="ms", utc=True),
                "setup_time_utc": pd.to_datetime(events["pivot_time_ms"], unit="ms", utc=True),
                "entry_time_utc": pd.to_datetime(events["entry_time_ms"], unit="ms", utc=True),
                "side": "long",
                "entry_price": pd.to_numeric(events["entry_price"], errors="coerce"),
                "source_file": "python_parity_engine",
                "truth_label": TruthLabel.VERIFIED_PYTHON_PARITY.value,
                "session_ok_pivot": pd.to_numeric(events.get("session_ok_pivot"), errors="coerce"),
                "session_ok_entry": pd.to_numeric(events.get("session_ok_entry"), errors="coerce"),
            }
        )
        signal_rows.append(sig)

        trades = _simulate_symbol_trades(bars=bars, events=events, cfg=cfg, symbol=str(symbol))
        if not trades.empty:
            trade_rows.append(trades)

    signals = pd.concat(signal_rows, ignore_index=True, sort=False) if signal_rows else _empty_signals()
    trades = pd.concat(trade_rows, ignore_index=True, sort=False) if trade_rows else _empty_trades()
    return ParityRunResult(signals=signals, trades=trades)


def _to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def compare_parity_to_exact(
    *,
    exact_trades: pd.DataFrame,
    parity_trades: pd.DataFrame,
    timeframe: str,
) -> ParityComparison:
    tf_minutes = timeframe_to_minutes(str(timeframe).lower())
    bar_seconds = float(tf_minutes * 60)

    ex = exact_trades.copy()
    py = parity_trades.copy()

    ex = ex[ex["trade_status"].fillna("CLOSED").astype(str).str.upper() != "OPEN"] if "trade_status" in ex.columns else ex
    py = py[py["trade_status"].fillna("CLOSED").astype(str).str.upper() != "OPEN"] if "trade_status" in py.columns else py

    ex["entry_time_utc"] = _to_utc(ex["entry_time_utc"])
    ex["exit_time_utc"] = _to_utc(ex["exit_time_utc"])
    py["entry_time_utc"] = _to_utc(py["entry_time_utc"])
    py["exit_time_utc"] = _to_utc(py["exit_time_utc"])

    ex = ex.sort_values(["symbol", "entry_time_utc"], kind="mergesort")
    py = py.sort_values(["symbol", "entry_time_utc"], kind="mergesort")

    merged_parts: List[pd.DataFrame] = []
    for sym in sorted(set(ex["symbol"].astype(str)).union(set(py["symbol"].astype(str)))):
        a = ex[ex["symbol"].astype(str) == sym].copy()
        b = py[py["symbol"].astype(str) == sym].copy()
        if a.empty or b.empty:
            continue
        a = a.rename(columns={"entry_time_utc": "entry_time_utc_exact"})
        b = b.rename(columns={"entry_time_utc": "entry_time_utc_parity"})
        m = pd.merge_asof(
            a.sort_values("entry_time_utc_exact"),
            b.sort_values("entry_time_utc_parity"),
            left_on="entry_time_utc_exact",
            right_on="entry_time_utc_parity",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=int(bar_seconds)),
            suffixes=("_exact", "_parity"),
        )
        merged_parts.append(m)

    merged = pd.concat(merged_parts, ignore_index=True, sort=False) if merged_parts else pd.DataFrame()

    cnt_mismatch = (abs(len(py) - len(ex)) / max(len(ex), 1)) * 100.0

    if merged.empty:
        entry_diff = np.inf
        exit_diff = np.inf
    else:
        entry_delta = (
            pd.to_datetime(merged["entry_time_utc_exact"], utc=True, errors="coerce")
            - pd.to_datetime(merged["entry_time_utc_parity"], utc=True, errors="coerce")
        ).dt.total_seconds()
        entry_diff = float(np.nanmax(np.abs(entry_delta) / bar_seconds)) if len(entry_delta) > 0 else np.inf
        if "exit_time_utc_exact" in merged.columns and "exit_time_utc_parity" in merged.columns:
            exit_delta = (
                pd.to_datetime(merged["exit_time_utc_exact"], utc=True, errors="coerce")
                - pd.to_datetime(merged["exit_time_utc_parity"], utc=True, errors="coerce")
            ).dt.total_seconds()
            exit_diff = float(np.nanmax(np.abs(exit_delta) / bar_seconds)) if exit_delta.notna().any() else np.inf
        else:
            exit_diff = np.inf

    ex_net_r = float(pd.to_numeric(ex.get("result_r"), errors="coerce").sum())
    py_net_r = float(pd.to_numeric(py.get("result_r"), errors="coerce").sum())
    net_r_mismatch = (abs(py_net_r - ex_net_r) / max(abs(ex_net_r), 1e-9)) * 100.0

    passed = bool(
        cnt_mismatch <= 0.5
        and entry_diff <= 1.0
        and exit_diff <= 1.0
        and net_r_mismatch <= 1.0
    )

    return ParityComparison(
        signal_count_mismatch_pct=float(cnt_mismatch),
        entry_timestamp_max_bar_diff=float(entry_diff),
        exit_timestamp_max_bar_diff=float(exit_diff),
        aggregate_net_r_mismatch_pct=float(net_r_mismatch),
        pass_thresholds=passed,
    )

