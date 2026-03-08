from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .config import RunConfig, effective_session_gate
from .indicators import (
    atr,
    dmi_adx,
    ema,
    macd,
    rolling_max,
    rolling_min,
    rolling_sum_via_sma,
    rsi_wilder,
    sma,
)
from .session_gate import SessionGateSpec, session_ok_at_pivot, session_ok_at_trigger


EVENT_COLUMNS = [
    "symbol",
    "timeframe",
    "signal_id",
    "event_time_ms",
    "entry_time_ms",
    "entry_price",
    "pivot_time_ms",
    "pivot_price",
    "mode",
    "trade_mode",
    "toggles_json",
    "strategy_params_json",
    "event_index",
    "entry_index",
    "setup_age_bars",
    "triggered_ok",
    "atr_entry",
    "osc_change_pct",
    # bars_gap is defined as: signal_index - previous_pivot_index.
    "bars_gap",
    # 1=Classic (pivot_price < prev_pivot_price), 0=Equal.
    "div_type",
    "loc_pivot",
    "vol_ratio_pivot",
    "rsi_pivot",
    "macd_pivot",
    "session_ok_pivot",
    "vol_ratio_entry",
    "atr_ratio_entry",
    "daily_close",
    "daily_ema200",
    "daily_ema_ok",
    "daily_slope_ok",
    "daily_adx",
    "daily_plus_di",
    "daily_minus_di",
    "daily_di_ok",
    "cvd_proxy_entry",
    "cvd_norm_entry",
    "cvd_z_entry",
    "cvd_pct_entry",
    "vol_behavior_ok_entry",
    "vol_spike_ok_entry",
    "session_ok_entry",
    "divergence_strength_ok",
    "min_pivot_gap_ok",
    "daily_ema_gate_ok",
    "daily_adx_gate_ok",
    "daily_di_gate_ok",
    "vol_ratio_pivot_ok",
    "vol_ratio_entry_ok",
    "atr_ratio_ok",
    "volume_behavior_gate_ok",
    "cvd_proxy_gate_ok",
    "cvd_z_gate_ok",
    "cvd_pct_gate_ok",
    "vol_spike_gate_ok",
    "wv70_gate_ok",
    "near_lower_ok",
    "bull_div_ok",
    "signal_allowed_ok",
]


EVENT_INT64_COLUMNS = {
    "event_time_ms",
    "entry_time_ms",
    "pivot_time_ms",
    "event_index",
    "entry_index",
    "setup_age_bars",
    "bars_gap",
}

EVENT_INT8_COLUMNS = {
    "triggered_ok",
    "div_type",
    "session_ok_pivot",
    "daily_ema_ok",
    "daily_slope_ok",
    "daily_di_ok",
    "vol_behavior_ok_entry",
    "vol_spike_ok_entry",
    "session_ok_entry",
    "divergence_strength_ok",
    "min_pivot_gap_ok",
    "daily_ema_gate_ok",
    "daily_adx_gate_ok",
    "daily_di_gate_ok",
    "vol_ratio_pivot_ok",
    "vol_ratio_entry_ok",
    "atr_ratio_ok",
    "volume_behavior_gate_ok",
    "cvd_proxy_gate_ok",
    "cvd_z_gate_ok",
    "cvd_pct_gate_ok",
    "vol_spike_gate_ok",
    "wv70_gate_ok",
    "near_lower_ok",
    "bull_div_ok",
    "signal_allowed_ok",
}

EVENT_FLOAT64_COLUMNS = {
    "entry_price",
    "pivot_price",
    "atr_entry",
    "osc_change_pct",
    "loc_pivot",
    "vol_ratio_pivot",
    "rsi_pivot",
    "macd_pivot",
    "vol_ratio_entry",
    "atr_ratio_entry",
    "daily_close",
    "daily_ema200",
    "daily_adx",
    "daily_plus_di",
    "daily_minus_di",
    "cvd_proxy_entry",
    "cvd_norm_entry",
    "cvd_z_entry",
    "cvd_pct_entry",
}

EVENT_DTYPE_MAP: Dict[str, str] = {}
for _c in EVENT_COLUMNS:
    if _c in EVENT_INT64_COLUMNS:
        EVENT_DTYPE_MAP[_c] = "int64"
    elif _c in EVENT_INT8_COLUMNS:
        EVENT_DTYPE_MAP[_c] = "int8"
    elif _c in EVENT_FLOAT64_COLUMNS:
        EVENT_DTYPE_MAP[_c] = "float64"
    else:
        EVENT_DTYPE_MAP[_c] = "object"


def empty_events_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=EVENT_DTYPE_MAP[c]) for c in EVENT_COLUMNS})


def _coerce_event_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in EVENT_FLOAT64_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.float64)
    for col in EVENT_INT64_COLUMNS:
        if col in out.columns:
            ser = pd.to_numeric(out[col], errors="coerce")
            if ser.isna().any():
                raise ValueError(f"Unexpected NaN in int64 event column: {col}")
            out[col] = ser.astype(np.int64)
    for col in EVENT_INT8_COLUMNS:
        if col in out.columns:
            ser = pd.to_numeric(out[col], errors="coerce")
            if ser.isna().any():
                raise ValueError(f"Unexpected NaN in int8 event column: {col}")
            out[col] = ser.astype(np.int8)
    for col in EVENT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.Series(dtype=EVENT_DTYPE_MAP[col])
    return out[EVENT_COLUMNS]


def compute_pivot_low_confirmations(low: np.ndarray, pivot_len: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(low)
    pivot_index_at_confirm = np.full(n, -1, dtype=int)
    pivot_price_at_confirm = np.full(n, np.nan, dtype=float)
    if n == 0 or pivot_len < 1:
        return pivot_index_at_confirm, pivot_price_at_confirm

    window = 2 * pivot_len + 1
    centered_min = (
        pd.Series(low).rolling(window=window, center=True, min_periods=window).min().to_numpy()
    )
    is_pivot = np.isfinite(centered_min) & np.isclose(low, centered_min, rtol=0.0, atol=1e-12)
    pivot_idx = np.flatnonzero(is_pivot)
    confirm_idx = pivot_idx + pivot_len
    valid = confirm_idx < n
    pivot_index_at_confirm[confirm_idx[valid]] = pivot_idx[valid]
    pivot_price_at_confirm[confirm_idx[valid]] = low[pivot_idx[valid]]
    return pivot_index_at_confirm, pivot_price_at_confirm


def _safe_ratio(numer: float, denom: float) -> float:
    if not np.isfinite(numer) or not np.isfinite(denom) or denom == 0.0:
        return np.nan
    return float(numer / denom)


def _as_i1(flag: bool) -> int:
    return int(bool(flag))


def _rolling_percentile_rank(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    w = max(2, int(window))
    for i in range(n):
        start = max(0, i - w + 1)
        win = values[start : i + 1]
        cur = values[i]
        if not np.isfinite(cur):
            continue
        finite = win[np.isfinite(win)]
        if len(finite) == 0:
            continue
        out[i] = float(100.0 * np.mean(finite <= cur))
    return out


def _build_daily_features(
    bars: pd.DataFrame,
    ema_gate_len: int,
    adx_len: int,
) -> Dict[str, np.ndarray]:
    daily = (
        bars.set_index("time")
        .resample("1D")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["close"])
    )
    n = len(bars)
    if daily.empty:
        nan = np.full(n, np.nan, dtype=float)
        return {
            "close": nan.copy(),
            "ema_gate": nan.copy(),
            "ema_gate_slope": nan.copy(),
            "ema200": nan.copy(),
            "ema200_slope": nan.copy(),
            "adx": nan.copy(),
            "plus_di": nan.copy(),
            "minus_di": nan.copy(),
        }

    d_close = daily["close"].to_numpy(dtype=float)
    d_ema_gate = ema(d_close, max(1, ema_gate_len))
    d_ema200 = ema(d_close, 200)
    d_ema_gate_slope = np.concatenate(([np.nan], np.diff(d_ema_gate)))
    d_ema200_slope = np.concatenate(([np.nan], np.diff(d_ema200)))
    d_plus, d_minus, d_adx = dmi_adx(
        daily["high"].to_numpy(dtype=float),
        daily["low"].to_numpy(dtype=float),
        d_close,
        max(1, adx_len),
    )
    feat = pd.DataFrame(
        {
            "close": d_close,
            "ema_gate": d_ema_gate,
            "ema_gate_slope": d_ema_gate_slope,
            "ema200": d_ema200,
            "ema200_slope": d_ema200_slope,
            "adx": d_adx,
            "plus_di": d_plus,
            "minus_di": d_minus,
        },
        index=daily.index,
    ).shift(1)

    by_day = bars["time"].dt.floor("D")
    mapped = feat.reindex(by_day.to_numpy(), method="ffill")
    return {k: mapped[k].to_numpy(dtype=float) for k in feat.columns}


def _legacy_session_mask(times: pd.Series, config: RunConfig) -> np.ndarray:
    toggles = config.strategy.toggles
    local_t = pd.to_datetime(times, utc=True) + pd.to_timedelta(int(toggles.session_tz_offset_hours), unit="h")
    h = local_t.dt.hour.to_numpy(dtype=int)
    sh = int(toggles.session_start_hour)
    eh = int(toggles.session_end_hour)
    if sh < eh:
        return (h >= sh) & (h < eh)
    if sh > eh:
        return (h >= sh) | (h < eh)
    return np.ones(len(times), dtype=bool)


def build_session_ok_mask(times: pd.Series, config: RunConfig) -> np.ndarray:
    spec = effective_session_gate(config.strategy)
    if not spec["enabled"]:
        return np.ones(len(times), dtype=bool)
    if spec["mode"] == "legacy_hour_range":
        return _legacy_session_mask(times=times, config=config)

    gate = SessionGateSpec(
        enabled=bool(spec["enabled"]),
        tz=str(spec["tz"]),
        ny=bool(spec["ny"]),
        london=bool(spec["london"]),
        tokyo=bool(spec["tokyo"]),
        sydney=bool(spec["sydney"]),
    )
    out = np.zeros(len(times), dtype=bool)
    ts = pd.to_datetime(times, utc=True)
    for i in range(len(ts)):
        out[i] = session_ok_at_trigger(ts.iloc[i], gate)
    return out


def extract_dfd05_events(
    bars: pd.DataFrame,
    symbol: str,
    timeframe: str,
    config: RunConfig,
    cvd_proxy: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    if bars.empty:
        return empty_events_frame()

    s_cfg = config.strategy
    toggles = s_cfg.toggles
    r_cfg = config.risk
    trade_mode = s_cfg.trade_mode.upper()
    mode = s_cfg.mode.upper()
    gate_vol_entry_at = s_cfg.normalized_gate_vol_entry_at()
    t = pd.to_datetime(bars["time"], utc=True)

    # Normalize to nanoseconds explicitly before deriving *_time_ms keys.
    t_ns = t.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    open_ = bars["open"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    volume = bars["volume"].to_numpy(dtype=float)
    n = len(bars)

    don_hi = rolling_max(high, s_cfg.don_len, min_periods=1)
    don_lo = rolling_min(low, s_cfg.don_len, min_periods=1)
    don_rng = don_hi - don_lo
    loc = np.full(n, 0.5, dtype=float)
    np.divide(close - don_lo, don_rng, out=loc, where=np.abs(don_rng) > 0.0)

    osc_src = (close - open_) * volume
    osc = ema(osc_src, s_cfg.osc_len)
    rsi14 = rsi_wilder(close, 14)
    macd_line, _macd_signal, _macd_hist = macd(close, fast_len=12, slow_len=26, signal_len=9)

    atr_base = atr(high, low, close, r_cfg.atr_len)
    atr14 = atr(high, low, close, 14)
    atr50 = atr(high, low, close, 50)
    long_trig_all = rolling_max(high, s_cfg.pivot_len, min_periods=1)
    piv_idx_confirm, piv_price_confirm = compute_pivot_low_confirmations(low, s_cfg.pivot_len)

    vol_sma20 = sma(volume, 20)
    vol_ratio_len_gate = max(1, toggles.vol_ratio_len)
    vol_sma_ratio_gate = sma(volume, vol_ratio_len_gate)

    atr_fast = atr(high, low, close, max(1, toggles.atr_ratio_fast_len))
    atr_slow = atr(high, low, close, max(1, toggles.atr_ratio_slow_len))

    vb_len = max(1, toggles.volume_behavior_len)
    up_vol = np.where(close > open_, volume, 0.0)
    down_vol = np.where(close < open_, volume, 0.0)
    up_sum = rolling_sum_via_sma(up_vol, vb_len)
    down_sum = rolling_sum_via_sma(down_vol, vb_len)
    vol_sum_short = rolling_sum_via_sma(volume, vb_len)
    vol_sum_long = rolling_sum_via_sma(volume, max(2, vb_len * 2))

    vol_spike_sma = sma(volume, max(1, toggles.vol_sma_len))

    wv_hi = rolling_max(close, max(1, toggles.wv70_len), min_periods=1)
    wv = np.full(n, np.nan, dtype=float)
    np.divide(100.0 * (wv_hi - low), wv_hi, out=wv, where=np.abs(wv_hi) > 0.0)

    daily = _build_daily_features(
        bars=bars,
        ema_gate_len=toggles.daily_ema_len,
        adx_len=toggles.daily_adx_len,
    )

    if cvd_proxy is None:
        cvd_proxy = np.full(n, np.nan, dtype=float)
    cvd_proxy = np.asarray(cvd_proxy, dtype=float)
    cvd_norm_len = max(2, int(toggles.cvd_norm_len))
    cvd_s = pd.Series(cvd_proxy)
    cvd_mean = cvd_s.rolling(cvd_norm_len, min_periods=min(200, cvd_norm_len)).mean().to_numpy()
    cvd_std = cvd_s.rolling(cvd_norm_len, min_periods=min(200, cvd_norm_len)).std(ddof=0).to_numpy()
    cvd_z = np.full(n, np.nan, dtype=float)
    np.divide(cvd_proxy - cvd_mean, cvd_std, out=cvd_z, where=np.isfinite(cvd_std) & (cvd_std > 0.0))
    cvd_pct = _rolling_percentile_rank(cvd_proxy, window=cvd_norm_len) if toggles.enable_cvd_pct_gate else np.full(n, np.nan, dtype=float)

    spec = effective_session_gate(s_cfg)
    use_multi_sessions = spec["mode"] != "legacy_hour_range"
    session_spec = SessionGateSpec(
        enabled=bool(spec["enabled"]),
        tz=str(spec["tz"]),
        ny=bool(spec.get("ny", False)),
        london=bool(spec.get("london", False)),
        tokyo=bool(spec.get("tokyo", False)),
        sydney=bool(spec.get("sydney", False)),
    )
    legacy_mask = (
        _legacy_session_mask(t, config=config) if spec["mode"] == "legacy_hour_range" else np.ones(n, dtype=bool)
    )

    def _session_ok_pivot(index: int) -> bool:
        if not bool(spec["enabled"]):
            return True
        if use_multi_sessions:
            return bool(session_ok_at_pivot(t.iloc[index], session_spec))
        return bool(legacy_mask[index])

    def _session_ok_trigger(index: int) -> bool:
        if not bool(spec["enabled"]):
            return True
        if use_multi_sessions:
            return bool(session_ok_at_trigger(t.iloc[index], session_spec))
        return bool(legacy_mask[index])

    def _volume_behavior_ok(index: int) -> bool:
        down = down_sum[index]
        v_long = vol_sum_long[index]
        if not np.isfinite(down) or down <= 0.0 or not np.isfinite(v_long) or v_long <= 0.0:
            return False
        up_down_ratio = up_sum[index] / down
        contraction = vol_sum_short[index] / v_long
        return bool(
            (up_down_ratio >= toggles.up_down_ratio_min)
            and (contraction <= toggles.pullback_contraction_max)
        )

    def _vol_spike_ok(index: int) -> bool:
        vma = vol_spike_sma[index]
        return bool(np.isfinite(vma) and (volume[index] >= toggles.vol_mult * vma))

    def _daily_feature_tuple(index: int) -> tuple[float, float, bool, bool, float, float, float, bool]:
        d_close = daily["close"][index]
        d_ema200 = daily["ema200"][index]
        d_ema200_slope = daily["ema200_slope"][index]
        d_adx = daily["adx"][index]
        d_plus = daily["plus_di"][index]
        d_minus = daily["minus_di"][index]
        daily_ema_ok = bool(np.isfinite(d_ema200) and close[index] >= d_ema200)
        daily_slope_ok = bool(np.isfinite(d_ema200_slope) and d_ema200_slope > toggles.daily_ema_slope_min)
        daily_di_ok = bool(np.isfinite(d_plus) and np.isfinite(d_minus) and (d_plus > d_minus))
        return d_close, d_ema200, daily_ema_ok, daily_slope_ok, d_adx, d_plus, d_minus, daily_di_ok

    toggles_json = config.toggles_json()
    strategy_params_json = json.dumps(
        {
            "strategy": {
                "don_len": s_cfg.don_len,
                "pivot_len": s_cfg.pivot_len,
                "osc_len": s_cfg.osc_len,
                "ext_band_pct": s_cfg.ext_band_pct,
                "warmup_bars": s_cfg.warmup_bars,
                "mode": mode,
                "use_bos_confirm": bool(s_cfg.use_bos_confirm),
                "bos_atr_buffer": float(s_cfg.bos_atr_buffer),
                "max_wait_bars": int(s_cfg.max_wait_bars),
                "trade_mode": trade_mode,
                "one_trade_at_a_time": bool(s_cfg.one_trade_at_a_time),
                "cooldown_bars": int(s_cfg.normalized_cooldown_bars()),
                "gate_vol_entry_at": gate_vol_entry_at,
            },
            "risk": {
                "atr_len": int(r_cfg.atr_len),
                "sl_atr_mult": float(r_cfg.sl_atr_mult),
                "rr_mult": float(r_cfg.rr_mult),
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    events: list[dict[str, Any]] = []
    last_pl_price = np.nan
    last_pl_osc = np.nan
    last_pl_bar = -1
    pending: Optional[Dict[str, Any]] = None

    def _evaluate_signal_gates(signal_i: int, pivot_i: int, pl_osc: float) -> tuple[bool, Dict[str, int]]:
        denom = abs(last_pl_osc)
        if denom == 0.0:
            osc_change_pct = np.inf if pl_osc > last_pl_osc else -np.inf
        else:
            osc_change_pct = 100.0 * (pl_osc - last_pl_osc) / denom
        divergence_strength_ok = bool(osc_change_pct >= toggles.min_osc_change_pct)
        min_pivot_gap_ok = bool((pivot_i - last_pl_bar) >= toggles.min_pivot_gap_bars)

        ema_gate_v = daily["ema_gate"][signal_i]
        ema_gate_slope = daily["ema_gate_slope"][signal_i]
        daily_ema_gate_ok = bool(np.isfinite(ema_gate_v) and np.isfinite(ema_gate_slope) and close[signal_i] >= ema_gate_v and ema_gate_slope > toggles.daily_ema_slope_min)
        adx_v = daily["adx"][signal_i]
        daily_adx_gate_ok = bool(np.isfinite(adx_v) and adx_v >= toggles.daily_adx_min)
        plus_v = daily["plus_di"][signal_i]
        minus_v = daily["minus_di"][signal_i]
        daily_di_gate_ok = bool(np.isfinite(plus_v) and np.isfinite(minus_v) and plus_v > minus_v)

        vol_ratio_pivot_v = _safe_ratio(volume[pivot_i], vol_sma_ratio_gate[pivot_i])
        vol_ratio_entry_v = _safe_ratio(volume[signal_i], vol_sma_ratio_gate[signal_i])
        vol_ratio_pivot_ok = bool(np.isfinite(vol_ratio_pivot_v) and vol_ratio_pivot_v >= toggles.pivot_vol_ratio_min)
        vol_ratio_entry_ok = bool(np.isfinite(vol_ratio_entry_v) and vol_ratio_entry_v >= toggles.entry_vol_ratio_min)
        enforce_entry_vol_at_signal = not (mode == "CONFIRM" and gate_vol_entry_at == "trigger")

        atr_ratio_v = _safe_ratio(atr_fast[signal_i], atr_slow[signal_i])
        atr_ratio_ok = bool(np.isfinite(atr_ratio_v) and atr_ratio_v <= toggles.atr_ratio_cap)
        volume_behavior_gate_ok = _volume_behavior_ok(pivot_i)
        cvd_val = cvd_proxy[signal_i]
        cvd_proxy_gate_ok = bool(np.isfinite(cvd_val) and (cvd_val >= toggles.cvd_min))
        cvd_z_val = cvd_z[signal_i]
        cvd_z_gate_ok = bool(np.isfinite(cvd_z_val) and (cvd_z_val >= toggles.cvd_z_min))
        cvd_pct_val = cvd_pct[signal_i]
        cvd_pct_gate_ok = bool(np.isfinite(cvd_pct_val) and (cvd_pct_val >= toggles.cvd_pct_min))
        vol_spike_gate_ok = _vol_spike_ok(signal_i)
        wv70_gate_ok = bool(np.isfinite(wv[signal_i]) and (wv[signal_i] >= toggles.wv70_min))

        flags = {
            "divergence_strength_ok": _as_i1(divergence_strength_ok),
            "min_pivot_gap_ok": _as_i1(min_pivot_gap_ok),
            "daily_ema_gate_ok": _as_i1(daily_ema_gate_ok),
            "daily_adx_gate_ok": _as_i1(daily_adx_gate_ok),
            "daily_di_gate_ok": _as_i1(daily_di_gate_ok),
            "vol_ratio_pivot_ok": _as_i1(vol_ratio_pivot_ok),
            "vol_ratio_entry_ok": _as_i1(vol_ratio_entry_ok),
            "atr_ratio_ok": _as_i1(atr_ratio_ok),
            "volume_behavior_gate_ok": _as_i1(volume_behavior_gate_ok),
            "cvd_proxy_gate_ok": _as_i1(cvd_proxy_gate_ok),
            "cvd_z_gate_ok": _as_i1(cvd_z_gate_ok),
            "cvd_pct_gate_ok": _as_i1(cvd_pct_gate_ok),
            "vol_spike_gate_ok": _as_i1(vol_spike_gate_ok),
            "wv70_gate_ok": _as_i1(wv70_gate_ok),
        }

        session_ok_pivot = _session_ok_pivot(pivot_i)
        if not session_ok_pivot:
            return False, flags

        if trade_mode != "GATED":
            return True, flags

        if toggles.enable_divergence_strength and not divergence_strength_ok:
            return False, flags
        if toggles.enable_min_pivot_gap and not min_pivot_gap_ok:
            return False, flags
        if toggles.enable_daily_ema_gate and not daily_ema_gate_ok:
            return False, flags
        if toggles.enable_daily_adx_gate and not daily_adx_gate_ok:
            return False, flags
        if toggles.enable_daily_di_gate and not daily_di_gate_ok:
            return False, flags
        if toggles.enable_vol_ratio_pivot_gate and not vol_ratio_pivot_ok:
            return False, flags
        if toggles.enable_vol_ratio_entry_gate and enforce_entry_vol_at_signal and not vol_ratio_entry_ok:
            return False, flags
        if toggles.enable_atr_ratio_cap and not atr_ratio_ok:
            return False, flags
        if toggles.enable_volume_behavior_gate and not volume_behavior_gate_ok:
            return False, flags
        if toggles.enable_cvd_proxy_gate and not cvd_proxy_gate_ok:
            return False, flags
        if toggles.enable_cvd_z_gate and not cvd_z_gate_ok:
            return False, flags
        if toggles.enable_cvd_pct_gate and not cvd_pct_gate_ok:
            return False, flags
        if toggles.enable_vol_spike_gate and not vol_spike_gate_ok:
            return False, flags
        if toggles.enable_wv70_gate and not wv70_gate_ok:
            return False, flags
        return True, flags

    def _append_event(
        setup: Dict[str, Any],
        entry_i: int,
        *,
        triggered_ok: bool = True,
        vol_ratio_entry_ok_override: Optional[bool] = None,
    ) -> None:
        signal_i = int(setup["signal_i"])
        pivot_i = int(setup["pivot_i"])
        event_time_ms = int(t_ns[signal_i] // 1_000_000)
        entry_time_ms = int(t_ns[entry_i] // 1_000_000)
        pivot_time_ms = int(t_ns[pivot_i] // 1_000_000)
        signal_id = f"{symbol}-{timeframe}-DFD05-{event_time_ms}"

        vol_ratio_entry = _safe_ratio(volume[entry_i], vol_sma20[entry_i])
        atr_ratio_entry = _safe_ratio(atr14[entry_i], atr50[entry_i])
        d_close, d_ema200, daily_ema_ok_entry, daily_slope_ok_entry, d_adx, d_plus, d_minus, d_di_ok = _daily_feature_tuple(entry_i)
        session_ok_entry = _session_ok_trigger(entry_i)

        row: Dict[str, Any] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "signal_id": signal_id,
            "event_time_ms": event_time_ms,
            "entry_time_ms": entry_time_ms,
            "entry_price": float(close[entry_i]),
            "pivot_time_ms": pivot_time_ms,
            "pivot_price": float(setup["pivot_price"]),
            "mode": mode,
            "trade_mode": trade_mode,
            "toggles_json": toggles_json,
            "strategy_params_json": strategy_params_json,
            "event_index": signal_i,
            "entry_index": entry_i,
            "setup_age_bars": int(max(0, entry_i - signal_i)),
            "triggered_ok": _as_i1(triggered_ok),
            "atr_entry": float(atr_base[entry_i]),
            "osc_change_pct": float(setup["osc_change_pct"]),
            "bars_gap": int(setup["bars_gap"]),
            "div_type": int(setup["div_type"]),
            "loc_pivot": float(setup["loc_pivot"]),
            "vol_ratio_pivot": float(setup["vol_ratio_pivot"]),
            "rsi_pivot": float(setup["rsi_pivot"]),
            "macd_pivot": float(setup["macd_pivot"]),
            "session_ok_pivot": int(setup["session_ok_pivot"]),
            "vol_ratio_entry": float(vol_ratio_entry),
            "atr_ratio_entry": float(atr_ratio_entry),
            "daily_close": float(d_close),
            "daily_ema200": float(d_ema200),
            "daily_ema_ok": _as_i1(daily_ema_ok_entry),
            "daily_slope_ok": _as_i1(daily_slope_ok_entry),
            "daily_adx": float(d_adx),
            "daily_plus_di": float(d_plus),
            "daily_minus_di": float(d_minus),
            "daily_di_ok": _as_i1(d_di_ok),
            "cvd_proxy_entry": float(cvd_proxy[entry_i]),
            "cvd_norm_entry": float(cvd_z[entry_i]),
            "cvd_z_entry": float(cvd_z[entry_i]),
            "cvd_pct_entry": float(cvd_pct[entry_i]),
            "vol_behavior_ok_entry": _as_i1(_volume_behavior_ok(entry_i)),
            "vol_spike_ok_entry": _as_i1(_vol_spike_ok(entry_i)),
            "session_ok_entry": _as_i1(session_ok_entry),
            "divergence_strength_ok": int(setup["divergence_strength_ok"]),
            "min_pivot_gap_ok": int(setup["min_pivot_gap_ok"]),
            "daily_ema_gate_ok": int(setup["daily_ema_gate_ok"]),
            "daily_adx_gate_ok": int(setup["daily_adx_gate_ok"]),
            "daily_di_gate_ok": int(setup["daily_di_gate_ok"]),
            "vol_ratio_pivot_ok": int(setup["vol_ratio_pivot_ok"]),
            "vol_ratio_entry_ok": int(
                _as_i1(vol_ratio_entry_ok_override)
                if vol_ratio_entry_ok_override is not None
                else setup["vol_ratio_entry_ok"]
            ),
            "atr_ratio_ok": int(setup["atr_ratio_ok"]),
            "volume_behavior_gate_ok": int(setup["volume_behavior_gate_ok"]),
            "cvd_proxy_gate_ok": int(setup["cvd_proxy_gate_ok"]),
            "cvd_z_gate_ok": int(setup["cvd_z_gate_ok"]),
            "cvd_pct_gate_ok": int(setup["cvd_pct_gate_ok"]),
            "vol_spike_gate_ok": int(setup["vol_spike_gate_ok"]),
            "wv70_gate_ok": int(setup["wv70_gate_ok"]),
            "near_lower_ok": int(setup["near_lower_ok"]),
            "bull_div_ok": int(setup["bull_div_ok"]),
            "signal_allowed_ok": int(setup["signal_allowed_ok"]),
        }
        events.append(row)

    for i in range(n):
        if mode == "CONFIRM" and pending is not None:
            setup_bar = int(pending["setup_bar"])
            age = i - setup_bar
            if age > s_cfg.max_wait_bars:
                pending = None
            elif age >= 1:
                if s_cfg.use_bos_confirm:
                    trigger_level = float(pending["long_trig"]) + atr_base[i] * s_cfg.bos_atr_buffer
                    trigger_price_ok = bool(close[i] > trigger_level)
                else:
                    trigger_price_ok = bool(close[i] > open_[i])

                trigger_session_ok = _session_ok_trigger(i)
                trigger_vol_ok = True
                if (
                    trade_mode == "GATED"
                    and toggles.enable_vol_ratio_entry_gate
                    and gate_vol_entry_at == "trigger"
                ):
                    vol_ratio_trigger_v = _safe_ratio(volume[i], vol_sma_ratio_gate[i])
                    trigger_vol_ok = bool(
                        np.isfinite(vol_ratio_trigger_v)
                        and vol_ratio_trigger_v >= toggles.entry_vol_ratio_min
                    )

                if trigger_price_ok and trigger_session_ok and trigger_vol_ok:
                    _append_event(
                        setup=pending,
                        entry_i=i,
                        triggered_ok=True,
                        vol_ratio_entry_ok_override=(
                            trigger_vol_ok if gate_vol_entry_at == "trigger" else None
                        ),
                    )
                    pending = None

        p = piv_idx_confirm[i]
        if p < 0:
            continue

        pl_price = piv_price_confirm[i]
        pl_osc = osc[p]
        has_prev = np.isfinite(last_pl_price) and np.isfinite(last_pl_osc) and last_pl_bar >= 0
        use_classic = trade_mode == "GATED" and toggles.enable_classic_only
        price_cmp = (pl_price < last_pl_price) if use_classic else (pl_price <= last_pl_price)
        bull_div = bool(has_prev and price_cmp and (pl_osc > last_pl_osc))
        near_lower = bool(np.isfinite(loc[p]) and (loc[p] <= s_cfg.ext_band_pct))
        signal_allowed = bool(i >= s_cfg.warmup_bars)

        gate_flags: Dict[str, int] = {
            "divergence_strength_ok": 0,
            "min_pivot_gap_ok": 0,
            "daily_ema_gate_ok": 0,
            "daily_adx_gate_ok": 0,
            "daily_di_gate_ok": 0,
            "vol_ratio_pivot_ok": 0,
            "vol_ratio_entry_ok": 0,
            "atr_ratio_ok": 0,
            "volume_behavior_gate_ok": 0,
            "cvd_proxy_gate_ok": 0,
            "cvd_z_gate_ok": 0,
            "cvd_pct_gate_ok": 0,
            "vol_spike_gate_ok": 0,
            "wv70_gate_ok": 0,
        }
        signal_ok = False
        if bull_div and near_lower and signal_allowed:
            signal_ok, gate_flags = _evaluate_signal_gates(signal_i=i, pivot_i=p, pl_osc=pl_osc)

        if has_prev:
            denom_prev = abs(last_pl_osc)
            if denom_prev == 0.0:
                osc_change_pct = np.inf if pl_osc > last_pl_osc else -np.inf
            else:
                osc_change_pct = 100.0 * (pl_osc - last_pl_osc) / denom_prev
            bars_gap = int(i - last_pl_bar)
            div_type = 1 if pl_price < last_pl_price else 0
        else:
            osc_change_pct = np.nan
            bars_gap = -1
            div_type = 0

        setup_payload: Dict[str, Any] = {
            "signal_i": i,
            "pivot_i": int(p),
            "pivot_price": float(pl_price),
            "long_trig": float(long_trig_all[i]),
            "setup_bar": i,
            "osc_change_pct": float(osc_change_pct),
            "bars_gap": int(bars_gap),
            "div_type": int(div_type),
            "loc_pivot": float(loc[p]),
            "vol_ratio_pivot": float(_safe_ratio(volume[p], vol_sma20[p])),
            "rsi_pivot": float(rsi14[p]),
            "macd_pivot": float(macd_line[p]),
            "session_ok_pivot": _as_i1(_session_ok_pivot(p)),
            "near_lower_ok": _as_i1(near_lower),
            "bull_div_ok": _as_i1(bull_div),
            "signal_allowed_ok": _as_i1(signal_allowed),
            **gate_flags,
        }

        if signal_ok:
            if mode == "RAW":
                _append_event(setup=setup_payload, entry_i=i)
            else:
                pending = setup_payload

        # Pine parity: update memory on every confirmed pivot low, not only divergences.
        last_pl_price = pl_price
        last_pl_osc = pl_osc
        last_pl_bar = p

    if not events:
        return empty_events_frame()
    out = _coerce_event_dtypes(pd.DataFrame(events))
    out = out.sort_values(["symbol", "event_time_ms", "entry_time_ms"]).reset_index(drop=True)
    return out


def toggles_as_dict(config: RunConfig) -> Dict[str, object]:
    return asdict(config.strategy.toggles)
