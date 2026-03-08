from __future__ import annotations

import numpy as np
import pandas as pd


def ema(values: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if length <= 1:
        return arr.copy()
    return pd.Series(arr).ewm(span=length, adjust=False, min_periods=1).mean().to_numpy()


def sma(values: np.ndarray, length: int, min_periods: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if length <= 1:
        return arr.copy()
    mp = length if min_periods is None else min_periods
    return pd.Series(arr).rolling(length, min_periods=mp).mean().to_numpy()


def rolling_max(values: np.ndarray, length: int, min_periods: int = 1) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if length <= 1:
        return arr.copy()
    return pd.Series(arr).rolling(length, min_periods=min_periods).max().to_numpy()


def rolling_min(values: np.ndarray, length: int, min_periods: int = 1) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if length <= 1:
        return arr.copy()
    return pd.Series(arr).rolling(length, min_periods=min_periods).min().to_numpy()


def rolling_sum_via_sma(values: np.ndarray, length: int) -> np.ndarray:
    return sma(values, length) * float(length)


def rma(values: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if length <= 1:
        return arr.copy()
    return (
        pd.Series(arr)
        .ewm(alpha=1.0 / float(length), adjust=False, min_periods=length)
        .mean()
        .to_numpy()
    )


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    return tr


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    tr = true_range(high, low, close)
    return rma(tr, length)


def dmi_adx(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)

    up_move = np.diff(h, prepend=h[0])
    down_move = -np.diff(l, prepend=l[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)

    tr = true_range(h, l, c)
    atr_len = rma(tr, length)
    plus_di = np.full_like(atr_len, np.nan)
    minus_di = np.full_like(atr_len, np.nan)
    plus_sm = rma(plus_dm, length)
    minus_sm = rma(minus_dm, length)

    valid = atr_len > 0
    plus_di[valid] = 100.0 * plus_sm[valid] / atr_len[valid]
    minus_di[valid] = 100.0 * minus_sm[valid] / atr_len[valid]

    denom = plus_di + minus_di
    dx = np.full_like(denom, np.nan)
    valid_dx = denom > 0
    dx[valid_dx] = 100.0 * np.abs(plus_di[valid_dx] - minus_di[valid_dx]) / denom[valid_dx]
    adx = rma(dx, length)
    return plus_di, minus_di, adx


def rsi_wilder(values: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if length <= 1:
        return np.full_like(arr, 50.0)

    delta = np.diff(arr, prepend=arr[0])
    up = np.where(delta > 0.0, delta, 0.0)
    down = np.where(delta < 0.0, -delta, 0.0)

    avg_up = rma(up, length)
    avg_down = rma(down, length)

    out = np.full_like(arr, np.nan)
    valid_down = avg_down > 0
    rs = np.full_like(arr, np.nan)
    rs[valid_down] = avg_up[valid_down] / avg_down[valid_down]
    out[valid_down] = 100.0 - (100.0 / (1.0 + rs[valid_down]))

    # Degenerate cases follow common RSI conventions.
    zero_down = (avg_down == 0.0) & np.isfinite(avg_up)
    out[zero_down] = 100.0
    both_zero = (avg_down == 0.0) & (avg_up == 0.0)
    out[both_zero] = 50.0
    return out


def macd(
    values: np.ndarray,
    fast_len: int = 12,
    slow_len: int = 26,
    signal_len: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    fast = ema(arr, max(1, fast_len))
    slow = ema(arr, max(1, slow_len))
    line = fast - slow
    signal = ema(line, max(1, signal_len))
    hist = line - signal
    return line, signal, hist
