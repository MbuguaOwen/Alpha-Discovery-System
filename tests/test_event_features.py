import numpy as np
import pandas as pd

from dfd05.config import RunConfig, SessionGateConfig, StrategyConfig
from dfd05.strategy import extract_dfd05_events


def test_event_feature_columns_present_and_populated():
    n = 1000
    times = pd.date_range("2024-01-01T00:00:00Z", periods=n, freq="1h", tz="UTC")

    open_ = np.full(n, 10.0, dtype=float)
    close = np.full(n, 10.0, dtype=float)
    low = 10.0 + 0.001 * np.arange(n, dtype=float)
    high = low + 1.0
    volume = 100.0 + (np.arange(n, dtype=float) % 30.0)

    # Two explicit pivots; second one forms bullish divergence vs first.
    low[830] = 8.0
    close[830] = 9.0
    low[930] = 7.0
    close[930] = 11.0

    bars = pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "symbol": ["XAUUSD"] * n,
        }
    )
    cvd_proxy = np.linspace(-2.0, 2.0, n, dtype=float)

    cfg = RunConfig(
        symbols=["XAUUSD"],
        timeframe="h1",
        strategy=StrategyConfig(
            don_len=20,
            pivot_len=1,
            osc_len=1,
            ext_band_pct=1.0,
            warmup_bars=0,
            mode="RAW",
            trade_mode="BASELINE_ALL",
            session_gate=SessionGateConfig(enabled=False),
        ),
    )

    events = extract_dfd05_events(
        bars=bars,
        symbol="XAUUSD",
        timeframe="h1",
        config=cfg,
        cvd_proxy=cvd_proxy,
    )
    assert len(events) >= 1
    ev = events.iloc[-1]

    required_cols = [
        "osc_change_pct",
        "bars_gap",
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
        "vol_behavior_ok_entry",
        "vol_spike_ok_entry",
        "session_ok_entry",
    ]
    for col in required_cols:
        assert col in events.columns

    finite_checks = [
        "osc_change_pct",
        "bars_gap",
        "loc_pivot",
        "vol_ratio_pivot",
        "rsi_pivot",
        "macd_pivot",
        "vol_ratio_entry",
        "atr_ratio_entry",
        "daily_adx",
        "daily_plus_di",
        "daily_minus_di",
        "cvd_proxy_entry",
        "cvd_norm_entry",
    ]
    for col in finite_checks:
        assert pd.notna(ev[col]), f"Expected non-null event feature: {col}"
    frac = events[finite_checks].notna().mean()
    assert (frac >= 0.5).all()

    assert ev["div_type"] in {0, 1}
    assert str(events["osc_change_pct"].dtype) == "float64"
    assert str(events["bars_gap"].dtype) == "int64"
    assert str(events["daily_adx"].dtype) == "float64"
    assert str(events["session_ok_pivot"].dtype) == "int8"
    assert str(events["daily_ema_ok"].dtype) == "int8"
    assert str(events["vol_behavior_ok_entry"].dtype) == "int8"
