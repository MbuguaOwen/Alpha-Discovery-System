import pandas as pd

from dfd05.config import RiskConfig, RunConfig, SessionGateConfig, StrategyConfig
from dfd05.strategy import extract_dfd05_events


def test_entry_is_signal_bar_close_not_pivot_bar_close():
    bars = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=8, freq="1h", tz="UTC"),
            "open": [10.0] * 8,
            "high": [11.0] * 8,
            "low": [10.0, 9.0, 10.0, 8.0, 10.0, 9.0, 10.0, 10.0],
            "close": [10.0, 9.0, 10.0, 11.0, 10.5, 10.0, 10.0, 10.0],
            "volume": [1.0] * 8,
            "symbol": ["XAUUSD"] * 8,
        }
    )

    cfg = RunConfig(
        symbols=["XAUUSD"],
        timeframe="h1",
        strategy=StrategyConfig(
            don_len=2,
            pivot_len=1,
            osc_len=1,
            ext_band_pct=1.0,
            warmup_bars=0,
            mode="RAW",
            trade_mode="BASELINE_ALL",
            session_gate=SessionGateConfig(enabled=False),
        ),
        risk=RiskConfig(atr_len=1, sl_atr_mult=1.0, rr_mult=2.0),
    )

    events = extract_dfd05_events(bars, symbol="XAUUSD", timeframe="h1", config=cfg)
    assert len(events) == 1

    ev = events.iloc[0]
    signal_bar_idx = 4
    pivot_bar_idx = 3

    assert ev["entry_price"] == bars.loc[signal_bar_idx, "close"]
    assert ev["entry_price"] != bars.loc[pivot_bar_idx, "close"]

    expected_entry_ms = int(bars.loc[signal_bar_idx, "time"].value // 1_000_000)
    expected_pivot_ms = int(bars.loc[pivot_bar_idx, "time"].value // 1_000_000)
    assert ev["entry_time_ms"] == expected_entry_ms
    assert ev["event_time_ms"] == expected_entry_ms
    assert ev["pivot_time_ms"] == expected_pivot_ms
