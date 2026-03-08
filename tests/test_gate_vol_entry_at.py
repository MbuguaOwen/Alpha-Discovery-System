import pandas as pd

from dfd05.config import RunConfig, SessionGateConfig, StrategyConfig
from dfd05.strategy import extract_dfd05_events


def _base_bars(volumes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=8, freq="1h", tz="UTC"),
            "open": [10.0] * 8,
            "high": [11.0] * 8,
            "low": [10.0, 9.0, 10.0, 8.0, 10.0, 9.0, 10.0, 10.0],
            "close": [10.0, 9.0, 10.0, 11.0, 10.5, 10.0, 11.0, 10.0],
            "volume": volumes,
            "symbol": ["XAUUSD"] * 8,
        }
    )


def _cfg(gate_vol_entry_at: str) -> RunConfig:
    return RunConfig(
        symbols=["XAUUSD"],
        timeframe="h1",
        strategy=StrategyConfig(
            don_len=2,
            pivot_len=1,
            osc_len=1,
            ext_band_pct=1.0,
            warmup_bars=0,
            mode="CONFIRM",
            use_bos_confirm=False,
            max_wait_bars=3,
            trade_mode="GATED",
            gate_vol_entry_at=gate_vol_entry_at,
            session_gate=SessionGateConfig(enabled=False),
        ),
    )


def test_gate_vol_entry_at_trigger_allows_signal_low_trigger_high_volume():
    # Signal bar ratio < 1.0, trigger bar ratio >= 1.0
    bars = _base_bars([1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 3.0, 1.0])

    cfg_signal = _cfg("signal")
    cfg_signal.strategy.toggles.enable_vol_ratio_entry_gate = True
    cfg_signal.strategy.toggles.entry_vol_ratio_min = 1.0
    cfg_signal.strategy.toggles.vol_ratio_len = 2
    ev_signal = extract_dfd05_events(bars, symbol="XAUUSD", timeframe="h1", config=cfg_signal)

    cfg_trigger = _cfg("trigger")
    cfg_trigger.strategy.toggles.enable_vol_ratio_entry_gate = True
    cfg_trigger.strategy.toggles.entry_vol_ratio_min = 1.0
    cfg_trigger.strategy.toggles.vol_ratio_len = 2
    ev_trigger = extract_dfd05_events(bars, symbol="XAUUSD", timeframe="h1", config=cfg_trigger)

    assert len(ev_signal) == 0
    assert len(ev_trigger) == 1
    assert int(ev_trigger.iloc[0]["setup_age_bars"]) == 2
    assert int(ev_trigger.iloc[0]["triggered_ok"]) == 1


def test_gate_vol_entry_at_trigger_blocks_signal_high_trigger_low_volume():
    # Signal bar ratio >= 1.0, trigger bar ratio < 1.0
    bars = _base_bars([1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 0.2, 1.0])

    cfg_signal = _cfg("signal")
    cfg_signal.strategy.toggles.enable_vol_ratio_entry_gate = True
    cfg_signal.strategy.toggles.entry_vol_ratio_min = 1.0
    cfg_signal.strategy.toggles.vol_ratio_len = 2
    ev_signal = extract_dfd05_events(bars, symbol="XAUUSD", timeframe="h1", config=cfg_signal)

    cfg_trigger = _cfg("trigger")
    cfg_trigger.strategy.toggles.enable_vol_ratio_entry_gate = True
    cfg_trigger.strategy.toggles.entry_vol_ratio_min = 1.0
    cfg_trigger.strategy.toggles.vol_ratio_len = 2
    ev_trigger = extract_dfd05_events(bars, symbol="XAUUSD", timeframe="h1", config=cfg_trigger)

    assert len(ev_signal) == 1
    assert len(ev_trigger) == 0
