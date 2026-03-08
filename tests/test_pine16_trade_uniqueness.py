import pandas as pd

from dfd05.pine16_config import load_pine16_exact_config
from dfd05.pine16_parity_engine import _simulate_symbol_trades


def test_trade_uniqueness_one_trade_at_a_time_blocks_overlap_entries():
    cfg = load_pine16_exact_config("configs/pine16_exact_prod_screenshot.yaml")
    bars = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01T00:00:00Z", periods=6, freq="15min", tz="UTC"),
            "open": [100.0] * 6,
            "high": [100.2, 100.2, 100.2, 103.2, 100.2, 100.2],
            "low": [99.8, 99.8, 99.8, 99.8, 99.8, 99.8],
            "close": [100.0] * 6,
            "volume": [1.0] * 6,
            "symbol": ["XAUUSD"] * 6,
        }
    )

    # Two entry attempts while first trade is still active.
    events = pd.DataFrame(
        [
            {"entry_index": 0, "event_time_ms": 1, "signal_id": "s1", "atr_entry": 1.0},
            {"entry_index": 1, "event_time_ms": 2, "signal_id": "s2", "atr_entry": 1.0},
        ]
    )
    trades = _simulate_symbol_trades(bars=bars, events=events, cfg=cfg, symbol="XAUUSD")

    assert len(trades) == 1
    assert trades.iloc[0]["trade_status"] == "CLOSED"

