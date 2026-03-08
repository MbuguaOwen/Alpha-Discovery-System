import pandas as pd

from dfd05.pine16_config import load_pine16_exact_config
from dfd05.pine16_parity_engine import _simulate_symbol_trades


def test_strategy_exit_same_bar_sl_first():
    cfg = load_pine16_exact_config("configs/pine16_exact_prod_screenshot.yaml")
    bars = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01T00:00:00Z", periods=3, freq="15min", tz="UTC"),
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 104.5, 100.0],
            "low": [100.0, 98.5, 100.0],
            "close": [100.0, 100.0, 100.0],
            "volume": [1.0, 1.0, 1.0],
            "symbol": ["XAUUSD"] * 3,
        }
    )
    events = pd.DataFrame(
        [
            {
                "entry_index": 0,
                "event_time_ms": 1704067200000,
                "signal_id": "s1",
                "atr_entry": 1.0,
            }
        ]
    )

    trades = _simulate_symbol_trades(bars=bars, events=events, cfg=cfg, symbol="XAUUSD")
    assert len(trades) == 1
    tr = trades.iloc[0]
    assert tr["trade_status"] == "CLOSED"
    assert tr["result_r"] == -1.0
    assert tr["exit_price"] == tr["sl_price"]

