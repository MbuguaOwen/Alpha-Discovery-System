import pandas as pd

from dfd05.pine16_config import load_pine16_exact_config
from dfd05.pine16_parity_engine import _simulate_symbol_trades


def _bars_for_confirm() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01T00:00:00Z", periods=4, freq="15min", tz="UTC"),
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.1, 103.5, 100.5, 100.2],
            "low": [99.9, 99.8, 99.7, 99.8],
            "close": [100.0, 103.0, 100.1, 100.0],
            "volume": [1.0, 1.0, 1.0, 1.0],
            "symbol": ["XAUUSD"] * 4,
        }
    )


def test_confirm_mode_trade_sim_closes_at_tp():
    cfg = load_pine16_exact_config("configs/pine16_exact_prod_screenshot.yaml")
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
    trades = _simulate_symbol_trades(bars=_bars_for_confirm(), events=events, cfg=cfg, symbol="XAUUSD")

    assert len(trades) == 1
    tr = trades.iloc[0]
    assert tr["trade_status"] == "CLOSED"
    assert tr["result_r"] == 3.0
    assert tr["exit_price"] == tr["tp_price"]

