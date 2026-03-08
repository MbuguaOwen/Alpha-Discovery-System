import pandas as pd

from dfd05.pine16_parity_engine import compare_parity_to_exact


def test_tv_reconciliation_thresholds_pass_for_near_match():
    exact = pd.DataFrame(
        {
            "symbol": ["XAUUSD", "XAUUSD"],
            "entry_time_utc": [
                pd.Timestamp("2024-01-01T00:00:00Z"),
                pd.Timestamp("2024-01-01T01:00:00Z"),
            ],
            "exit_time_utc": [
                pd.Timestamp("2024-01-01T00:15:00Z"),
                pd.Timestamp("2024-01-01T01:15:00Z"),
            ],
            "result_r": [3.0, -1.0],
        }
    )
    parity = pd.DataFrame(
        {
            "symbol": ["XAUUSD", "XAUUSD"],
            "entry_time_utc": [
                pd.Timestamp("2024-01-01T00:00:00Z"),
                pd.Timestamp("2024-01-01T01:00:00Z"),
            ],
            "exit_time_utc": [
                pd.Timestamp("2024-01-01T00:15:00Z"),
                pd.Timestamp("2024-01-01T01:15:00Z"),
            ],
            "result_r": [3.0, -1.0],
        }
    )

    cmp = compare_parity_to_exact(exact_trades=exact, parity_trades=parity, timeframe="m15")
    assert cmp.pass_thresholds is True
    assert cmp.signal_count_mismatch_pct == 0.0
    assert cmp.aggregate_net_r_mismatch_pct == 0.0


def test_tv_reconciliation_fails_on_large_count_or_r_drift():
    exact = pd.DataFrame(
        {
            "symbol": ["XAUUSD"],
            "entry_time_utc": [pd.Timestamp("2024-01-01T00:00:00Z")],
            "exit_time_utc": [pd.Timestamp("2024-01-01T00:15:00Z")],
            "result_r": [3.0],
        }
    )
    parity = pd.DataFrame(
        {
            "symbol": ["XAUUSD", "XAUUSD"],
            "entry_time_utc": [
                pd.Timestamp("2024-01-01T00:00:00Z"),
                pd.Timestamp("2024-01-01T01:00:00Z"),
            ],
            "exit_time_utc": [
                pd.Timestamp("2024-01-01T00:15:00Z"),
                pd.Timestamp("2024-01-01T01:15:00Z"),
            ],
            "result_r": [3.0, -1.0],
        }
    )

    cmp = compare_parity_to_exact(exact_trades=exact, parity_trades=parity, timeframe="m15")
    assert cmp.pass_thresholds is False

