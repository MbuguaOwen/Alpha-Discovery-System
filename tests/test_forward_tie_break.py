import pandas as pd

from dfd05.config import ForwardConfig, RiskConfig, RunConfig
from dfd05.forward import compute_forward_outcomes


def test_forward_first_touch_tie_break_sl_first():
    bars = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC"),
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 105.0, 103.0],
            "low": [100.0, 95.0, 97.0],
            "close": [100.0, 101.0, 102.0],
            "volume": [1.0, 1.0, 1.0],
            "symbol": ["XAUUSD"] * 3,
        }
    )

    events = pd.DataFrame(
        [
            {
                "symbol": "XAUUSD",
                "timeframe": "h1",
                "signal_id": "XAUUSD-h1-DFD05-1704067200000",
                "event_time_ms": int(bars.loc[0, "time"].value // 1_000_000),
                "entry_time_ms": int(bars.loc[0, "time"].value // 1_000_000),
                "entry_price": 100.0,
                "pivot_time_ms": int(bars.loc[0, "time"].value // 1_000_000),
                "pivot_price": 99.0,
                "mode": "RAW",
                "trade_mode": "BASELINE_ALL",
                "toggles_json": "{}",
                "event_index": 0,
                "entry_index": 0,
                "atr_entry": 2.0,
            }
        ]
    )

    cfg = RunConfig(
        forward=ForwardConfig(
            mode="barrier",
            forward_horizons_hours=[1, 4],
            tie_break="sl",
            emit_resolved=True,
        ),
        risk=RiskConfig(atr_len=14, sl_atr_mult=1.0, rr_mult=2.0),
    )

    out = compute_forward_outcomes(bars=bars, events=events, config=cfg, timeframe_minutes=60)
    assert len(out) == 1
    row = out.iloc[0]

    # entry=100, atr=2 -> SL=98, TP=104. Bar 1 hits both high>=104 and low<=98.
    assert row["tp_first_1h"] == 0
    assert row["sl_first_1h"] == 0
    assert row["both_samebar_1h"] == 1
    assert row["no_hit_1h"] == 0
    assert row["is_truncated_1h"] == 0
    assert row["tp_first_resolved_1h"] == 0
    assert row["sl_first_resolved_1h"] == 1
    assert row["mfe_1h"] == 0.05
    assert row["mae_1h"] == -0.05

    # 4h horizon is truncated with only 2 bars after entry in this dataset.
    assert row["is_truncated_4h"] == 1
    assert pd.isna(row["tp_first_4h"])
    assert pd.isna(row["sl_first_4h"])
    assert pd.isna(row["both_samebar_4h"])
    assert pd.isna(row["no_hit_4h"])
    assert pd.isna(row["mfe_4h"])
    assert pd.isna(row["mae_4h"])
