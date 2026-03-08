import pandas as pd

from dfd05.config import ForwardConfig, RiskConfig, RunConfig
from dfd05.forward import compute_forward_outcomes, validate_forward_outcomes


def test_forward_validation_passes_and_summarizes():
    bars = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC"),
            "open": [100.0] * 5,
            "high": [100.0, 104.0, 103.0, 101.0, 101.0],
            "low": [100.0, 98.0, 99.0, 99.5, 99.5],
            "close": [100.0, 100.0, 100.0, 100.0, 100.0],
            "volume": [1.0] * 5,
            "symbol": ["XAUUSD"] * 5,
        }
    )
    events = pd.DataFrame(
        [
            {
                "symbol": "XAUUSD",
                "timeframe": "h1",
                "signal_id": "XAUUSD-h1-DFD05-1",
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
            forward_horizons_hours=[1],
            tie_break="sl",
            emit_resolved=True,
        ),
        risk=RiskConfig(atr_len=14, sl_atr_mult=1.0, rr_mult=2.0),
    )

    fwd = compute_forward_outcomes(bars=bars, events=events, config=cfg, timeframe_minutes=60)
    summary = validate_forward_outcomes(
        forward_df=fwd,
        horizons_hours=[1],
        rr_mult=cfg.risk.rr_mult,
        emit_resolved=True,
        mode="barrier",
    )
    assert len(summary) == 1
    s = summary[0]
    assert s["n_events"] == 1.0
    assert s["truncated_rate"] == 0.0
