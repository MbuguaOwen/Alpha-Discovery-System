import math

import pandas as pd
import pytest

from dfd05.config import AtrTargetConfig, ForwardConfig, RunConfig
from dfd05.forward import compute_forward_outcomes, validate_forward_outcomes


def test_time_only_forward_outputs_and_truncation_behavior():
    bars = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC"),
            "open": [100.0, 100.0, 101.0, 102.0, 103.0],
            "high": [100.0, 102.0, 103.0, 104.0, 105.0],
            "low": [100.0, 99.0, 98.0, 97.0, 96.0],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "volume": [1.0, 2.0, 3.0, 4.0, 5.0],
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
                "atr_entry": 1.0,
            }
        ]
    )
    cfg = RunConfig(
        forward=ForwardConfig(
            mode="time_only",
            horizons_hours=[1, 5],
            ret_thresholds_bps=[0, 100],
            percentile_targets=[0.8],
            mfe_thresholds_bps=[100, 300],
            mae_thresholds_bps=[50, 100],
            mfe_percentile_targets=[0.8],
            atr_targets=AtrTargetConfig(
                mfe_atr_thresholds=[0.5, 1.0],
                mae_atr_thresholds=[0.5, 1.0],
                pct_targets=[0.8],
                quality_mfe_threshold=1.0,
                quality_mae_threshold=1.0,
            ),
            truncate_policy="nan",
        )
    )

    out = compute_forward_outcomes(bars=bars, events=events, config=cfg, timeframe_minutes=60)
    assert len(out) == 1
    row = out.iloc[0]

    assert row["is_truncated_1h"] == 0
    assert row["ret_1h"] == pytest.approx(0.01)
    assert row["logret_1h"] == pytest.approx(math.log(1.01))
    assert row["mfe_1h"] == pytest.approx(0.02)
    assert row["mae_1h"] == pytest.approx(-0.01)
    assert row["ret_atr_1h"] == pytest.approx(1.0)
    assert row["mfe_atr_1h"] == pytest.approx(2.0)
    assert row["mae_atr_1h"] == pytest.approx(-1.0)
    assert row["rr_like_1h"] == pytest.approx(2.0)
    assert row["max_ru_1h"] == pytest.approx(0.02)
    assert row["max_dd_1h"] == pytest.approx(-0.01)
    assert row["up_1h"] == 1
    assert row["dn_1h"] == 0
    assert row["flat_1h"] == 0
    assert row["worked_1h_ge_0bps"] == 1
    assert row["worked_1h_ge_100bps"] == 1
    assert row["ret_pct_1h"] == pytest.approx(1.0)
    assert row["worked_1h_top_q80"] == 1
    assert row["worked_mfe_1h_ge_100bps"] == 1
    assert row["worked_mfe_1h_ge_300bps"] == 0
    assert row["mfe_pct_1h"] == pytest.approx(1.0)
    assert row["worked_mfe_1h_top_q80"] == 1
    assert row["safe_mae_1h_le_50bps"] == 0
    assert row["safe_mae_1h_le_100bps"] == 1
    assert row["worked_mfeatr_1h_ge_0p5"] == 1
    assert row["worked_mfeatr_1h_ge_1"] == 1
    assert row["safe_maeatr_1h_le_0p5"] == 0
    assert row["safe_maeatr_1h_le_1"] == 1
    assert row["good_1h_mfe1_mae1"] == 1
    assert row["mfeatr_pct_1h"] == pytest.approx(1.0)
    assert row["worked_mfeatr_1h_top_q80"] == 1

    assert row["is_truncated_5h"] == 1
    assert pd.isna(row["ret_5h"])
    assert pd.isna(row["logret_5h"])
    assert pd.isna(row["mfe_5h"])
    assert pd.isna(row["mae_5h"])
    assert pd.isna(row["ret_atr_5h"])
    assert pd.isna(row["mfe_atr_5h"])
    assert pd.isna(row["mae_atr_5h"])
    assert pd.isna(row["rr_like_5h"])
    assert pd.isna(row["worked_5h_ge_0bps"])
    assert pd.isna(row["ret_pct_5h"])
    assert pd.isna(row["worked_5h_top_q80"])
    assert pd.isna(row["worked_mfe_5h_ge_100bps"])
    assert pd.isna(row["mfe_pct_5h"])
    assert pd.isna(row["worked_mfe_5h_top_q80"])
    assert pd.isna(row["safe_mae_5h_le_100bps"])
    assert pd.isna(row["worked_mfeatr_5h_ge_1"])
    assert pd.isna(row["safe_maeatr_5h_le_1"])
    assert pd.isna(row["good_5h_mfe1_mae1"])
    assert pd.isna(row["mfeatr_pct_5h"])
    assert pd.isna(row["worked_mfeatr_5h_top_q80"])

    summary = validate_forward_outcomes(
        forward_df=out,
        horizons_hours=[1, 5],
        rr_mult=2.0,
        emit_resolved=True,
        mode="time_only",
    )
    assert len(summary) == 2
    s1 = [s for s in summary if s["horizon_h"] == 1.0][0]
    s5 = [s for s in summary if s["horizon_h"] == 5.0][0]
    assert s1["n_valid"] == 1.0
    assert s1["up_rate"] == pytest.approx(1.0)
    assert s5["n_valid"] == 0.0
    assert math.isnan(s5["mean_ret"])
