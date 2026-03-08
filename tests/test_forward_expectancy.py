import pandas as pd
import pytest

from dfd05.forward import validate_forward_outcomes


def test_forward_expectancy_with_no_hit_uses_total_and_hit_only():
    forward_df = pd.DataFrame(
        [
            {
                "tp_first_24h": 1.0,
                "sl_first_24h": 0.0,
                "both_samebar_24h": 0.0,
                "no_hit_24h": 0.0,
                "is_truncated_24h": 0,
                "mfe_24h": 0.03,
                "mae_24h": -0.01,
                "tp_first_resolved_24h": 1.0,
                "sl_first_resolved_24h": 0.0,
            },
            {
                "tp_first_24h": 0.0,
                "sl_first_24h": 1.0,
                "both_samebar_24h": 0.0,
                "no_hit_24h": 0.0,
                "is_truncated_24h": 0,
                "mfe_24h": 0.01,
                "mae_24h": -0.02,
                "tp_first_resolved_24h": 0.0,
                "sl_first_resolved_24h": 1.0,
            },
            {
                "tp_first_24h": 0.0,
                "sl_first_24h": 0.0,
                "both_samebar_24h": 0.0,
                "no_hit_24h": 1.0,
                "is_truncated_24h": 0,
                "mfe_24h": 0.005,
                "mae_24h": -0.006,
                "tp_first_resolved_24h": 0.0,
                "sl_first_resolved_24h": 0.0,
            },
        ]
    )

    summary = validate_forward_outcomes(
        forward_df=forward_df,
        horizons_hours=[24],
        rr_mult=3.0,
        emit_resolved=True,
        mode="barrier",
    )
    assert len(summary) == 1
    s = summary[0]

    assert s["tp_rate"] == pytest.approx(1.0 / 3.0)
    assert s["sl_rate"] == pytest.approx(1.0 / 3.0)
    assert s["no_hit_rate"] == pytest.approx(1.0 / 3.0)
    assert s["tp_resolved_rate"] == pytest.approx(1.0 / 3.0)
    assert s["sl_resolved_rate"] == pytest.approx(1.0 / 3.0)
    assert s["hit_rate"] == pytest.approx(2.0 / 3.0)
    assert s["expR_total"] == pytest.approx(2.0 / 3.0)
    assert s["expR_hit_only"] == pytest.approx(1.0)
