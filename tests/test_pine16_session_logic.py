import pandas as pd

from dfd05.pine16_session import Pine16SessionSpec, in_hm_range_inclusive, session_ok


def test_pine16_session_inclusive_end_boundary():
    spec = Pine16SessionSpec(useSessionGate=True, useNY=True, useLondon=False, useTokyo=False, useSydney=False)

    assert in_hm_range_inclusive(1630, 1630, 2300) is True
    assert in_hm_range_inclusive(2300, 1630, 2300) is True

    # UTC 20:00 == UTC+3 23:00 -> inclusive end should still pass.
    assert session_ok(pd.Timestamp("2024-01-01T20:00:00Z"), spec) is True
    assert session_ok(pd.Timestamp("2024-01-01T20:01:00Z"), spec) is False


def test_pine16_session_gate_true_with_no_region_selected_defaults_to_pass():
    spec = Pine16SessionSpec(useSessionGate=True, useNY=False, useLondon=False, useTokyo=False, useSydney=False)
    assert session_ok(pd.Timestamp("2024-01-01T00:00:00Z"), spec) is True

