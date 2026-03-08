import numpy as np

from dfd05.strategy import compute_pivot_low_confirmations


def test_pivot_low_confirms_after_right_bars():
    low = np.array([5.0, 4.0, 3.0, 4.0, 5.0], dtype=float)
    pivot_len = 2

    piv_idx, piv_price = compute_pivot_low_confirmations(low, pivot_len)

    # Pivot at index 2 can only be known at index 4 (2 bars to the right).
    assert np.all(piv_idx[:4] == -1)
    assert np.isnan(piv_price[:4]).all()
    assert piv_idx[4] == 2
    assert piv_price[4] == 3.0

