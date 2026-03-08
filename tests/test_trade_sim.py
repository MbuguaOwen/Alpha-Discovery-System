import pandas as pd

from dfd05.trade_sim import select_executed_trades


def test_select_executed_trades_one_trade_at_a_time_by_horizon():
    events = pd.DataFrame(
        [
            {"symbol": "A", "entry_index": 10, "event_time_ms": 1},
            {"symbol": "A", "entry_index": 12, "event_time_ms": 2},
            {"symbol": "A", "entry_index": 20, "event_time_ms": 3},
            {"symbol": "B", "entry_index": 5, "event_time_ms": 1},
            {"symbol": "B", "entry_index": 6, "event_time_ms": 2},
        ]
    )

    selected = select_executed_trades(
        events_df=events,
        tf_minutes=60,
        horizons_hours=[1, 4],
        one_trade_at_a_time=True,
        cooldown_bars=0,
    )

    assert sorted(selected.keys()) == [1, 4]
    assert len(selected[1]) == 4  # A:10,12,20 and B:5
    assert len(selected[4]) == 3  # A:10,20 and B:5

    sel_h1_a = selected[1][selected[1]["symbol"] == "A"]["entry_index"].tolist()
    sel_h4_a = selected[4][selected[4]["symbol"] == "A"]["entry_index"].tolist()
    assert sel_h1_a == [10, 12, 20]
    assert sel_h4_a == [10, 20]


def test_select_executed_trades_returns_all_when_disabled():
    events = pd.DataFrame(
        [
            {"symbol": "A", "entry_index": 10},
            {"symbol": "A", "entry_index": 11},
            {"symbol": "A", "entry_index": 12},
        ]
    )
    selected = select_executed_trades(
        events_df=events,
        tf_minutes=60,
        horizons_hours=[1, 4],
        one_trade_at_a_time=False,
        cooldown_bars=5,
    )
    assert len(selected[1]) == 3
    assert len(selected[4]) == 3
