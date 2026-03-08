import pandas as pd

from dfd05.config import RunConfig, SessionGateConfig, StrategyConfig
from dfd05.session_gate import SessionGateSpec, in_hm_range, session_ok_at_pivot, session_ok_at_trigger
from dfd05.strategy import build_session_ok_mask, extract_dfd05_events


def test_session_gate_parity_utc_to_utc3_boundaries():
    times = pd.Series(
        pd.to_datetime(
            [
                "2024-01-01T13:29:00Z",  # 16:29 UTC+3
                "2024-01-01T13:30:00Z",  # 16:30 UTC+3
                "2024-01-01T19:59:00Z",  # 22:59 UTC+3
                "2024-01-01T20:00:00Z",  # 23:00 UTC+3
            ],
            utc=True,
        )
    )
    cfg = RunConfig(
        strategy=StrategyConfig(
            session_gate=SessionGateConfig(
                enabled=True,
                tz="UTC+3",
                ny=True,
                london=False,
                tokyo=False,
                sydney=False,
            )
        )
    )
    got = build_session_ok_mask(times=times, config=cfg).tolist()
    assert got == [False, True, True, True]

    cfg_all = RunConfig(
        strategy=StrategyConfig(
            session_gate=SessionGateConfig(
                enabled=True,
                tz="UTC+3",
                ny=False,
                london=False,
                tokyo=False,
                sydney=False,
            )
        )
    )
    got_all = build_session_ok_mask(times=times, config=cfg_all).tolist()
    assert got_all == [True, True, True, True]


def test_session_gate_hm_helpers_with_utc3_clock():
    spec = SessionGateSpec(enabled=True, tz="Etc/GMT-3", ny=True, london=False, tokyo=False, sydney=False)
    assert in_hm_range(1630, 1630, 2300) is True
    assert in_hm_range(2259, 1630, 2300) is True
    assert in_hm_range(2300, 1630, 2300) is True

    # UTC 13:30 => UTC+3 16:30, inside NY.
    ts_in = pd.Timestamp("2024-01-01T13:30:00Z")
    # UTC 20:00 => UTC+3 23:00, inside NY (end inclusive).
    ts_out = pd.Timestamp("2024-01-01T20:00:00Z")
    assert session_ok_at_pivot(ts_in, spec) is True
    assert session_ok_at_trigger(ts_in, spec) is True
    assert session_ok_at_pivot(ts_out, spec) is True
    assert session_ok_at_trigger(ts_out, spec) is True


def test_raw_mode_uses_pivot_time_for_session_gate():
    bars = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01T12:00:00Z", periods=8, freq="15min", tz="UTC"),
            "open": [10.0] * 8,
            "high": [11.0] * 8,
            "low": [10.0, 9.0, 10.0, 8.0, 10.0, 7.0, 10.0, 10.0],
            "close": [10.0, 9.0, 10.0, 9.0, 10.0, 11.0, 10.0, 10.0],
            "volume": [1.0] * 8,
            "symbol": ["XAUUSD"] * 8,
        }
    )

    cfg_off = RunConfig(
        symbols=["XAUUSD"],
        timeframe="m15",
        strategy=StrategyConfig(
            don_len=2,
            pivot_len=1,
            osc_len=1,
            ext_band_pct=1.0,
            warmup_bars=0,
            mode="RAW",
            trade_mode="BASELINE_ALL",
            session_gate=SessionGateConfig(enabled=False),
        ),
    )
    ev_off = extract_dfd05_events(bars, symbol="XAUUSD", timeframe="m15", config=cfg_off)
    assert len(ev_off) == 1

    cfg_on = RunConfig(
        symbols=["XAUUSD"],
        timeframe="m15",
        strategy=StrategyConfig(
            don_len=2,
            pivot_len=1,
            osc_len=1,
            ext_band_pct=1.0,
            warmup_bars=0,
            mode="RAW",
            trade_mode="BASELINE_ALL",
            session_gate=SessionGateConfig(
                enabled=True,
                tz="UTC+3",
                ny=True,
                london=False,
                tokyo=False,
                sydney=False,
            ),
        ),
    )
    ev_on = extract_dfd05_events(bars, symbol="XAUUSD", timeframe="m15", config=cfg_on)
    assert len(ev_on) == 0


def test_confirm_trigger_uses_current_bar_time_for_session_gate():
    bars = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01T07:15:00Z", periods=8, freq="15min", tz="UTC"),
            "open": [10.0] * 8,
            "high": [11.0] * 8,
            "low": [10.0, 9.0, 10.0, 8.0, 10.0, 7.0, 10.0, 10.0],
            "close": [10.0, 9.0, 10.0, 9.0, 10.0, 11.0, 10.0, 11.0],
            "volume": [1.0] * 8,
            "symbol": ["XAUUSD"] * 8,
        }
    )

    cfg_off = RunConfig(
        symbols=["XAUUSD"],
        timeframe="m15",
        strategy=StrategyConfig(
            don_len=2,
            pivot_len=1,
            osc_len=1,
            ext_band_pct=1.0,
            warmup_bars=0,
            mode="CONFIRM",
            use_bos_confirm=False,
            max_wait_bars=1,
            trade_mode="BASELINE_ALL",
            session_gate=SessionGateConfig(enabled=False),
        ),
    )
    ev_off = extract_dfd05_events(bars, symbol="XAUUSD", timeframe="m15", config=cfg_off)
    assert len(ev_off) == 1

    cfg_on = RunConfig(
        symbols=["XAUUSD"],
        timeframe="m15",
        strategy=StrategyConfig(
            don_len=2,
            pivot_len=1,
            osc_len=1,
            ext_band_pct=1.0,
            warmup_bars=0,
            mode="CONFIRM",
            use_bos_confirm=False,
            max_wait_bars=1,
            trade_mode="BASELINE_ALL",
            session_gate=SessionGateConfig(
                enabled=True,
                tz="UTC+3",
                ny=False,
                london=False,
                tokyo=True,
                sydney=False,
            ),
        ),
    )
    ev_on = extract_dfd05_events(bars, symbol="XAUUSD", timeframe="m15", config=cfg_on)
    # With inclusive session bounds, trigger at exact end-minute still passes.
    assert len(ev_on) == 1
