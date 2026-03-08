from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SessionGateSpec:
    enabled: bool
    tz: str = "Etc/GMT-3"
    ny: bool = True
    london: bool = False
    tokyo: bool = False
    sydney: bool = False


_SESSION_HM: dict[str, tuple[int, int]] = {
    "ny": (1630, 2300),
    "london": (1100, 1930),
    "tokyo": (300, 1200),
    "sydney": (100, 1000),
}


def _to_hm(ts_utc: pd.Timestamp, tz: str) -> int:
    ts = pd.Timestamp(ts_utc)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    local = ts.tz_convert(tz)
    return int(local.hour * 100 + local.minute)


def hm_from_utc_series(times_utc: Iterable[pd.Timestamp] | pd.Series, tz: str) -> np.ndarray:
    ser = pd.to_datetime(pd.Series(times_utc), utc=True)
    local = ser.dt.tz_convert(tz)
    return (local.dt.hour.to_numpy(dtype=int) * 100) + local.dt.minute.to_numpy(dtype=int)


def in_hm_range(hm: int, start_hm: int, end_hm: int) -> bool:
    start = int(start_hm)
    end = int(end_hm)
    v = int(hm)
    if start <= end:
        return start <= v <= end
    return (v >= start) or (v <= end)


def _selected_sessions(spec: SessionGateSpec) -> list[str]:
    out: list[str] = []
    if spec.ny:
        out.append("ny")
    if spec.london:
        out.append("london")
    if spec.tokyo:
        out.append("tokyo")
    if spec.sydney:
        out.append("sydney")
    return out


def session_ok_hm(hm: int, spec: SessionGateSpec) -> bool:
    if not spec.enabled:
        return True
    selected = _selected_sessions(spec)
    if not selected:
        return True
    for name in selected:
        start_hm, end_hm = _SESSION_HM[name]
        if in_hm_range(hm=hm, start_hm=start_hm, end_hm=end_hm):
            return True
    return False


def session_ok_at_pivot(event_pivot_timestamp_utc: pd.Timestamp, spec: SessionGateSpec) -> bool:
    hm = _to_hm(pd.Timestamp(event_pivot_timestamp_utc), tz=spec.tz)
    return session_ok_hm(hm=hm, spec=spec)


def session_ok_at_trigger(trigger_timestamp_utc: pd.Timestamp, spec: SessionGateSpec) -> bool:
    hm = _to_hm(pd.Timestamp(trigger_timestamp_utc), tz=spec.tz)
    return session_ok_hm(hm=hm, spec=spec)


def session_ok_mask(times_utc: pd.Series, spec: SessionGateSpec) -> np.ndarray:
    if not spec.enabled:
        return np.ones(len(times_utc), dtype=bool)
    selected = _selected_sessions(spec)
    if not selected:
        return np.ones(len(times_utc), dtype=bool)
    hm = hm_from_utc_series(times_utc, tz=spec.tz)
    out = np.zeros(len(hm), dtype=bool)
    for name in selected:
        start_hm, end_hm = _SESSION_HM[name]
        if start_hm <= end_hm:
            out |= (hm >= start_hm) & (hm <= end_hm)
        else:
            out |= (hm >= start_hm) | (hm <= end_hm)
    return out
