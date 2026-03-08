from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Pine16SessionSpec:
    useSessionGate: bool = True
    tz: str = "Etc/GMT-3"
    useNY: bool = True
    useLondon: bool = False
    useTokyo: bool = False
    useSydney: bool = False


SESSION_HM = {
    "ny": (1630, 2300),
    "london": (1100, 1930),
    "tokyo": (300, 1200),
    "sydney": (100, 1000),
}


def in_hm_range_inclusive(hm: int, start_hm: int, end_hm: int) -> bool:
    start = int(start_hm)
    end = int(end_hm)
    v = int(hm)
    if start <= end:
        return start <= v <= end
    return (v >= start) or (v <= end)


def hm_from_utc(ts_utc: pd.Timestamp, tz: str) -> int:
    ts = pd.Timestamp(ts_utc)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    loc = ts.tz_convert(tz)
    return int(loc.hour * 100 + loc.minute)


def session_ok(ts_utc: pd.Timestamp, spec: Pine16SessionSpec) -> bool:
    if not bool(spec.useSessionGate):
        return True
    selected = []
    if spec.useNY:
        selected.append("ny")
    if spec.useLondon:
        selected.append("london")
    if spec.useTokyo:
        selected.append("tokyo")
    if spec.useSydney:
        selected.append("sydney")
    if not selected:
        return True

    hm = hm_from_utc(ts_utc, spec.tz)
    for sess in selected:
        sh, eh = SESSION_HM[sess]
        if in_hm_range_inclusive(hm, sh, eh):
            return True
    return False

