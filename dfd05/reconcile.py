from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class ReconcileSummary:
    model_trades: int
    tv_trades: int
    matched: int
    unmatched_model: int
    unmatched_tv: int
    mean_abs_price_diff: float

    def as_dict(self) -> Dict[str, float | int]:
        return {
            "model_trades": self.model_trades,
            "tv_trades": self.tv_trades,
            "matched": self.matched,
            "unmatched_model": self.unmatched_model,
            "unmatched_tv": self.unmatched_tv,
            "mean_abs_price_diff": self.mean_abs_price_diff,
        }


def reconcile_with_tradingview_csv(
    events: pd.DataFrame,
    tv_csv_path: str,
    tv_time_col: str = "entry_time",
    tv_price_col: str = "entry_price",
    time_tolerance_seconds: int = 0,
    price_tolerance: float = 1e-8,
) -> ReconcileSummary:
    if events.empty:
        return ReconcileSummary(0, 0, 0, 0, 0, np.nan)

    tv = pd.read_csv(tv_csv_path)
    if tv_time_col not in tv.columns or tv_price_col not in tv.columns:
        raise ValueError(
            f"TradingView CSV must contain columns '{tv_time_col}' and '{tv_price_col}'."
        )

    tv_t = pd.to_datetime(tv[tv_time_col], utc=True).view("int64").to_numpy() // 1_000_000
    tv_p = pd.to_numeric(tv[tv_price_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(tv_p)
    tv_t = tv_t[valid]
    tv_p = tv_p[valid]

    model_t = events["entry_time_ms"].to_numpy(dtype=np.int64)
    model_p = events["entry_price"].to_numpy(dtype=float)

    used_tv = np.zeros(len(tv_t), dtype=bool)
    matched = 0
    price_diffs = []

    tol_ms = int(time_tolerance_seconds * 1000)
    for i in range(len(model_t)):
        t0 = model_t[i]
        p0 = model_p[i]
        idx = np.where(~used_tv & (np.abs(tv_t - t0) <= tol_ms))[0]
        if idx.size == 0:
            continue
        diffs = np.abs(tv_p[idx] - p0)
        j_local = int(np.argmin(diffs))
        j = int(idx[j_local])
        if diffs[j_local] <= price_tolerance:
            used_tv[j] = True
            matched += 1
            price_diffs.append(float(diffs[j_local]))

    summary = ReconcileSummary(
        model_trades=len(model_t),
        tv_trades=len(tv_t),
        matched=matched,
        unmatched_model=len(model_t) - matched,
        unmatched_tv=int((~used_tv).sum()),
        mean_abs_price_diff=float(np.mean(price_diffs)) if price_diffs else np.nan,
    )
    return summary

