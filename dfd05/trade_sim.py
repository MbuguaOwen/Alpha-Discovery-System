from __future__ import annotations

from typing import Dict, List

import pandas as pd


def _horizon_to_bars(hours: int, tf_minutes: int) -> int:
    if tf_minutes <= 0:
        raise ValueError(f"Invalid tf_minutes={tf_minutes}")
    total_minutes = int(hours) * 60
    bars = total_minutes // int(tf_minutes)
    if bars <= 0:
        raise ValueError(f"horizon={hours}h yields zero bars for tf_minutes={tf_minutes}")
    return int(bars)


def select_executed_trades(
    events_df: pd.DataFrame,
    tf_minutes: int,
    horizons_hours: List[int],
    one_trade_at_a_time: bool,
    cooldown_bars: int,
) -> Dict[int, pd.DataFrame]:
    horizons = sorted(set(int(h) for h in horizons_hours if int(h) > 0))
    if not horizons:
        return {}

    if events_df.empty:
        return {h: events_df.copy() for h in horizons}

    required = ["symbol", "entry_index"]
    missing = [c for c in required if c not in events_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for trade simulation: {missing}")

    sort_cols = ["symbol", "entry_index"]
    for c in ["event_time_ms", "entry_time_ms", "signal_id"]:
        if c in events_df.columns:
            sort_cols.append(c)
    base = events_df.copy().sort_values(sort_cols).reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    base["entry_index"] = pd.to_numeric(base["entry_index"], errors="coerce")
    if base["entry_index"].isna().any():
        raise ValueError("entry_index contains NaN/non-numeric values.")
    base["entry_index"] = base["entry_index"].astype("int64")

    if not bool(one_trade_at_a_time):
        return {h: base.drop(columns=["_orig_idx"]).copy().reset_index(drop=True) for h in horizons}

    cooldown = max(0, int(cooldown_bars))
    out: Dict[int, pd.DataFrame] = {}
    for h in horizons:
        horizon_bars = _horizon_to_bars(hours=int(h), tf_minutes=int(tf_minutes))
        keep_orig_idx: List[int] = []

        for _sym, grp in base.groupby("symbol", sort=False):
            current_exit = -10**18
            for _, row in grp.iterrows():
                entry_index = int(row["entry_index"])
                if entry_index <= current_exit:
                    continue
                keep_orig_idx.append(int(row["_orig_idx"]))
                current_exit = int(entry_index + horizon_bars + cooldown)

        selected = events_df.loc[sorted(keep_orig_idx)].copy() if keep_orig_idx else events_df.iloc[0:0].copy()
        selected = selected.sort_values(sort_cols).reset_index(drop=True)
        out[int(h)] = selected
    return out
