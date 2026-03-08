from __future__ import annotations

import re
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from .forward import infer_forward_mode


FORWARD_KEY_COLS = [
    "symbol",
    "timeframe",
    "signal_id",
    "event_time_ms",
    "entry_time_ms",
    "entry_price",
    "pivot_time_ms",
    "pivot_price",
    "mode",
    "trade_mode",
    "toggles_json",
]


def horizon_suffix(hours: float | int) -> str:
    if float(hours).is_integer():
        return f"{int(hours)}h"
    return f"{str(hours).replace('.', 'p')}h"


def _suffix_to_hours(suffix: str) -> float:
    token = suffix.strip().lower()
    if not token.endswith("h"):
        raise ValueError(f"Invalid horizon suffix: {suffix}")
    val = float(token[:-1].replace("p", "."))
    if float(val).is_integer():
        return float(int(val))
    return float(val)


def infer_horizons_hours(columns: Iterable[str], mode: str = "auto") -> List[int]:
    mode_norm = (mode or "").strip().lower()
    if mode_norm == "auto":
        mode_norm = infer_forward_mode(list(columns))

    found: List[float] = []
    if mode_norm == "time_only":
        pat = re.compile(r"^ret_(?P<sfx>\d+(?:p\d+)?h)$")
    else:
        pat = re.compile(r"^tp_first_(?P<sfx>\d+(?:p\d+)?h)$")

    for col in columns:
        m = pat.match(col)
        if not m:
            continue
        found.append(_suffix_to_hours(m.group("sfx")))
    uniq = sorted(set(found))
    out: List[int] = []
    for h in uniq:
        if float(h).is_integer() and int(h) > 0:
            out.append(int(h))
    return out


def merge_events_forward(events_df: pd.DataFrame, forward_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty and forward_df.empty:
        return pd.DataFrame()
    if events_df.empty:
        return forward_df.copy()
    if forward_df.empty:
        return events_df.copy()

    left = events_df.copy()
    right = forward_df.copy()
    right_drop = [c for c in right.columns if c in left.columns and c not in FORWARD_KEY_COLS]
    right = right.drop(columns=right_drop, errors="ignore")
    return left.merge(right, on=FORWARD_KEY_COLS, how="inner")


def _derive_year(frame: pd.DataFrame) -> pd.Series:
    if "year" in frame.columns:
        year = pd.to_numeric(frame["year"], errors="coerce")
        return year.astype("Int64")
    if "event_time_ms" in frame.columns:
        dt = pd.to_datetime(frame["event_time_ms"], unit="ms", utc=True, errors="coerce")
        return dt.dt.year.astype("Int64")
    if "time" in frame.columns:
        dt = pd.to_datetime(frame["time"], utc=True, errors="coerce")
        return dt.dt.year.astype("Int64")
    return pd.Series(pd.array([pd.NA] * len(frame), dtype="Int64"), index=frame.index)


def _horizon_metrics_barrier(frame: pd.DataFrame, hours: int, rr_mult: float) -> dict[str, float]:
    suffix = horizon_suffix(hours)
    tp_col = f"tp_first_{suffix}"
    sl_col = f"sl_first_{suffix}"
    same_col = f"both_samebar_{suffix}"
    nh_col = f"no_hit_{suffix}"
    trunc_col = f"is_truncated_{suffix}"
    tp_res_col = f"tp_first_resolved_{suffix}"
    sl_res_col = f"sl_first_resolved_{suffix}"

    for c in [tp_col, sl_col, same_col, nh_col, trunc_col]:
        if c not in frame.columns:
            raise ValueError(f"Missing required dashboard column: {c}")

    n = int(len(frame))
    if n == 0:
        return {
            "n": 0.0,
            "tp_rate": np.nan,
            "sl_rate": np.nan,
            "samebar_rate": np.nan,
            "no_hit_rate": np.nan,
            "tp_resolved_rate": np.nan,
            "sl_resolved_rate": np.nan,
            "hit_rate": np.nan,
            "trunc_rate": np.nan,
            "truncated_rate": np.nan,
            "expR_total": np.nan,
            "expR_hit_only": np.nan,
        }

    trunc = frame[trunc_col].fillna(0).to_numpy(dtype=np.int8)
    non_mask = trunc == 0
    trunc_rate = float(np.mean(trunc))

    if non_mask.any():
        tp_rate = float(frame.loc[non_mask, tp_col].mean())
        sl_rate = float(frame.loc[non_mask, sl_col].mean())
        same_rate = float(frame.loc[non_mask, same_col].mean())
        nh_rate = float(frame.loc[non_mask, nh_col].mean())
        hit_rate = float(1.0 - nh_rate)
        if tp_res_col in frame.columns and sl_res_col in frame.columns:
            tp_res = float(frame.loc[non_mask, tp_res_col].mean())
            sl_res = float(frame.loc[non_mask, sl_res_col].mean())
        else:
            tp_res = tp_rate
            sl_res = sl_rate
        exp_total = float(rr_mult * tp_res - sl_res)
        if hit_rate > 0:
            tp_hit = float(tp_res / hit_rate)
            sl_hit = float(sl_res / hit_rate)
            exp_hit_only = float(rr_mult * tp_hit - sl_hit)
        else:
            exp_hit_only = np.nan
    else:
        tp_rate = sl_rate = same_rate = nh_rate = np.nan
        tp_res = sl_res = hit_rate = exp_total = exp_hit_only = np.nan

    return {
        "n": float(n),
        "tp_rate": tp_rate,
        "sl_rate": sl_rate,
        "samebar_rate": same_rate,
        "no_hit_rate": nh_rate,
        "tp_resolved_rate": tp_res,
        "sl_resolved_rate": sl_res,
        "hit_rate": hit_rate,
        "trunc_rate": trunc_rate,
        "truncated_rate": trunc_rate,
        "expR_total": exp_total,
        "expR_hit_only": exp_hit_only,
    }


def _horizon_metrics_time_only(frame: pd.DataFrame, hours: int) -> dict[str, float]:
    suffix = horizon_suffix(hours)
    ret_col = f"ret_{suffix}"
    up_col = f"up_{suffix}"
    mfe_col = f"mfe_{suffix}"
    mae_col = f"mae_{suffix}"
    trunc_col = f"is_truncated_{suffix}"
    for c in [ret_col, up_col, mfe_col, mae_col, trunc_col]:
        if c not in frame.columns:
            raise ValueError(f"Missing required time-only dashboard column: {c}")

    n_total = int(len(frame))
    if n_total == 0:
        return {
            "n_total": 0.0,
            "n_valid": 0.0,
            "trunc_rate": np.nan,
            "mean_ret": np.nan,
            "median_ret": np.nan,
            "up_rate": np.nan,
            "mean_mfe": np.nan,
            "mean_mae": np.nan,
            "p25_ret": np.nan,
            "p50_ret": np.nan,
            "p75_ret": np.nan,
        }

    trunc = frame[trunc_col].fillna(0).to_numpy(dtype=np.int8)
    valid = trunc == 0
    n_valid = int(valid.sum())
    trunc_rate = float(np.mean(trunc))

    if n_valid > 0:
        ret = frame.loc[valid, ret_col].astype(float)
        mean_ret = float(ret.mean())
        median_ret = float(ret.median())
        up_rate = float(frame.loc[valid, up_col].astype(float).mean())
        mean_mfe = float(frame.loc[valid, mfe_col].astype(float).mean())
        mean_mae = float(frame.loc[valid, mae_col].astype(float).mean())
        p25 = float(ret.quantile(0.25))
        p50 = float(ret.quantile(0.50))
        p75 = float(ret.quantile(0.75))
    else:
        mean_ret = median_ret = up_rate = mean_mfe = mean_mae = p25 = p50 = p75 = np.nan

    return {
        "n_total": float(n_total),
        "n_valid": float(n_valid),
        "trunc_rate": trunc_rate,
        "mean_ret": mean_ret,
        "median_ret": median_ret,
        "up_rate": up_rate,
        "mean_mfe": mean_mfe,
        "mean_mae": mean_mae,
        "p25_ret": p25,
        "p50_ret": p50,
        "p75_ret": p75,
    }


def _build_barrier_dashboard(
    frame: pd.DataFrame,
    horizons: List[int],
    rr_mult: float,
) -> pd.DataFrame:
    with_year = frame.copy()
    with_year["_year"] = _derive_year(with_year)

    overall_rows: list[dict[str, object]] = []
    symbol_year_rows: list[dict[str, object]] = []
    for h in horizons:
        overall = _horizon_metrics_barrier(with_year, hours=h, rr_mult=rr_mult)
        overall_rows.append(
            {
                "forward_mode": "barrier",
                "section": "overall",
                "horizon_h": int(h),
                "symbol": None,
                "year": None,
                **overall,
                "worst_year_expR": np.nan,
                "overall_expR": np.nan,
                "rank": np.nan,
            }
        )

        grouped = with_year.groupby(["symbol", "_year"], dropna=False)
        for (sym, yr), g in grouped:
            met = _horizon_metrics_barrier(g, hours=h, rr_mult=rr_mult)
            symbol_year_rows.append(
                {
                    "forward_mode": "barrier",
                    "section": "per_symbol_year",
                    "horizon_h": int(h),
                    "symbol": str(sym) if pd.notna(sym) else None,
                    "year": int(yr) if pd.notna(yr) else None,
                    **met,
                    "worst_year_expR": np.nan,
                    "overall_expR": np.nan,
                    "rank": np.nan,
                }
            )

    overall_df = pd.DataFrame(overall_rows)
    sy_df = pd.DataFrame(symbol_year_rows)
    worst_year = (
        sy_df.groupby("horizon_h", dropna=False)["expR_total"].min().rename("worst_year_expR").reset_index()
    )
    rank_df = overall_df[["horizon_h", "expR_total"]].rename(columns={"expR_total": "overall_expR"})
    rank_df = rank_df.merge(worst_year, on="horizon_h", how="left")
    rank_df = rank_df.sort_values(
        ["worst_year_expR", "overall_expR", "horizon_h"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    rank_df["rank"] = np.arange(1, len(rank_df) + 1, dtype=int)
    rank_df["forward_mode"] = "barrier"
    rank_df["section"] = "ranking"
    rank_df["symbol"] = None
    rank_df["year"] = None
    rank_df["n"] = np.nan
    rank_df["tp_rate"] = np.nan
    rank_df["sl_rate"] = np.nan
    rank_df["samebar_rate"] = np.nan
    rank_df["no_hit_rate"] = np.nan
    rank_df["tp_resolved_rate"] = np.nan
    rank_df["sl_resolved_rate"] = np.nan
    rank_df["hit_rate"] = np.nan
    rank_df["trunc_rate"] = np.nan
    rank_df["truncated_rate"] = np.nan
    rank_df["expR_total"] = np.nan
    rank_df["expR_hit_only"] = np.nan

    out = pd.concat([overall_df, sy_df, rank_df], ignore_index=True, sort=False)
    ordered = [
        "forward_mode",
        "section",
        "horizon_h",
        "symbol",
        "year",
        "n",
        "tp_rate",
        "sl_rate",
        "samebar_rate",
        "no_hit_rate",
        "tp_resolved_rate",
        "sl_resolved_rate",
        "hit_rate",
        "trunc_rate",
        "truncated_rate",
        "expR_total",
        "expR_hit_only",
        "worst_year_expR",
        "overall_expR",
        "rank",
    ]
    return out[ordered]


def _build_time_only_dashboard(
    frame: pd.DataFrame,
    horizons: List[int],
) -> pd.DataFrame:
    with_year = frame.copy()
    with_year["_year"] = _derive_year(with_year)

    overall_rows: list[dict[str, object]] = []
    symbol_year_rows: list[dict[str, object]] = []
    for h in horizons:
        met = _horizon_metrics_time_only(with_year, hours=h)
        overall_rows.append(
            {
                "forward_mode": "time_only",
                "section": "overall",
                "horizon_h": int(h),
                "symbol": None,
                "year": None,
                **met,
                "rank": np.nan,
            }
        )
        grouped = with_year.groupby(["symbol", "_year"], dropna=False)
        for (sym, yr), g in grouped:
            gm = _horizon_metrics_time_only(g, hours=h)
            symbol_year_rows.append(
                {
                    "forward_mode": "time_only",
                    "section": "per_symbol_year",
                    "horizon_h": int(h),
                    "symbol": str(sym) if pd.notna(sym) else None,
                    "year": int(yr) if pd.notna(yr) else None,
                    **gm,
                    "rank": np.nan,
                }
            )

    overall_df = pd.DataFrame(overall_rows)
    sy_df = pd.DataFrame(symbol_year_rows)

    year_group = sy_df.groupby(["horizon_h", "year"], dropna=False).agg(
        mean_ret=("mean_ret", "mean"),
        up_rate=("up_rate", "mean"),
    ).reset_index()
    worst_year = (
        year_group.groupby("horizon_h", dropna=False).agg(
            worst_year_mean_ret=("mean_ret", "min"),
            worst_year_up_rate=("up_rate", "min"),
        ).reset_index()
    )
    overall_rank = overall_df[["horizon_h", "mean_ret", "up_rate"]].rename(
        columns={"mean_ret": "overall_mean_ret", "up_rate": "overall_up_rate"}
    )
    rank_df = overall_rank.merge(worst_year, on="horizon_h", how="left")
    rank_df = rank_df.sort_values(
        ["worst_year_mean_ret", "overall_mean_ret", "horizon_h"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    rank_df["rank"] = np.arange(1, len(rank_df) + 1, dtype=int)
    rank_df["forward_mode"] = "time_only"
    rank_df["section"] = "ranking"
    rank_df["symbol"] = None
    rank_df["year"] = None
    rank_df["n_total"] = np.nan
    rank_df["n_valid"] = np.nan
    rank_df["trunc_rate"] = np.nan
    rank_df["mean_ret"] = np.nan
    rank_df["median_ret"] = np.nan
    rank_df["up_rate"] = np.nan
    rank_df["mean_mfe"] = np.nan
    rank_df["mean_mae"] = np.nan
    rank_df["p25_ret"] = np.nan
    rank_df["p50_ret"] = np.nan
    rank_df["p75_ret"] = np.nan

    overall_df = overall_df.merge(worst_year, on="horizon_h", how="left")
    overall_df = overall_df.merge(overall_rank, on="horizon_h", how="left")
    sy_df = sy_df.merge(worst_year, on="horizon_h", how="left")
    sy_df = sy_df.merge(overall_rank, on="horizon_h", how="left")
    out = pd.concat([overall_df, sy_df, rank_df], ignore_index=True, sort=False)
    ordered = [
        "forward_mode",
        "section",
        "horizon_h",
        "symbol",
        "year",
        "n_total",
        "n_valid",
        "trunc_rate",
        "mean_ret",
        "median_ret",
        "up_rate",
        "mean_mfe",
        "mean_mae",
        "p25_ret",
        "p50_ret",
        "p75_ret",
        "worst_year_mean_ret",
        "worst_year_up_rate",
        "overall_mean_ret",
        "overall_up_rate",
        "rank",
    ]
    return out[ordered]


def build_horizon_dashboard(
    frame: pd.DataFrame,
    horizons_hours: Optional[List[int]] = None,
    rr_mult: float = 2.0,
    mode: str = "auto",
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    mode_norm = (mode or "").strip().lower()
    if mode_norm == "auto":
        mode_norm = infer_forward_mode(frame.columns)

    if horizons_hours is None:
        horizons = infer_horizons_hours(frame.columns, mode=mode_norm)
    else:
        horizons = sorted(set(int(h) for h in horizons_hours if int(h) > 0))
    if not horizons:
        raise ValueError("No horizons found for dashboard.")

    if mode_norm == "time_only":
        return _build_time_only_dashboard(frame=frame, horizons=horizons)
    return _build_barrier_dashboard(frame=frame, horizons=horizons, rr_mult=rr_mult)
