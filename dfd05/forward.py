from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import RunConfig


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


def _horizon_suffix(hours: float | int) -> str:
    if float(hours).is_integer():
        return f"{int(hours)}h"
    return f"{str(hours).replace('.', 'p')}h"


def _horizon_to_bars(
    hours: int,
    timeframe_minutes: int,
    warn_cb: Callable[[str], None] | None = None,
) -> int:
    total_minutes = int(hours) * 60
    if timeframe_minutes <= 0:
        raise ValueError(f"Invalid timeframe_minutes={timeframe_minutes}")
    bars = total_minutes // timeframe_minutes
    rem = total_minutes % timeframe_minutes
    if rem != 0 and warn_cb is not None:
        warn_cb(
            f"[forward] horizon={hours}h not divisible by tf={timeframe_minutes}m; "
            f"using floor bars={bars}."
        )
    return int(bars)


def infer_forward_mode(columns: List[str] | pd.Index) -> str:
    cols = list(columns)
    has_timeonly = any(c.startswith("ret_") and c.endswith("h") for c in cols) and any(
        c.startswith("up_") and c.endswith("h") for c in cols
    )
    if has_timeonly:
        return "time_only"
    return "barrier"


def _quantile_label_suffix(q: float) -> str:
    return f"q{int(round(100.0 * float(q)))}"


def _threshold_label(value: float) -> str:
    token = f"{float(value):.10g}"
    if token.startswith("-"):
        token = f"m{token[1:]}"
    return token.replace(".", "p")


def time_only_horizon_columns(
    hours: int,
    ret_thresholds_bps: List[int] | None = None,
    percentile_targets: List[float] | None = None,
    mfe_thresholds_bps: List[int] | None = None,
    mae_thresholds_bps: List[int] | None = None,
    mfe_percentile_targets: List[float] | None = None,
    mfe_atr_thresholds: List[float] | None = None,
    mae_atr_thresholds: List[float] | None = None,
    mfeatr_percentile_targets: List[float] | None = None,
    quality_mfe_threshold: float = 1.0,
    quality_mae_threshold: float = 1.0,
) -> List[str]:
    suffix = _horizon_suffix(hours)
    cols = [
        f"ret_{suffix}",
        f"logret_{suffix}",
        f"mfe_{suffix}",
        f"mae_{suffix}",
        f"ret_atr_{suffix}",
        f"mfe_atr_{suffix}",
        f"mae_atr_{suffix}",
        f"rr_like_{suffix}",
        f"max_dd_{suffix}",
        f"max_ru_{suffix}",
        f"up_{suffix}",
        f"dn_{suffix}",
        f"flat_{suffix}",
        f"is_truncated_{suffix}",
    ]
    for bps in sorted(set(int(x) for x in (ret_thresholds_bps or []))):
        cols.append(f"worked_{suffix}_ge_{int(bps)}bps")
    if percentile_targets:
        cols.append(f"ret_pct_{suffix}")
        for q in sorted(set(float(x) for x in percentile_targets)):
            if 0.0 < q <= 1.0:
                cols.append(f"worked_{suffix}_top_{_quantile_label_suffix(q)}")
    for bps in sorted(set(int(x) for x in (mfe_thresholds_bps or []))):
        cols.append(f"worked_mfe_{suffix}_ge_{int(bps)}bps")
    if mfe_percentile_targets:
        cols.append(f"mfe_pct_{suffix}")
        for q in sorted(set(float(x) for x in mfe_percentile_targets)):
            if 0.0 < q <= 1.0:
                cols.append(f"worked_mfe_{suffix}_top_{_quantile_label_suffix(q)}")
    for bps in sorted(set(int(x) for x in (mae_thresholds_bps or []))):
        cols.append(f"safe_mae_{suffix}_le_{int(bps)}bps")
    for thr in sorted(set(float(x) for x in (mfe_atr_thresholds or []))):
        if thr > 0.0:
            cols.append(f"worked_mfeatr_{suffix}_ge_{_threshold_label(thr)}")
    for thr in sorted(set(float(x) for x in (mae_atr_thresholds or []))):
        if thr > 0.0:
            cols.append(f"safe_maeatr_{suffix}_le_{_threshold_label(thr)}")
    cols.append(
        f"good_{suffix}_mfe{_threshold_label(quality_mfe_threshold)}_mae{_threshold_label(quality_mae_threshold)}"
    )
    if mfeatr_percentile_targets:
        cols.append(f"mfeatr_pct_{suffix}")
        for q in sorted(set(float(x) for x in mfeatr_percentile_targets)):
            if 0.0 < q <= 1.0:
                cols.append(f"worked_mfeatr_{suffix}_top_{_quantile_label_suffix(q)}")
    return cols


def barrier_horizon_columns(hours: int, emit_resolved: bool) -> List[str]:
    suffix = _horizon_suffix(hours)
    cols = [
        f"tp_first_{suffix}",
        f"sl_first_{suffix}",
        f"both_samebar_{suffix}",
        f"no_hit_{suffix}",
        f"is_truncated_{suffix}",
        f"mfe_{suffix}",
        f"mae_{suffix}",
    ]
    if emit_resolved:
        cols.extend([f"tp_first_resolved_{suffix}", f"sl_first_resolved_{suffix}"])
    return cols


def horizon_columns(
    hours: int,
    emit_resolved: bool = True,
    mode: str = "barrier",
    ret_thresholds_bps: List[int] | None = None,
    percentile_targets: List[float] | None = None,
    mfe_thresholds_bps: List[int] | None = None,
    mae_thresholds_bps: List[int] | None = None,
    mfe_percentile_targets: List[float] | None = None,
    mfe_atr_thresholds: List[float] | None = None,
    mae_atr_thresholds: List[float] | None = None,
    mfeatr_percentile_targets: List[float] | None = None,
    quality_mfe_threshold: float = 1.0,
    quality_mae_threshold: float = 1.0,
) -> List[str]:
    if (mode or "").strip().lower() == "time_only":
        return time_only_horizon_columns(
            hours=hours,
            ret_thresholds_bps=ret_thresholds_bps,
            percentile_targets=percentile_targets,
            mfe_thresholds_bps=mfe_thresholds_bps,
            mae_thresholds_bps=mae_thresholds_bps,
            mfe_percentile_targets=mfe_percentile_targets,
            mfe_atr_thresholds=mfe_atr_thresholds,
            mae_atr_thresholds=mae_atr_thresholds,
            mfeatr_percentile_targets=mfeatr_percentile_targets,
            quality_mfe_threshold=quality_mfe_threshold,
            quality_mae_threshold=quality_mae_threshold,
        )
    return barrier_horizon_columns(hours=hours, emit_resolved=emit_resolved)


def scan_forward_first_touch(
    high: np.ndarray,
    low: np.ndarray,
    entry_i: int,
    tp: float,
    sl: float,
    n_fwd_bars: int,
    entry_price: float,
) -> Tuple[str, int, float, float]:
    start = int(entry_i) + 1
    end = int(entry_i) + int(n_fwd_bars)

    if (
        n_fwd_bars <= 0
        or start < 0
        or end >= len(high)
        or not np.isfinite(tp)
        or not np.isfinite(sl)
        or not np.isfinite(entry_price)
        or entry_price == 0.0
    ):
        return "truncated", -1, np.nan, np.nan

    highs = high[start : end + 1]
    lows = low[start : end + 1]
    mfe = float(np.max((highs - entry_price) / entry_price))
    mae = float(np.min((lows - entry_price) / entry_price))

    for i in range(start, end + 1):
        hit_tp = high[i] >= tp
        hit_sl = low[i] <= sl
        if hit_tp and hit_sl:
            return "same", i, mfe, mae
        if hit_tp:
            return "tp", i, mfe, mae
        if hit_sl:
            return "sl", i, mfe, mae
    return "no_hit", -1, mfe, mae


def _resolved_from_outcome(outcome: str, tie_break: str) -> tuple[float, float]:
    if outcome == "tp":
        return 1.0, 0.0
    if outcome == "sl":
        return 0.0, 1.0
    if outcome == "same":
        return (1.0, 0.0) if tie_break == "tp" else (0.0, 1.0)
    if outcome == "no_hit":
        return 0.0, 0.0
    return np.nan, np.nan


def _compute_barrier_forward_for_events(
    bars_df: pd.DataFrame,
    events_df: pd.DataFrame,
    horizons_hours: List[int],
    tf_minutes: int,
    tie_break: str,
    emit_resolved: bool,
    sl_atr_mult: float,
    rr_mult: float,
    warn_cb: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()

    high = bars_df["high"].to_numpy(dtype=float)
    low = bars_df["low"].to_numpy(dtype=float)
    entry_i = events_df["entry_index"].to_numpy(dtype=np.int64)
    entry_price = events_df["entry_price"].to_numpy(dtype=float)
    atr_entry = events_df["atr_entry"].to_numpy(dtype=float)
    sl_price = entry_price - float(sl_atr_mult) * atr_entry
    tp_price = entry_price + float(sl_atr_mult) * float(rr_mult) * atr_entry

    out = events_df[
        [
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
    ].copy()
    out["sl_price"] = sl_price
    out["tp_price"] = tp_price

    n_events = len(events_df)
    tie_break_norm = tie_break.strip().lower()
    if tie_break_norm not in {"sl", "tp"}:
        tie_break_norm = "sl"

    for h in horizons_hours:
        bars_fwd = _horizon_to_bars(hours=h, timeframe_minutes=tf_minutes, warn_cb=warn_cb)
        suffix = _horizon_suffix(h)

        tp_first = np.full(n_events, np.nan, dtype=float)
        sl_first = np.full(n_events, np.nan, dtype=float)
        both_same = np.full(n_events, np.nan, dtype=float)
        no_hit = np.full(n_events, np.nan, dtype=float)
        is_truncated = np.zeros(n_events, dtype=np.int8)
        mfe = np.full(n_events, np.nan, dtype=float)
        mae = np.full(n_events, np.nan, dtype=float)
        if emit_resolved:
            tp_res = np.full(n_events, np.nan, dtype=float)
            sl_res = np.full(n_events, np.nan, dtype=float)

        for j in range(n_events):
            outcome, _touch_i, mfe_v, mae_v = scan_forward_first_touch(
                high=high,
                low=low,
                entry_i=int(entry_i[j]),
                tp=float(tp_price[j]),
                sl=float(sl_price[j]),
                n_fwd_bars=bars_fwd,
                entry_price=float(entry_price[j]),
            )
            if outcome == "truncated":
                is_truncated[j] = 1
                continue
            mfe[j] = mfe_v
            mae[j] = mae_v
            if outcome == "tp":
                tp_first[j], sl_first[j], both_same[j], no_hit[j] = 1.0, 0.0, 0.0, 0.0
            elif outcome == "sl":
                tp_first[j], sl_first[j], both_same[j], no_hit[j] = 0.0, 1.0, 0.0, 0.0
            elif outcome == "same":
                tp_first[j], sl_first[j], both_same[j], no_hit[j] = 0.0, 0.0, 1.0, 0.0
            else:
                tp_first[j], sl_first[j], both_same[j], no_hit[j] = 0.0, 0.0, 0.0, 1.0
            if emit_resolved:
                tpr, slr = _resolved_from_outcome(outcome=outcome, tie_break=tie_break_norm)
                tp_res[j] = tpr
                sl_res[j] = slr

        out[f"tp_first_{suffix}"] = tp_first
        out[f"sl_first_{suffix}"] = sl_first
        out[f"both_samebar_{suffix}"] = both_same
        out[f"no_hit_{suffix}"] = no_hit
        out[f"is_truncated_{suffix}"] = is_truncated
        out[f"mfe_{suffix}"] = mfe
        out[f"mae_{suffix}"] = mae
        if emit_resolved:
            out[f"tp_first_resolved_{suffix}"] = tp_res
            out[f"sl_first_resolved_{suffix}"] = sl_res
    return out.sort_values(["symbol", "event_time_ms", "entry_time_ms"]).reset_index(drop=True)


def _compute_time_only_forward_for_events(
    bars_df: pd.DataFrame,
    events_df: pd.DataFrame,
    horizons_hours: List[int],
    tf_minutes: int,
    ret_thresholds_bps: List[int],
    percentile_targets: List[float],
    mfe_thresholds_bps: List[int],
    mae_thresholds_bps: List[int],
    mfe_percentile_targets: List[float],
    mfe_atr_thresholds: List[float],
    mae_atr_thresholds: List[float],
    mfeatr_percentile_targets: List[float],
    quality_mfe_threshold: float,
    quality_mae_threshold: float,
    truncate_policy: str,
    warn_cb: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()
    if (truncate_policy or "").strip().lower() != "nan":
        raise ValueError("Only truncate_policy='nan' is currently supported.")

    high = bars_df["high"].to_numpy(dtype=float)
    low = bars_df["low"].to_numpy(dtype=float)
    close = bars_df["close"].to_numpy(dtype=float)
    entry_i = events_df["entry_index"].to_numpy(dtype=np.int64)
    entry_price = events_df["entry_price"].to_numpy(dtype=float)
    if "atr_entry" in events_df.columns:
        atr_entry = events_df["atr_entry"].to_numpy(dtype=float)
    else:
        atr_entry = np.full(len(events_df), np.nan, dtype=float)

    out = events_df[
        [
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
    ].copy()
    new_cols: Dict[str, np.ndarray] = {}

    n_events = len(events_df)
    symbols = events_df["symbol"].astype(str).to_numpy()
    thresholds = sorted(set(int(x) for x in ret_thresholds_bps))
    quantiles = sorted(set(float(q) for q in percentile_targets if 0.0 < float(q) <= 1.0))
    mfe_thresholds = sorted(set(int(x) for x in mfe_thresholds_bps))
    mae_thresholds = sorted(set(int(x) for x in mae_thresholds_bps))
    mfe_quantiles = sorted(set(float(q) for q in mfe_percentile_targets if 0.0 < float(q) <= 1.0))
    mfe_atr_thresholds_norm = sorted(set(float(x) for x in mfe_atr_thresholds if float(x) > 0.0))
    mae_atr_thresholds_norm = sorted(set(float(x) for x in mae_atr_thresholds if float(x) > 0.0))
    mfeatr_quantiles = sorted(
        set(float(q) for q in mfeatr_percentile_targets if 0.0 < float(q) <= 1.0)
    )
    quality_mfe_thr = float(quality_mfe_threshold)
    quality_mae_thr = float(quality_mae_threshold)

    for h in horizons_hours:
        bars_fwd = _horizon_to_bars(hours=h, timeframe_minutes=tf_minutes, warn_cb=warn_cb)
        suffix = _horizon_suffix(h)

        ret = np.full(n_events, np.nan, dtype=float)
        logret = np.full(n_events, np.nan, dtype=float)
        mfe = np.full(n_events, np.nan, dtype=float)
        mae = np.full(n_events, np.nan, dtype=float)
        ret_atr = np.full(n_events, np.nan, dtype=float)
        mfe_atr = np.full(n_events, np.nan, dtype=float)
        mae_atr = np.full(n_events, np.nan, dtype=float)
        rr_like = np.full(n_events, np.nan, dtype=float)
        max_dd = np.full(n_events, np.nan, dtype=float)
        max_ru = np.full(n_events, np.nan, dtype=float)
        up = np.full(n_events, np.nan, dtype=float)
        dn = np.full(n_events, np.nan, dtype=float)
        flat = np.full(n_events, np.nan, dtype=float)
        is_truncated = np.zeros(n_events, dtype=np.int8)

        for j in range(n_events):
            start = int(entry_i[j]) + 1
            end = int(entry_i[j]) + int(bars_fwd)
            ep = float(entry_price[j])
            if (
                bars_fwd <= 0
                or start < 0
                or end >= len(close)
                or not np.isfinite(ep)
                or ep == 0.0
            ):
                is_truncated[j] = 1
                continue

            highs = high[start : end + 1]
            lows = low[start : end + 1]
            c_end = float(close[end])

            r = float((c_end - ep) / ep)
            ret[j] = r
            if c_end > 0.0 and ep > 0.0:
                logret[j] = float(np.log(c_end / ep))
            ru = float(np.max((highs - ep) / ep))
            dd = float(np.min((lows - ep) / ep))
            mfe[j] = ru
            mae[j] = dd
            atr_e = float(atr_entry[j])
            if np.isfinite(atr_e) and atr_e != 0.0:
                ret_atr[j] = float((c_end - ep) / atr_e)
                mfe_atr[j] = float((float(np.max(highs)) - ep) / atr_e)
                mae_atr[j] = float((float(np.min(lows)) - ep) / atr_e)
                if np.isfinite(mae_atr[j]) and mae_atr[j] != 0.0 and np.isfinite(mfe_atr[j]):
                    rr_like[j] = float(mfe_atr[j] / abs(mae_atr[j]))
            max_ru[j] = ru
            max_dd[j] = dd
            if r > 0.0:
                up[j], dn[j], flat[j] = 1.0, 0.0, 0.0
            elif r < 0.0:
                up[j], dn[j], flat[j] = 0.0, 1.0, 0.0
            else:
                up[j], dn[j], flat[j] = 0.0, 0.0, 1.0

        new_cols[f"ret_{suffix}"] = ret
        new_cols[f"logret_{suffix}"] = logret
        new_cols[f"mfe_{suffix}"] = mfe
        new_cols[f"mae_{suffix}"] = mae
        new_cols[f"ret_atr_{suffix}"] = ret_atr
        new_cols[f"mfe_atr_{suffix}"] = mfe_atr
        new_cols[f"mae_atr_{suffix}"] = mae_atr
        new_cols[f"rr_like_{suffix}"] = rr_like
        new_cols[f"max_dd_{suffix}"] = max_dd
        new_cols[f"max_ru_{suffix}"] = max_ru
        new_cols[f"up_{suffix}"] = up
        new_cols[f"dn_{suffix}"] = dn
        new_cols[f"flat_{suffix}"] = flat
        new_cols[f"is_truncated_{suffix}"] = is_truncated

        valid = (is_truncated == 0) & np.isfinite(ret)
        for bps in thresholds:
            col = f"worked_{suffix}_ge_{int(bps)}bps"
            lab = np.full(n_events, np.nan, dtype=float)
            if valid.any():
                lab[valid] = (ret[valid] >= (float(bps) / 10000.0)).astype(float)
            new_cols[col] = lab

        if quantiles:
            ret_pct = np.full(n_events, np.nan, dtype=float)
            if valid.any():
                tmp = pd.DataFrame({"symbol": symbols, "ret": ret})
                idx_valid = np.flatnonzero(valid)
                valid_df = tmp.iloc[idx_valid].copy()
                ranks = valid_df.groupby("symbol")["ret"].rank(method="average", pct=True)
                ret_pct[idx_valid] = ranks.to_numpy(dtype=float)
            new_cols[f"ret_pct_{suffix}"] = ret_pct
            valid_pct = (is_truncated == 0) & np.isfinite(ret_pct)
            for q in quantiles:
                qlabel = _quantile_label_suffix(q)
                col = f"worked_{suffix}_top_{qlabel}"
                lab = np.full(n_events, np.nan, dtype=float)
                if valid_pct.any():
                    lab[valid_pct] = (ret_pct[valid_pct] >= q).astype(float)
                new_cols[col] = lab

        valid_mfe = (is_truncated == 0) & np.isfinite(mfe)
        for bps in mfe_thresholds:
            col = f"worked_mfe_{suffix}_ge_{int(bps)}bps"
            lab = np.full(n_events, np.nan, dtype=float)
            if valid_mfe.any():
                lab[valid_mfe] = (mfe[valid_mfe] >= (float(bps) / 10000.0)).astype(float)
            new_cols[col] = lab

        if mfe_quantiles:
            mfe_pct = np.full(n_events, np.nan, dtype=float)
            if valid_mfe.any():
                tmp = pd.DataFrame({"symbol": symbols, "mfe": mfe})
                idx_valid = np.flatnonzero(valid_mfe)
                valid_df = tmp.iloc[idx_valid].copy()
                ranks = valid_df.groupby("symbol")["mfe"].rank(method="average", pct=True)
                mfe_pct[idx_valid] = ranks.to_numpy(dtype=float)
            new_cols[f"mfe_pct_{suffix}"] = mfe_pct
            valid_mfe_pct = (is_truncated == 0) & np.isfinite(mfe_pct)
            for q in mfe_quantiles:
                qlabel = _quantile_label_suffix(q)
                col = f"worked_mfe_{suffix}_top_{qlabel}"
                lab = np.full(n_events, np.nan, dtype=float)
                if valid_mfe_pct.any():
                    lab[valid_mfe_pct] = (mfe_pct[valid_mfe_pct] >= q).astype(float)
                new_cols[col] = lab

        valid_mae = (is_truncated == 0) & np.isfinite(mae)
        for bps in mae_thresholds:
            col = f"safe_mae_{suffix}_le_{int(bps)}bps"
            lab = np.full(n_events, np.nan, dtype=float)
            if valid_mae.any():
                lab[valid_mae] = (mae[valid_mae] >= -(float(bps) / 10000.0)).astype(float)
            new_cols[col] = lab

        valid_mfe_atr = (is_truncated == 0) & np.isfinite(mfe_atr)
        for thr in mfe_atr_thresholds_norm:
            col = f"worked_mfeatr_{suffix}_ge_{_threshold_label(thr)}"
            lab = np.full(n_events, np.nan, dtype=float)
            if valid_mfe_atr.any():
                lab[valid_mfe_atr] = (mfe_atr[valid_mfe_atr] >= thr).astype(float)
            new_cols[col] = lab

        valid_mae_atr = (is_truncated == 0) & np.isfinite(mae_atr)
        for thr in mae_atr_thresholds_norm:
            col = f"safe_maeatr_{suffix}_le_{_threshold_label(thr)}"
            lab = np.full(n_events, np.nan, dtype=float)
            if valid_mae_atr.any():
                lab[valid_mae_atr] = (mae_atr[valid_mae_atr] >= -thr).astype(float)
            new_cols[col] = lab

        good_col = (
            f"good_{suffix}_mfe{_threshold_label(quality_mfe_thr)}_mae{_threshold_label(quality_mae_thr)}"
        )
        good_lab = np.full(n_events, np.nan, dtype=float)
        valid_good = (is_truncated == 0) & np.isfinite(mfe_atr) & np.isfinite(mae_atr)
        if valid_good.any():
            good_lab[valid_good] = (
                (mfe_atr[valid_good] >= quality_mfe_thr)
                & (mae_atr[valid_good] >= -quality_mae_thr)
            ).astype(float)
        new_cols[good_col] = good_lab

        if mfeatr_quantiles:
            mfeatr_pct = np.full(n_events, np.nan, dtype=float)
            if valid_mfe_atr.any():
                tmp = pd.DataFrame({"symbol": symbols, "mfe_atr": mfe_atr})
                idx_valid = np.flatnonzero(valid_mfe_atr)
                valid_df = tmp.iloc[idx_valid].copy()
                ranks = valid_df.groupby("symbol")["mfe_atr"].rank(method="average", pct=True)
                mfeatr_pct[idx_valid] = ranks.to_numpy(dtype=float)
            new_cols[f"mfeatr_pct_{suffix}"] = mfeatr_pct
            valid_mfeatr_pct = (is_truncated == 0) & np.isfinite(mfeatr_pct)
            for q in mfeatr_quantiles:
                col = f"worked_mfeatr_{suffix}_top_{_quantile_label_suffix(q)}"
                lab = np.full(n_events, np.nan, dtype=float)
                if valid_mfeatr_pct.any():
                    lab[valid_mfeatr_pct] = (mfeatr_pct[valid_mfeatr_pct] >= q).astype(float)
                new_cols[col] = lab

    if new_cols:
        out = pd.concat([out.reset_index(drop=True), pd.DataFrame(new_cols)], axis=1)
    return out.sort_values(["symbol", "event_time_ms", "entry_time_ms"]).reset_index(drop=True)


def compute_forward_for_events(
    bars_df: pd.DataFrame,
    events_df: pd.DataFrame,
    horizons_hours: List[int],
    tf_minutes: int,
    tie_break: str,
    emit_resolved: bool,
    sl_atr_mult: float,
    rr_mult: float,
    warn_cb: Callable[[str], None] | None = None,
    mode: str = "barrier",
    ret_thresholds_bps: List[int] | None = None,
    percentile_targets: List[float] | None = None,
    mfe_thresholds_bps: List[int] | None = None,
    mae_thresholds_bps: List[int] | None = None,
    mfe_percentile_targets: List[float] | None = None,
    mfe_atr_thresholds: List[float] | None = None,
    mae_atr_thresholds: List[float] | None = None,
    mfeatr_percentile_targets: List[float] | None = None,
    quality_mfe_threshold: float = 1.0,
    quality_mae_threshold: float = 1.0,
    truncate_policy: str = "nan",
    emit_barrier: bool = False,
) -> pd.DataFrame:
    mode_norm = (mode or "").strip().lower()
    if mode_norm == "time_only":
        time_df = _compute_time_only_forward_for_events(
            bars_df=bars_df,
            events_df=events_df,
            horizons_hours=horizons_hours,
            tf_minutes=tf_minutes,
            ret_thresholds_bps=ret_thresholds_bps or [],
            percentile_targets=percentile_targets or [],
            mfe_thresholds_bps=mfe_thresholds_bps or [],
            mae_thresholds_bps=mae_thresholds_bps or [],
            mfe_percentile_targets=mfe_percentile_targets or [],
            mfe_atr_thresholds=mfe_atr_thresholds or [],
            mae_atr_thresholds=mae_atr_thresholds or [],
            mfeatr_percentile_targets=mfeatr_percentile_targets or [],
            quality_mfe_threshold=float(quality_mfe_threshold),
            quality_mae_threshold=float(quality_mae_threshold),
            truncate_policy=truncate_policy,
            warn_cb=warn_cb,
        )
        if not emit_barrier or time_df.empty:
            return time_df

        barrier_df = _compute_barrier_forward_for_events(
            bars_df=bars_df,
            events_df=events_df,
            horizons_hours=horizons_hours,
            tf_minutes=tf_minutes,
            tie_break=tie_break,
            emit_resolved=emit_resolved,
            sl_atr_mult=sl_atr_mult,
            rr_mult=rr_mult,
            warn_cb=warn_cb,
        )
        if barrier_df.empty:
            return time_df
        extra_cols = [c for c in barrier_df.columns if c not in FORWARD_KEY_COLS]
        return time_df.merge(
            barrier_df[FORWARD_KEY_COLS + extra_cols],
            on=FORWARD_KEY_COLS,
            how="left",
        )
    return _compute_barrier_forward_for_events(
        bars_df=bars_df,
        events_df=events_df,
        horizons_hours=horizons_hours,
        tf_minutes=tf_minutes,
        tie_break=tie_break,
        emit_resolved=emit_resolved,
        sl_atr_mult=sl_atr_mult,
        rr_mult=rr_mult,
        warn_cb=warn_cb,
    )


def _validate_barrier_forward_outcomes(
    forward_df: pd.DataFrame,
    horizons_hours: List[int],
    rr_mult: float,
    emit_resolved: bool,
) -> List[Dict[str, float]]:
    summaries: List[Dict[str, float]] = []
    if forward_df.empty:
        return summaries

    for h in horizons_hours:
        suffix = _horizon_suffix(h)
        tp_col = f"tp_first_{suffix}"
        sl_col = f"sl_first_{suffix}"
        same_col = f"both_samebar_{suffix}"
        nh_col = f"no_hit_{suffix}"
        trunc_col = f"is_truncated_{suffix}"
        mfe_col = f"mfe_{suffix}"
        mae_col = f"mae_{suffix}"

        required = [tp_col, sl_col, same_col, nh_col, trunc_col, mfe_col, mae_col]
        missing = [c for c in required if c not in forward_df.columns]
        if missing:
            raise ValueError(f"Missing forward columns for {suffix}: {missing}")

        trunc = forward_df[trunc_col].fillna(0).to_numpy(dtype=np.int8)
        non_mask = trunc == 0
        trunc_mask = trunc == 1

        if non_mask.any():
            non_df = forward_df.loc[non_mask, [tp_col, sl_col, same_col, nh_col]]
            if non_df.isna().any().any():
                raise ValueError(f"NaN outcome flags found for non-truncated rows at {suffix}")
            sums = non_df.sum(axis=1).to_numpy(dtype=float)
            if not np.allclose(sums, 1.0):
                raise ValueError(
                    f"Mutual exclusivity failed at {suffix}; expected tp+sl+same+no_hit==1."
                )
            if forward_df.loc[non_mask, [mfe_col, mae_col]].isna().any().any():
                raise ValueError(f"NaN MFE/MAE found for non-truncated rows at {suffix}")

        if trunc_mask.any():
            trunc_df = forward_df.loc[
                trunc_mask, [tp_col, sl_col, same_col, nh_col, mfe_col, mae_col]
            ]
            if not trunc_df.isna().all().all():
                raise ValueError(
                    f"Truncated rows must have NaN flags and NaN MFE/MAE at {suffix}."
                )

        n_all = len(forward_df)
        n_non = int(non_mask.sum())
        if n_non > 0:
            tp_rate = float(forward_df.loc[non_mask, tp_col].mean())
            sl_rate = float(forward_df.loc[non_mask, sl_col].mean())
            same_rate = float(forward_df.loc[non_mask, same_col].mean())
            nh_rate = float(forward_df.loc[non_mask, nh_col].mean())
            hit_rate = float(1.0 - nh_rate)
            if emit_resolved:
                tp_res_col = f"tp_first_resolved_{suffix}"
                sl_res_col = f"sl_first_resolved_{suffix}"
                if tp_res_col not in forward_df.columns or sl_res_col not in forward_df.columns:
                    raise ValueError(f"Missing resolved columns for {suffix}")
                tp_res_rate = float(forward_df.loc[non_mask, tp_res_col].mean())
                sl_res_rate = float(forward_df.loc[non_mask, sl_res_col].mean())
            else:
                tp_res_rate = tp_rate
                sl_res_rate = sl_rate

            exp_total = float(rr_mult * tp_res_rate - sl_res_rate)
            if hit_rate > 0:
                tp_hit = float(tp_res_rate / hit_rate)
                sl_hit = float(sl_res_rate / hit_rate)
                exp_hit_only = float(rr_mult * tp_hit - sl_hit)
            else:
                exp_hit_only = np.nan
        else:
            tp_rate = sl_rate = same_rate = nh_rate = np.nan
            tp_res_rate = sl_res_rate = np.nan
            hit_rate = np.nan
            exp_total = np.nan
            exp_hit_only = np.nan

        summaries.append(
            {
                "mode": "barrier",
                "horizon_h": float(h),
                "n_events": float(n_all),
                "n_valid": float(n_non),
                "tp_rate": tp_rate,
                "sl_rate": sl_rate,
                "samebar_rate": same_rate,
                "no_hit_rate": nh_rate,
                "tp_resolved_rate": tp_res_rate,
                "sl_resolved_rate": sl_res_rate,
                "hit_rate": hit_rate,
                "trunc_rate": float(trunc_mask.mean()),
                "truncated_rate": float(trunc_mask.mean()),
                "expR_total": exp_total,
                "expR_hit_only": exp_hit_only,
                "expectancy_r": exp_total,
            }
        )
    return summaries


def _validate_time_only_forward_outcomes(
    forward_df: pd.DataFrame,
    horizons_hours: List[int],
) -> List[Dict[str, float]]:
    summaries: List[Dict[str, float]] = []
    if forward_df.empty:
        return summaries

    for h in horizons_hours:
        suffix = _horizon_suffix(h)
        ret_col = f"ret_{suffix}"
        logret_col = f"logret_{suffix}"
        mfe_col = f"mfe_{suffix}"
        mae_col = f"mae_{suffix}"
        ret_atr_col = f"ret_atr_{suffix}"
        mfe_atr_col = f"mfe_atr_{suffix}"
        mae_atr_col = f"mae_atr_{suffix}"
        rr_like_col = f"rr_like_{suffix}"
        max_dd_col = f"max_dd_{suffix}"
        max_ru_col = f"max_ru_{suffix}"
        up_col = f"up_{suffix}"
        dn_col = f"dn_{suffix}"
        flat_col = f"flat_{suffix}"
        trunc_col = f"is_truncated_{suffix}"
        base_cols = [
            ret_col,
            logret_col,
            mfe_col,
            mae_col,
            ret_atr_col,
            mfe_atr_col,
            mae_atr_col,
            rr_like_col,
            max_dd_col,
            max_ru_col,
            up_col,
            dn_col,
            flat_col,
            trunc_col,
        ]
        missing = [c for c in base_cols if c not in forward_df.columns]
        if missing:
            raise ValueError(f"Missing time-only forward columns for {suffix}: {missing}")

        trunc = forward_df[trunc_col].fillna(0).to_numpy(dtype=np.int8)
        non_mask = trunc == 0
        trunc_mask = trunc == 1

        if non_mask.any():
            non_df = forward_df.loc[
                non_mask,
                [
                    ret_col,
                    mfe_col,
                    mae_col,
                    max_dd_col,
                    max_ru_col,
                    up_col,
                    dn_col,
                    flat_col,
                ],
            ]
            # ATR-normalized metrics can be NaN when atr_entry==0 and rr_like can be NaN when mae_atr==0.
            if non_df.isna().any().any():
                raise ValueError(f"NaN base time-only fields for non-truncated rows at {suffix}")
            ud_sum = forward_df.loc[non_mask, [up_col, dn_col, flat_col]].sum(axis=1).to_numpy(dtype=float)
            if not np.allclose(ud_sum, 1.0):
                raise ValueError(f"Mutual exclusivity failed at {suffix}; expected up+dn+flat==1.")

        if trunc_mask.any():
            trunc_required = [
                ret_col,
                logret_col,
                mfe_col,
                mae_col,
                ret_atr_col,
                mfe_atr_col,
                mae_atr_col,
                rr_like_col,
                max_dd_col,
                max_ru_col,
                up_col,
                dn_col,
                flat_col,
            ]
            if not forward_df.loc[trunc_mask, trunc_required].isna().all().all():
                raise ValueError(f"Truncated rows must be NaN for time-only outcome fields at {suffix}.")

            # Optional labels should also be NaN when truncated.
            opt_cols = [
                c
                for c in forward_df.columns
                if c in {f"ret_pct_{suffix}", f"mfe_pct_{suffix}", f"mfeatr_pct_{suffix}"}
                or c.startswith(f"worked_{suffix}_")
                or c.startswith(f"worked_mfe_{suffix}_")
                or c.startswith(f"worked_mfeatr_{suffix}_")
                or c.startswith(f"safe_mae_{suffix}_")
                or c.startswith(f"safe_maeatr_{suffix}_")
                or c.startswith(f"good_{suffix}_")
            ]
            if opt_cols:
                if not forward_df.loc[trunc_mask, opt_cols].isna().all().all():
                    raise ValueError(f"Truncated rows must be NaN for optional labels at {suffix}.")

        n_all = len(forward_df)
        n_non = int(non_mask.sum())
        if n_non > 0:
            ret_valid = forward_df.loc[non_mask, ret_col].astype(float)
            mean_ret = float(ret_valid.mean())
            median_ret = float(ret_valid.median())
            up_rate = float(forward_df.loc[non_mask, up_col].astype(float).mean())
            mean_mfe = float(forward_df.loc[non_mask, mfe_col].astype(float).mean())
            mean_mae = float(forward_df.loc[non_mask, mae_col].astype(float).mean())
            p25 = float(ret_valid.quantile(0.25))
            p50 = float(ret_valid.quantile(0.50))
            p75 = float(ret_valid.quantile(0.75))
        else:
            mean_ret = median_ret = up_rate = mean_mfe = mean_mae = p25 = p50 = p75 = np.nan

        summaries.append(
            {
                "mode": "time_only",
                "horizon_h": float(h),
                "n_events": float(n_all),
                "n_total": float(n_all),
                "n_valid": float(n_non),
                "trunc_rate": float(trunc_mask.mean()),
                "truncated_rate": float(trunc_mask.mean()),
                "mean_ret": mean_ret,
                "median_ret": median_ret,
                "up_rate": up_rate,
                "mean_mfe": mean_mfe,
                "mean_mae": mean_mae,
                "p25_ret": p25,
                "p50_ret": p50,
                "p75_ret": p75,
            }
        )
    return summaries


def validate_forward_outcomes(
    forward_df: pd.DataFrame,
    horizons_hours: List[int],
    rr_mult: float,
    emit_resolved: bool,
    mode: str = "barrier",
) -> List[Dict[str, float]]:
    mode_norm = (mode or "").strip().lower()
    if mode_norm == "time_only":
        return _validate_time_only_forward_outcomes(
            forward_df=forward_df,
            horizons_hours=horizons_hours,
        )
    return _validate_barrier_forward_outcomes(
        forward_df=forward_df,
        horizons_hours=horizons_hours,
        rr_mult=rr_mult,
        emit_resolved=emit_resolved,
    )


def compute_forward_outcomes(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    config: RunConfig,
    timeframe_minutes: int,
) -> pd.DataFrame:
    return compute_forward_for_events(
        bars_df=bars,
        events_df=events,
        horizons_hours=config.forward.normalized_horizons_hours(),
        tf_minutes=timeframe_minutes,
        tie_break=config.forward.normalized_tie_break(),
        emit_resolved=config.forward.emit_resolved,
        sl_atr_mult=config.risk.sl_atr_mult,
        rr_mult=config.risk.rr_mult,
        warn_cb=None,
        mode=config.forward.normalized_mode(),
        ret_thresholds_bps=config.forward.normalized_ret_thresholds_bps(),
        percentile_targets=config.forward.normalized_percentile_targets(),
        mfe_thresholds_bps=config.forward.normalized_mfe_thresholds_bps(),
        mae_thresholds_bps=config.forward.normalized_mae_thresholds_bps(),
        mfe_percentile_targets=config.forward.normalized_mfe_percentile_targets(),
        mfe_atr_thresholds=config.forward.normalized_mfe_atr_thresholds(),
        mae_atr_thresholds=config.forward.normalized_mae_atr_thresholds(),
        mfeatr_percentile_targets=config.forward.normalized_mfeatr_percentile_targets(),
        quality_mfe_threshold=config.forward.normalized_quality_mfe_threshold(),
        quality_mae_threshold=config.forward.normalized_quality_mae_threshold(),
        truncate_policy=config.forward.truncate_policy,
        emit_barrier=bool(config.forward.emit_barrier),
    )
