from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _parse_bool_arg(raw: str) -> bool:
    token = str(raw).strip().lower()
    if token in {"1", "true", "t", "yes", "y"}:
        return True
    if token in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {raw}")


def _parse_horizons_csv(raw: str) -> List[int]:
    out: List[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    if not out:
        raise ValueError("No valid horizons parsed from --horizons")
    return sorted(set(out))


def _derive_year(df: pd.DataFrame) -> pd.Series:
    if "year" in df.columns:
        return pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "event_time_ms" in df.columns:
        dt = pd.to_datetime(df["event_time_ms"], unit="ms", utc=True, errors="coerce")
        return dt.dt.year.astype("Int64")
    if "time" in df.columns:
        dt = pd.to_datetime(df["time"], utc=True, errors="coerce")
        return dt.dt.year.astype("Int64")
    return pd.Series(pd.array([pd.NA] * len(df), dtype="Int64"), index=df.index)


def _safe_target_rate(df: pd.DataFrame, target_col: str) -> float:
    y = pd.to_numeric(df[target_col], errors="coerce")
    if len(y) == 0:
        return float("nan")
    return float(np.nanmean(y.to_numpy(dtype=float)))


def _summarize(df: pd.DataFrame, target_col: str, ret_col: Optional[str]) -> Dict[str, float]:
    y = pd.to_numeric(df[target_col], errors="coerce")
    valid_mask = np.isfinite(y.to_numpy(dtype=float))
    valid = y[valid_mask]
    out: Dict[str, float] = {
        "n": float(len(df)),
        "n_target_valid": float(valid_mask.sum()),
        "target_rate": float(valid.mean()) if len(valid) else float("nan"),
    }
    if ret_col and ret_col in df.columns:
        r = pd.to_numeric(df[ret_col], errors="coerce")
        out["mean_ret"] = float(np.nanmean(r.to_numpy(dtype=float))) if len(r) else float("nan")
        out["median_ret"] = float(np.nanmedian(r.to_numpy(dtype=float))) if len(r) else float("nan")
    return out


def _is_bool_feature(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    numeric = pd.to_numeric(series, errors="coerce")
    clean = numeric.dropna()
    if len(clean) == 0:
        return False
    vals = set(clean.unique().tolist())
    return vals.issubset({0.0, 1.0})


def _feature_columns(
    df: pd.DataFrame,
    target_col: str,
    include_ok_flags: bool,
) -> tuple[List[str], List[str]]:
    ignore = {
        "symbol",
        "timeframe",
        "signal_id",
        "event_time_ms",
        "entry_time_ms",
        "pivot_time_ms",
        "time",
        "year",
        "_year",
        target_col,
    }
    outcome_prefixes = (
        "tp_first_",
        "sl_first_",
        "both_samebar_",
        "no_hit_",
        "is_truncated_",
        "ret_",
        "logret_",
        "mfe_",
        "mae_",
        "ret_atr_",
        "mfe_atr_",
        "mae_atr_",
        "rr_like_",
        "max_dd_",
        "max_ru_",
        "up_",
        "dn_",
        "flat_",
        "worked_",
        "worked_mfe_",
        "worked_mfeatr_",
        "ret_pct_",
        "mfe_pct_",
        "mfeatr_pct_",
        "safe_mae_",
        "safe_maeatr_",
        "good_",
    )
    num_feats: List[str] = []
    bool_feats: List[str] = []
    for c in df.columns:
        if c in ignore or any(c.startswith(pref) for pref in outcome_prefixes):
            continue
        if not include_ok_flags and c.endswith("_ok"):
            continue
        s = df[c]
        if _is_bool_feature(s):
            bool_feats.append(c)
            continue
        if pd.api.types.is_numeric_dtype(s):
            num_feats.append(c)
            continue
        if pd.api.types.is_object_dtype(s):
            num = pd.to_numeric(s, errors="coerce")
            if np.isfinite(num.to_numpy(dtype=float)).any():
                if _is_bool_feature(num):
                    bool_feats.append(c)
                else:
                    num_feats.append(c)
            continue
        if pd.api.types.is_extension_array_dtype(s):
            num = pd.to_numeric(s, errors="coerce")
            if np.isfinite(num.to_numpy(dtype=float)).any():
                if _is_bool_feature(num):
                    bool_feats.append(c)
                else:
                    num_feats.append(c)
    return num_feats, bool_feats


def _coverage_ok(
    subset: pd.DataFrame,
    eligible_buckets: pd.Index,
    min_cover_global: int,
    min_cover_per_symbol_year: int,
) -> bool:
    if len(subset) < min_cover_global:
        return False
    counts = subset.groupby(["symbol", "_year"], dropna=False).size()
    for bucket in eligible_buckets:
        if int(counts.get(bucket, 0)) < min_cover_per_symbol_year:
            return False
    return True


def _rule_mask(df: pd.DataFrame, rule: Dict[str, Any]) -> np.ndarray:
    feature = str(rule["feature"])
    kind = str(rule["kind"])
    if kind == "numeric":
        thr = float(rule["threshold"])
        x = pd.to_numeric(df[feature], errors="coerce")
        return (np.isfinite(x.to_numpy(dtype=float)) & (x >= thr)).to_numpy(dtype=bool)
    if kind == "bool_true":
        x = pd.to_numeric(df[feature], errors="coerce")
        return (x == 1).to_numpy(dtype=bool)
    return np.zeros(len(df), dtype=bool)


def _best_numeric_rule(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
    qgrid: np.ndarray,
    eligible_buckets: pd.Index,
    min_cover_global: int,
    min_cover_per_symbol_year: int,
) -> Optional[Dict[str, Any]]:
    x = pd.to_numeric(df[feature], errors="coerce")
    valid = np.isfinite(x.to_numpy(dtype=float))
    if int(valid.sum()) < max(min_cover_global, 50):
        return None

    base_rate = _safe_target_rate(df=df, target_col=target_col)
    thresholds = np.unique(np.quantile(x[valid], qgrid))
    best: Optional[Dict[str, Any]] = None
    for thr in thresholds:
        mask = valid & (x >= thr)
        sub = df.loc[mask]
        if not _coverage_ok(
            subset=sub,
            eligible_buckets=eligible_buckets,
            min_cover_global=min_cover_global,
            min_cover_per_symbol_year=min_cover_per_symbol_year,
        ):
            continue
        rate = _safe_target_rate(df=sub, target_col=target_col)
        rec = {
            "feature": feature,
            "kind": "numeric",
            "rule": f"{feature} >= {thr:.10g}",
            "threshold": float(thr),
            "cover": float(len(sub)),
            "cover_pct": float(len(sub) / len(df)),
            "target_rate": rate,
            "lift_vs_base": float(rate - base_rate),
        }
        if best is None or float(rec["lift_vs_base"]) > float(best["lift_vs_base"]):
            best = rec
    return best


def _best_bool_true_rule(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
    eligible_buckets: pd.Index,
    min_cover_global: int,
    min_cover_per_symbol_year: int,
) -> Optional[Dict[str, Any]]:
    x = pd.to_numeric(df[feature], errors="coerce")
    mask = x == 1
    sub = df.loc[mask]
    if not _coverage_ok(
        subset=sub,
        eligible_buckets=eligible_buckets,
        min_cover_global=min_cover_global,
        min_cover_per_symbol_year=min_cover_per_symbol_year,
    ):
        return None
    base_rate = _safe_target_rate(df=df, target_col=target_col)
    rate = _safe_target_rate(df=sub, target_col=target_col)
    return {
        "feature": feature,
        "kind": "bool_true",
        "rule": f"{feature} == True",
        "threshold": 1.0,
        "cover": float(len(sub)),
        "cover_pct": float(len(sub) / len(df)),
        "target_rate": rate,
        "lift_vs_base": float(rate - base_rate),
    }


def _greedy_combo(
    df: pd.DataFrame,
    candidates: pd.DataFrame,
    target_col: str,
    eligible_buckets: pd.Index,
    min_cover_global: int,
    min_cover_per_symbol_year: int,
    max_steps: int,
) -> tuple[list[Dict[str, Any]], pd.DataFrame]:
    cur = df.copy()
    chosen: list[Dict[str, Any]] = []
    prev_rate = _safe_target_rate(df=cur, target_col=target_col)
    used: set[str] = set()

    for _ in range(max_steps):
        best: Optional[Dict[str, Any]] = None
        best_df: Optional[pd.DataFrame] = None
        for rec in candidates.to_dict("records"):
            rule_str = str(rec["rule"])
            if rule_str in used:
                continue
            mask = _rule_mask(cur, rec)
            nxt = cur.loc[mask]
            if not _coverage_ok(
                subset=nxt,
                eligible_buckets=eligible_buckets,
                min_cover_global=min_cover_global,
                min_cover_per_symbol_year=min_cover_per_symbol_year,
            ):
                continue
            rate = _safe_target_rate(df=nxt, target_col=target_col)
            lift = float(rate - prev_rate)
            cand = {
                **rec,
                "n_after": float(len(nxt)),
                "target_rate_after": rate,
                "lift_vs_prev": lift,
            }
            if best is None or float(cand["lift_vs_prev"]) > float(best["lift_vs_prev"]):
                best = cand
                best_df = nxt

        if best is None or float(best["lift_vs_prev"]) <= 0.0 or best_df is None:
            break
        chosen.append(best)
        used.add(str(best["rule"]))
        cur = best_df
        prev_rate = float(best["target_rate_after"])
    return chosen, cur


def _infer_target_horizon(target: str) -> Optional[str]:
    m = re.search(r"_(\d+(?:p\d+)?h)(?:_|$)", target)
    if not m:
        return None
    return m.group(1)


def _default_target_for_horizon(df: pd.DataFrame, h: int) -> Optional[str]:
    suffix = f"{h}h"
    candidates = [
        f"worked_{suffix}_top_q80",
        f"worked_{suffix}_ge_0bps",
        f"up_{suffix}",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _sanitize_filename_token(token: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", token)


def _truncate_col_for_target(df: pd.DataFrame, target: str) -> Optional[str]:
    sfx = _infer_target_horizon(target)
    if sfx is None:
        return None
    trunc_col = f"is_truncated_{sfx}"
    if trunc_col in df.columns:
        return trunc_col
    return None


def _ret_col_for_target(df: pd.DataFrame, target: str) -> Optional[str]:
    sfx = _infer_target_horizon(target)
    if sfx is None:
        return None
    ret_col = f"ret_{sfx}"
    if ret_col in df.columns:
        return ret_col
    return None


def _normalize_target_token(token: str) -> str:
    raw = token.strip()
    if raw == "":
        return raw
    candidate = raw.replace("p", ".")
    try:
        val = float(candidate)
        if np.isfinite(val):
            return f"{val:.10g}".replace(".", "p")
    except ValueError:
        pass
    return raw


def _normalize_target_name(name: str) -> str:
    parts = [p for p in str(name).strip().split("_") if p != ""]
    if not parts:
        return ""
    return "_".join(_normalize_target_token(p) for p in parts)


def _resolve_target_column(columns: List[str], requested: str) -> str:
    if requested in columns:
        return requested

    norm_req = _normalize_target_name(requested)
    norm_map: Dict[str, List[str]] = {}
    for col in columns:
        k = _normalize_target_name(col)
        norm_map.setdefault(k, []).append(col)

    cands = norm_map.get(norm_req, [])
    if len(cands) == 1:
        return cands[0]
    if len(cands) > 1:
        raise SystemExit(
            "Ambiguous target alias. Matches: "
            + ", ".join(sorted(cands))
            + f". Requested: {requested}"
        )

    raise SystemExit(
        f"Target column not found: {requested}. "
        "Tip: run a quick column listing for worked_mfeatr/good targets."
    )


def _prepare_frame_for_target(
    labeled: pd.DataFrame,
    target: str,
    min_cover_global: int,
    min_cover_per_symbol_year: int,
) -> tuple[pd.DataFrame, pd.Index, str]:
    target = _resolve_target_column(list(labeled.columns), target)

    frame = labeled.copy()
    trunc_col = _truncate_col_for_target(frame, target)
    if trunc_col is not None:
        frame = frame[frame[trunc_col].fillna(0) == 0].copy()

    y = pd.to_numeric(frame[target], errors="coerce")
    frame = frame[np.isfinite(y.to_numpy(dtype=float))].copy()
    if frame.empty:
        raise SystemExit(f"No valid rows after filtering target/truncation for {target}.")

    y_vals = set(pd.to_numeric(frame[target], errors="coerce").dropna().unique().tolist())
    if not y_vals.issubset({0.0, 1.0}):
        raise SystemExit(f"Target must be binary 0/1 for lift search: {target}")

    bucket_counts = frame.groupby(["symbol", "_year"], dropna=False).size()
    eligible = bucket_counts[bucket_counts >= min_cover_per_symbol_year].index
    if len(eligible) == 0:
        raise SystemExit(
            f"No (symbol,year) buckets with >= {min_cover_per_symbol_year} rows for target={target}."
        )

    idx = pd.MultiIndex.from_frame(frame[["symbol", "_year"]])
    frame = frame[idx.isin(eligible)].reset_index(drop=True)
    if len(frame) < min_cover_global:
        raise SystemExit(
            f"Filtered frame has {len(frame)} rows < min_cover_global={min_cover_global} "
            f"for target={target}."
        )
    return frame, eligible, target


def _run_for_target(
    labeled: pd.DataFrame,
    target: str,
    args: argparse.Namespace,
    qgrid: np.ndarray,
    outdir: Path,
) -> None:
    frame, eligible, resolved_target = _prepare_frame_for_target(
        labeled=labeled,
        target=target,
        min_cover_global=args.min_cover_global,
        min_cover_per_symbol_year=args.min_cover_per_symbol_year,
    )
    if resolved_target != target:
        print(f"Resolved target alias: {target} -> {resolved_target}")
    target = resolved_target

    ret_col = _ret_col_for_target(frame, target)
    base = _summarize(df=frame, target_col=target, ret_col=ret_col)
    print(f"BASE {target}: {base}")

    per_sy_rows: list[dict[str, Any]] = []
    for (sym, yr), g in frame.groupby(["symbol", "_year"], dropna=False):
        s = _summarize(df=g, target_col=target, ret_col=ret_col)
        per_sy_rows.append(
            {
                "symbol": str(sym) if pd.notna(sym) else None,
                "year": int(yr) if pd.notna(yr) else None,
                **s,
            }
        )
    per_sy = pd.DataFrame(per_sy_rows).sort_values(["symbol", "year"], na_position="last")

    num_feats, bool_feats = _feature_columns(
        frame,
        target_col=target,
        include_ok_flags=bool(args.include_ok_flags),
    )
    candidates: list[Dict[str, Any]] = []
    for feat in num_feats:
        best = _best_numeric_rule(
            df=frame,
            feature=feat,
            target_col=target,
            qgrid=qgrid,
            eligible_buckets=eligible,
            min_cover_global=args.min_cover_global,
            min_cover_per_symbol_year=args.min_cover_per_symbol_year,
        )
        if best:
            candidates.append(best)
    for feat in bool_feats:
        best_bool = _best_bool_true_rule(
            df=frame,
            feature=feat,
            target_col=target,
            eligible_buckets=eligible,
            min_cover_global=args.min_cover_global,
            min_cover_per_symbol_year=args.min_cover_per_symbol_year,
        )
        if best_bool:
            candidates.append(best_bool)

    if candidates:
        cand_df = pd.DataFrame(candidates).sort_values(
            ["lift_vs_base", "target_rate"],
            ascending=[False, False],
        ).head(args.topk)
    else:
        cand_df = pd.DataFrame(
            columns=[
                "feature",
                "kind",
                "rule",
                "threshold",
                "cover",
                "cover_pct",
                "target_rate",
                "lift_vs_base",
            ]
        )

    chosen: list[Dict[str, Any]]
    final_df: pd.DataFrame
    if cand_df.empty:
        chosen = []
        final_df = frame
    else:
        chosen, final_df = _greedy_combo(
            df=frame,
            candidates=cand_df,
            target_col=target,
            eligible_buckets=eligible,
            min_cover_global=args.min_cover_global,
            min_cover_per_symbol_year=args.min_cover_per_symbol_year,
            max_steps=args.max_steps,
        )

    safe_target = _sanitize_filename_token(target)
    per_sy_out = outdir / f"per_symbol_year_{safe_target}.csv"
    uni_out = outdir / f"top_univariate_rules_{safe_target}.csv"
    combo_out = outdir / f"greedy_combo_{safe_target}.json"
    per_sy.to_csv(per_sy_out, index=False)
    cand_df.to_csv(uni_out, index=False)

    combo = {
        "target": target,
        "horizon_suffix": _infer_target_horizon(target),
        "base": base,
        "chosen_rules": chosen,
        "final": _summarize(df=final_df, target_col=target, ret_col=ret_col),
        "final_n": int(len(final_df)),
        "min_cover_global": int(args.min_cover_global),
        "min_cover_per_symbol_year": int(args.min_cover_per_symbol_year),
    }
    combo_out.write_text(json.dumps(combo, indent=2), encoding="utf-8")

    print(f"Wrote {per_sy_out}")
    print(f"Wrote {uni_out}")
    print(f"Wrote {combo_out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="DFD05 lift search using a binary target column")
    ap.add_argument("--labeled", required=True, help="Path to labeled parquet")
    ap.add_argument("--forward", required=False, help="Optional forward parquet to merge")
    ap.add_argument(
        "--target",
        default=None,
        help=(
            "Binary target column (e.g. up_24h, worked_24h_ge_50bps, worked_24h_top_q80, "
            "worked_mfe_24h_ge_100bps, worked_mfe_24h_top_q80, safe_mae_24h_le_100bps, "
            "worked_mfeatr_24h_ge_1, worked_mfeatr_24h_top_q80, safe_maeatr_24h_le_1, good_24h_mfe1_mae1)"
        ),
    )
    ap.add_argument("--horizons", default="24", help='CSV horizons list, e.g. "4,24,72"')
    ap.add_argument("--min_cover_global", type=int, default=200)
    ap.add_argument("--min_cover_per_symbol_year", type=int, default=20)
    ap.add_argument("--min_cover", type=int, default=None, help="Deprecated alias for --min_cover_global")
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--max_steps", type=int, default=6)
    ap.add_argument(
        "--include_ok_flags",
        type=_parse_bool_arg,
        default=False,
        help="Include *_ok boolean flags as candidate features (default: false)",
    )
    ap.add_argument("--outdir", default="data/reports/dfd05_lift")
    args = ap.parse_args()
    if args.min_cover is not None:
        args.min_cover_global = int(args.min_cover)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    labeled = pd.read_parquet(args.labeled)
    if args.forward:
        forward = pd.read_parquet(args.forward)
        keys = ["symbol", "timeframe", "signal_id", "event_time_ms"]
        labeled = labeled.merge(forward, on=keys, how="left", suffixes=("", "_fwd"))
    labeled["_year"] = _derive_year(labeled)

    horizons = _parse_horizons_csv(args.horizons)
    qgrid = np.linspace(0.50, 0.95, 24)

    targets: List[str] = []
    if args.target:
        targets = [str(args.target)]
    else:
        for h in horizons:
            t = _default_target_for_horizon(labeled, h)
            if t is None:
                print(f"Skipping {h}h: no default target column found.")
                continue
            targets.append(t)
    if not targets:
        raise SystemExit("No target columns selected for lift analysis.")

    for target in targets:
        _run_for_target(labeled=labeled, target=target, args=args, qgrid=qgrid, outdir=outdir)


if __name__ == "__main__":
    main()
