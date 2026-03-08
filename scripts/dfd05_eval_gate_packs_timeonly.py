from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


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
    raise SystemExit(f"Target column not found: {requested}")


DEFAULT_PACKS_24H: Dict[str, List[str]] = {
    "Pack_A": [
        "daily_adx >= 42",
        "daily_di_ok == 1",
        "osc_change_pct >= 70",
        "bars_gap >= 16",
        "vol_ratio_entry >= 1.5",
    ],
    "Pack_B": [
        "daily_adx >= 42",
        "daily_di_ok == 1",
        "osc_change_pct >= 70",
        "bars_gap >= 16",
        "vol_ratio_entry >= 1.5",
        "vol_spike_ok_entry == 1",
    ],
    "Pack_C": [
        "daily_adx >= 42",
        "daily_di_ok == 1",
        "osc_change_pct >= 70",
        "bars_gap >= 16",
        "vol_ratio_entry >= 1.5",
        "vol_behavior_ok_entry == 1",
    ],
    "Pack_D": [
        "daily_adx >= 42",
        "daily_di_ok == 1",
        "osc_change_pct >= 70",
        "bars_gap >= 16",
        "vol_ratio_entry >= 1.5",
        "vol_spike_ok_entry == 1",
        "vol_behavior_ok_entry == 1",
    ],
    "Pack_E": [
        "daily_adx >= 42",
        "daily_di_ok == 1",
        "osc_change_pct >= 70",
        "bars_gap >= 16",
        "vol_ratio_entry >= 1.5",
        "vol_spike_ok_entry == 1",
        "vol_behavior_ok_entry == 1",
        "session_ok_pivot == 1",
    ],
    "Pack_ATR_A": [
        "daily_adx >= 25",
        "daily_di_ok == 1",
        "osc_change_pct >= 30",
        "bars_gap >= 10",
    ],
    "Pack_ATR_B": [
        "daily_adx >= 25",
        "daily_di_ok == 1",
        "osc_change_pct >= 30",
        "bars_gap >= 10",
        "vol_ratio_entry >= 1.2",
    ],
    "Pack_ATR_C": [
        "daily_adx >= 25",
        "daily_di_ok == 1",
        "osc_change_pct >= 30",
        "bars_gap >= 10",
        "vol_ratio_entry >= 1.2",
        "vol_behavior_ok_entry == 1",
    ],
    "Pack_ATR_D": [
        "daily_adx >= 25",
        "daily_di_ok == 1",
        "osc_change_pct >= 30",
        "bars_gap >= 10",
        "vol_ratio_entry >= 1.2",
        "vol_spike_ok_entry == 1",
    ],
    "Pack_ATR_E": [
        "daily_adx >= 25",
        "daily_di_ok == 1",
        "osc_change_pct >= 30",
        "bars_gap >= 10",
        "vol_ratio_entry >= 1.2",
        "session_ok_pivot == 1",
    ],
}


def _load_packs(path: str | None) -> Dict[str, List[str]]:
    if not path:
        return DEFAULT_PACKS_24H
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("PyYAML is required for --packs_yaml") from exc
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if isinstance(raw, dict) and "packs" in raw and isinstance(raw["packs"], dict):
        raw = raw["packs"]
    if not isinstance(raw, dict):
        raise SystemExit("Invalid packs YAML format. Expected mapping of pack name -> rules list.")
    out: Dict[str, List[str]] = {}
    for k, v in raw.items():
        if not isinstance(v, list):
            continue
        out[str(k)] = [str(r) for r in v]
    if not out:
        raise SystemExit("No valid packs found in YAML.")
    return out


def _derive_year(df: pd.DataFrame) -> pd.Series:
    if "event_time_ms" in df.columns:
        dt = pd.to_datetime(df["event_time_ms"], unit="ms", utc=True, errors="coerce")
    else:
        dt = pd.to_datetime(df.get("time"), utc=True, errors="coerce")
    return dt.dt.year.astype("Int64")


def _infer_tf_runid(path: Path, frame: pd.DataFrame) -> tuple[str, str]:
    m = re.match(r"^labeled_dfd05_(?P<tf>[^_]+)_(?P<rid>.+)$", path.stem)
    if m:
        return m.group("tf"), m.group("rid")
    tf = "unknown"
    if "timeframe" in frame.columns:
        vals = frame["timeframe"].dropna().astype(str).unique().tolist()
        if len(vals) == 1:
            tf = vals[0]
    rid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return tf, rid


def _sanitize_filename_token(token: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", token)


def _rule_lhs(rule: str) -> str | None:
    if " >= " in rule:
        lhs, _ = rule.split(" >= ", 1)
        return lhs.strip()
    if " == " in rule:
        lhs, _ = rule.split(" == ", 1)
        return lhs.strip()
    return None


def _assert_rule_columns_exist(
    df: pd.DataFrame,
    packs: Dict[str, List[str]],
    dump_missing_cols: bool,
) -> None:
    required: set[str] = set()
    for rules in packs.values():
        for rule in rules:
            lhs = _rule_lhs(rule)
            if lhs:
                required.add(lhs)
    missing = sorted(c for c in required if c not in df.columns)
    if not missing:
        return
    lines = [
        "Missing rule columns in labeled frame:",
        *[f"  - {c}" for c in missing],
    ]
    if dump_missing_cols:
        lines.append("Available columns:")
        for c in sorted(df.columns):
            lines.append(f"  - {c}")
    raise SystemExit("\n".join(lines))


def _apply_rule(df: pd.DataFrame, rule: str) -> pd.Series:
    if " >= " in rule:
        lhs, rhs = rule.split(" >= ", 1)
        if lhs not in df.columns:
            return pd.Series(False, index=df.index)
        x = pd.to_numeric(df[lhs], errors="coerce")
        xv = x.to_numpy(dtype=float)
        out = np.isfinite(xv) & (xv >= float(rhs.strip()))
        return pd.Series(out, index=df.index)
    if " == " in rule:
        lhs, rhs = rule.split(" == ", 1)
        if lhs not in df.columns:
            return pd.Series(False, index=df.index)
        rhs_tok = rhs.strip()
        if rhs_tok.lower() in {"true", "false"}:
            target = 1.0 if rhs_tok.lower() == "true" else 0.0
        else:
            target = float(rhs_tok)
        x = pd.to_numeric(df[lhs], errors="coerce")
        xv = x.to_numpy(dtype=float)
        out = np.isfinite(xv) & (xv == target)
        return pd.Series(out, index=df.index)
    return pd.Series(False, index=df.index)


def _safe_mean(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or len(df) == 0:
        return float("nan")
    x = pd.to_numeric(df[col], errors="coerce")
    if len(x) == 0:
        return float("nan")
    return float(np.nanmean(x.to_numpy(dtype=float)))


def _prepare_frame(df: pd.DataFrame, target_col: str, h: int) -> pd.DataFrame:
    suffix = f"{h}h"
    trunc_col = f"is_truncated_{suffix}"
    if target_col not in df.columns:
        raise SystemExit(f"Target column not found: {target_col}")
    for req in [f"ret_{suffix}", f"mfe_{suffix}", f"mae_{suffix}"]:
        if req not in df.columns:
            raise SystemExit(f"Missing required metric column for {h}h: {req}")
    if trunc_col not in df.columns:
        raise SystemExit(f"Missing truncation column for {h}h: {trunc_col}")

    out = df.copy()
    out = out[out[trunc_col].fillna(0) == 0].copy()
    y = pd.to_numeric(out[target_col], errors="coerce")
    out = out[np.isfinite(y.to_numpy(dtype=float))].copy()
    if out.empty:
        raise SystemExit("No valid rows after truncation/target filtering.")
    out["_year"] = _derive_year(out)
    return out.reset_index(drop=True)


def _coverage_flags(
    sub: pd.DataFrame,
    eligible_buckets: pd.Index,
    min_cover_global: int,
    min_cover_per_symbol_year: int,
) -> tuple[bool, bool, bool]:
    global_ok = len(sub) >= min_cover_global
    if len(eligible_buckets) == 0:
        return global_ok, False, False
    counts = sub.groupby(["symbol", "_year"], dropna=False).size()
    per_bucket_ok = True
    for bucket in eligible_buckets:
        if int(counts.get(bucket, 0)) < min_cover_per_symbol_year:
            per_bucket_ok = False
            break
    return global_ok, per_bucket_ok, bool(global_ok and per_bucket_ok)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate DFD05 time-only gate packs for a binary target")
    ap.add_argument("--labeled", required=True, help="Path to labeled parquet")
    ap.add_argument("--target", required=True, help="Binary target column (e.g. worked_mfe_24h_ge_100bps)")
    ap.add_argument("--horizon", type=int, default=24, help="Horizon hours to evaluate")
    ap.add_argument("--packs_yaml", required=False, help="Optional YAML with pack definitions")
    ap.add_argument("--outdir", default="data/reports", help="Output directory")
    ap.add_argument("--min_cover_per_symbol_year", type=int, default=20)
    ap.add_argument("--min_cover_global", type=int, default=200)
    ap.add_argument(
        "--dump_missing_cols",
        action="store_true",
        help="When rule columns are missing, include available-column dump in error",
    )
    args = ap.parse_args()

    labeled_path = Path(args.labeled)
    raw = pd.read_parquet(labeled_path)
    target_col = _resolve_target_column(list(raw.columns), args.target)
    if target_col != args.target:
        print(f"Resolved target alias: {args.target} -> {target_col}")
    df = _prepare_frame(raw, target_col=target_col, h=args.horizon)
    packs = _load_packs(args.packs_yaml)
    _assert_rule_columns_exist(df, packs, dump_missing_cols=bool(args.dump_missing_cols))

    bucket_counts = df.groupby(["symbol", "_year"], dropna=False).size()
    eligible_buckets = bucket_counts[bucket_counts >= int(args.min_cover_per_symbol_year)].index

    suffix = f"{int(args.horizon)}h"
    ret_col = f"ret_{suffix}"
    mfe_col = f"mfe_{suffix}"
    mae_col = f"mae_{suffix}"

    rows: list[dict[str, Any]] = []
    for pack_name, rules in packs.items():
        mask = pd.Series(True, index=df.index)
        for rule in rules:
            mask &= _apply_rule(df, rule=rule)
        sub = df.loc[mask].copy()

        target_rate = _safe_mean(sub, target_col)
        mean_ret = _safe_mean(sub, ret_col)
        mean_mfe = _safe_mean(sub, mfe_col)
        mean_mae = _safe_mean(sub, mae_col)

        year_stats = (
            sub.groupby("_year", dropna=False)
            .agg(
                target_rate=(target_col, lambda s: pd.to_numeric(s, errors="coerce").mean()),
                mean_ret=(ret_col, lambda s: pd.to_numeric(s, errors="coerce").mean()),
            )
            .reset_index()
            if len(sub)
            else pd.DataFrame(columns=["_year", "target_rate", "mean_ret"])
        )
        worst_year_target_rate = (
            float(year_stats["target_rate"].min()) if len(year_stats) else float("nan")
        )
        worst_year_mean_ret = float(year_stats["mean_ret"].min()) if len(year_stats) else float("nan")

        sy_rows: list[dict[str, Any]] = []
        for (sym, yr), g in sub.groupby(["symbol", "_year"], dropna=False):
            sy_rows.append(
                {
                    "section": "per_symbol_year",
                    "pack": pack_name,
                    "symbol": str(sym) if pd.notna(sym) else None,
                    "year": int(yr) if pd.notna(yr) else None,
                    "n_valid": float(len(g)),
                    "target_rate": _safe_mean(g, target_col),
                    f"mean_ret_{suffix}": _safe_mean(g, ret_col),
                    f"mean_mfe_{suffix}": _safe_mean(g, mfe_col),
                    f"mean_mae_{suffix}": _safe_mean(g, mae_col),
                    "worst_year_target_rate": np.nan,
                    "worst_year_mean_ret": np.nan,
                    "min_symbol_year_n": np.nan,
                    "mean_symbol_year_n": np.nan,
                    "cover_global_ok": np.nan,
                    "cover_per_symbol_year_ok": np.nan,
                    "coverage_ok": np.nan,
                    "rules": "; ".join(rules),
                }
            )
        sy_df = pd.DataFrame(sy_rows)
        min_symbol_year_n = float(sy_df["n_valid"].min()) if len(sy_df) else float("nan")
        mean_symbol_year_n = float(sy_df["n_valid"].mean()) if len(sy_df) else float("nan")

        cover_global_ok, cover_per_symbol_year_ok, coverage_ok = _coverage_flags(
            sub=sub,
            eligible_buckets=eligible_buckets,
            min_cover_global=int(args.min_cover_global),
            min_cover_per_symbol_year=int(args.min_cover_per_symbol_year),
        )

        rows.append(
            {
                "section": "overall",
                "pack": pack_name,
                "symbol": None,
                "year": None,
                "n_valid": float(len(sub)),
                "target_rate": target_rate,
                f"mean_ret_{suffix}": mean_ret,
                f"mean_mfe_{suffix}": mean_mfe,
                f"mean_mae_{suffix}": mean_mae,
                "worst_year_target_rate": worst_year_target_rate,
                "worst_year_mean_ret": worst_year_mean_ret,
                "min_symbol_year_n": min_symbol_year_n,
                "mean_symbol_year_n": mean_symbol_year_n,
                "cover_global_ok": bool(cover_global_ok),
                "cover_per_symbol_year_ok": bool(cover_per_symbol_year_ok),
                "coverage_ok": bool(coverage_ok),
                "rules": "; ".join(rules),
            }
        )
        rows.extend(sy_rows)

    out_df = pd.DataFrame(rows)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tf, rid = _infer_tf_runid(labeled_path, df)
    safe_target = _sanitize_filename_token(target_col)
    out_path = out_dir / f"dfd05_gate_packs_timeonly_{tf}_{rid}_{safe_target}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"gate_packs_timeonly: {out_path}")


if __name__ == "__main__":
    main()
