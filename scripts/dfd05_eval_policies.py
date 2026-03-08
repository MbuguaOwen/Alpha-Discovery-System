from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RunMeta:
    labeled_path: str
    variant: str
    timeframe: str
    session_mode: str
    run_id: str


def _parse_csv_tokens(raw: str) -> List[str]:
    out: List[str] = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if t:
            out.append(t)
    return out


def _parse_int_csv(raw: str, field_name: str) -> List[int]:
    vals: List[int] = []
    for token in _parse_csv_tokens(raw):
        vals.append(int(token))
    if not vals:
        raise ValueError(f"No valid values parsed from --{field_name}")
    return vals


def _parse_run_id_from_path(path: str) -> RunMeta:
    p = Path(path)
    name = p.name
    pat = re.compile(
        r"^labeled_dfd05_[^_]+_(?P<runid>\d{8}T\d{6}Z_(?P<variant>.+)_(?P<timeframe>[a-z0-9]+)_(?P<session_mode>session_on|session_off|config_default))\.parquet$"
    )
    m = pat.match(name)
    if m is None:
        raise ValueError(
            f"Could not parse variant/timeframe/session from labeled file name: {name}. "
            "Provide --run_logs for explicit metadata mapping."
        )
    return RunMeta(
        labeled_path=str(p),
        variant=str(m.group("variant")),
        timeframe=str(m.group("timeframe")),
        session_mode=str(m.group("session_mode")),
        run_id=str(m.group("runid")),
    )


def _collect_runs(
    run_logs: Optional[str],
    labeled_paths_csv: Optional[str],
    labeled_dir: Optional[str],
) -> List[RunMeta]:
    runs: List[RunMeta] = []
    if run_logs is not None and str(run_logs).strip() != "":
        df = pd.read_csv(run_logs)
        required = {"labeled_path", "variant", "timeframe", "session_mode"}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"run_logs is missing required columns: {missing}")
        for _, r in df.iterrows():
            lp = str(r["labeled_path"])
            rid = str(r["run_id"]) if "run_id" in df.columns else Path(lp).stem
            runs.append(
                RunMeta(
                    labeled_path=lp,
                    variant=str(r["variant"]),
                    timeframe=str(r["timeframe"]),
                    session_mode=str(r["session_mode"]),
                    run_id=rid,
                )
            )

    if labeled_paths_csv is not None and str(labeled_paths_csv).strip() != "":
        for raw in _parse_csv_tokens(labeled_paths_csv):
            runs.append(_parse_run_id_from_path(raw))

    if labeled_dir is not None and str(labeled_dir).strip() != "":
        for p in sorted(Path(labeled_dir).glob("labeled_dfd05_*.parquet")):
            runs.append(_parse_run_id_from_path(str(p)))

    if not runs:
        raise ValueError("No labeled runs discovered. Provide --run_logs, --labeled_paths, or --labeled_dir.")

    dedup: Dict[str, RunMeta] = {}
    for r in runs:
        key = f"{Path(r.labeled_path).resolve()}::{r.variant}::{r.timeframe}::{r.session_mode}"
        dedup[key] = r
    out = list(dedup.values())
    out.sort(key=lambda x: (x.variant, x.timeframe, x.session_mode, x.run_id, x.labeled_path))
    return out


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    if not mask.any():
        return np.nan
    return float(np.average(v[mask], weights=w[mask]))


def _bool_series_from_numeric(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").fillna(1.0).to_numpy(dtype=float)
    return pd.Series(x == 0.0, index=series.index)


def _required_cols_for_horizons(h_early: int, h_mid: int, h_late: int) -> List[str]:
    return [
        "symbol",
        "event_time_ms",
        f"ret_{h_early}h",
        f"ret_{h_mid}h",
        f"ret_{h_late}h",
        f"mae_{h_early}h",
        f"is_truncated_{h_early}h",
        f"is_truncated_{h_mid}h",
        f"is_truncated_{h_late}h",
    ]


def _safe_year_col(df: pd.DataFrame) -> pd.Series:
    if "event_time_ms" in df.columns:
        dt = pd.to_datetime(df["event_time_ms"], unit="ms", utc=True, errors="coerce")
    elif "entry_time_ms" in df.columns:
        dt = pd.to_datetime(df["entry_time_ms"], unit="ms", utc=True, errors="coerce")
    else:
        raise ValueError("Labeled frame must contain event_time_ms or entry_time_ms.")
    if dt.isna().any():
        raise ValueError("Timestamp parsing failed: invalid event_time_ms/entry_time_ms values.")
    return dt.dt.year.astype("int64")


def _compute_policy_vector(
    policy_name: str,
    ret_early: np.ndarray,
    ret_mid: np.ndarray,
    ret_late: np.ndarray,
    mae_early: np.ndarray,
    x_bps: float,
    y_bps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x = float(x_bps) / 10000.0
    y = float(y_bps) / 10000.0
    bad_early = (ret_early <= -x) | (mae_early <= -x)

    if policy_name == "P1_EarlyCutHold":
        policy_ret = np.where(bad_early, ret_early, ret_late)
    elif policy_name == "P2_EarlyCutMidTake":
        mid_take = ret_mid >= y
        policy_ret = np.where(bad_early, ret_early, np.where(mid_take, ret_mid, ret_late))
    elif policy_name == "P3_QuantileBased":
        mid_take = ret_mid >= y
        policy_ret = np.where(bad_early, ret_early, np.where(mid_take, ret_mid, ret_late))
    else:
        raise ValueError(f"Unsupported policy_name: {policy_name}")
    return policy_ret.astype("float64"), bad_early.astype(bool)


def _summarize_policy(
    policy_name: str,
    df_valid: pd.DataFrame,
    policy_ret: np.ndarray,
    early_exit_mask: np.ndarray,
    x_bps: float,
    y_bps: float,
    min_n_valid_global: int,
    min_n_valid_per_symbol_year: int,
) -> Dict[str, object]:
    n_valid = int(len(df_valid))
    if n_valid <= 0:
        raise ValueError("Internal error: empty df_valid passed to _summarize_policy.")

    work = df_valid.copy()
    work["policy_ret"] = policy_ret
    work["year"] = _safe_year_col(work)

    sy = (
        work.groupby(["symbol", "year"], dropna=False)["policy_ret"]
        .agg(n_valid="size", mean_ret="mean")
        .reset_index()
        .sort_values(["symbol", "year"], kind="mergesort")
    )
    yearly = (
        sy.groupby("year", dropna=False)
        .apply(lambda g: pd.Series({"year_mean_ret": _weighted_mean(g["mean_ret"], g["n_valid"])}))
        .reset_index()
        .sort_values("year", kind="mergesort")
    )

    mean_ret = float(np.mean(policy_ret))
    win_rate = float(np.mean(policy_ret > 0.0))
    out: Dict[str, object] = {
        "policy": policy_name,
        "x_bps": float(x_bps),
        "y_bps": float(y_bps),
        "n_valid": int(n_valid),
        "mean_policy_ret": mean_ret,
        "mean_policy_ret_bps": float(mean_ret * 10000.0),
        "median_policy_ret": float(np.median(policy_ret)),
        "p25_policy_ret": float(np.quantile(policy_ret, 0.25)),
        "p75_policy_ret": float(np.quantile(policy_ret, 0.75)),
        "win_rate": win_rate,
        "down_rate": float(1.0 - win_rate),
        "early_exit_rate": float(np.mean(early_exit_mask.astype(float))),
        "worst_year_mean_policy_ret": float(yearly["year_mean_ret"].min()) if len(yearly) > 0 else np.nan,
        "worst_year_mean_policy_ret_bps": float(yearly["year_mean_ret"].min() * 10000.0)
        if len(yearly) > 0
        else np.nan,
        "min_symbol_year_n_valid": int(sy["n_valid"].min()) if len(sy) > 0 else 0,
        "eligible_year_count": int(len(yearly)),
    }
    out["selection_eligible"] = bool(
        out["n_valid"] >= int(min_n_valid_global)
        and out["eligible_year_count"] > 0
        and out["min_symbol_year_n_valid"] >= int(min_n_valid_per_symbol_year)
    )
    return out


def evaluate_policies(
    runs: Sequence[RunMeta],
    horizons: Sequence[int],
    p1_cut_bps: float,
    p2_take_bps: float,
    min_n_valid_global: int,
    min_n_valid_per_symbol_year: int,
) -> Tuple[pd.DataFrame, List[str]]:
    if len(horizons) != 3:
        raise ValueError("--horizons must provide exactly 3 values for policy evaluation, e.g. 4,24,72.")
    h_early, h_mid, h_late = [int(h) for h in horizons]
    rows: List[Dict[str, object]] = []
    warnings: List[str] = []

    for meta in runs:
        labeled_path = Path(meta.labeled_path)
        if not labeled_path.exists():
            warnings.append(f"Missing labeled parquet, skipped: {labeled_path}")
            continue
        cols = _required_cols_for_horizons(h_early, h_mid, h_late)
        frame = pd.read_parquet(labeled_path)
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise ValueError(f"{labeled_path} missing required policy columns: {missing}")
        n_total = int(len(frame))
        if n_total <= 0:
            warnings.append(f"Empty labeled frame, skipped: {labeled_path}")
            continue

        trunc_ok = (
            _bool_series_from_numeric(frame[f"is_truncated_{h_early}h"])
            & _bool_series_from_numeric(frame[f"is_truncated_{h_mid}h"])
            & _bool_series_from_numeric(frame[f"is_truncated_{h_late}h"])
        )
        req_num_cols = [
            f"ret_{h_early}h",
            f"ret_{h_mid}h",
            f"ret_{h_late}h",
            f"mae_{h_early}h",
        ]
        finite_mask = np.ones(n_total, dtype=bool)
        for c in req_num_cols:
            x = pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float)
            finite_mask &= np.isfinite(x)
        valid_mask = trunc_ok.to_numpy(dtype=bool) & finite_mask
        n_valid = int(valid_mask.sum())
        if n_total != len(frame):
            raise ValueError(f"n_total mismatch for {labeled_path}: n_total={n_total}, len(df)={len(frame)}")
        if n_valid < 0 or n_valid > n_total:
            raise ValueError(f"Invalid n_valid for {labeled_path}: n_total={n_total}, n_valid={n_valid}")
        if n_valid == 0:
            warnings.append(f"No valid rows after truncation/finite filtering: {labeled_path}")
            continue

        work = frame.loc[valid_mask, ["symbol", "event_time_ms"] + req_num_cols].copy()
        ret_early = pd.to_numeric(work[f"ret_{h_early}h"], errors="coerce").to_numpy(dtype=float)
        ret_mid = pd.to_numeric(work[f"ret_{h_mid}h"], errors="coerce").to_numpy(dtype=float)
        ret_late = pd.to_numeric(work[f"ret_{h_late}h"], errors="coerce").to_numpy(dtype=float)
        mae_early = pd.to_numeric(work[f"mae_{h_early}h"], errors="coerce").to_numpy(dtype=float)

        p25_early_bps = float(np.quantile(ret_early, 0.25) * 10000.0)
        p75_mid_bps = float(np.quantile(ret_mid, 0.75) * 10000.0)
        p3_x_bps = float(abs(p25_early_bps))
        p3_y_bps = float(p75_mid_bps)

        pol_defs = [
            ("P1_EarlyCutHold", float(p1_cut_bps), np.nan),
            ("P2_EarlyCutMidTake", float(p1_cut_bps), float(p2_take_bps)),
            ("P3_QuantileBased", float(p3_x_bps), float(p3_y_bps)),
        ]
        for name, x_bps, y_bps in pol_defs:
            y_use = float(y_bps if np.isfinite(y_bps) else p2_take_bps)
            policy_ret, early_exit = _compute_policy_vector(
                policy_name=name,
                ret_early=ret_early,
                ret_mid=ret_mid,
                ret_late=ret_late,
                mae_early=mae_early,
                x_bps=float(x_bps),
                y_bps=float(y_use),
            )
            stats = _summarize_policy(
                policy_name=name,
                df_valid=work,
                policy_ret=policy_ret,
                early_exit_mask=early_exit,
                x_bps=float(x_bps),
                y_bps=float(y_use),
                min_n_valid_global=min_n_valid_global,
                min_n_valid_per_symbol_year=min_n_valid_per_symbol_year,
            )
            rows.append(
                {
                    "variant": meta.variant,
                    "timeframe": meta.timeframe,
                    "session_mode": meta.session_mode,
                    "run_id": meta.run_id,
                    "labeled_path": str(labeled_path),
                    "h_early": int(h_early),
                    "h_mid": int(h_mid),
                    "h_late": int(h_late),
                    "n_total": int(n_total),
                    "n_valid": int(n_valid),
                    **stats,
                }
            )
            if int(stats["min_symbol_year_n_valid"]) < int(min_n_valid_per_symbol_year):
                warnings.append(
                    f"Low symbol-year coverage: variant={meta.variant} tf={meta.timeframe} "
                    f"session={meta.session_mode} policy={name} "
                    f"min_symbol_year_n_valid={int(stats['min_symbol_year_n_valid'])} "
                    f"< {int(min_n_valid_per_symbol_year)}"
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out, warnings
    out = out.sort_values(
        ["variant", "timeframe", "session_mode", "policy", "run_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return out, warnings


def _build_markdown_summary(
    policy_df: pd.DataFrame,
    warnings: Sequence[str],
    min_n_valid_global: int,
    min_n_valid_per_symbol_year: int,
) -> str:
    lines: List[str] = []
    lines.append("# DFD05 Policy Evaluation (Time-Based)")
    lines.append("")
    lines.append(f"- Generated UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(
        f"- Eligibility constraints: min_n_valid_global={int(min_n_valid_global)}, "
        f"min_n_valid_per_symbol_year={int(min_n_valid_per_symbol_year)}"
    )
    lines.append("")
    if policy_df.empty:
        lines.append("## Result")
        lines.append("")
        lines.append("- No policy rows were produced.")
    else:
        elig = policy_df[policy_df["selection_eligible"].astype(bool)].copy()
        lines.append("## Best Policies (Eligible Only)")
        lines.append("")
        if elig.empty:
            lines.append("- No eligible policy rows met minimum thresholds.")
        else:
            top_mean = elig.sort_values(
                ["mean_policy_ret", "worst_year_mean_policy_ret", "win_rate"],
                ascending=[False, False, False],
                kind="mergesort",
            ).iloc[0]
            top_robust = elig.sort_values(
                ["worst_year_mean_policy_ret", "mean_policy_ret", "win_rate"],
                ascending=[False, False, False],
                kind="mergesort",
            ).iloc[0]
            lines.append(
                f"- Max mean policy return: {top_mean['policy']} ({top_mean['variant']}, "
                f"{top_mean['timeframe']}, {top_mean['session_mode']}) "
                f"mean={float(top_mean['mean_policy_ret_bps']):.2f} bps, "
                f"worst_year={float(top_mean['worst_year_mean_policy_ret_bps']):.2f} bps."
            )
            lines.append(
                f"- Max worst-year robustness: {top_robust['policy']} ({top_robust['variant']}, "
                f"{top_robust['timeframe']}, {top_robust['session_mode']}) "
                f"worst_year={float(top_robust['worst_year_mean_policy_ret_bps']):.2f} bps, "
                f"mean={float(top_robust['mean_policy_ret_bps']):.2f} bps."
            )
    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    if warnings:
        lines.extend([f"- {w}" for w in warnings])
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Policies")
    lines.append("")
    lines.append("- P1_EarlyCutHold: exit early if 4h is bad, else hold to 72h.")
    lines.append("- P2_EarlyCutMidTake: early cut, else take at 24h if strong, else hold to 72h.")
    lines.append("- P3_QuantileBased: thresholds from in-sample quantiles (p25/p75).")
    return "\n".join(lines).rstrip() + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate time-based exit policies on DFD05 labeled outputs.")
    ap.add_argument("--run_logs", default=None, help="Path to run_logs.csv from master/ablation outputs.")
    ap.add_argument("--labeled_paths", default=None, help="CSV list of labeled parquet paths.")
    ap.add_argument("--labeled_dir", default=None, help="Directory containing labeled_dfd05_*.parquet files.")
    ap.add_argument("--horizons", default="4,24,72", help="Three policy horizons in hours, e.g. 4,24,72.")
    ap.add_argument("--p1_cut_bps", type=float, default=25.0, help="X threshold (bps) for early cut in P1/P2.")
    ap.add_argument("--p2_take_bps", type=float, default=25.0, help="Y threshold (bps) for 24h take in P2.")
    ap.add_argument("--min_n_valid_global", type=int, default=200, help="Global minimum n_valid for eligibility.")
    ap.add_argument(
        "--min_n_valid_per_symbol_year",
        type=int,
        default=10,
        help="Minimum symbol-year n_valid for eligibility.",
    )
    ap.add_argument("--outdir", required=True, help="Output directory for policy artifacts.")
    return ap


def run_policy_evaluation(args: argparse.Namespace) -> Dict[str, Path]:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    runs = _collect_runs(
        run_logs=args.run_logs,
        labeled_paths_csv=args.labeled_paths,
        labeled_dir=args.labeled_dir,
    )
    horizons = _parse_int_csv(args.horizons, field_name="horizons")
    if int(args.min_n_valid_global) < 1:
        raise ValueError("--min_n_valid_global must be >= 1")
    if int(args.min_n_valid_per_symbol_year) < 1:
        raise ValueError("--min_n_valid_per_symbol_year must be >= 1")

    policy_df, warnings = evaluate_policies(
        runs=runs,
        horizons=horizons,
        p1_cut_bps=float(args.p1_cut_bps),
        p2_take_bps=float(args.p2_take_bps),
        min_n_valid_global=int(args.min_n_valid_global),
        min_n_valid_per_symbol_year=int(args.min_n_valid_per_symbol_year),
    )

    results_path = outdir / "policy_results.csv"
    summary_path = outdir / "policy_summary.md"
    policy_df.to_csv(results_path, index=False)
    summary_path.write_text(
        _build_markdown_summary(
            policy_df=policy_df,
            warnings=warnings,
            min_n_valid_global=int(args.min_n_valid_global),
            min_n_valid_per_symbol_year=int(args.min_n_valid_per_symbol_year),
        ),
        encoding="utf-8",
    )
    return {"policy_results": results_path, "policy_summary": summary_path}


def main() -> None:
    args = build_arg_parser().parse_args()
    outputs = run_policy_evaluation(args)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
