from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.dfd05_eval_master as em


ABLATION_PRESETS: Dict[str, str] = {
    "baseline": "configs/dfd05_pine16_baseline.yaml",
    "gate_only": "configs/dfd05_ablate_gate_only.yaml",
    "bos_only": "configs/dfd05_ablate_bos_only.yaml",
    "bos_plus_vol": "configs/dfd05_ablate_bos_plus_vol.yaml",
    "prod": "configs/dfd05_pine16_prod_bos_vol.yaml",
}


def _parse_csv_tokens(raw: str) -> List[str]:
    out: List[str] = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if t:
            out.append(t)
    return out


def _parse_ablation_configs(raw: str) -> List[str]:
    items: List[str] = []
    for token in _parse_csv_tokens(raw):
        key = token.strip()
        low = key.lower()
        if low in ABLATION_PRESETS:
            items.append(low)
            continue
        p = Path(key)
        if p.exists():
            items.append(str(p))
            continue
        raise ValueError(
            f"Unknown ablation config token: {token}. Use one of {list(ABLATION_PRESETS)} or a config path."
        )
    if not items:
        raise ValueError("No configs resolved from --configs")
    dedup: Dict[str, str] = {}
    for token in items:
        dedup[token.lower()] = token
    return list(dedup.values())


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    lowered = series.astype(str).str.strip().str.lower()
    out = lowered.isin({"1", "true", "yes", "y", "t"})
    out = out.where(~series.isna(), False)
    return out


def _compute_lift_vs_baseline(
    overall: pd.DataFrame,
    baseline_variant: str = "baseline",
    level: str = "executed",
) -> pd.DataFrame:
    if overall.empty:
        return pd.DataFrame()

    scoped = overall.copy()
    scoped["level"] = scoped["level"].astype(str).str.lower()
    scoped = scoped[scoped["level"] == level.lower()].copy()
    if scoped.empty:
        return pd.DataFrame()

    keys = ["timeframe", "session_mode", "horizon_h"]
    metric_cols = [
        "mean_ret",
        "worst_year_mean_ret",
        "good_rate",
        "worst_year_good_rate",
        "up_rate",
    ]
    for col in metric_cols + ["n_valid", "min_symbol_year_n_valid"]:
        if col in scoped.columns:
            scoped[col] = pd.to_numeric(scoped[col], errors="coerce")
    scoped["selection_eligible"] = _bool_series(scoped.get("selection_eligible", pd.Series(False, index=scoped.index)))

    baseline = scoped[scoped["variant"] == baseline_variant][
        keys + metric_cols + ["n_valid", "min_symbol_year_n_valid", "selection_eligible"]
    ].copy()
    baseline = baseline.rename(
        columns={
            "mean_ret": "baseline_mean_ret",
            "worst_year_mean_ret": "baseline_worst_year_mean_ret",
            "good_rate": "baseline_good_rate",
            "worst_year_good_rate": "baseline_worst_year_good_rate",
            "up_rate": "baseline_up_rate",
            "n_valid": "baseline_n_valid",
            "min_symbol_year_n_valid": "baseline_min_symbol_year_n_valid",
            "selection_eligible": "baseline_selection_eligible",
        }
    )

    merged = scoped.merge(baseline, on=keys, how="left")
    merged["config"] = merged["variant"]
    merged["delta_mean_ret"] = merged["mean_ret"] - merged["baseline_mean_ret"]
    merged["delta_worst_year_mean_ret"] = merged["worst_year_mean_ret"] - merged["baseline_worst_year_mean_ret"]
    merged["delta_good_rate"] = merged["good_rate"] - merged["baseline_good_rate"]
    merged["delta_worst_year_good_rate"] = (
        merged["worst_year_good_rate"] - merged["baseline_worst_year_good_rate"]
    )
    merged["delta_up_rate"] = merged["up_rate"] - merged["baseline_up_rate"]

    out_cols = [
        "config",
        "timeframe",
        "session_mode",
        "horizon_h",
        "delta_mean_ret",
        "delta_worst_year_mean_ret",
        "delta_good_rate",
        "delta_worst_year_good_rate",
        "delta_up_rate",
        "n_valid",
        "baseline_n_valid",
        "selection_eligible",
        "baseline_selection_eligible",
    ]
    out = merged[out_cols].copy()
    return out.sort_values(["config", "timeframe", "session_mode", "horizon_h"]).reset_index(drop=True)


def _pairwise_delta(
    overall: pd.DataFrame,
    lhs_config: str,
    rhs_config: str,
    level: str = "executed",
) -> pd.DataFrame:
    scoped = overall.copy()
    scoped["level"] = scoped["level"].astype(str).str.lower()
    scoped = scoped[scoped["level"] == level.lower()].copy()
    if scoped.empty:
        return pd.DataFrame()

    keys = ["timeframe", "session_mode", "horizon_h"]
    metric_cols = [
        "mean_ret",
        "worst_year_mean_ret",
        "good_rate",
        "worst_year_good_rate",
        "up_rate",
        "n_valid",
        "min_symbol_year_n_valid",
        "selection_eligible",
    ]
    lhs = scoped[scoped["variant"] == lhs_config][keys + metric_cols].copy()
    rhs = scoped[scoped["variant"] == rhs_config][keys + metric_cols].copy()
    if lhs.empty or rhs.empty:
        return pd.DataFrame()

    lhs = lhs.rename(columns={c: f"{c}_lhs" for c in metric_cols})
    rhs = rhs.rename(columns={c: f"{c}_rhs" for c in metric_cols})
    m = lhs.merge(rhs, on=keys, how="inner")
    if m.empty:
        return m

    for col in [
        "mean_ret_lhs",
        "mean_ret_rhs",
        "worst_year_mean_ret_lhs",
        "worst_year_mean_ret_rhs",
        "good_rate_lhs",
        "good_rate_rhs",
        "worst_year_good_rate_lhs",
        "worst_year_good_rate_rhs",
        "up_rate_lhs",
        "up_rate_rhs",
    ]:
        m[col] = pd.to_numeric(m[col], errors="coerce")
    m["selection_eligible_lhs"] = _bool_series(m["selection_eligible_lhs"])
    m["selection_eligible_rhs"] = _bool_series(m["selection_eligible_rhs"])
    m["delta_mean_ret"] = m["mean_ret_lhs"] - m["mean_ret_rhs"]
    m["delta_worst_year_mean_ret"] = m["worst_year_mean_ret_lhs"] - m["worst_year_mean_ret_rhs"]
    m["delta_good_rate"] = m["good_rate_lhs"] - m["good_rate_rhs"]
    m["delta_worst_year_good_rate"] = m["worst_year_good_rate_lhs"] - m["worst_year_good_rate_rhs"]
    m["delta_up_rate"] = m["up_rate_lhs"] - m["up_rate_rhs"]
    return m


def _fmt_bps(v: float) -> str:
    if not np.isfinite(v):
        return "n/a"
    return f"{(float(v) * 10000.0):+.2f}"


def _fmt_pp(v: float) -> str:
    if not np.isfinite(v):
        return "n/a"
    return f"{(float(v) * 100.0):+.2f}"


def _pick_best_row(df: pd.DataFrame, metric: str) -> Optional[pd.Series]:
    if df.empty or metric not in df.columns:
        return None
    valid = df[pd.to_numeric(df[metric], errors="coerce").notna()].copy()
    if valid.empty:
        return None
    ranked = valid.sort_values(metric, ascending=False, kind="mergesort")
    return ranked.iloc[0]


def _write_summary(
    out_path: Path,
    overall: pd.DataFrame,
    best: pd.DataFrame,
    lift: pd.DataFrame,
    min_n_valid_global: int,
    min_n_valid_per_symbol_year: int,
) -> None:
    lines: List[str] = []
    lines.append("# DFD05 Ablation Evaluation")
    lines.append("")
    lines.append(f"- Generated UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Selection level default: executed")
    lines.append(
        f"- Eligibility constraints: min_n_valid_global={int(min_n_valid_global)}, "
        f"min_n_valid_per_symbol_year={int(min_n_valid_per_symbol_year)}"
    )
    lines.append("")

    best_exec = best.copy()
    if not best_exec.empty:
        best_exec["level"] = best_exec["level"].astype(str).str.lower()
        best_exec = best_exec[best_exec["level"] == "executed"].copy()
        best_exec = best_exec[best_exec["selection_status"].astype(str) == "selected"].copy()
    if best_exec.empty:
        lines.append("## Final Recommendation")
        lines.append("")
        lines.append("- No eligible executed rows met minimum sample constraints.")
    else:
        for col in [
            "worst_year_mean_ret",
            "worst_year_good_rate",
            "overall_mean_ret",
            "overall_good_rate",
            "selected_horizon_h",
        ]:
            best_exec[col] = pd.to_numeric(best_exec[col], errors="coerce")
        pick = best_exec.sort_values(
            ["worst_year_mean_ret", "worst_year_good_rate", "overall_mean_ret", "overall_good_rate"],
            ascending=[False, False, False, False],
            na_position="last",
        ).iloc[0]
        lines.append("## Final Recommendation")
        lines.append("")
        lines.append(
            "- Recommendation tuple: "
            f"({pick['timeframe']}, {pick['session_mode']}, {int(pick['selected_horizon_h'])}h, {pick['variant']})"
        )
        lines.append(
            f"- Drivers: worst_year_mean_ret={_fmt_bps(float(pick['worst_year_mean_ret']))} bps, "
            f"worst_year_good_rate={_fmt_pp(float(pick['worst_year_good_rate']))} pp, "
            f"overall_mean_ret={_fmt_bps(float(pick['overall_mean_ret']))} bps."
        )

    lines.append("")
    lines.append("## Lift Attribution")
    lines.append("")

    robust_overall = overall.copy()
    robust_overall["level"] = robust_overall["level"].astype(str).str.lower()
    robust_overall = robust_overall[robust_overall["level"] == "executed"].copy()
    robust_overall["selection_eligible"] = _bool_series(
        robust_overall.get("selection_eligible", pd.Series(False, index=robust_overall.index))
    )
    robust_overall = robust_overall[robust_overall["selection_eligible"]].copy()

    bos_vs_base = _pairwise_delta(robust_overall, lhs_config="bos_only", rhs_config="baseline")
    bos_pick = _pick_best_row(bos_vs_base, "delta_mean_ret")
    if bos_pick is not None:
        lines.append(
            f"- BOS only adds {_fmt_bps(float(bos_pick['delta_mean_ret']))} bps at {int(bos_pick['horizon_h'])}h "
            f"({bos_pick['timeframe']}, {bos_pick['session_mode']}); "
            f"worst-year mean-ret shift={_fmt_bps(float(bos_pick['delta_worst_year_mean_ret']))} bps."
        )
    else:
        lines.append("- BOS only attribution unavailable under current robustness filters.")

    vol_vs_bos = _pairwise_delta(robust_overall, lhs_config="bos_plus_vol", rhs_config="bos_only")
    vol_pick = _pick_best_row(vol_vs_bos, "delta_worst_year_good_rate")
    if vol_pick is not None:
        lines.append(
            f"- Adding vol gate to BOS changes worst-year good-rate by "
            f"{_fmt_pp(float(vol_pick['delta_worst_year_good_rate']))} pp at {int(vol_pick['horizon_h'])}h "
            f"({vol_pick['timeframe']}, {vol_pick['session_mode']})."
        )
    else:
        lines.append("- Vol-gate stability attribution unavailable under current robustness filters.")

    prod_vs_bosvol = _pairwise_delta(robust_overall, lhs_config="prod", rhs_config="bos_plus_vol")
    prod_pick = _pick_best_row(prod_vs_bosvol, "delta_mean_ret")
    if prod_pick is not None:
        lines.append(
            f"- One-trade-at-a-time (prod vs bos_plus_vol) shifts mean-ret by "
            f"{_fmt_bps(float(prod_pick['delta_mean_ret']))} bps at {int(prod_pick['horizon_h'])}h "
            f"({prod_pick['timeframe']}, {prod_pick['session_mode']}); "
            f"delta up-rate={_fmt_pp(float(prod_pick['delta_up_rate']))} pp."
        )
    else:
        lines.append("- One-trade-at-a-time attribution unavailable under current robustness filters.")

    if not best.empty:
        best_tmp = best.copy()
        best_tmp["level"] = best_tmp["level"].astype(str).str.lower()
        best_tmp = best_tmp[
            (best_tmp["level"] == "executed") & (best_tmp["selection_status"].astype(str) == "selected")
        ]
        h_a = best_tmp[best_tmp["variant"] == "bos_plus_vol"][
            ["timeframe", "session_mode", "selected_horizon_h"]
        ].rename(columns={"selected_horizon_h": "h_bos_plus_vol"})
        h_b = best_tmp[best_tmp["variant"] == "prod"][["timeframe", "session_mode", "selected_horizon_h"]].rename(
            columns={"selected_horizon_h": "h_prod"}
        )
        h_merge = h_a.merge(h_b, on=["timeframe", "session_mode"], how="inner")
        if not h_merge.empty:
            changed = int((h_merge["h_bos_plus_vol"] != h_merge["h_prod"]).sum())
            lines.append(
                f"- One-trade horizon impact: selected horizon changed in {changed}/{len(h_merge)} "
                "matched timeframe/session cells."
            )

    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    low_cells = overall.copy()
    low_cells["level"] = low_cells["level"].astype(str).str.lower()
    low_cells = low_cells[low_cells["level"] == "executed"].copy()
    low_cells["selection_eligible"] = _bool_series(
        low_cells.get("selection_eligible", pd.Series(False, index=low_cells.index))
    )
    low_cells = low_cells[~low_cells["selection_eligible"]].copy()
    if low_cells.empty:
        lines.append("- None")
    else:
        low_cells = low_cells.sort_values(["variant", "timeframe", "session_mode", "horizon_h"])
        for _, r in low_cells.iterrows():
            n_valid_raw = pd.to_numeric(r.get("n_valid"), errors="coerce")
            n_valid = int(n_valid_raw) if pd.notna(n_valid_raw) else 0
            min_sy = pd.to_numeric(r.get("min_symbol_year_n_valid"), errors="coerce")
            min_sy_txt = "nan" if not np.isfinite(min_sy) else str(int(min_sy))
            lines.append(
                f"- Ineligible cell: config={r['variant']} tf={r['timeframe']} session={r['session_mode']} "
                f"h={int(r['horizon_h'])} n_valid={n_valid} min_symbol_year_n_valid={min_sy_txt}"
            )

    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `overall.csv`")
    lines.append("- `per_symbol_year.csv`")
    lines.append("- `best_selection.csv`")
    lines.append("- `lift_vs_baseline.csv`")
    lines.append("- `run_logs.csv`")
    lines.append("- `summary.md`")
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_ablation_evaluation(args: argparse.Namespace) -> Dict[str, Path]:
    resolved_configs = _parse_ablation_configs(args.configs)
    config_tokens = ",".join(resolved_configs)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    master_args = argparse.Namespace(
        configs=config_tokens,
        parity=args.parity,
        timeframes=args.timeframes,
        horizons=args.horizons,
        compare_sessions=bool(args.compare_sessions),
        time_scale_params=args.time_scale_params,
        outdir=str(outdir),
        selection_levels=args.selection_levels,
        min_n_valid_global=int(args.min_n_valid_global),
        min_n_valid_per_symbol_year=int(args.min_n_valid_per_symbol_year),
        prefer_horizon=args.prefer_horizon,
        gate_vol_entry_at=args.gate_vol_entry_at,
        session_on_regions=args.session_on_regions,
        session_tz=args.session_tz,
    )
    master_outputs = em.run_master_evaluation(master_args)
    overall = pd.read_csv(master_outputs["overall"])
    per_sy = pd.read_csv(master_outputs["per_symbol_year"])
    best = pd.read_csv(master_outputs["best_horizon_selection"])

    lift = _compute_lift_vs_baseline(overall=overall, baseline_variant="baseline", level="executed")

    overall_path = outdir / "overall.csv"
    per_sy_path = outdir / "per_symbol_year.csv"
    best_path = outdir / "best_selection.csv"
    lift_path = outdir / "lift_vs_baseline.csv"
    run_logs_path = outdir / "run_logs.csv"
    summary_path = outdir / "summary.md"

    overall.to_csv(overall_path, index=False)
    per_sy.to_csv(per_sy_path, index=False)
    best.to_csv(best_path, index=False)
    lift.to_csv(lift_path, index=False)
    if "run_logs" in master_outputs:
        run_logs = pd.read_csv(master_outputs["run_logs"])
        run_logs.to_csv(run_logs_path, index=False)
    else:
        pd.DataFrame().to_csv(run_logs_path, index=False)

    _write_summary(
        out_path=summary_path,
        overall=overall,
        best=best,
        lift=lift,
        min_n_valid_global=int(args.min_n_valid_global),
        min_n_valid_per_symbol_year=int(args.min_n_valid_per_symbol_year),
    )

    return {
        "overall": overall_path,
        "per_symbol_year": per_sy_path,
        "best_selection": best_path,
        "lift_vs_baseline": lift_path,
        "run_logs": run_logs_path,
        "summary": summary_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="DFD05 ablation matrix evaluation with robust selection and lift attribution."
    )
    ap.add_argument(
        "--configs",
        default="baseline,gate_only,bos_only,bos_plus_vol,prod",
        help="CSV of ablation configs to run.",
    )
    ap.add_argument(
        "--parity",
        required=False,
        choices=["pine16"],
        help="Optional parity check for baseline config.",
    )
    ap.add_argument("--timeframes", required=True, help="CSV list, e.g. m15,h1,h4")
    ap.add_argument("--horizons", required=True, help="CSV list in hours, e.g. 4,24,72")
    ap.add_argument(
        "--compare_sessions",
        action="store_true",
        help="Run each config/timeframe with session_on and session_off.",
    )
    ap.add_argument(
        "--outdir",
        default="data/reports/dfd05_ablation_eval",
        help="Output directory for ablation artifacts.",
    )
    ap.add_argument(
        "--time_scale_params",
        default=None,
        help="Optional params scaling, e.g. reference_tf=m15",
    )
    ap.add_argument(
        "--selection_levels",
        default="executed",
        help="CSV of levels eligible for best selection. Defaults to executed.",
    )
    ap.add_argument(
        "--min_n_valid_global",
        type=int,
        default=200,
        help="Minimum n_valid per row to be eligible for best selection.",
    )
    ap.add_argument(
        "--min_n_valid_per_symbol_year",
        type=int,
        default=10,
        help="Minimum n_valid across symbol-year rows to be eligible for best selection.",
    )
    ap.add_argument(
        "--prefer_horizon",
        type=int,
        default=24,
        help="Optional preferred horizon tie-breaker (applied only when metrics are extremely close).",
    )
    ap.add_argument(
        "--gate_vol_entry_at",
        default=None,
        help="Optional override for strategy.gate_vol_entry_at across runs: signal or trigger.",
    )
    ap.add_argument(
        "--session_on_regions",
        "--session_on_regions_csv",
        dest="session_on_regions",
        default="ny",
        help="CSV session regions to enable in session_on mode: ny,london,tokyo,sydney",
    )
    ap.add_argument(
        "--session_tz",
        default="Etc/GMT-3",
        help="Timezone string for session_gate when session_on is enabled.",
    )
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    outputs = run_ablation_evaluation(args)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
