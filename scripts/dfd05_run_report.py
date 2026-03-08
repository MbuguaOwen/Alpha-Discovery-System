from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.dfd05_build_html_report as build_html
import scripts.dfd05_eval_ablation as eval_ablation
import scripts.dfd05_eval_barrier as eval_barrier
import scripts.dfd05_eval_policies as eval_policies

CLAIM_EXPECTED: Dict[int, Tuple[int, float]] = {
    4: (226, 51.33),
    24: (222, 50.45),
    72: (201, 49.25),
}


def _resolve_outdir(raw_outdir: str, run_id: str) -> Path:
    token = str(raw_outdir).strip()
    if "<RUN_ID>" in token:
        token = token.replace("<RUN_ID>", run_id)
    return Path(token)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return bool(value)
    token = str(value).strip().lower()
    return token in {"1", "true", "yes", "y", "t"}


def _compute_claim_verification(overall: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    if overall.empty:
        return pd.DataFrame(), False
    req = ["variant", "timeframe", "level", "session_mode", "horizon_h", "n_valid", "up_rate", "mean_ret", "worst_year_mean_ret"]
    missing = [c for c in req if c not in overall.columns]
    if missing:
        raise ValueError(f"overall.csv missing required claim-verification columns: {missing}")

    scoped = overall.copy()
    scoped["variant"] = scoped["variant"].astype(str).str.lower()
    scoped["timeframe"] = scoped["timeframe"].astype(str).str.lower()
    scoped["level"] = scoped["level"].astype(str).str.lower()
    scoped["session_mode"] = scoped["session_mode"].astype(str).str.lower()
    scoped["horizon_h"] = pd.to_numeric(scoped["horizon_h"], errors="coerce").astype("Int64")
    scoped["n_valid"] = pd.to_numeric(scoped["n_valid"], errors="coerce")
    for col in ["up_rate", "mean_ret", "worst_year_mean_ret"]:
        scoped[col] = pd.to_numeric(scoped[col], errors="coerce")

    scoped = scoped[
        (scoped["variant"] == "prod")
        & (scoped["timeframe"] == "m15")
        & (scoped["level"] == "executed")
        & (scoped["session_mode"] == "session_on")
        & (scoped["horizon_h"].isin(list(CLAIM_EXPECTED.keys())))
    ].copy()
    scoped = scoped.sort_values("horizon_h", kind="mergesort")

    rows: List[Dict[str, object]] = []
    all_match = True
    for horizon_h, (exp_n, exp_up_pct) in CLAIM_EXPECTED.items():
        row = scoped[scoped["horizon_h"] == int(horizon_h)]
        if row.empty:
            rows.append(
                {
                    "horizon_h": int(horizon_h),
                    "n_valid": np.nan,
                    "up_rate": np.nan,
                    "up_rate_pct": np.nan,
                    "mean_ret_bps": np.nan,
                    "worst_year_mean_ret_bps": np.nan,
                    "expected_n_valid": int(exp_n),
                    "expected_up_rate_pct": float(exp_up_pct),
                    "n_valid_match": False,
                    "up_rate_match": False,
                    "status": "missing",
                }
            )
            all_match = False
            continue
        r = row.iloc[0]
        n_valid_raw = float(pd.to_numeric(r["n_valid"], errors="coerce"))
        up_rate = float(pd.to_numeric(r["up_rate"], errors="coerce"))
        mean_ret = float(pd.to_numeric(r["mean_ret"], errors="coerce"))
        worst_year_mean_ret = float(pd.to_numeric(r["worst_year_mean_ret"], errors="coerce"))
        if not np.isfinite(n_valid_raw) or not np.isfinite(up_rate):
            rows.append(
                {
                    "horizon_h": int(horizon_h),
                    "n_valid": np.nan,
                    "up_rate": np.nan,
                    "up_rate_pct": np.nan,
                    "mean_ret_bps": mean_ret * 10000.0 if np.isfinite(mean_ret) else np.nan,
                    "worst_year_mean_ret_bps": worst_year_mean_ret * 10000.0 if np.isfinite(worst_year_mean_ret) else np.nan,
                    "expected_n_valid": int(exp_n),
                    "expected_up_rate_pct": float(exp_up_pct),
                    "n_valid_match": False,
                    "up_rate_match": False,
                    "status": "invalid",
                }
            )
            all_match = False
            continue
        n_valid = int(n_valid_raw)
        up_pct = up_rate * 100.0
        n_ok = n_valid == int(exp_n)
        up_ok = abs(up_pct - float(exp_up_pct)) <= 0.01
        status = "pass" if (n_ok and up_ok) else "fail"
        all_match = all_match and bool(n_ok and up_ok)
        rows.append(
            {
                "horizon_h": int(horizon_h),
                "n_valid": int(n_valid),
                "up_rate": up_rate,
                "up_rate_pct": up_pct,
                "mean_ret_bps": mean_ret * 10000.0,
                "worst_year_mean_ret_bps": worst_year_mean_ret * 10000.0,
                "expected_n_valid": int(exp_n),
                "expected_up_rate_pct": float(exp_up_pct),
                "n_valid_match": bool(n_ok),
                "up_rate_match": bool(up_ok),
                "status": status,
            }
        )
    out = pd.DataFrame(rows).sort_values("horizon_h", kind="mergesort").reset_index(drop=True)
    return out, bool(all_match and len(out) == len(CLAIM_EXPECTED))


def _compute_baseline_vs_prod(overall: pd.DataFrame) -> pd.DataFrame:
    if overall.empty:
        return pd.DataFrame()
    req = ["variant", "timeframe", "level", "session_mode", "horizon_h", "n_valid", "up_rate", "mean_ret", "worst_year_mean_ret"]
    missing = [c for c in req if c not in overall.columns]
    if missing:
        raise ValueError(f"overall.csv missing required baseline-vs-prod columns: {missing}")
    work = overall.copy()
    work["variant"] = work["variant"].astype(str).str.lower()
    work["timeframe"] = work["timeframe"].astype(str).str.lower()
    work["level"] = work["level"].astype(str).str.lower()
    work["session_mode"] = work["session_mode"].astype(str).str.lower()
    work["horizon_h"] = pd.to_numeric(work["horizon_h"], errors="coerce").astype("Int64")
    work["n_valid"] = pd.to_numeric(work["n_valid"], errors="coerce")
    for col in ["up_rate", "mean_ret", "worst_year_mean_ret"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work[
        (work["variant"].isin(["baseline", "prod"]))
        & (work["timeframe"] == "m15")
        & (work["level"] == "executed")
        & (work["session_mode"] == "session_on")
        & (work["horizon_h"].isin(list(CLAIM_EXPECTED.keys())))
    ].copy()
    if work.empty:
        return work
    work["up_rate_pct"] = work["up_rate"] * 100.0
    work["mean_ret_bps"] = work["mean_ret"] * 10000.0
    work["worst_year_mean_ret_bps"] = work["worst_year_mean_ret"] * 10000.0
    cols = [
        "variant",
        "horizon_h",
        "n_valid",
        "up_rate",
        "up_rate_pct",
        "mean_ret_bps",
        "worst_year_mean_ret_bps",
    ]
    return work[cols].sort_values(["horizon_h", "variant"], kind="mergesort").reset_index(drop=True)


def _infer_claim_closest_variant(overall: pd.DataFrame) -> str:
    if overall.empty:
        return "unknown"
    work = overall.copy()
    work["variant"] = work["variant"].astype(str).str.lower()
    work["timeframe"] = work["timeframe"].astype(str).str.lower()
    work["level"] = work["level"].astype(str).str.lower()
    work["session_mode"] = work["session_mode"].astype(str).str.lower()
    work["horizon_h"] = pd.to_numeric(work["horizon_h"], errors="coerce").astype("Int64")
    work["n_valid"] = pd.to_numeric(work["n_valid"], errors="coerce")
    work["up_rate"] = pd.to_numeric(work["up_rate"], errors="coerce")
    work = work[
        (work["variant"].isin(["baseline", "prod"]))
        & (work["timeframe"] == "m15")
        & (work["level"] == "executed")
        & (work["session_mode"] == "session_on")
        & (work["horizon_h"].isin(list(CLAIM_EXPECTED.keys())))
    ].copy()
    if work.empty:
        return "unknown"

    scores: Dict[str, float] = {}
    for variant, g in work.groupby("variant", dropna=False):
        score = 0.0
        for h, (exp_n, exp_up_pct) in CLAIM_EXPECTED.items():
            row = g[g["horizon_h"] == int(h)]
            if row.empty:
                score += 10_000.0
                continue
            rr = row.iloc[0]
            n_valid = float(pd.to_numeric(rr["n_valid"], errors="coerce"))
            up_pct = float(pd.to_numeric(rr["up_rate"], errors="coerce")) * 100.0
            if not np.isfinite(n_valid) or not np.isfinite(up_pct):
                score += 10_000.0
                continue
            score += abs(n_valid - float(exp_n)) + abs(up_pct - float(exp_up_pct))
        scores[str(variant)] = score
    if not scores:
        return "unknown"
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    return ranked[0][0]


def _trade_mode_label(trade_mode: str, use_vol_entry_gate: bool) -> str:
    tm = str(trade_mode).strip().upper()
    if tm == "BASELINE_ALL":
        return "Loose"
    if tm == "GATED":
        return "Gated" if bool(use_vol_entry_gate) else "Hybrid"
    return tm


def _entry_mode_label(strategy_mode: str, use_bos_confirm: bool) -> str:
    sm = str(strategy_mode).strip().upper()
    if sm == "RAW":
        return "Divergence"
    if sm == "CONFIRM" and bool(use_bos_confirm):
        return "Confirm(BOS)"
    if sm == "CONFIRM":
        return "Confirm"
    return sm


def _build_config_audit(run_logs: pd.DataFrame) -> pd.DataFrame:
    if run_logs.empty:
        return pd.DataFrame()
    required = [
        "variant",
        "timeframe",
        "session_mode",
        "trade_mode",
        "strategy_mode",
        "use_bos_confirm",
        "enable_vol_ratio_entry_gate",
        "entry_vol_ratio_min",
        "gate_vol_entry_at",
        "bos_atr_buffer",
        "max_wait_bars",
        "one_trade_at_a_time",
        "cooldown_bars",
    ]
    missing = [c for c in required if c not in run_logs.columns]
    if missing:
        raise ValueError(f"run_logs.csv missing required config-audit columns: {missing}")

    work = run_logs.copy()
    work["variant"] = work["variant"].astype(str).str.lower()
    work["timeframe"] = work["timeframe"].astype(str).str.lower()
    work["session_mode"] = work["session_mode"].astype(str).str.lower()
    work["_tf_rank"] = (work["timeframe"] != "m15").astype("int64")
    work["_sess_rank"] = (work["session_mode"] != "session_on").astype("int64")
    work = work.sort_values(["variant", "_tf_rank", "_sess_rank"], kind="mergesort")
    rep = work.groupby("variant", dropna=False, as_index=False).first()

    rep["use_bos_confirm"] = rep["use_bos_confirm"].map(_to_bool)
    rep["enable_vol_ratio_entry_gate"] = rep["enable_vol_ratio_entry_gate"].map(_to_bool)
    rep["one_trade_at_a_time"] = rep["one_trade_at_a_time"].map(_to_bool)
    rep["entry_vol_ratio_min"] = pd.to_numeric(rep["entry_vol_ratio_min"], errors="coerce")
    rep["bos_atr_buffer"] = pd.to_numeric(rep["bos_atr_buffer"], errors="coerce")
    rep["max_wait_bars"] = pd.to_numeric(rep["max_wait_bars"], errors="coerce").astype("Int64")
    rep["cooldown_bars"] = pd.to_numeric(rep["cooldown_bars"], errors="coerce").astype("Int64")

    rep["trade_mode_label"] = [
        _trade_mode_label(trade_mode=str(tm), use_vol_entry_gate=bool(vg))
        for tm, vg in zip(rep["trade_mode"], rep["enable_vol_ratio_entry_gate"])
    ]
    rep["entry_mode_label"] = [
        _entry_mode_label(strategy_mode=str(sm), use_bos_confirm=bool(bc))
        for sm, bc in zip(rep["strategy_mode"], rep["use_bos_confirm"])
    ]
    rep["gate_vol_entry_at_label"] = rep["gate_vol_entry_at"].astype(str).str.lower().map(
        lambda x: "entry" if x == "signal" else x
    )
    cols = [
        "variant",
        "trade_mode_label",
        "entry_mode_label",
        "enable_vol_ratio_entry_gate",
        "entry_vol_ratio_min",
        "gate_vol_entry_at_label",
        "bos_atr_buffer",
        "max_wait_bars",
        "one_trade_at_a_time",
        "cooldown_bars",
        "trade_mode",
        "strategy_mode",
        "use_bos_confirm",
    ]
    out = rep[cols].copy().sort_values("variant", kind="mergesort").reset_index(drop=True)
    out = out.rename(
        columns={
            "enable_vol_ratio_entry_gate": "useVolEntryGate",
            "entry_vol_ratio_min": "minEntryVolRatio",
            "gate_vol_entry_at_label": "gate_vol_entry_at",
            "bos_atr_buffer": "bosAtrBuffer",
            "max_wait_bars": "maxWaitBars",
            "one_trade_at_a_time": "oneTradeAtTime",
            "cooldown_bars": "cooldownBars",
        }
    )
    return out


def _fmt_md_cell(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(float(v)):
            return ""
        return f"{float(v):.6f}"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if pd.isna(v):
        return ""
    return str(v)


def _markdown_table(df: pd.DataFrame, columns: Sequence[str]) -> str:
    if df.empty:
        return "_No rows._"
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return "_No columns._"
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, r in df[cols].iterrows():
        lines.append("| " + " | ".join(_fmt_md_cell(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _append_summary_sections(
    summary_path: Path,
    claim_df: pd.DataFrame,
    claim_pass: bool,
    baseline_vs_prod: pd.DataFrame,
    selected_baseline_variant: str,
    config_audit: pd.DataFrame,
    barrier_results: pd.DataFrame,
) -> None:
    base = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    lines: List[str] = [base.rstrip(), "", "## Claim Verification", ""]
    lines.append(f"- Status: {'PASS' if claim_pass else 'FAIL'}")
    lines.append("- Slice: variant=prod, timeframe=m15, level=executed, session_mode=session_on")
    lines.append("")
    lines.append(
        _markdown_table(
            claim_df,
            [
                "horizon_h",
                "n_valid",
                "up_rate_pct",
                "mean_ret_bps",
                "worst_year_mean_ret_bps",
                "expected_n_valid",
                "expected_up_rate_pct",
                "status",
            ],
        )
    )
    lines.extend(["", "## Baseline vs Prod (Same Slice)", ""])
    lines.append(
        _markdown_table(
            baseline_vs_prod,
            ["variant", "horizon_h", "n_valid", "up_rate_pct", "mean_ret_bps", "worst_year_mean_ret_bps"],
        )
    )
    lines.append("")
    lines.append(
        f"- Selected baseline strategy corresponds to: `{selected_baseline_variant}` configuration."
    )
    if selected_baseline_variant == "prod":
        lines.append("- Naming recommendation: rename `baseline` -> `baseline_raw` and `prod` -> `baseline_selected`.")
    else:
        lines.append("- Naming recommendation: keep `baseline` as-is, and label `prod` as production/toggled variant.")

    lines.extend(["", "## Config Audit", ""])
    lines.append(
        _markdown_table(
            config_audit,
            [
                "variant",
                "trade_mode_label",
                "entry_mode_label",
                "useVolEntryGate",
                "minEntryVolRatio",
                "gate_vol_entry_at",
                "bosAtrBuffer",
                "maxWaitBars",
                "oneTradeAtTime",
                "cooldownBars",
            ],
        )
    )
    lines.extend(["", "## Barrier 1:3 (Separate From Up Rate)", ""])
    lines.append(
        _markdown_table(
            barrier_results,
            [
                "variant",
                "timeframe",
                "session_mode",
                "n_trades",
                "win_rate_1to3",
                "avg_R",
                "worst_year_win_rate",
                "selected_horizon_h",
            ],
        )
    )
    summary_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_handoff_zip(outdir: Path, files: Sequence[Path]) -> Path:
    zip_path = outdir / "handoff_artifacts.zip"
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            if p.exists():
                zf.write(p, arcname=p.name)
    return zip_path


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run DFD05 multi-variant horizon evaluation and build an interactive HTML report."
    )
    ap.add_argument(
        "--configs",
        default="baseline,gate_only,bos_only,bos_plus_vol,prod",
        help="CSV of variants/configs.",
    )
    ap.add_argument("--parity", required=False, choices=["pine16"], help="Optional parity check for baseline.")
    ap.add_argument("--timeframes", required=True, help="CSV list, e.g. m15,h1,h4")
    ap.add_argument("--horizons", required=True, help="CSV list in hours, e.g. 4,24,72")
    ap.add_argument("--compare_sessions", action="store_true", help="Evaluate session_on and session_off.")
    ap.add_argument(
        "--time_scale_params",
        default=None,
        help="Optional time-scaling params, e.g. reference_tf=m15",
    )
    ap.add_argument(
        "--outdir",
        default="data/reports/dfd05_report_<RUN_ID>",
        help="Output directory. Use <RUN_ID> placeholder for timestamped reports.",
    )
    ap.add_argument("--min_n_valid_global", type=int, default=200, help="Eligibility threshold.")
    ap.add_argument(
        "--min_n_valid_per_symbol_year",
        type=int,
        default=10,
        help="Eligibility threshold for symbol-year cells.",
    )
    ap.add_argument("--selection_levels", default="executed", help="Levels eligible for horizon selection.")
    ap.add_argument(
        "--prefer_horizon",
        type=int,
        default=24,
        help="Optional tie-bias toward this horizon when metrics are very close.",
    )
    ap.add_argument(
        "--gate_vol_entry_at",
        default=None,
        help="Optional override for gate volume check location: signal or trigger.",
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
    ap.add_argument(
        "--objective_c_worst_year_min_bps",
        type=float,
        default=0.0,
        help="Objective C threshold in bps for worst_year_mean_ret.",
    )
    ap.add_argument("--p1_cut_bps", type=float, default=25.0, help="Policy P1/P2 early-cut X threshold (bps).")
    ap.add_argument("--p2_take_bps", type=float, default=25.0, help="Policy P2 24h take Y threshold (bps).")
    return ap


def run_report_pipeline(args: argparse.Namespace) -> Dict[str, Path]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = _resolve_outdir(args.outdir, run_id=run_id)
    outdir.mkdir(parents=True, exist_ok=True)
    legacy_note = outdir / "LEGACY_NON_EXACT_WARNING.txt"
    legacy_note.write_text(
        "This report pipeline is legacy and must not be presented as exact Pine-exported truth.\\n"
        "Truth label: UNVERIFIED_PYTHON_APPROXIMATION\\n",
        encoding="utf-8",
    )

    ablation_args = argparse.Namespace(
        configs=args.configs,
        parity=args.parity,
        timeframes=args.timeframes,
        horizons=args.horizons,
        compare_sessions=bool(args.compare_sessions),
        outdir=str(outdir),
        time_scale_params=args.time_scale_params,
        selection_levels=args.selection_levels,
        min_n_valid_global=int(args.min_n_valid_global),
        min_n_valid_per_symbol_year=int(args.min_n_valid_per_symbol_year),
        prefer_horizon=args.prefer_horizon,
        gate_vol_entry_at=args.gate_vol_entry_at,
        session_on_regions=args.session_on_regions,
        session_tz=args.session_tz,
    )
    ablation_outputs = eval_ablation.run_ablation_evaluation(ablation_args)

    policies_args = argparse.Namespace(
        run_logs=str(ablation_outputs["run_logs"]),
        labeled_paths=None,
        labeled_dir=None,
        horizons=args.horizons,
        p1_cut_bps=float(args.p1_cut_bps),
        p2_take_bps=float(args.p2_take_bps),
        min_n_valid_global=int(args.min_n_valid_global),
        min_n_valid_per_symbol_year=int(args.min_n_valid_per_symbol_year),
        outdir=str(outdir),
    )
    policy_outputs = eval_policies.run_policy_evaluation(policies_args)

    barrier_args = argparse.Namespace(
        run_logs=str(ablation_outputs["run_logs"]),
        best_selection=str(ablation_outputs["best_selection"]),
        outdir=str(outdir),
        sl_atr_mult=1.0,
        rr_mult=3.0,
    )
    barrier_outputs = eval_barrier.run_barrier_evaluation(barrier_args)

    overall_df = _load_csv(Path(ablation_outputs["overall"]))
    run_logs_df = _load_csv(Path(ablation_outputs["run_logs"]))
    barrier_df = _load_csv(Path(barrier_outputs["barrier_results"]))

    claim_df, claim_pass = _compute_claim_verification(overall_df)
    claim_path = outdir / "claim_verification.csv"
    claim_df.to_csv(claim_path, index=False)

    baseline_vs_prod = _compute_baseline_vs_prod(overall_df)
    baseline_vs_prod_path = outdir / "baseline_vs_prod.csv"
    baseline_vs_prod.to_csv(baseline_vs_prod_path, index=False)

    selected_baseline_variant = _infer_claim_closest_variant(overall_df)
    mapping_df = pd.DataFrame(
        [
            {
                "selected_baseline_variant": selected_baseline_variant,
                "recommend_rename": "baseline_raw vs baseline_selected" if selected_baseline_variant == "prod" else "keep_baseline_name",
            }
        ]
    )
    mapping_path = outdir / "baseline_identity.csv"
    mapping_df.to_csv(mapping_path, index=False)

    config_audit = _build_config_audit(run_logs_df)
    config_audit_path = outdir / "config_audit.csv"
    config_audit.to_csv(config_audit_path, index=False)

    summary_path = Path(ablation_outputs["summary"])
    _append_summary_sections(
        summary_path=summary_path,
        claim_df=claim_df,
        claim_pass=claim_pass,
        baseline_vs_prod=baseline_vs_prod,
        selected_baseline_variant=selected_baseline_variant,
        config_audit=config_audit,
        barrier_results=barrier_df,
    )

    html_args = argparse.Namespace(
        outdir=str(outdir),
        objective_c_worst_year_min_bps=float(args.objective_c_worst_year_min_bps),
        min_n_valid_global=int(args.min_n_valid_global),
        min_n_valid_per_symbol_year=int(args.min_n_valid_per_symbol_year),
    )
    html_outputs = build_html.run_build_html_report(html_args)

    handoff_zip = _write_handoff_zip(
        outdir=outdir,
        files=[
            Path(ablation_outputs["overall"]),
            Path(ablation_outputs["best_selection"]),
            Path(ablation_outputs["run_logs"]),
            Path(ablation_outputs["summary"]),
            Path(html_outputs["index_html"]),
            Path(barrier_outputs["barrier_results"]),
            claim_path,
            config_audit_path,
        ],
    )

    out: Dict[str, Path] = {"report_outdir": outdir}
    out.update(ablation_outputs)
    out.update(policy_outputs)
    out.update(barrier_outputs)
    out.update(
        {
            "claim_verification": claim_path,
            "baseline_vs_prod": baseline_vs_prod_path,
            "baseline_identity": mapping_path,
            "config_audit": config_audit_path,
            "legacy_warning": legacy_note,
            "handoff_zip": handoff_zip,
        }
    )
    out.update(html_outputs)
    return out


def main() -> None:
    args = build_arg_parser().parse_args()
    outputs = run_report_pipeline(args)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
