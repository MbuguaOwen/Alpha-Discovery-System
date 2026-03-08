from __future__ import annotations

import argparse
from pathlib import Path

from dfd05.pine16_m30_horizon import run_m30_horizon_research
from dfd05.pine16_truth import TruthMode


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run Pine16 M30 deployment candidate export.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--truth-mode", required=True, choices=[m.value for m in TruthMode])
    ap.add_argument("--exact-dir", default="data/derived/pine16_exact")
    ap.add_argument("--output-dir", default="outputs/reports")
    ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--min-n-robust", type=int, default=30)
    ap.add_argument("--compare-m15-config", default="configs/pine16_exact_prod_all_sessions.yaml")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    outs = run_m30_horizon_research(
        config_path=args.config,
        truth_mode_raw=args.truth_mode,
        exact_dir=Path(args.exact_dir),
        output_dir=Path(args.output_dir),
        min_n=int(args.min_n),
        min_n_robust=int(args.min_n_robust),
        export_html=False,
        compare_m15_config=args.compare_m15_config,
        run_audit=False,
    )
    print(f"deployment_candidates_csv: {outs.deployment_candidates_csv}")
    print(f"deployment_candidates_md: {outs.deployment_candidates_md}")
    print(f"keep_watch_cut_csv: {outs.keep_watch_cut_csv}")


if __name__ == "__main__":
    main()

