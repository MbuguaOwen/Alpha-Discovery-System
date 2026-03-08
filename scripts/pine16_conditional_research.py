from __future__ import annotations

import argparse
from pathlib import Path

from dfd05.pine16_research import run_conditional_research
from dfd05.pine16_truth import TruthMode


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run Pine16 deep conditional research pipeline.")
    ap.add_argument("--config", required=True, help="Pine16 exact config path.")
    ap.add_argument("--truth-mode", required=True, choices=[m.value for m in TruthMode], help="Truth mode selector.")
    ap.add_argument("--exact-dir", default="data/derived/pine16_exact", help="Exact/parity artifacts directory.")
    ap.add_argument("--output-dir", default="outputs/reports", help="Reports output directory.")
    ap.add_argument("--min-n", type=int, default=20, help="Minimum sample size for conditional combos.")
    ap.add_argument("--min-n-robust", type=int, default=30, help="Minimum sample size for robust keep/deploy flags.")
    ap.add_argument("--export-html", action="store_true", help="Write HTML report alongside markdown.")
    ap.add_argument("--include-feature-importance", action="store_true", help="Generate interpretable feature-importance outputs.")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    outs = run_conditional_research(
        config_path=args.config,
        truth_mode_raw=args.truth_mode,
        exact_dir=Path(args.exact_dir),
        output_dir=Path(args.output_dir),
        min_n=int(args.min_n),
        min_n_robust=int(args.min_n_robust),
        export_html=bool(args.export_html),
        include_feature_importance=bool(args.include_feature_importance),
    )
    for k, v in outs.__dict__.items():
        if v is not None:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
