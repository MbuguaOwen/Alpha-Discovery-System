from __future__ import annotations

import argparse
from pathlib import Path

from dfd05.pine16_research import run_conditional_research
from dfd05.pine16_truth import TruthMode


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run Pine16 feature-bucket research export.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--truth-mode", required=True, choices=[m.value for m in TruthMode])
    ap.add_argument("--exact-dir", default="data/derived/pine16_exact")
    ap.add_argument("--output-dir", default="outputs/reports")
    ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--min-n-robust", type=int, default=30)
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
        export_html=False,
        include_feature_importance=False,
    )
    print(f"research_master: {outs.research_master}")
    print(f"by_feature_bucket: {outs.research_by_feature_bucket}")


if __name__ == "__main__":
    main()
