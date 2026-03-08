from __future__ import annotations

import argparse
from pathlib import Path

from dfd05.pine16_m30_horizon import run_m30_horizon_research
from dfd05.pine16_truth import TruthMode


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run Pine16 M30 horizon-grid research pipeline.")
    ap.add_argument("--config", required=True, help="Pine16 M30 config path.")
    ap.add_argument("--truth-mode", required=True, choices=[m.value for m in TruthMode], help="Truth mode selector.")
    ap.add_argument("--exact-dir", default="data/derived/pine16_exact", help="Exact/parity artifacts directory.")
    ap.add_argument("--output-dir", default="outputs/reports", help="Reports output directory.")
    ap.add_argument("--min-n", type=int, default=20, help="Minimum sample size.")
    ap.add_argument("--min-n-robust", type=int, default=30, help="Minimum robust sample size.")
    ap.add_argument("--compare-m15-config", default="configs/pine16_exact_prod_all_sessions.yaml", help="M15 config used for M15 vs M30 comparison.")
    ap.add_argument("--export-html", action="store_true", help="Write HTML report.")
    ap.add_argument("--run-audit", action="store_true", help="Write/refresh audit as part of run.")
    ap.add_argument("--audit-path", default="outputs/audit_pine16_m30_horizon_grid.md", help="Audit markdown output path.")
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
        export_html=bool(args.export_html),
        compare_m15_config=args.compare_m15_config,
        run_audit=bool(args.run_audit),
        audit_path=Path(args.audit_path),
    )
    for k, v in outs.__dict__.items():
        if v is not None:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()

