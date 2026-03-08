from __future__ import annotations

import argparse
from pathlib import Path

from dfd05.pine16_m30_horizon import write_m30_horizon_audit


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Write Pine16 M30 horizon-grid audit.")
    ap.add_argument("--audit-path", default="outputs/audit_pine16_m30_horizon_grid.md", help="Audit markdown output path.")
    ap.add_argument("--exact-dir", default="data/derived/pine16_exact", help="Exact/parity artifacts directory.")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    out = write_m30_horizon_audit(audit_path=Path(args.audit_path), exact_dir=Path(args.exact_dir))
    print(f"audit_md: {out}")


if __name__ == "__main__":
    main()

