from __future__ import annotations

import argparse
from pathlib import Path

from dfd05.pine16_config import load_pine16_exact_config
from dfd05.pine16_export_ingest import ingest_tv_exports


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Ingest TradingView exports into normalized Pine16 exact parquet schemas.")
    ap.add_argument("--imports-dir", default="tv_exports", help="Directory containing TradingView CSV exports.")
    ap.add_argument(
        "--config",
        default="configs/pine16_exact_prod_screenshot.yaml",
        help="Pine16 exact config pack used for metadata context.",
    )
    ap.add_argument(
        "--output-dir",
        default="data/derived/pine16_exact",
        help="Output directory for normalized parquet outputs.",
    )
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = load_pine16_exact_config(args.config)
    outs = ingest_tv_exports(
        imports_dir=Path(args.imports_dir),
        output_dir=Path(args.output_dir),
        cfg=cfg,
        config_pack=cfg.metadata.config_pack,
    )
    print(f"trades_exact_pine: {outs.trades_path}")
    print(f"signals_exact_pine: {outs.signals_path}")
    print(f"summary_exact_pine: {outs.summary_path}")
    print(f"imports_inventory: {outs.inventory_path}")


if __name__ == "__main__":
    main()

