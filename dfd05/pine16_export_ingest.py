from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .pine16_config import Pine16ExactConfig
from .pine16_export_schema import (
    SIGNAL_REQUIRED_COLUMNS,
    SUMMARY_REQUIRED_COLUMNS,
    TRADE_REQUIRED_COLUMNS,
    discover_export_files,
    merge_normalized_exports,
)


@dataclass
class IngestOutputs:
    trades_path: Path
    signals_path: Path
    summary_path: Path
    inventory_path: Path


def _read_csvs(paths: List[Path]) -> List[tuple[pd.DataFrame, str]]:
    out: List[tuple[pd.DataFrame, str]] = []
    for p in paths:
        try:
            frame = pd.read_csv(p)
        except UnicodeDecodeError:
            frame = pd.read_csv(p, encoding="utf-8-sig")
        out.append((frame, str(p)))
    return out


def ingest_tv_exports(
    *,
    imports_dir: Path,
    output_dir: Path,
    cfg: Pine16ExactConfig,
    config_pack: Optional[str] = None,
) -> IngestOutputs:
    files = discover_export_files(imports_dir)

    trades_parts = _read_csvs(files["trades"])
    signal_parts = _read_csvs(files["signals"])
    summary_parts = _read_csvs(files["summary"])

    normalized = merge_normalized_exports(
        trades_parts=trades_parts,
        signal_parts=signal_parts,
        summary_parts=summary_parts,
        config_pack=str(config_pack or cfg.metadata.config_pack),
        timeframe=str(cfg.timeframe),
        tv_strategy_name=str(cfg.metadata.tv_strategy_name),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    trades_path = output_dir / "trades_exact_pine.parquet"
    signals_path = output_dir / "signals_exact_pine.parquet"
    summary_path = output_dir / "summary_exact_pine.parquet"
    inventory_path = output_dir / "imports_inventory.csv"

    trades = normalized.trades
    signals = normalized.signals
    summary = normalized.summary

    if trades.empty:
        trades = pd.DataFrame(columns=TRADE_REQUIRED_COLUMNS)
    if signals.empty:
        signals = pd.DataFrame(columns=SIGNAL_REQUIRED_COLUMNS)
    if summary.empty:
        summary = pd.DataFrame(columns=SUMMARY_REQUIRED_COLUMNS)

    trades.to_parquet(trades_path, index=False)
    signals.to_parquet(signals_path, index=False)
    summary.to_parquet(summary_path, index=False)

    inventory_rows: List[Dict[str, object]] = []
    for kind, paths in files.items():
        for p in paths:
            inventory_rows.append({"kind": kind, "file": str(p)})
    pd.DataFrame(inventory_rows, columns=["kind", "file"]).to_csv(inventory_path, index=False)

    return IngestOutputs(
        trades_path=trades_path,
        signals_path=signals_path,
        summary_path=summary_path,
        inventory_path=inventory_path,
    )

