from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from dfd05.pine16_config import load_pine16_exact_config
from dfd05.pine16_parity_engine import compare_parity_to_exact, run_python_parity
from dfd05.pine16_truth import TruthLabel


def _load_reference_trades(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    return pd.DataFrame()


def _load_exact_trades_from_dir(exact_dir: Path) -> pd.DataFrame:
    p = exact_dir / "trades_exact_pine.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def run_verify_parity(config_path: str, exact_dir: Path, reference_trades: Optional[str]) -> Dict[str, Path]:
    cfg = load_pine16_exact_config(config_path)
    exact_dir.mkdir(parents=True, exist_ok=True)

    parity = run_python_parity(cfg)
    parity_signals_path = exact_dir / "parity_signals.parquet"
    parity_trades_path = exact_dir / "parity_trades.parquet"
    parity.signals.to_parquet(parity_signals_path, index=False)
    parity.trades.to_parquet(parity_trades_path, index=False)

    exact_trades = _load_exact_trades_from_dir(exact_dir)
    if exact_trades.empty:
        exact_trades = _load_reference_trades(reference_trades)

    verification_path = exact_dir / "parity_verification.json"
    if exact_trades.empty:
        payload = {
            "config": str(config_path),
            "timeframe": str(cfg.timeframe),
            "reference_available": False,
            "pass_thresholds": False,
            "signal_count_mismatch_pct": None,
            "entry_timestamp_max_bar_diff": None,
            "exit_timestamp_max_bar_diff": None,
            "aggregate_net_r_mismatch_pct": None,
            "truth_label": TruthLabel.UNVERIFIED_PYTHON_APPROXIMATION.value,
            "reason": "no_pine_reference_artifacts",
            "parity_trades_path": str(parity_trades_path),
            "parity_signals_path": str(parity_signals_path),
        }
    else:
        cmp = compare_parity_to_exact(
            exact_trades=exact_trades,
            parity_trades=parity.trades,
            timeframe=cfg.timeframe,
        )
        payload = {
            "config": str(config_path),
            "timeframe": str(cfg.timeframe),
            "reference_available": True,
            "pass_thresholds": bool(cmp.pass_thresholds),
            "signal_count_mismatch_pct": float(cmp.signal_count_mismatch_pct),
            "entry_timestamp_max_bar_diff": float(cmp.entry_timestamp_max_bar_diff),
            "exit_timestamp_max_bar_diff": float(cmp.exit_timestamp_max_bar_diff),
            "aggregate_net_r_mismatch_pct": float(cmp.aggregate_net_r_mismatch_pct),
            "truth_label": (
                TruthLabel.VERIFIED_PYTHON_PARITY.value
                if cmp.pass_thresholds
                else TruthLabel.UNVERIFIED_PYTHON_APPROXIMATION.value
            ),
            "reason": "thresholds_passed" if cmp.pass_thresholds else "thresholds_failed",
            "parity_trades_path": str(parity_trades_path),
            "parity_signals_path": str(parity_signals_path),
        }
    verification_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "parity_signals": parity_signals_path,
        "parity_trades": parity_trades_path,
        "parity_verification": verification_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run strict Pine16 parity verification against Pine-exported references.")
    ap.add_argument("--config", required=True, help="Pine16 exact config path.")
    ap.add_argument(
        "--exact-dir",
        default="data/derived/pine16_exact",
        help="Directory with normalized exact artifacts and parity outputs.",
    )
    ap.add_argument(
        "--reference-trades",
        default=None,
        help="Optional reference trades artifact (CSV or parquet) when exact_dir has no trades_exact_pine.parquet.",
    )
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    outs = run_verify_parity(
        config_path=args.config,
        exact_dir=Path(args.exact_dir),
        reference_trades=args.reference_trades,
    )
    for k, v in outs.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
