from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from dfd05.data import load_bars_for_symbol, timeframe_to_minutes
from dfd05.indicators import atr
from dfd05.pine16_config import load_pine16_exact_config, to_legacy_run_config
from dfd05.pine16_export_schema import summarize_trades


def _match_bar_indices(
    entry_times: pd.Series,
    bars: pd.DataFrame,
    *,
    timeframe: str,
) -> pd.Series:
    if entry_times.empty:
        return pd.Series(dtype="Int64")

    tf_seconds = timeframe_to_minutes(timeframe) * 60
    left = pd.DataFrame({"entry_time_utc": pd.to_datetime(entry_times, utc=True, errors="coerce")}).reset_index()
    right = pd.DataFrame({"bar_time_utc": pd.to_datetime(bars["time"], utc=True), "bar_index": np.arange(len(bars), dtype=np.int64)})

    merged = pd.merge_asof(
        left.sort_values("entry_time_utc"),
        right.sort_values("bar_time_utc"),
        left_on="entry_time_utc",
        right_on="bar_time_utc",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tf_seconds),
    )
    out = merged.sort_values("index")["bar_index"]
    return out.astype("Int64")


def _reconstruct_trades(trades: pd.DataFrame, cfg_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_pine16_exact_config(cfg_path)
    legacy_cfg = to_legacy_run_config(cfg)

    if trades.empty:
        empty_summary = pd.DataFrame(
            columns=[
                "symbol",
                "timeframe",
                "tv_strategy_name",
                "config_pack",
                "n_trades",
                "win_rate",
                "net_r",
                "expectancy_r",
                "profit_factor",
                "year",
                "source_file",
                "truth_label",
            ]
        )
        return trades, empty_summary

    out = trades.copy()
    out["entry_time_utc"] = pd.to_datetime(out["entry_time_utc"], utc=True, errors="coerce")
    out["exit_time_utc"] = pd.to_datetime(out["exit_time_utc"], utc=True, errors="coerce")
    out["entry_price"] = pd.to_numeric(out["entry_price"], errors="coerce")
    out["exit_price"] = pd.to_numeric(out["exit_price"], errors="coerce")
    out["atr_entry"] = pd.to_numeric(out["atr_entry"], errors="coerce")
    out["sl_price"] = pd.to_numeric(out["sl_price"], errors="coerce")
    out["tp_price"] = pd.to_numeric(out["tp_price"], errors="coerce")
    out["result_r"] = pd.to_numeric(out["result_r"], errors="coerce")
    out["result_pct"] = pd.to_numeric(out["result_pct"], errors="coerce")
    out["bars_held"] = pd.to_numeric(out["bars_held"], errors="coerce")
    out["rr_multiple"] = pd.to_numeric(out["rr_multiple"], errors="coerce")

    rec_status = []
    rec_note = []

    for symbol, idx in out.groupby("symbol", dropna=False).groups.items():
        rows = out.loc[idx].copy()
        try:
            bars = load_bars_for_symbol(legacy_cfg, symbol=str(symbol), timeframe=str(cfg.timeframe))
        except FileNotFoundError:
            for i in rows.index:
                rec_status.append((i, "missing_bars"))
                rec_note.append((i, "No local bars found for symbol; reconstruction skipped."))
            continue

        if bars.empty:
            for i in rows.index:
                rec_status.append((i, "missing_bars"))
                rec_note.append((i, "No local bars found for symbol; reconstruction skipped."))
            continue

        high = bars["high"].to_numpy(dtype=float)
        low = bars["low"].to_numpy(dtype=float)
        close = bars["close"].to_numpy(dtype=float)
        atr_arr = atr(high, low, close, int(cfg.risk.atrLen))

        entry_idx = _match_bar_indices(rows["entry_time_utc"], bars, timeframe=str(cfg.timeframe))
        exit_idx = _match_bar_indices(rows["exit_time_utc"], bars, timeframe=str(cfg.timeframe))

        for local_pos, row_idx in enumerate(rows.index):
            eidx = entry_idx.iloc[local_pos]
            xidx = exit_idx.iloc[local_pos] if local_pos < len(exit_idx) else pd.NA
            reconstructed_any = False

            if pd.notna(eidx):
                ei = int(eidx)
                if not np.isfinite(out.at[row_idx, "atr_entry"]):
                    out.at[row_idx, "atr_entry"] = float(atr_arr[ei])
                    reconstructed_any = True

                atr_e = pd.to_numeric(out.at[row_idx, "atr_entry"], errors="coerce")
                entry_px = pd.to_numeric(out.at[row_idx, "entry_price"], errors="coerce")
                if np.isfinite(atr_e) and atr_e > 0 and np.isfinite(entry_px):
                    if not np.isfinite(out.at[row_idx, "sl_price"]):
                        out.at[row_idx, "sl_price"] = float(entry_px - float(cfg.risk.slAtrMult) * atr_e)
                        reconstructed_any = True
                    if not np.isfinite(out.at[row_idx, "tp_price"]):
                        out.at[row_idx, "tp_price"] = float(entry_px + float(cfg.risk.slAtrMult) * float(cfg.risk.rrMult) * atr_e)
                        reconstructed_any = True

                    if not np.isfinite(out.at[row_idx, "rr_multiple"]):
                        out.at[row_idx, "rr_multiple"] = float(cfg.risk.rrMult)
                        reconstructed_any = True

                    exit_px = pd.to_numeric(out.at[row_idx, "exit_price"], errors="coerce")
                    if np.isfinite(exit_px) and not np.isfinite(out.at[row_idx, "result_r"]):
                        out.at[row_idx, "result_r"] = float((exit_px - entry_px) / (float(cfg.risk.slAtrMult) * atr_e))
                        reconstructed_any = True
                    if np.isfinite(exit_px) and np.isfinite(entry_px) and entry_px != 0 and not np.isfinite(out.at[row_idx, "result_pct"]):
                        out.at[row_idx, "result_pct"] = float(((exit_px - entry_px) / entry_px) * 100.0)
                        reconstructed_any = True

                if pd.notna(xidx) and not np.isfinite(out.at[row_idx, "bars_held"]):
                    out.at[row_idx, "bars_held"] = int(max(0, int(xidx) - int(ei)))
                    reconstructed_any = True

            if not np.isfinite(pd.to_numeric(out.at[row_idx, "rr_multiple"], errors="coerce")):
                out.at[row_idx, "rr_multiple"] = float(cfg.risk.rrMult)
                reconstructed_any = True

            rec_status.append((row_idx, "reconstructed" if reconstructed_any else "raw_export"))
            rec_note.append(
                (
                    row_idx,
                    "ATR/SL/TP/result fields reconstructed from local OHLCV using canonical Pine risk settings."
                    if reconstructed_any
                    else "",
                )
            )

    status_map = dict(rec_status)
    note_map = dict(rec_note)
    out["reconstruction_status"] = [status_map.get(i, "raw_export") for i in out.index]
    out["reconstruction_assumptions"] = [note_map.get(i, "") for i in out.index]

    out["bar_time_utc"] = out["bar_time_utc"].where(out["bar_time_utc"].notna(), out["entry_time_utc"])
    out["year"] = out["entry_time_utc"].dt.year.astype("Int64")
    out["month"] = out["entry_time_utc"].dt.month.astype("Int64")

    summary = summarize_trades(out, source_file="reconciled_exact_pine")
    return out, summary


def _rebuild_signals_from_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "timeframe",
                "tv_strategy_name",
                "config_pack",
                "bar_time_utc",
                "entry_time_utc",
                "side",
                "entry_price",
                "source_file",
                "truth_label",
            ]
        )
    out = trades[
        [
            "symbol",
            "timeframe",
            "tv_strategy_name",
            "config_pack",
            "bar_time_utc",
            "entry_time_utc",
            "side",
            "entry_price",
            "source_file",
            "truth_label",
        ]
    ].copy()
    out = out.drop_duplicates(["symbol", "timeframe", "entry_time_utc", "entry_price"], keep="first")
    return out.reset_index(drop=True)


def run_reconcile(config_path: str, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trades_path = output_dir / "trades_exact_pine.parquet"
    signals_path = output_dir / "signals_exact_pine.parquet"
    summary_path = output_dir / "summary_exact_pine.parquet"

    trades = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()
    signals = pd.read_parquet(signals_path) if signals_path.exists() else pd.DataFrame()

    trades_rec, summary = _reconstruct_trades(trades, cfg_path=config_path)
    if signals.empty:
        signals_rec = _rebuild_signals_from_trades(trades_rec)
    else:
        signals_rec = signals.copy()

    trades_rec.to_parquet(trades_path, index=False)
    signals_rec.to_parquet(signals_path, index=False)
    summary.to_parquet(summary_path, index=False)

    status_counts = (
        trades_rec["reconstruction_status"].value_counts(dropna=False).rename_axis("reconstruction_status").reset_index(name="n")
        if not trades_rec.empty
        else pd.DataFrame(columns=["reconstruction_status", "n"])
    )
    status_path = output_dir / "reconcile_status.csv"
    status_counts.to_csv(status_path, index=False)

    context_path = output_dir / "reconcile_context.json"
    context_path.write_text(
        pd.Series(
            {
                "config": str(config_path),
                "risk": asdict(load_pine16_exact_config(config_path).risk),
                "timeframe": load_pine16_exact_config(config_path).timeframe,
            }
        ).to_json(),
        encoding="utf-8",
    )

    return {
        "trades_exact_pine": trades_path,
        "signals_exact_pine": signals_path,
        "summary_exact_pine": summary_path,
        "reconcile_status": status_path,
        "reconcile_context": context_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Reconcile and reconstruct Pine16 exact exports against local OHLCV.")
    ap.add_argument("--config", required=True, help="Pine16 exact config path.")
    ap.add_argument("--output-dir", default="data/derived/pine16_exact", help="Normalized export directory.")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    outs = run_reconcile(config_path=args.config, output_dir=Path(args.output_dir))
    for k, v in outs.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

