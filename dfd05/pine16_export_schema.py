from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .pine16_truth import TruthLabel


TRADE_REQUIRED_COLUMNS = [
    "symbol",
    "timeframe",
    "tv_strategy_name",
    "config_pack",
    "bar_time_utc",
    "entry_time_utc",
    "exit_time_utc",
    "entry_price",
    "exit_price",
    "side",
    "atr_entry",
    "sl_price",
    "tp_price",
    "rr_multiple",
    "bars_held",
    "result_r",
    "result_pct",
    "year",
    "month",
    "source_file",
    "truth_label",
    "reconstruction_status",
    "reconstruction_assumptions",
]

SIGNAL_REQUIRED_COLUMNS = [
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

SUMMARY_REQUIRED_COLUMNS = [
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


def _safe_name(text: str) -> str:
    return (
        str(text)
        .strip()
        .lower()
        .replace("%", " pct")
        .replace("#", " num")
        .replace("/", " ")
        .replace("-", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace(".", " ")
        .replace("__", " ")
    )


def _canonical_cols(frame: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for c in frame.columns:
        out[_safe_name(c)] = c
    return out


def _pick(frame: pd.DataFrame, *aliases: str) -> pd.Series | None:
    cmap = _canonical_cols(frame)
    for alias in aliases:
        key = _safe_name(alias)
        if key in cmap:
            return frame[cmap[key]]
    return None


def _to_utc(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return pd.to_datetime(series, utc=True, errors="coerce")


def _ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[list(cols)]


@dataclass
class NormalizedExports:
    trades: pd.DataFrame
    signals: pd.DataFrame
    summary: pd.DataFrame


def normalize_trade_export(
    raw: pd.DataFrame,
    source_file: str,
    *,
    config_pack: str,
    timeframe: str,
    tv_strategy_name: str,
) -> pd.DataFrame:
    entry_t = _to_utc(_pick(raw, "entry_time_utc", "entry time", "entry date", "entry_timestamp"))
    exit_t = _to_utc(_pick(raw, "exit_time_utc", "exit time", "exit date", "exit_timestamp"))
    bar_t = _to_utc(_pick(raw, "bar_time_utc", "bar time", "signal time", "time"))
    if bar_t.empty and not entry_t.empty:
        bar_t = entry_t

    entry_px = pd.to_numeric(_pick(raw, "entry_price", "entry price", "entry"), errors="coerce")
    exit_px = pd.to_numeric(_pick(raw, "exit_price", "exit price", "exit"), errors="coerce")

    side_src = _pick(raw, "side", "direction", "type")
    if side_src is None:
        side = pd.Series(["long"] * len(raw), index=raw.index)
    else:
        side = side_src.astype(str).str.lower().replace({"buy": "long", "long": "long", "sell": "short", "short": "short"})

    result_pct = pd.to_numeric(
        _pick(raw, "result_pct", "profit pct", "profit %", "pnl pct", "net profit pct"),
        errors="coerce",
    )
    result_r = pd.to_numeric(
        _pick(raw, "result_r", "r", "r multiple", "r mult", "result r"),
        errors="coerce",
    )

    symbol = _pick(raw, "symbol", "ticker", "market")
    if symbol is None:
        symbol = pd.Series(["UNKNOWN"] * len(raw), index=raw.index)

    bars_held = pd.to_numeric(_pick(raw, "bars_held", "bars held", "bars"), errors="coerce")

    out = pd.DataFrame(
        {
            "symbol": symbol.astype(str),
            "timeframe": [str(timeframe)] * len(raw),
            "tv_strategy_name": [str(tv_strategy_name)] * len(raw),
            "config_pack": [str(config_pack)] * len(raw),
            "bar_time_utc": bar_t,
            "entry_time_utc": entry_t,
            "exit_time_utc": exit_t,
            "entry_price": entry_px,
            "exit_price": exit_px,
            "side": side,
            "atr_entry": pd.to_numeric(_pick(raw, "atr_entry", "atr entry"), errors="coerce"),
            "sl_price": pd.to_numeric(_pick(raw, "sl_price", "stop_price", "stop price", "sl"), errors="coerce"),
            "tp_price": pd.to_numeric(_pick(raw, "tp_price", "target_price", "target price", "tp"), errors="coerce"),
            "rr_multiple": pd.to_numeric(_pick(raw, "rr_multiple", "rr", "r multiple"), errors="coerce"),
            "bars_held": bars_held,
            "result_r": result_r,
            "result_pct": result_pct,
            "year": entry_t.dt.year.astype("Int64"),
            "month": entry_t.dt.month.astype("Int64"),
            "source_file": [str(source_file)] * len(raw),
            "truth_label": [TruthLabel.EXACT_PINE_EXPORTED.value] * len(raw),
            "reconstruction_status": ["raw_export"] * len(raw),
            "reconstruction_assumptions": [""] * len(raw),
        }
    )
    return _ensure_columns(out, TRADE_REQUIRED_COLUMNS)


def normalize_signal_export(
    raw: pd.DataFrame,
    source_file: str,
    *,
    config_pack: str,
    timeframe: str,
    tv_strategy_name: str,
) -> pd.DataFrame:
    bar_t = _to_utc(_pick(raw, "bar_time_utc", "bar time", "time", "signal time"))
    entry_t = _to_utc(_pick(raw, "entry_time_utc", "entry time", "entry date", "time"))
    entry_px = pd.to_numeric(_pick(raw, "entry_price", "entry price", "entry"), errors="coerce")
    side = _pick(raw, "side", "direction", "type")
    if side is None:
        side = pd.Series(["long"] * len(raw), index=raw.index)

    sym = _pick(raw, "symbol", "ticker", "market")
    if sym is None:
        sym = pd.Series(["UNKNOWN"] * len(raw), index=raw.index)

    out = pd.DataFrame(
        {
            "symbol": sym.astype(str),
            "timeframe": [str(timeframe)] * len(raw),
            "tv_strategy_name": [str(tv_strategy_name)] * len(raw),
            "config_pack": [str(config_pack)] * len(raw),
            "bar_time_utc": bar_t,
            "entry_time_utc": entry_t,
            "side": side.astype(str).str.lower(),
            "entry_price": entry_px,
            "source_file": [str(source_file)] * len(raw),
            "truth_label": [TruthLabel.EXACT_PINE_EXPORTED.value] * len(raw),
        }
    )
    return _ensure_columns(out, SIGNAL_REQUIRED_COLUMNS)


def summarize_trades(trades: pd.DataFrame, source_file: str) -> pd.DataFrame:
    if trades.empty:
        return _ensure_columns(pd.DataFrame(), SUMMARY_REQUIRED_COLUMNS)

    frame = trades.copy()
    frame["result_r"] = pd.to_numeric(frame["result_r"], errors="coerce")
    grouped = frame.groupby(["symbol", "timeframe", "tv_strategy_name", "config_pack", "year"], dropna=False)
    rows: List[Dict[str, object]] = []
    for key, g in grouped:
        symbol, timeframe, tv_strategy_name, config_pack, year = key
        r = pd.to_numeric(g["result_r"], errors="coerce")
        wins = (r > 0).sum()
        losses = (r < 0).sum()
        gross_win = float(r[r > 0].sum()) if wins > 0 else 0.0
        gross_loss = float((-r[r < 0]).sum()) if losses > 0 else 0.0
        pf = float(gross_win / gross_loss) if gross_loss > 0 else np.nan
        rows.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "tv_strategy_name": tv_strategy_name,
                "config_pack": config_pack,
                "n_trades": int(len(g)),
                "win_rate": float((r > 0).mean()) if len(g) > 0 else np.nan,
                "net_r": float(r.sum()) if len(g) > 0 else np.nan,
                "expectancy_r": float(r.mean()) if len(g) > 0 else np.nan,
                "profit_factor": pf,
                "year": int(year) if pd.notna(year) else np.nan,
                "source_file": str(source_file),
                "truth_label": TruthLabel.EXACT_PINE_EXPORTED.value,
            }
        )
    out = pd.DataFrame(rows)
    return _ensure_columns(out, SUMMARY_REQUIRED_COLUMNS)


def normalize_summary_export(
    raw: pd.DataFrame,
    source_file: str,
    *,
    config_pack: str,
    timeframe: str,
    tv_strategy_name: str,
) -> pd.DataFrame:
    sym = _pick(raw, "symbol", "ticker", "market")
    if sym is None:
        sym = pd.Series(["UNKNOWN"] * len(raw), index=raw.index)
    year = pd.to_numeric(_pick(raw, "year"), errors="coerce").astype("Int64")

    out = pd.DataFrame(
        {
            "symbol": sym.astype(str),
            "timeframe": [str(timeframe)] * len(raw),
            "tv_strategy_name": [str(tv_strategy_name)] * len(raw),
            "config_pack": [str(config_pack)] * len(raw),
            "n_trades": pd.to_numeric(_pick(raw, "n_trades", "trades", "total trades"), errors="coerce"),
            "win_rate": pd.to_numeric(_pick(raw, "win_rate", "win rate", "percent profitable"), errors="coerce"),
            "net_r": pd.to_numeric(_pick(raw, "net_r", "net r"), errors="coerce"),
            "expectancy_r": pd.to_numeric(_pick(raw, "expectancy_r", "expectancy r"), errors="coerce"),
            "profit_factor": pd.to_numeric(_pick(raw, "profit_factor", "pf"), errors="coerce"),
            "year": year,
            "source_file": [str(source_file)] * len(raw),
            "truth_label": [TruthLabel.EXACT_PINE_EXPORTED.value] * len(raw),
        }
    )
    return _ensure_columns(out, SUMMARY_REQUIRED_COLUMNS)


def merge_normalized_exports(
    trades_parts: List[Tuple[pd.DataFrame, str]],
    signal_parts: List[Tuple[pd.DataFrame, str]],
    summary_parts: List[Tuple[pd.DataFrame, str]],
    *,
    config_pack: str,
    timeframe: str,
    tv_strategy_name: str,
) -> NormalizedExports:
    trade_frames = [
        normalize_trade_export(
            raw=df,
            source_file=src,
            config_pack=config_pack,
            timeframe=timeframe,
            tv_strategy_name=tv_strategy_name,
        )
        for df, src in trades_parts
    ]
    signal_frames = [
        normalize_signal_export(
            raw=df,
            source_file=src,
            config_pack=config_pack,
            timeframe=timeframe,
            tv_strategy_name=tv_strategy_name,
        )
        for df, src in signal_parts
    ]
    summary_frames = [
        normalize_summary_export(
            raw=df,
            source_file=src,
            config_pack=config_pack,
            timeframe=timeframe,
            tv_strategy_name=tv_strategy_name,
        )
        for df, src in summary_parts
    ]

    trades = pd.concat(trade_frames, ignore_index=True, sort=False) if trade_frames else _ensure_columns(pd.DataFrame(), TRADE_REQUIRED_COLUMNS)
    signals = pd.concat(signal_frames, ignore_index=True, sort=False) if signal_frames else _ensure_columns(pd.DataFrame(), SIGNAL_REQUIRED_COLUMNS)

    if signals.empty and not trades.empty:
        signals = _ensure_columns(
            trades[[
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
            ]].copy(),
            SIGNAL_REQUIRED_COLUMNS,
        )

    if summary_frames:
        summary = pd.concat(summary_frames, ignore_index=True, sort=False)
        summary = _ensure_columns(summary, SUMMARY_REQUIRED_COLUMNS)
    else:
        summary = summarize_trades(trades, source_file="normalized_from_ingestion")
    return NormalizedExports(trades=trades, signals=signals, summary=summary)


def discover_export_files(imports_dir: Path) -> Dict[str, List[Path]]:
    if not imports_dir.exists():
        raise FileNotFoundError(f"Imports directory does not exist: {imports_dir}")

    out: Dict[str, List[Path]] = {"trades": [], "signals": [], "summary": []}
    for path in sorted(imports_dir.rglob("*.csv")):
        name = path.name.lower()
        if "trade" in name:
            out["trades"].append(path)
        elif any(tok in name for tok in ["signal", "alert", "event"]):
            out["signals"].append(path)
        elif any(tok in name for tok in ["summary", "performance", "tester"]):
            out["summary"].append(path)

    return out

