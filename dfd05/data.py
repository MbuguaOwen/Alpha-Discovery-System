from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .config import RunConfig


TF_TO_DIR: Dict[str, str] = {
    "m1": "parquet_m1",
    "m15": "parquet_m15",
    "m30": "parquet_m30",
    "h1": "parquet_h1",
    "h4": "parquet_h4",
}

TF_TO_MINUTES: Dict[str, int] = {"m1": 1, "m15": 15, "m30": 30, "h1": 60, "h4": 240}
TF_TO_PANDAS_RULE: Dict[str, str] = {
    "m1": "1min",
    "m15": "15min",
    "m30": "30min",
    "h1": "1h",
    "h4": "4h",
}


def timeframe_to_minutes(timeframe: str) -> int:
    tf = timeframe.lower()
    if tf not in TF_TO_MINUTES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return TF_TO_MINUTES[tf]


def _normalize_time_column(series: pd.Series) -> pd.Series:
    if is_numeric_dtype(series):
        vmax = float(series.max())
        if vmax > 1e12:
            return pd.to_datetime(series, unit="ms", utc=True)
        if vmax > 1e10:
            return pd.to_datetime(series, unit="s", utc=True)
        return pd.to_datetime(series, unit="s", utc=True)
    return pd.to_datetime(series, utc=True)


def _find_parquet_files(roots: Iterable[str], timeframe: str, symbol: str) -> List[Path]:
    tf_dir = TF_TO_DIR[timeframe]
    for root in roots:
        base = Path(root) / tf_dir / f"symbol={symbol}"
        part0 = base / "part-0000.parquet"
        if part0.exists():
            return [part0]
        if base.exists():
            files = sorted(base.rglob("*.parquet"))
            if files:
                return files
    return []


def _read_parquet_with_schema(files: List[Path]) -> pd.DataFrame:
    cols = ["time", "open", "high", "low", "close", "volume"]
    try:
        return pd.read_parquet([str(p) for p in files], columns=cols)
    except Exception:
        # Some datasets contain schema drift across partitions (for example
        # dictionary vs string encoding on non-required columns). Read per-file.
        parts = []
        for path in files:
            try:
                parts.append(pd.read_parquet(str(path), columns=cols))
            except Exception:
                parts.append(pd.read_parquet(str(path)))
        return pd.concat(parts, ignore_index=True)


def _resample_ohlcv(bars: pd.DataFrame, target_tf: str, symbol: str) -> pd.DataFrame:
    rule = TF_TO_PANDAS_RULE[target_tf]
    if bars.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "symbol"])

    src = bars.sort_values("time").set_index("time")
    agg = src.resample(rule, label="left", closed="left", origin="epoch").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    agg = agg.dropna(subset=["open", "high", "low", "close"]).reset_index()
    agg["symbol"] = symbol
    return agg[["time", "open", "high", "low", "close", "volume", "symbol"]]


def _choose_resample_source_tfs(target_tf: str) -> List[str]:
    # Use the finest available source among {m1, m15}, preferring m1.
    target_minutes = timeframe_to_minutes(target_tf)
    out: List[str] = []
    for source_tf in ("m1", "m15"):
        if source_tf == target_tf:
            continue
        source_minutes = timeframe_to_minutes(source_tf)
        if target_minutes % source_minutes == 0:
            out.append(source_tf)
    return out


def _coerce_schema(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    cols_l = {c.lower(): c for c in df.columns}

    def pick(*names: str) -> Optional[str]:
        for n in names:
            if n in cols_l:
                return cols_l[n]
        return None

    time_col = pick("time", "timestamp", "datetime", "date")
    open_col = pick("open", "o")
    high_col = pick("high", "h")
    low_col = pick("low", "l")
    close_col = pick("close", "c")
    volume_col = pick("volume", "vol", "v", "tick_volume")
    symbol_col = pick("symbol")

    missing = [
        name
        for name, col in [
            ("time", time_col),
            ("open", open_col),
            ("high", high_col),
            ("low", low_col),
            ("close", close_col),
            ("volume", volume_col),
        ]
        if col is None
    ]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "time": _normalize_time_column(df[time_col]),
            "open": pd.to_numeric(df[open_col], errors="coerce"),
            "high": pd.to_numeric(df[high_col], errors="coerce"),
            "low": pd.to_numeric(df[low_col], errors="coerce"),
            "close": pd.to_numeric(df[close_col], errors="coerce"),
            "volume": pd.to_numeric(df[volume_col], errors="coerce"),
            "symbol": df[symbol_col].astype(str) if symbol_col is not None else symbol,
        }
    )
    out = out.dropna(subset=["time", "open", "high", "low", "close", "volume"])
    out = out.sort_values("time").drop_duplicates(subset=["time"], keep="last")
    return out.reset_index(drop=True)


def load_bars_for_symbol(config: RunConfig, symbol: str, timeframe: str) -> pd.DataFrame:
    tf = timeframe.lower()
    if tf not in TF_TO_MINUTES:
        raise ValueError(f"Unsupported timeframe {timeframe}. Expected one of {list(TF_TO_MINUTES)}")

    csv_key = f"{symbol}:{tf}"
    csv_path = config.data.csv_inputs.get(csv_key) or config.data.csv_inputs.get(symbol)
    if csv_path:
        df = pd.read_csv(csv_path)
        bars = _coerce_schema(df, symbol=symbol)
    else:
        files: List[Path] = []
        if tf in TF_TO_DIR:
            files = _find_parquet_files(config.data.parquet_roots, tf, symbol)
        if files:
            bars = _coerce_schema(_read_parquet_with_schema(files), symbol=symbol)
        else:
            source_tfs = _choose_resample_source_tfs(tf)
            if not source_tfs:
                raise FileNotFoundError(
                    f"No canonical parquet files found for symbol={symbol}, timeframe={tf}, "
                    f"and no compatible resample source in {{m1,m15}}."
                )
            src_files: List[Path] = []
            source_tf_used: Optional[str] = None
            for source_tf in source_tfs:
                src_files = _find_parquet_files(config.data.parquet_roots, source_tf, symbol)
                if src_files:
                    source_tf_used = source_tf
                    break
            if not src_files or source_tf_used is None:
                raise FileNotFoundError(
                    f"No canonical parquet files found for symbol={symbol}, timeframe={tf}, "
                    f"or source timeframe in {source_tfs}. roots={config.data.parquet_roots}"
                )
            src_bars = _coerce_schema(_read_parquet_with_schema(src_files), symbol=symbol)
            bars = _resample_ohlcv(src_bars, target_tf=tf, symbol=symbol)

    if config.data.years:
        bars = bars[bars["time"].dt.year.isin(config.data.years)]
    bars = bars.reset_index(drop=True)
    return bars
