from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from dfd05.indicators import dmi_adx, ema


H_RE = re.compile(r"^forward_return_(\d+)h_pct$")


def _extract_horizons(df: pd.DataFrame) -> List[int]:
    hs: List[int] = []
    for c in df.columns:
        m = H_RE.match(str(c))
        if m:
            hs.append(int(m.group(1)))
    return sorted(set(hs))


def _load_master(master_paths: Sequence[Path], horizons_hours: Sequence[int] | None) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    use_h = sorted({int(h) for h in (horizons_hours or []) if int(h) > 0})
    for p in master_paths:
        df = pd.read_parquet(p)
        if df.empty:
            continue
        hs = _extract_horizons(df)
        hs_eff = use_h if use_h else hs
        id_cols = [c for c in ["signal_id", "trade_id", "symbol", "timeframe", "entry_time", "entry_price", "truth_label", "config_pack"] if c in df.columns]
        for h in hs_eff:
            ret_col = f"forward_return_{h}h_pct"
            if ret_col not in df.columns:
                continue
            sub = df[id_cols].copy()
            sub["horizon_h"] = int(h)
            sub["forward_return_pct"] = pd.to_numeric(df[ret_col], errors="coerce")
            parts.append(sub)
    if not parts:
        return pd.DataFrame(columns=["symbol", "timeframe", "entry_time", "horizon_h", "forward_return_pct"])
    out = pd.concat(parts, ignore_index=True, sort=False)
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    out["forward_return_pct"] = pd.to_numeric(out["forward_return_pct"], errors="coerce")
    out = out.dropna(subset=["entry_time", "forward_return_pct", "symbol", "timeframe"])
    # Deduplicate overlapping signal rows from combined masters.
    out["_k"] = (
        out["symbol"].astype(str)
        + "|"
        + out["timeframe"].astype(str)
        + "|"
        + out["entry_time"].astype(str)
        + "|"
        + out["horizon_h"].astype(str)
    )
    out = out.drop_duplicates(subset=["_k"], keep="first").drop(columns=["_k"])
    out["entry_day"] = out["entry_time"].dt.floor("D")
    out["win"] = (out["forward_return_pct"] > 0).astype(int)
    return out.reset_index(drop=True)


def _bars_path(data_root: Path, symbol: str, timeframe: str) -> Path:
    return data_root / f"parquet_{str(timeframe).lower()}" / f"symbol={symbol}" / "part-0000.parquet"


def _build_daily_regime(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame(columns=["entry_day", "daily_close_prev", "ema200_prev", "adx14_prev", "regime_ema200", "regime_adx18"])
    b = bars.copy()
    b["time"] = pd.to_datetime(b["time"], utc=True, errors="coerce")
    for c in ("open", "high", "low", "close", "volume"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b = b.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time")
    if b.empty:
        return pd.DataFrame(columns=["entry_day", "daily_close_prev", "ema200_prev", "adx14_prev", "regime_ema200", "regime_adx18"])

    d = (
        b.set_index("time")
        .resample("1D", label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    if d.empty:
        return pd.DataFrame(columns=["entry_day", "daily_close_prev", "ema200_prev", "adx14_prev", "regime_ema200", "regime_adx18"])

    close = d["close"].to_numpy(dtype=float)
    high = d["high"].to_numpy(dtype=float)
    low = d["low"].to_numpy(dtype=float)
    ema200 = ema(close, 200)
    _, _, adx14 = dmi_adx(high, low, close, 14)

    d["daily_close_prev"] = pd.Series(close).shift(1)
    d["ema200_prev"] = pd.Series(ema200).shift(1)
    d["adx14_prev"] = pd.Series(adx14).shift(1)

    d["regime_ema200"] = np.where(
        d["daily_close_prev"].notna() & d["ema200_prev"].notna(),
        np.where(d["daily_close_prev"] > d["ema200_prev"], "above_ema200", "below_ema200"),
        "unknown",
    )
    d["regime_adx18"] = np.where(
        d["adx14_prev"].notna(),
        np.where(d["adx14_prev"] > 18.0, "adx_gt_18", "adx_le_18"),
        "unknown",
    )
    d = d.rename(columns={"time": "entry_day"})
    return d[["entry_day", "daily_close_prev", "ema200_prev", "adx14_prev", "regime_ema200", "regime_adx18"]]


def _enrich_regimes(master_long: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    if master_long.empty:
        return master_long.copy()
    parts: List[pd.DataFrame] = []
    for (symbol, timeframe), g in master_long.groupby(["symbol", "timeframe"], dropna=False, sort=True):
        p = _bars_path(data_root=data_root, symbol=str(symbol), timeframe=str(timeframe))
        if not p.exists():
            tmp = g.copy()
            tmp["regime_ema200"] = "unknown"
            tmp["regime_adx18"] = "unknown"
            tmp["daily_close_prev"] = np.nan
            tmp["ema200_prev"] = np.nan
            tmp["adx14_prev"] = np.nan
            parts.append(tmp)
            continue
        bars = pd.read_parquet(p)
        regimes = _build_daily_regime(bars)
        tmp = g.merge(regimes, on="entry_day", how="left")
        tmp["regime_ema200"] = tmp["regime_ema200"].fillna("unknown")
        tmp["regime_adx18"] = tmp["regime_adx18"].fillna("unknown")
        parts.append(tmp)
    out = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    out["regime_combo"] = out["regime_adx18"].astype(str) + "|" + out["regime_ema200"].astype(str)
    return out


def _agg_group(g: pd.DataFrame) -> Dict[str, float]:
    r = pd.to_numeric(g["forward_return_pct"], errors="coerce").dropna()
    n = int(r.shape[0])
    if n == 0:
        return {
            "n": 0,
            "win_rate": np.nan,
            "expectancy_pct": np.nan,
            "median_return_pct": np.nan,
            "std_return_pct": np.nan,
            "sharpe_signal": np.nan,
        }
    m = float(r.mean())
    sd = float(r.std(ddof=1)) if n > 1 else np.nan
    return {
        "n": n,
        "win_rate": float((r > 0).mean()),
        "expectancy_pct": m,
        "median_return_pct": float(r.median()),
        "std_return_pct": sd,
        "sharpe_signal": float(m / sd) if np.isfinite(sd) and sd > 0 else np.nan,
    }


def _aggregate(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for k, g in df.groupby(list(keys), dropna=False, sort=True):
        if not isinstance(k, tuple):
            k = (k,)
        row: Dict[str, object] = {c: v for c, v in zip(keys, k)}
        row.update(_agg_group(g))
        rows.append(row)
    return pd.DataFrame(rows)


def _top_tables(grid: pd.DataFrame, min_n: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if grid.empty:
        return pd.DataFrame(), pd.DataFrame()
    work = grid[pd.to_numeric(grid["n"], errors="coerce") >= int(min_n)].copy()
    if work.empty:
        work = grid.copy()
    top_exp = (
        work.sort_values(["expectancy_pct", "win_rate", "n"], ascending=[False, False, False], kind="mergesort")
        .groupby(["timeframe", "horizon_h"], as_index=False, sort=True)
        .head(5)
        .reset_index(drop=True)
    )
    top_wr = (
        work.sort_values(["win_rate", "expectancy_pct", "n"], ascending=[False, False, False], kind="mergesort")
        .groupby(["timeframe", "horizon_h"], as_index=False, sort=True)
        .head(5)
        .reset_index(drop=True)
    )
    return top_exp, top_wr


def run_grid(
    master_paths: Sequence[Path],
    data_root: Path,
    output_dir: Path,
    horizons_hours: Sequence[int] | None,
    min_n: int,
    out_prefix: str,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    master_long = _load_master(master_paths=master_paths, horizons_hours=horizons_hours)
    enriched = _enrich_regimes(master_long=master_long, data_root=data_root)

    keys = ["timeframe", "symbol", "horizon_h", "regime_adx18", "regime_ema200", "regime_combo"]
    grid = _aggregate(enriched, keys=keys)
    by_adx = _aggregate(enriched, keys=["timeframe", "symbol", "horizon_h", "regime_adx18"])
    by_ema = _aggregate(enriched, keys=["timeframe", "symbol", "horizon_h", "regime_ema200"])

    top_exp, top_wr = _top_tables(grid=grid, min_n=min_n)

    paths = {
        "grid": output_dir / f"{out_prefix}_grid.csv",
        "top_expectancy": output_dir / f"{out_prefix}_top_expectancy.csv",
        "top_winrate": output_dir / f"{out_prefix}_top_winrate.csv",
        "by_adx": output_dir / f"{out_prefix}_by_adx.csv",
        "by_ema": output_dir / f"{out_prefix}_by_ema.csv",
        "summary_md": output_dir / f"{out_prefix}_summary.md",
    }
    grid.to_csv(paths["grid"], index=False)
    top_exp.to_csv(paths["top_expectancy"], index=False)
    top_wr.to_csv(paths["top_winrate"], index=False)
    by_adx.to_csv(paths["by_adx"], index=False)
    by_ema.to_csv(paths["by_ema"], index=False)

    def _tbl(df: pd.DataFrame, cols: Sequence[str]) -> str:
        cols2 = [c for c in cols if c in df.columns]
        if not cols2:
            return "_No columns._"
        if df.empty:
            return "_No rows._"
        lines = [
            "| " + " | ".join(cols2) + " |",
            "| " + " | ".join(["---"] * len(cols2)) + " |",
        ]
        for _, row in df[cols2].iterrows():
            vals: List[str] = []
            for c in cols2:
                v = row[c]
                if isinstance(v, (float, np.floating)):
                    vals.append("" if not np.isfinite(float(v)) else f"{float(v):.6f}")
                elif pd.isna(v):
                    vals.append("")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines = [
        "# Pine16 Regime Grid Search",
        "",
        f"- master_paths_n: `{len(master_paths)}`",
        f"- data_root: `{data_root}`",
        f"- min_n: `{int(min_n)}`",
        "- regimes: `ADX(14) > 18 vs <= 18`, `prev close above/below daily EMA200`",
        "",
        "## Top By Expectancy",
        _tbl(top_exp, ["timeframe", "symbol", "horizon_h", "regime_adx18", "regime_ema200", "n", "win_rate", "expectancy_pct", "median_return_pct", "sharpe_signal"]),
        "",
        "## Top By Win Rate",
        _tbl(top_wr, ["timeframe", "symbol", "horizon_h", "regime_adx18", "regime_ema200", "n", "win_rate", "expectancy_pct", "median_return_pct", "sharpe_signal"]),
    ]
    paths["summary_md"].write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Regime grid search for forward-sign masters (ADX/EMA regimes).")
    ap.add_argument("--master-path", action="append", required=True, help="Forward-sign master parquet path; pass multiple times.")
    ap.add_argument("--data-root", default="data/derived/dukascopy", help="Canonical bars root containing parquet_{tf}/symbol=<SYM>/part-0000.parquet.")
    ap.add_argument("--output-dir", default="outputs/reports", help="Directory for grid outputs.")
    ap.add_argument("--out-prefix", default="pine16_regime_grid", help="Output file prefix.")
    ap.add_argument("--horizons-hours", nargs="+", type=int, default=None, help="Optional horizons filter, e.g. 24 336.")
    ap.add_argument("--min-n", type=int, default=20, help="Minimum sample size for top leaderboards.")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = run_grid(
        master_paths=[Path(p) for p in args.master_path],
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        horizons_hours=args.horizons_hours,
        min_n=int(args.min_n),
        out_prefix=str(args.out_prefix),
    )
    print(f"grid_csv: {paths['grid']}")
    print(f"top_expectancy_csv: {paths['top_expectancy']}")
    print(f"top_winrate_csv: {paths['top_winrate']}")
    print(f"by_adx_csv: {paths['by_adx']}")
    print(f"by_ema_csv: {paths['by_ema']}")
    print(f"summary_md: {paths['summary_md']}")


if __name__ == "__main__":
    main()
