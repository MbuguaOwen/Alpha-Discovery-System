from __future__ import annotations

import argparse
import asyncio
import dataclasses
import lzma
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import aiohttp
import numpy as np
import pandas as pd


DUKASCOPY_BASE = "https://datafeed.dukascopy.com/datafeed"
SUPPORTED_TFS = ("m1", "m15", "m30", "h1", "h4")
TF_RULES = {
    "m1": "1min",
    "m15": "15min",
    "m30": "30min",
    "h1": "1h",
    "h4": "4h",
}


@dataclasses.dataclass(frozen=True)
class DownloadTask:
    symbol: str
    day: date


@dataclasses.dataclass
class DownloadResult:
    symbol: str
    day: date
    status: str
    http_status: int | None
    bars: pd.DataFrame
    error: str = ""


def _parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def _date_range(start: date, end_exclusive: date) -> list[date]:
    out: list[date] = []
    d = start
    while d < end_exclusive:
        out.append(d)
        d = d + timedelta(days=1)
    return out


def _price_scale(symbol: str) -> float:
    s = symbol.upper().strip()
    if s.startswith("XAU") or s.startswith("XAG") or s.endswith("JPY"):
        return 1_000.0
    return 100_000.0


def _day_url(symbol: str, d: date) -> str:
    # Dukascopy month in URL is zero-indexed.
    m0 = d.month - 1
    return f"{DUKASCOPY_BASE}/{symbol}/{d.year}/{m0:02d}/{d.day:02d}/BID_candles_min_1.bi5"


def _decode_bi5_day(payload: bytes, symbol: str, d: date) -> pd.DataFrame:
    decoded = lzma.decompress(payload)
    if len(decoded) % 24 != 0:
        raise ValueError(f"Unexpected BI5 payload length for {symbol} {d}: {len(decoded)} bytes (not multiple of 24)")

    arr = np.frombuffer(
        decoded,
        dtype=np.dtype(
            [
                ("offset_s", ">u4"),
                ("open_raw", ">u4"),
                ("close_raw", ">u4"),
                ("low_raw", ">u4"),
                ("high_raw", ">u4"),
                ("volume", ">f4"),
            ]
        ),
    )
    if arr.size == 0:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "symbol"])

    # Use naive UTC anchor for numpy datetime64 compatibility.
    day_start = datetime(d.year, d.month, d.day, 0, 0, 0)
    base = np.datetime64(day_start, "s")
    times = pd.to_datetime(base + arr["offset_s"].astype("timedelta64[s]"), utc=True)
    scale = _price_scale(symbol)

    out = pd.DataFrame(
        {
            "time": times,
            "open": arr["open_raw"].astype(np.float64) / scale,
            "high": arr["high_raw"].astype(np.float64) / scale,
            "low": arr["low_raw"].astype(np.float64) / scale,
            "close": arr["close_raw"].astype(np.float64) / scale,
            "volume": arr["volume"].astype(np.float64),
            "symbol": symbol,
        }
    )
    return out


async def _download_one(
    session: aiohttp.ClientSession,
    task: DownloadTask,
    retries: int,
    timeout_sec: int,
) -> DownloadResult:
    url = _day_url(task.symbol, task.day)
    last_err = ""
    for attempt in range(max(1, retries) + 1):
        try:
            timeout = aiohttp.ClientTimeout(total=timeout_sec)
            async with session.get(url, timeout=timeout) as resp:
                if resp.status == 404:
                    return DownloadResult(
                        symbol=task.symbol,
                        day=task.day,
                        status="not_found",
                        http_status=404,
                        bars=pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "symbol"]),
                    )
                if resp.status != 200:
                    last_err = f"http_{resp.status}"
                    await asyncio.sleep(min(2 + attempt, 5))
                    continue
                payload = await resp.read()
                bars = _decode_bi5_day(payload, symbol=task.symbol, d=task.day)
                return DownloadResult(
                    symbol=task.symbol,
                    day=task.day,
                    status="ok",
                    http_status=200,
                    bars=bars,
                )
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(min(2 + attempt, 5))

    return DownloadResult(
        symbol=task.symbol,
        day=task.day,
        status="error",
        http_status=None,
        bars=pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "symbol"]),
        error=last_err,
    )


async def _download_symbol_days(
    symbol: str,
    days: Sequence[date],
    max_concurrency: int,
    retries: int,
    timeout_sec: int,
    show_progress: bool = True,
    progress_every: int = 50,
) -> list[DownloadResult]:
    connector = aiohttp.TCPConnector(limit=max(1, max_concurrency))
    sem = asyncio.Semaphore(max(1, max_concurrency))
    out: list[DownloadResult] = []
    total = int(len(days))
    done = 0
    ok_n = 0
    nf_n = 0
    err_n = 0
    t0 = time.perf_counter()
    async with aiohttp.ClientSession(connector=connector) as session:
        async def run_task(d: date) -> DownloadResult:
            async with sem:
                return await _download_one(session, DownloadTask(symbol=symbol, day=d), retries=retries, timeout_sec=timeout_sec)

        coros = [run_task(d) for d in days]
        for fut in asyncio.as_completed(coros):
            r = await fut
            out.append(r)
            done += 1
            if r.status == "ok":
                ok_n += 1
            elif r.status == "not_found":
                nf_n += 1
            else:
                err_n += 1
            if show_progress and (done == total or done % max(1, int(progress_every)) == 0):
                elapsed = max(1e-9, time.perf_counter() - t0)
                rate = float(done) / elapsed
                eta = (float(total - done) / rate) if rate > 0 else float("nan")
                pct = (100.0 * float(done) / float(total)) if total > 0 else 100.0
                eta_str = f"{eta:.1f}s" if np.isfinite(eta) else "n/a"
                print(
                    f"[restore] {symbol} {done}/{total} ({pct:.1f}%) ok={ok_n} not_found={nf_n} error={err_n} rate={rate:.2f}/s eta={eta_str}",
                    flush=True,
                )
    out.sort(key=lambda x: x.day)
    return out


def _resample_ohlcv(df_m1: pd.DataFrame, tf: str) -> pd.DataFrame:
    if tf == "m1":
        return df_m1.copy()
    if tf not in TF_RULES:
        raise ValueError(f"Unsupported timeframe: {tf}")
    if df_m1.empty:
        return df_m1.copy()

    rule = TF_RULES[tf]
    src = df_m1.sort_values("time").set_index("time")
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
    agg["symbol"] = df_m1["symbol"].iloc[0]
    return agg[["time", "open", "high", "low", "close", "volume", "symbol"]]


def _dedupe_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    for c in ("open", "high", "low", "close", "volume"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["time", "open", "high", "low", "close", "volume"])
    out = out.sort_values("time").drop_duplicates(subset=["time"], keep="last")
    out = out[(out["high"] >= out[["open", "close", "low"]].max(axis=1)) & (out["low"] <= out[["open", "close", "high"]].min(axis=1))]
    return out.reset_index(drop=True)


def _gap_report(df_m1: pd.DataFrame, symbol: str) -> pd.DataFrame:
    cols = [
        "symbol",
        "gap_after_time",
        "next_time",
        "missing_minutes",
        "contains_weekend",
        "unexpected_gap",
    ]
    if df_m1.empty:
        return pd.DataFrame(columns=cols)

    work = df_m1.sort_values("time").copy()
    prev_t = work["time"].shift(1)
    delta_m = (work["time"] - prev_t).dt.total_seconds() / 60.0
    mask = delta_m > 1.0
    if not mask.any():
        return pd.DataFrame(columns=cols)

    rows: list[dict[str, object]] = []
    for i in work.index[mask]:
        t_prev = prev_t.loc[i]
        t_next = work.at[i, "time"]
        miss = int(round((t_next - t_prev).total_seconds() / 60.0)) - 1
        if miss <= 0:
            continue
        missing_idx = pd.date_range(t_prev + pd.Timedelta(minutes=1), t_next - pd.Timedelta(minutes=1), freq="1min", tz="UTC")
        w = missing_idx.weekday
        contains_weekend = bool((w >= 5).any())
        unexpected = bool((w < 5).any())
        rows.append(
            {
                "symbol": symbol,
                "gap_after_time": t_prev,
                "next_time": t_next,
                "missing_minutes": miss,
                "contains_weekend": int(contains_weekend),
                "unexpected_gap": int(unexpected),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def _fill_short_unexpected_gaps(df_m1: pd.DataFrame, max_fill_minutes: int) -> tuple[pd.DataFrame, int]:
    if df_m1.empty or max_fill_minutes <= 0:
        return df_m1.copy(), 0
    work = df_m1.sort_values("time").reset_index(drop=True)
    fills: list[dict[str, object]] = []
    filled = 0

    for i in range(1, len(work)):
        prev_row = work.iloc[i - 1]
        cur_row = work.iloc[i]
        t_prev = prev_row["time"]
        t_cur = cur_row["time"]
        miss = int(round((t_cur - t_prev).total_seconds() / 60.0)) - 1
        if miss <= 0 or miss > int(max_fill_minutes):
            continue
        missing_idx = pd.date_range(t_prev + pd.Timedelta(minutes=1), t_cur - pd.Timedelta(minutes=1), freq="1min", tz="UTC")
        if (missing_idx.weekday >= 5).any():
            continue
        px = float(prev_row["close"])
        for ts in missing_idx:
            fills.append(
                {
                    "time": ts,
                    "open": px,
                    "high": px,
                    "low": px,
                    "close": px,
                    "volume": 0.0,
                    "symbol": str(prev_row["symbol"]),
                }
            )
        filled += len(missing_idx)

    if not fills:
        return work, 0
    out = pd.concat([work, pd.DataFrame(fills)], ignore_index=True, sort=False)
    out = out.sort_values("time").drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)
    return out, filled


def _write_symbol_timeframe(df: pd.DataFrame, output_root: Path, tf: str, symbol: str) -> Path:
    out_dir = output_root / f"parquet_{tf}" / f"symbol={symbol}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "part-0000.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def _build_tasks(symbols: Iterable[str], start_d: date, end_d_exclusive: date) -> dict[str, list[date]]:
    days = _date_range(start_d, end_d_exclusive)
    return {s: days for s in symbols}


async def _run_restore_async(
    symbols: Sequence[str],
    start_date: date,
    end_date_exclusive: date,
    warmup_days: int,
    output_root: Path,
    build_tfs: Sequence[str],
    max_concurrency: int,
    retries: int,
    timeout_sec: int,
    fill_unexpected_gaps: bool,
    max_fill_minutes: int,
    retry_manifest: Path | None = None,
    show_progress: bool = True,
    progress_every: int = 50,
) -> dict[str, Path]:
    if start_date >= end_date_exclusive:
        raise ValueError("start-date must be before end-date.")
    for tf in build_tfs:
        if tf not in SUPPORTED_TFS:
            raise ValueError(f"Unsupported timeframe in --build-timeframes: {tf}")

    effective_start = start_date - timedelta(days=max(0, int(warmup_days)))
    if retry_manifest is not None:
        if not retry_manifest.exists():
            raise FileNotFoundError(f"Retry manifest not found: {retry_manifest}")
        man = pd.read_csv(retry_manifest)
        if man.empty or not {"symbol", "day", "status"}.issubset(man.columns):
            raise ValueError("Retry manifest must contain symbol/day/status columns.")
        err = man[man["status"].astype(str).str.lower() == "error"].copy()
        err["symbol"] = err["symbol"].astype(str).str.upper()
        err["day"] = pd.to_datetime(err["day"], errors="coerce").dt.date
        err = err[err["symbol"].isin(symbols) & err["day"].notna()]
        task_days = {s: sorted(err.loc[err["symbol"] == s, "day"].tolist()) for s in symbols}
    else:
        task_days = _build_tasks(symbols=symbols, start_d=effective_start, end_d_exclusive=end_date_exclusive)

    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "restore_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifests: list[pd.DataFrame] = []
    gap_reports: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for symbol in symbols:
        if show_progress:
            print(
                f"[restore] symbol={symbol} days={len(task_days[symbol])} "
                f"mode={'retry_manifest' if retry_manifest is not None else 'full_range'}",
                flush=True,
            )
        existing_m1_path = output_root / "parquet_m1" / f"symbol={symbol}" / "part-0000.parquet"
        existing_m1 = (
            pd.read_parquet(existing_m1_path)
            if retry_manifest is not None and existing_m1_path.exists()
            else pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "symbol"])
        )

        results = await _download_symbol_days(
            symbol=symbol,
            days=task_days[symbol],
            max_concurrency=max_concurrency,
            retries=retries,
            timeout_sec=timeout_sec,
            show_progress=show_progress,
            progress_every=progress_every,
        )

        manifest = pd.DataFrame(
            [
                {
                    "symbol": r.symbol,
                    "day": r.day.isoformat(),
                    "status": r.status,
                    "http_status": r.http_status,
                    "bars": int(len(r.bars)),
                    "error": r.error,
                }
                for r in results
            ],
            columns=["symbol", "day", "status", "http_status", "bars", "error"],
        )
        manifests.append(manifest)

        bars_parts = [existing_m1] if not existing_m1.empty else []
        bars_parts.extend([r.bars for r in results if r.status == "ok" and not r.bars.empty])
        bars_m1 = pd.concat(bars_parts, ignore_index=True, sort=False) if bars_parts else pd.DataFrame(
            columns=["time", "open", "high", "low", "close", "volume", "symbol"]
        )
        bars_m1 = _dedupe_and_validate(bars_m1)

        gap_before = _gap_report(bars_m1, symbol=symbol)
        gap_reports.append(gap_before.assign(stage="before_fill"))
        filled_rows = 0
        if fill_unexpected_gaps:
            bars_m1, filled_rows = _fill_short_unexpected_gaps(bars_m1, max_fill_minutes=max_fill_minutes)
            bars_m1 = _dedupe_and_validate(bars_m1)
        gap_after = _gap_report(bars_m1, symbol=symbol)
        gap_reports.append(gap_after.assign(stage="after_fill"))

        for tf in build_tfs:
            out_tf = _resample_ohlcv(bars_m1, tf=tf)
            _write_symbol_timeframe(out_tf, output_root=output_root, tf=tf, symbol=symbol)

        ok_days = int((manifest["status"] == "ok").sum())
        nf_days = int((manifest["status"] == "not_found").sum())
        err_days = int((manifest["status"] == "error").sum())
        summary_rows.append(
            {
                "symbol": symbol,
                "effective_start": effective_start.isoformat(),
                "requested_start": start_date.isoformat(),
                "end_exclusive": end_date_exclusive.isoformat(),
                "requested_days": int(len(task_days[symbol])),
                "ok_days": ok_days,
                "not_found_days": nf_days,
                "error_days": err_days,
                "bars_m1": int(len(bars_m1)),
                "filled_rows": int(filled_rows),
                "unexpected_gaps_after_fill": int(pd.to_numeric(gap_after.get("unexpected_gap", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
            }
        )
        if show_progress:
            print(
                f"[restore] symbol={symbol} complete ok_days={ok_days} not_found_days={nf_days} "
                f"error_days={err_days} bars_m1={len(bars_m1)}",
                flush=True,
            )

    manifest_all = pd.concat(manifests, ignore_index=True, sort=False) if manifests else pd.DataFrame()
    gaps_all = pd.concat(gap_reports, ignore_index=True, sort=False) if gap_reports else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)

    manifest_path = logs_dir / "download_manifest.csv"
    gaps_path = logs_dir / "gap_report.csv"
    summary_path = logs_dir / "restore_summary.csv"
    manifest_all.to_csv(manifest_path, index=False)
    gaps_all.to_csv(gaps_path, index=False)
    summary.to_csv(summary_path, index=False)

    return {
        "manifest": manifest_path,
        "gaps": gaps_path,
        "summary": summary_path,
        "output_root": output_root,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Restore Dukascopy bars and write canonical parquet_{tf}/symbol=<SYM>/part-0000.parquet files."
    )
    ap.add_argument("--symbols", nargs="+", default=["XAUUSD", "XAGUSD", "EURUSD"], help="Symbols to restore.")
    ap.add_argument("--start-date", default="2022-01-01", help="Requested start date (inclusive), YYYY-MM-DD.")
    ap.add_argument("--end-date", default="2026-01-01", help="End date (exclusive), YYYY-MM-DD.")
    ap.add_argument("--warmup-days", type=int, default=30, help="Extra history before start-date for strategy warmup.")
    ap.add_argument("--output-root", default="data/derived/dukascopy", help="Output root for canonical parquet folders.")
    ap.add_argument(
        "--build-timeframes",
        nargs="+",
        default=["m1", "m15", "m30", "h1", "h4"],
        help="Timeframes to write. Supported: m1 m15 m30 h1 h4.",
    )
    ap.add_argument("--max-concurrency", type=int, default=16, help="Concurrent download tasks.")
    ap.add_argument("--retries", type=int, default=2, help="Retries per day file after the first attempt.")
    ap.add_argument("--timeout-sec", type=int, default=20, help="HTTP timeout seconds per request.")
    ap.add_argument(
        "--retry-manifest",
        default=None,
        help="Optional path to a previous download_manifest.csv; only rows with status=error are retried and merged into existing parquet.",
    )
    ap.add_argument("--progress-every", type=int, default=50, help="Emit progress every N completed day downloads per symbol.")
    ap.add_argument("--quiet-progress", action="store_true", help="Disable live download progress logs.")
    ap.add_argument("--fill-unexpected-gaps", action="store_true", help="Fill small unexpected weekday minute gaps with flat zero-volume bars.")
    ap.add_argument("--max-fill-minutes", type=int, default=3, help="Only fill unexpected gaps with missing minutes <= this threshold.")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    symbols = [str(s).strip().upper() for s in args.symbols if str(s).strip()]
    if not symbols:
        raise SystemExit("No symbols provided.")
    start_d = _parse_date(args.start_date)
    end_d = _parse_date(args.end_date)

    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    artifacts = asyncio.run(
        _run_restore_async(
            symbols=symbols,
            start_date=start_d,
            end_date_exclusive=end_d,
            warmup_days=int(args.warmup_days),
            output_root=Path(args.output_root),
            build_tfs=[str(tf).strip().lower() for tf in args.build_timeframes if str(tf).strip()],
            max_concurrency=int(args.max_concurrency),
            retries=int(args.retries),
            timeout_sec=int(args.timeout_sec),
            fill_unexpected_gaps=bool(args.fill_unexpected_gaps),
            max_fill_minutes=int(args.max_fill_minutes),
            retry_manifest=Path(args.retry_manifest) if args.retry_manifest else None,
            show_progress=not bool(args.quiet_progress),
            progress_every=max(1, int(args.progress_every)),
        )
    )

    print(f"output_root: {artifacts['output_root']}")
    print(f"manifest_csv: {artifacts['manifest']}")
    print(f"gap_report_csv: {artifacts['gaps']}")
    print(f"restore_summary_csv: {artifacts['summary']}")


if __name__ == "__main__":
    main()
