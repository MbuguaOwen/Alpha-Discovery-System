from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfd05.dashboard import build_horizon_dashboard, merge_events_forward
from dfd05.forward import infer_forward_mode


def _parse_horizons_csv(raw: str) -> List[int]:
    out: List[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    if not out:
        raise ValueError("No valid horizons parsed from --horizons")
    return sorted(set(out))


def _infer_tf_runid(paths: list[Path], frame: pd.DataFrame) -> tuple[str, str]:
    pat = re.compile(r"^(?:labeled|forward|events)_dfd05_(?P<tf>[^_]+)_(?P<rid>.+)$")
    for p in paths:
        m = pat.match(p.stem)
        if m:
            return m.group("tf"), m.group("rid")

    timeframe = "unknown"
    if "timeframe" in frame.columns and not frame.empty:
        vals = frame["timeframe"].dropna().astype(str).unique().tolist()
        if len(vals) == 1:
            timeframe = vals[0]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return timeframe, run_id


def main() -> None:
    ap = argparse.ArgumentParser(description="Build DFD05 horizon dashboard CSV")
    ap.add_argument("--labeled", required=False, help="Path to labeled_dfd05 parquet")
    ap.add_argument("--forward", required=False, help="Path to forward_dfd05 parquet")
    ap.add_argument("--events", required=False, help="Path to events_dfd05 parquet")
    ap.add_argument("--horizons", required=False, help='Optional override, e.g. "4,24,72"')
    ap.add_argument("--rr", type=float, default=2.0, help="RR multiplier for expR metrics")
    ap.add_argument(
        "--mode",
        choices=["auto", "time_only", "barrier"],
        default="auto",
        help="Forward mode for dashboard interpretation",
    )
    ap.add_argument("--outdir", default="data/reports", help="Output directory for CSV")
    args = ap.parse_args()

    input_paths: list[Path] = []
    if args.labeled:
        labeled_path = Path(args.labeled)
        input_paths.append(labeled_path)
        frame = pd.read_parquet(labeled_path)
    else:
        if not args.forward or not args.events:
            raise SystemExit("Provide --labeled OR both --forward and --events.")
        events_path = Path(args.events)
        forward_path = Path(args.forward)
        input_paths.extend([events_path, forward_path])
        events_df = pd.read_parquet(events_path)
        forward_df = pd.read_parquet(forward_path)
        frame = merge_events_forward(events_df=events_df, forward_df=forward_df)

    horizons = _parse_horizons_csv(args.horizons) if args.horizons else None
    mode = args.mode
    if mode == "auto":
        mode = infer_forward_mode(frame.columns)
    dashboard = build_horizon_dashboard(
        frame=frame,
        horizons_hours=horizons,
        rr_mult=args.rr,
        mode=mode,
    )

    timeframe, run_id = _infer_tf_runid(input_paths, frame)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "time_only":
        out_path = out_dir / f"dfd05_dashboard_timeonly_{timeframe}_{run_id}.csv"
    else:
        out_path = out_dir / f"dfd05_dashboard_{timeframe}_{run_id}.csv"
    dashboard.to_csv(out_path, index=False)

    print(f"dashboard: {out_path}")
    if dashboard.empty:
        return

    overall = dashboard[dashboard["section"] == "overall"].sort_values("horizon_h")
    ranking = dashboard[dashboard["section"] == "ranking"].sort_values("rank")
    if not overall.empty:
        if mode == "time_only":
            for _, row in overall.iterrows():
                print(
                    f"[overall] h={int(row['horizon_h'])}h "
                    f"n_total={int(row['n_total'])} n_valid={int(row['n_valid'])} "
                    f"mean_ret={row['mean_ret']:.6f} median_ret={row['median_ret']:.6f} "
                    f"up_rate={row['up_rate']:.6f} "
                    f"mean_mfe={row['mean_mfe']:.6f} mean_mae={row['mean_mae']:.6f} "
                    f"p25={row['p25_ret']:.6f} p50={row['p50_ret']:.6f} p75={row['p75_ret']:.6f} "
                    f"worst_year_mean_ret={row['worst_year_mean_ret']:.6f} "
                    f"worst_year_up_rate={row['worst_year_up_rate']:.6f}"
                )
        else:
            for _, row in overall.iterrows():
                print(
                    f"[overall] h={int(row['horizon_h'])}h n={int(row['n'])} "
                    f"tp={row['tp_rate']:.6f} sl={row['sl_rate']:.6f} "
                    f"same={row['samebar_rate']:.6f} no_hit={row['no_hit_rate']:.6f} "
                    f"tp_res={row['tp_resolved_rate']:.6f} sl_res={row['sl_resolved_rate']:.6f} "
                    f"hit={row['hit_rate']:.6f} trunc={row['trunc_rate']:.6f} "
                    f"expR_total={row['expR_total']:.6f} expR_hit={row['expR_hit_only']:.6f}"
                )
    if not ranking.empty:
        top = ranking.iloc[0]
        if mode == "time_only":
            print(
                f"[ranking] best_h={int(top['horizon_h'])}h "
                f"worst_year_mean_ret={top['worst_year_mean_ret']:.6f} "
                f"overall_mean_ret={top['overall_mean_ret']:.6f}"
            )
        else:
            print(
                f"[ranking] best_h={int(top['horizon_h'])}h "
                f"worst_year_expR={top['worst_year_expR']:.6f} overall_expR={top['overall_expR']:.6f}"
            )


if __name__ == "__main__":
    main()
