from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


DEFAULT_SYMBOLS = [
    "XAUUSD",
    "XAGUSD",
    "LIGHTCMDUSD",
    "BRENTCMDUSD",
    "EURJPY",
    "GBPJPY",
    "USDJPY",
]


def _action_24_336(w24: float, m24: float, w336: float, m336: float) -> str:
    keep = (
        (np.isfinite(w24) and w24 > 0.52 and np.isfinite(m24) and m24 > 0.0)
        or (np.isfinite(w336) and w336 > 0.52 and np.isfinite(m336) and m336 > 0.0)
    )
    cut = (
        (not np.isfinite(w24) or w24 < 0.50)
        and (not np.isfinite(w336) or w336 < 0.50)
        and (not np.isfinite(m24) or m24 <= 0.0)
        and (not np.isfinite(m336) or m336 <= 0.0)
    )
    if keep:
        return "KEEP"
    if cut:
        return "CUT"
    return "WATCH"


def _build_summary(
    by_symbol_session: pd.DataFrame,
    symbols: Iterable[str],
    timeframe: str,
    scope: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    use = by_symbol_session[
        (by_symbol_session["analysis_session_scope"].astype(str) == scope)
        & (by_symbol_session["symbol"].astype(str).isin(list(symbols)))
    ].copy()
    for symbol in symbols:
        sub = use[use["symbol"].astype(str) == str(symbol)]
        row: Dict[str, object] = {
            "timeframe": str(timeframe),
            "analysis_session_scope": str(scope),
            "symbol": str(symbol),
            "truth_label_block": (sub["truth_label_block"].iloc[0] if not sub.empty else "UNVERIFIED_PYTHON_APPROXIMATION"),
        }
        for h in (24, 168, 336):
            hs = sub[pd.to_numeric(sub["horizon_h"], errors="coerce") == int(h)]
            row[f"n_{h}h"] = int(pd.to_numeric(hs["n_signals"], errors="coerce").iloc[0]) if not hs.empty else 0
            row[f"win_rate_{h}h"] = float(pd.to_numeric(hs["win_rate"], errors="coerce").iloc[0]) if not hs.empty else np.nan
            row[f"mean_return_{h}h_pct"] = (
                float(pd.to_numeric(hs["mean_forward_return_pct"], errors="coerce").iloc[0]) if not hs.empty else np.nan
            )
            row[f"median_return_{h}h_pct"] = (
                float(pd.to_numeric(hs["median_forward_return_pct"], errors="coerce").iloc[0]) if not hs.empty else np.nan
            )
        row["session_action_24h_336h"] = _action_24_336(
            float(row["win_rate_24h"]),
            float(row["mean_return_24h_pct"]),
            float(row["win_rate_336h"]),
            float(row["mean_return_336h_pct"]),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _build_changes(summary_all: pd.DataFrame, summary_lny: pd.DataFrame) -> pd.DataFrame:
    rank = {"CUT": 0, "WATCH": 1, "KEEP": 2}
    rows: List[Dict[str, object]] = []
    tfs = sorted(set(summary_all["timeframe"].astype(str)) & set(summary_lny["timeframe"].astype(str)))
    for tf in tfs:
        a_tf = summary_all[summary_all["timeframe"].astype(str) == str(tf)].copy()
        l_tf = summary_lny[summary_lny["timeframe"].astype(str) == str(tf)].copy()
        amap = {str(r["symbol"]): r for _, r in a_tf.iterrows()}
        lmap = {str(r["symbol"]): r for _, r in l_tf.iterrows()}
        for symbol in sorted(set(amap.keys()) & set(lmap.keys())):
            a = amap[symbol]
            l = lmap[symbol]
            a_act = str(a["session_action_24h_336h"])
            l_act = str(l["session_action_24h_336h"])
            delta = int(rank.get(l_act, 0) - rank.get(a_act, 0))
            change = "UNCHANGED"
            if delta > 0:
                change = "UPGRADE"
            elif delta < 0:
                change = "DOWNGRADE"
            rows.append(
                {
                    "timeframe": str(tf),
                    "symbol": symbol,
                    "action_all_sessions_24h_336h": a_act,
                    "action_london_or_newyork_24h_336h": l_act,
                    "change_type": change,
                    "delta_rank": delta,
                    "all_336h_win_rate": float(a["win_rate_336h"]),
                    "london_ny_336h_win_rate": float(l["win_rate_336h"]),
                    "all_336h_mean_return_pct": float(a["mean_return_336h_pct"]),
                    "london_ny_336h_mean_return_pct": float(l["mean_return_336h_pct"]),
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["timeframe", "delta_rank", "symbol"], kind="mergesort").reset_index(drop=True)


def _build_matrix(summary_lny: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    syms = sorted(summary_lny["symbol"].astype(str).unique().tolist())
    for symbol in syms:
        s15 = summary_lny[(summary_lny["timeframe"] == "m15") & (summary_lny["symbol"] == symbol)]
        s30 = summary_lny[(summary_lny["timeframe"] == "m30") & (summary_lny["symbol"] == symbol)]
        if s15.empty or s30.empty:
            continue
        r15 = s15.iloc[0]
        r30 = s30.iloc[0]
        a15 = str(r15["session_action_24h_336h"])
        a30 = str(r30["session_action_24h_336h"])
        allow15 = bool(a15 == "KEEP")
        allow30 = bool(a30 == "KEEP")
        tier = "C"
        if allow15 and allow30:
            tier = "A"
        elif allow15 or allow30:
            tier = "B"
        rows.append(
            {
                "symbol": symbol,
                "session_restriction": "london_or_newyork",
                "m15_action_24h_336h": a15,
                "m30_action_24h_336h": a30,
                "m15_allowed": allow15,
                "m30_allowed": allow30,
                "confidence_tier": tier,
                "m15_win_rate_336h": float(r15["win_rate_336h"]),
                "m30_win_rate_336h": float(r30["win_rate_336h"]),
                "m15_mean_return_336h_pct": float(r15["mean_return_336h_pct"]),
                "m30_mean_return_336h_pct": float(r30["mean_return_336h_pct"]),
                "m15_win_rate_168h": float(r15["win_rate_168h"]),
                "m30_win_rate_168h": float(r30["win_rate_168h"]),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["confidence_tier", "symbol"], kind="mergesort").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build London+NY validation tables (24h/168h/336h) from forward-sign outputs.")
    ap.add_argument(
        "--m15-dir",
        default="outputs/reports_watchlist_m15_allsessions_24h_168h_336h",
        help="Directory containing m15 forward-sign outputs.",
    )
    ap.add_argument(
        "--m30-dir",
        default="outputs/reports_watchlist_m30_allsessions_24h_168h_336h",
        help="Directory containing m30 forward-sign outputs.",
    )
    ap.add_argument(
        "--output-dir",
        default="outputs/reports",
        help="Directory for validation outputs.",
    )
    ap.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Watchlist symbols in reporting order.",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = {
        "m15": Path(args.m15_dir),
        "m30": Path(args.m30_dir),
    }

    summaries_all: List[pd.DataFrame] = []
    summaries_lny: List[pd.DataFrame] = []
    yearly_parts: List[pd.DataFrame] = []

    for tf, run_dir in runs.items():
        by_symbol_session = pd.read_csv(run_dir / "pine16_forward_sign_24h_168h_336h_by_symbol_session.csv")
        by_symbol_year_session = pd.read_csv(run_dir / "pine16_forward_sign_24h_168h_336h_by_symbol_year_session.csv")

        s_all = _build_summary(by_symbol_session, args.symbols, timeframe=tf, scope="all_sessions")
        s_lny = _build_summary(by_symbol_session, args.symbols, timeframe=tf, scope="london_or_newyork")
        summaries_all.append(s_all)
        summaries_lny.append(s_lny)

        y = by_symbol_year_session[
            (by_symbol_year_session["analysis_session_scope"].astype(str) == "london_or_newyork")
            & (pd.to_numeric(by_symbol_year_session["horizon_h"], errors="coerce") == 336)
            & (by_symbol_year_session["symbol"].astype(str).isin(list(args.symbols)))
        ].copy()
        y.insert(0, "timeframe", tf)
        yearly_parts.append(
            y[
                [
                    "timeframe",
                    "symbol",
                    "year",
                    "n_signals",
                    "win_rate",
                    "mean_forward_return_pct",
                    "median_forward_return_pct",
                    "truth_label_block",
                ]
            ]
        )

    summary_all = pd.concat(summaries_all, ignore_index=True)
    summary_lny = pd.concat(summaries_lny, ignore_index=True)
    summary = summary_lny.sort_values(["timeframe", "symbol"], kind="mergesort").reset_index(drop=True)
    changes = _build_changes(summary_all, summary_lny)
    yearly = pd.concat(yearly_parts, ignore_index=True).sort_values(["timeframe", "symbol", "year"], kind="mergesort")
    matrix = _build_matrix(summary_lny)

    summary_csv = out_dir / "pine16_watchlist_london_ny_summary_24h_168h_336h.csv"
    changes_csv = out_dir / "pine16_watchlist_london_ny_action_changes_vs_all_sessions_24h_168h_336h.csv"
    yearly_csv = out_dir / "pine16_watchlist_london_ny_336h_yearly_consistency.csv"
    matrix_csv = out_dir / "pine16_watchlist_execution_matrix_london_ny.csv"
    note_md = out_dir / "pine16_watchlist_london_ny_validation_24h_168h_336h.md"

    summary.to_csv(summary_csv, index=False)
    changes.to_csv(changes_csv, index=False)
    yearly.to_csv(yearly_csv, index=False)
    matrix.to_csv(matrix_csv, index=False)

    m30 = summary[summary["timeframe"] == "m30"][
        [
            "symbol",
            "n_24h",
            "win_rate_24h",
            "mean_return_24h_pct",
            "n_168h",
            "win_rate_168h",
            "mean_return_168h_pct",
            "n_336h",
            "win_rate_336h",
            "mean_return_336h_pct",
            "session_action_24h_336h",
        ]
    ].copy()
    for c in ("win_rate_24h", "win_rate_168h", "win_rate_336h"):
        m30[c] = (pd.to_numeric(m30[c], errors="coerce") * 100.0).round(3)
    for c in ("mean_return_24h_pct", "mean_return_168h_pct", "mean_return_336h_pct"):
        m30[c] = pd.to_numeric(m30[c], errors="coerce").round(3)

    chg = changes[
        [
            "timeframe",
            "symbol",
            "action_all_sessions_24h_336h",
            "action_london_or_newyork_24h_336h",
            "change_type",
            "all_336h_win_rate",
            "london_ny_336h_win_rate",
            "all_336h_mean_return_pct",
            "london_ny_336h_mean_return_pct",
        ]
    ].copy()
    for c in ("all_336h_win_rate", "london_ny_336h_win_rate"):
        chg[c] = (pd.to_numeric(chg[c], errors="coerce") * 100.0).round(3)
    for c in ("all_336h_mean_return_pct", "london_ny_336h_mean_return_pct"):
        chg[c] = pd.to_numeric(chg[c], errors="coerce").round(3)

    lines = [
        "# Pine16 London+NY Validation (24h/168h/336h)",
        "",
        "Truth stamp: `UNVERIFIED_PYTHON_APPROXIMATION`",
        "",
        "## Core outputs",
        f"- summary_csv: `{summary_csv.as_posix()}`",
        f"- changes_csv: `{changes_csv.as_posix()}`",
        f"- yearly_336_csv: `{yearly_csv.as_posix()}`",
        f"- execution_matrix_csv: `{matrix_csv.as_posix()}`",
        "",
        "## M30 London+New York quick table (24h / 168h / 336h)",
        m30.to_csv(index=False).strip(),
        "",
        "## Downgrades/Upgrades vs all_sessions (24h/336h action)",
        chg.to_csv(index=False).strip(),
        "",
    ]
    note_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"summary_csv: {summary_csv}")
    print(f"changes_csv: {changes_csv}")
    print(f"yearly_336_csv: {yearly_csv}")
    print(f"execution_matrix_csv: {matrix_csv}")
    print(f"validation_md: {note_md}")


if __name__ == "__main__":
    main()
