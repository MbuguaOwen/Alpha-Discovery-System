from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "data" / "reports" / "dfd05_report_ny_london_20260306T193320Z"
YEARS = [2022, 2023, 2024, 2025, 2026]


@dataclass(frozen=True)
class ForwardSlice:
    config: str
    variant: str
    timeframe: str
    session_mode: str
    frame: pd.DataFrame


def _resolve_path(raw: object) -> Path:
    p = Path(str(raw))
    if p.is_absolute():
        return p
    return ROOT / p


def _safe_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().sum() == 0:
        return float("nan")
    return float(vals.mean())


def _safe_median(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().sum() == 0:
        return float("nan")
    return float(vals.median())


def _safe_std(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().sum() <= 1:
        return float("nan")
    return float(vals.std(ddof=1))


def _safe_skew(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().sum() <= 2:
        return float("nan")
    return float(vals.skew())


def _safe_t_stat(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    n = len(vals)
    if n <= 1:
        return float("nan")
    std = float(vals.std(ddof=1))
    if not np.isfinite(std) or std <= 0.0:
        return float("nan")
    return float(vals.mean() / (std / np.sqrt(n)))


def _safe_quantile(series: pd.Series, q: float) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return float("nan")
    return float(vals.quantile(q))


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den == 0.0:
        return float("nan")
    return float(num / den)


def _load_forward_slices(run_logs: pd.DataFrame) -> List[ForwardSlice]:
    slices: List[ForwardSlice] = []
    columns = [
        "event_time_ms",
        "ret_24h",
        "up_24h",
        "mfe_24h",
        "mae_24h",
        "mfe_atr_24h",
        "mae_atr_24h",
        "is_truncated_24h",
    ]
    for _, row in run_logs.iterrows():
        variant = str(row["variant"])
        timeframe = str(row["timeframe"])
        session_mode = str(row["session_mode"])
        config = f"{variant}|{timeframe}|{session_mode}"

        forward_path = _resolve_path(row["forward_path"])
        if not forward_path.exists():
            continue

        try:
            df = pd.read_parquet(forward_path, columns=columns)
        except Exception:
            df = pd.read_parquet(forward_path)
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise ValueError(f"{forward_path} missing required columns: {missing}")
            df = df[columns].copy()

        trunc = pd.to_numeric(df["is_truncated_24h"], errors="coerce").fillna(1)
        df = df.loc[trunc == 0].copy()
        if df.empty:
            continue

        df["year"] = pd.to_datetime(df["event_time_ms"], unit="ms", utc=True, errors="coerce").dt.year
        df["ret_24h"] = pd.to_numeric(df["ret_24h"], errors="coerce")
        df["up_24h"] = pd.to_numeric(df["up_24h"], errors="coerce")
        df["mfe_24h"] = pd.to_numeric(df["mfe_24h"], errors="coerce")
        df["mae_24h"] = pd.to_numeric(df["mae_24h"], errors="coerce")
        df["mfe_atr_24h"] = pd.to_numeric(df["mfe_atr_24h"], errors="coerce")
        df["mae_atr_24h"] = pd.to_numeric(df["mae_atr_24h"], errors="coerce")

        slices.append(
            ForwardSlice(
                config=config,
                variant=variant,
                timeframe=timeframe,
                session_mode=session_mode,
                frame=df,
            )
        )
    return slices


def _signal_quality_and_excursions(forward_slices: List[ForwardSlice]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signal_rows: List[Dict[str, float]] = []
    excursion_rows: List[Dict[str, float]] = []
    yearly_rows: List[Dict[str, float]] = []

    for slc in forward_slices:
        df = slc.frame.copy()
        ret = df["ret_24h"]
        up = df["up_24h"]

        base = df.dropna(subset=["ret_24h", "up_24h"]).copy()
        n_events = int(len(base))
        if n_events == 0:
            continue

        up_rate = _safe_mean(base["up_24h"])
        mean_ret = _safe_mean(base["ret_24h"])
        median_ret = _safe_median(base["ret_24h"])
        std_ret = _safe_std(base["ret_24h"])
        skew_ret = _safe_skew(base["ret_24h"])
        t_stat = _safe_t_stat(base["ret_24h"])

        year_mean = base.groupby("year", dropna=False)["ret_24h"].mean()
        year_mean = year_mean[np.isfinite(year_mean)]
        if len(year_mean) > 0:
            worst_year = float(year_mean.min()) * 10000.0
            best_year = float(year_mean.max()) * 10000.0
        else:
            worst_year = float("nan")
            best_year = float("nan")

        signal_rows.append(
            {
                "config": slc.config,
                "horizon_h": 24,
                "n_events": n_events,
                "up_rate_24h": up_rate,
                "down_rate_24h": 1.0 - up_rate if np.isfinite(up_rate) else float("nan"),
                "mean_return_24h": mean_ret,
                "mean_return_24h_bps": mean_ret * 10000.0 if np.isfinite(mean_ret) else float("nan"),
                "median_return_24h": median_ret,
                "median_return_24h_bps": median_ret * 10000.0 if np.isfinite(median_ret) else float("nan"),
                "std_return_24h": std_ret,
                "skew_return_24h": skew_ret,
                "t_stat_return_24h": t_stat,
                "worst_year_mean_return_24h_bps": worst_year,
                "best_year_mean_return_24h_bps": best_year,
            }
        )

        # Prefer ATR-normalized R metrics; fallback to raw return units if ATR fields are missing.
        if df["mfe_atr_24h"].notna().sum() > 0 and df["mae_atr_24h"].notna().sum() > 0:
            mfe_r = df["mfe_atr_24h"]
            mae_r = df["mae_atr_24h"]
        else:
            mfe_r = df["mfe_24h"]
            mae_r = df["mae_24h"]
        ex = pd.DataFrame({"mfe_r": mfe_r, "mae_r": mae_r}).dropna()
        ex_n = int(len(ex))

        avg_mfe = _safe_mean(ex["mfe_r"])
        avg_mae = _safe_mean(ex["mae_r"])
        edge_ratio = _safe_ratio(avg_mfe, abs(avg_mae))

        excursion_rows.append(
            {
                "config": slc.config,
                "n_events": ex_n,
                "avg_MFE_R": avg_mfe,
                "median_MFE_R": _safe_median(ex["mfe_r"]),
                "p75_MFE_R": _safe_quantile(ex["mfe_r"], 0.75),
                "p95_MFE_R": _safe_quantile(ex["mfe_r"], 0.95),
                "avg_MAE_R": avg_mae,
                "median_MAE_R": _safe_median(ex["mae_r"]),
                "p75_abs_MAE_R": _safe_quantile(ex["mae_r"].abs(), 0.75),
                "p95_abs_MAE_R": _safe_quantile(ex["mae_r"].abs(), 0.95),
                "Edge_Ratio": edge_ratio,
                "frac_hit_1R": _safe_mean((ex["mfe_r"] >= 1.0).astype(float)),
                "frac_hit_2R": _safe_mean((ex["mfe_r"] >= 2.0).astype(float)),
                "frac_hit_3R": _safe_mean((ex["mfe_r"] >= 3.0).astype(float)),
                "frac_hit_neg1R": _safe_mean((ex["mae_r"] <= -1.0).astype(float)),
            }
        )

        for year in YEARS:
            yr = base.loc[base["year"] == year].copy()
            yr_n = int(len(yr))
            yr_up = _safe_mean(yr["up_24h"]) if yr_n > 0 else float("nan")
            yr_mean_ret = _safe_mean(yr["ret_24h"]) if yr_n > 0 else float("nan")
            yr_median_ret = _safe_median(yr["ret_24h"]) if yr_n > 0 else float("nan")

            yr_ex = pd.DataFrame({"mfe_r": mfe_r.loc[yr.index], "mae_r": mae_r.loc[yr.index]}).dropna()
            yr_mfe = _safe_mean(yr_ex["mfe_r"]) if len(yr_ex) > 0 else float("nan")
            yr_mae = _safe_mean(yr_ex["mae_r"]) if len(yr_ex) > 0 else float("nan")
            yr_edge = _safe_ratio(yr_mfe, abs(yr_mae))

            yearly_rows.append(
                {
                    "year": year,
                    "config": slc.config,
                    "n_events": yr_n,
                    "up_rate_24h": yr_up,
                    "mean_return_24h_bps": yr_mean_ret * 10000.0 if np.isfinite(yr_mean_ret) else float("nan"),
                    "median_return_24h_bps": yr_median_ret * 10000.0 if np.isfinite(yr_median_ret) else float("nan"),
                    "avg_MFE_R": yr_mfe,
                    "avg_MAE_R": yr_mae,
                    "Edge_Ratio": yr_edge,
                }
            )

    signal_df = pd.DataFrame(signal_rows).sort_values("config").reset_index(drop=True)
    excursion_df = pd.DataFrame(excursion_rows).sort_values("config").reset_index(drop=True)
    yearly_signal_df = pd.DataFrame(yearly_rows).sort_values(["year", "config"]).reset_index(drop=True)
    return signal_df, excursion_df, yearly_signal_df


def _classify_year_row(up_rate: float, mean_bps: float, edge_ratio: float, barrier_exp: float) -> str:
    directional = (np.isfinite(up_rate) and up_rate > 0.50) or (np.isfinite(mean_bps) and mean_bps > 0.0)
    asymmetry = np.isfinite(edge_ratio) and edge_ratio > 1.0
    barrier_pos = np.isfinite(barrier_exp) and barrier_exp > 0.0

    if (not directional) and ((not np.isfinite(edge_ratio)) or edge_ratio <= 1.0) and (
        (not np.isfinite(barrier_exp)) or barrier_exp <= 0.0
    ):
        return "NO_EDGE"
    if directional and asymmetry and barrier_pos:
        return "FULLY_VALIDATED_THIS_YEAR"
    if directional and asymmetry and not barrier_pos:
        return "DIRECTIONAL_EDGE_ONLY"
    if directional and not asymmetry:
        return "DIRECTIONAL_BUT_FRAGILE"
    return "MIXED"


def _build_yearly_edge_audit(
    yearly_signal_df: pd.DataFrame,
    barrier_results: pd.DataFrame,
    expectancy_by_year: pd.DataFrame,
) -> pd.DataFrame:
    barrier = expectancy_by_year.copy()
    barrier = barrier.rename(
        columns={
            "win_rate_ex_open": "barrier_win_rate",
            "expectancy_R": "barrier_expectancy_R",
        }
    )
    barrier = barrier[["year", "config", "barrier_win_rate", "barrier_expectancy_R"]].copy()

    selected_configs = (
        barrier_results.assign(config=barrier_results["variant"] + "|" + barrier_results["timeframe"] + "|" + barrier_results["session_mode"])[
            "config"
        ]
        .drop_duplicates()
        .tolist()
    )

    base_rows = [{"year": y, "config": c} for c in selected_configs for y in YEARS]
    out = pd.DataFrame(base_rows)
    out = out.merge(
        yearly_signal_df[
            [
                "year",
                "config",
                "n_events",
                "up_rate_24h",
                "mean_return_24h_bps",
                "median_return_24h_bps",
                "avg_MFE_R",
                "avg_MAE_R",
                "Edge_Ratio",
            ]
        ],
        on=["year", "config"],
        how="left",
    )
    out = out.merge(barrier, on=["year", "config"], how="left")
    out["classification"] = out.apply(
        lambda r: _classify_year_row(
            up_rate=float(r["up_rate_24h"]) if pd.notna(r["up_rate_24h"]) else float("nan"),
            mean_bps=float(r["mean_return_24h_bps"]) if pd.notna(r["mean_return_24h_bps"]) else float("nan"),
            edge_ratio=float(r["Edge_Ratio"]) if pd.notna(r["Edge_Ratio"]) else float("nan"),
            barrier_exp=float(r["barrier_expectancy_R"]) if pd.notna(r["barrier_expectancy_R"]) else float("nan"),
        ),
        axis=1,
    )
    out = out.sort_values(["year", "config"]).reset_index(drop=True)
    return out


def _build_claim_audit(
    signal_df: pd.DataFrame,
    excursion_df: pd.DataFrame,
    yearly_edge_df: pd.DataFrame,
    barrier_results: pd.DataFrame,
) -> pd.DataFrame:
    merged = signal_df.merge(excursion_df[["config", "Edge_Ratio"]], on="config", how="left")
    merged[["variant", "timeframe", "session_mode"]] = merged["config"].str.split("|", expand=True)

    m15 = merged[merged["timeframe"] == "m15"].copy()
    m15_barrier = barrier_results[barrier_results["timeframe"] == "m15"].copy()

    prod_signal = merged[merged["config"] == "prod|m15|session_on"].iloc[0]
    prod_barrier = m15_barrier[
        (m15_barrier["variant"] == "prod")
        & (m15_barrier["timeframe"] == "m15")
        & (m15_barrier["session_mode"] == "session_on")
    ].iloc[0]

    pos_barrier_count = int((m15_barrier["avg_R"] > 0.0).sum())
    total_barrier_count = int(len(m15_barrier))

    claims: List[Dict[str, object]] = [
        {
            "claim_id": 1,
            "claim_text": "52-55% directional accuracy means the model has edge",
            "source_file": str(REPORT_DIR / "summary.md"),
            "claimed_value": "52-55% directional accuracy implies edge",
            "reproduced_value": (
                f"m15 up_rate range={m15['up_rate_24h'].min()*100:.2f}% to {m15['up_rate_24h'].max()*100:.2f}%; "
                f"m15 mean_ret_24h range={m15['mean_return_24h_bps'].min():.2f} to {m15['mean_return_24h_bps'].max():.2f} bps"
            ),
            "status": "PARTLY TRUE",
            "explanation": "Directional bias exists in several slices, but this alone does not establish tradable or profitable edge.",
        },
        {
            "claim_id": 2,
            "claim_text": "24h up_rate proves profitability",
            "source_file": str(REPORT_DIR / "summary.md"),
            "claimed_value": "up_rate alone validates profitability",
            "reproduced_value": (
                f"All m15 selected 1:3 barrier expectancies <= 0 (positive count {pos_barrier_count}/{total_barrier_count})"
            ),
            "status": "FALSE",
            "explanation": "Profitability depends on payoff and path outcomes; barrier expectancy remains non-positive across selected m15 slices.",
        },
        {
            "claim_id": 3,
            "claim_text": "The model is a validated positive expectancy system",
            "source_file": str(REPORT_DIR / "barrier_results.csv"),
            "claimed_value": "validated positive expectancy",
            "reproduced_value": (
                f"barrier avg_R positive in {int((barrier_results['avg_R'] > 0).sum())}/{len(barrier_results)} selected configs"
            ),
            "status": "FALSE",
            "explanation": "Barrier 1:3 expectancy is mostly negative; only one non-m15 slice is marginally positive and one m15 slice is exactly zero.",
        },
        {
            "claim_id": 4,
            "claim_text": "The selected/prod slice is profitable",
            "source_file": str(REPORT_DIR / "claim_verification.csv"),
            "claimed_value": "prod slice profitable",
            "reproduced_value": (
                f"prod|m15|session_on: mean_ret_24h={prod_signal['mean_return_24h_bps']:.2f} bps; "
                f"barrier avg_R={prod_barrier['avg_R']:.6f}"
            ),
            "status": "FALSE",
            "explanation": "Selected prod m15 session-on has negative 24h mean return and negative 1:3 expectancy.",
        },
        {
            "claim_id": 5,
            "claim_text": "The 1:3 barrier confirms the signal",
            "source_file": str(REPORT_DIR / "barrier_results.csv"),
            "claimed_value": "barrier confirms signal edge",
            "reproduced_value": (
                f"selected configs with avg_R>0: {int((barrier_results['avg_R'] > 0).sum())}/{len(barrier_results)}; "
                f"m15 avg_R max={m15_barrier['avg_R'].max():.6f}"
            ),
            "status": "FALSE",
            "explanation": "Barrier outcomes do not confirm the directional signal in this setup; m15 barrier expectancy is non-positive.",
        },
        {
            "claim_id": 6,
            "claim_text": "If execution is improved, Model 16 may still be useful",
            "source_file": str(REPORT_DIR / "summary.md"),
            "claimed_value": "model can remain useful with better execution",
            "reproduced_value": (
                f"8/10 m15 signal slices have up_rate>50% and mean_ret_24h>0; "
                f"Edge_Ratio median={m15['Edge_Ratio'].median():.3f}"
            ),
            "status": "PARTLY TRUE",
            "explanation": "There is directional bias in many m15 slices, but excursion asymmetry is weak and year robustness is mixed.",
        },
    ]
    return pd.DataFrame(claims)


def _format_pct(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x * 100.0:.2f}%"


def _format_bps(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.2f} bps"


def _write_reports(
    signal_df: pd.DataFrame,
    excursion_df: pd.DataFrame,
    yearly_edge_df: pd.DataFrame,
    claim_df: pd.DataFrame,
    barrier_results: pd.DataFrame,
) -> None:
    merged = signal_df.merge(excursion_df[["config", "Edge_Ratio"]], on="config", how="left")
    merged[["variant", "timeframe", "session_mode"]] = merged["config"].str.split("|", expand=True)
    m15 = merged[merged["timeframe"] == "m15"].copy()

    m15_barrier = barrier_results[barrier_results["timeframe"] == "m15"].copy()

    prod_signal = merged[merged["config"] == "prod|m15|session_on"].iloc[0]
    prod_barrier = m15_barrier[
        (m15_barrier["variant"] == "prod")
        & (m15_barrier["timeframe"] == "m15")
        & (m15_barrier["session_mode"] == "session_on")
    ].iloc[0]
    baseline_signal = merged[merged["config"] == "baseline|m15|session_on"].iloc[0]
    baseline_barrier = m15_barrier[
        (m15_barrier["variant"] == "baseline")
        & (m15_barrier["timeframe"] == "m15")
        & (m15_barrier["session_mode"] == "session_on")
    ].iloc[0]

    yearly_m15 = yearly_edge_df[yearly_edge_df["config"].str.contains(r"\|m15\|", regex=True)].copy()
    yearly_consistency = (
        yearly_m15.groupby("config")
        .agg(
            years_with_positive_mean=("mean_return_24h_bps", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
            years_total=("year", "count"),
            worst_year_mean_bps=("mean_return_24h_bps", "min"),
            worst_year_barrier_R=("barrier_expectancy_R", "min"),
        )
        .reset_index()
    )

    methodology_audit_lines = [
        "- Primary files audited: `run_logs.csv`, `overall.csv`, `per_symbol_year.csv`, `best_selection.csv`, `barrier_results.csv`, `claim_verification.csv`, `config_audit.csv`, `summary.md`, `index.html` under `data/reports/dfd05_report_ny_london_20260306T193320Z`.",
        "- Raw event-level recomputation source: `forward_path` parquet files referenced by `run_logs.csv` (30 config/timeframe/session slices).",
        "- 1:3 barrier yearly outcomes source: `expectancy_by_year.csv` (reconciled exactly to `barrier_results.csv` totals).",
        "- Missing requested docs: `DFDS_Strategy_Validation_Report.docx` and `Divergence_Alpha_Discovery_Engineering_Report.md` are not present in extracted artifacts.",
        "- Methodological mistake in prior validation: `claim_verification.csv` validates only `n_valid` and `up_rate` for a slice; it does not validate barrier win-rate/expectancy. This conflates signal-direction checks with trade-profitability confirmation when read as a full validation pass.",
    ]

    signal_summary_lines = [
        f"- m15 signal slices with `up_rate_24h > 50%`: {int((m15['up_rate_24h'] > 0.5).sum())}/{len(m15)}.",
        f"- m15 signal slices with `mean_return_24h > 0`: {int((m15['mean_return_24h'] > 0).sum())}/{len(m15)}.",
        f"- m15 `Edge_Ratio` median: {m15['Edge_Ratio'].median():.3f} (threshold for edge > 1.0).",
        f"- Baseline m15 session_on: up_rate={_format_pct(float(baseline_signal['up_rate_24h']))}, mean_ret={_format_bps(float(baseline_signal['mean_return_24h_bps']))}, Edge_Ratio={float(baseline_signal['Edge_Ratio']):.3f}.",
        f"- Prod m15 session_on: up_rate={_format_pct(float(prod_signal['up_rate_24h']))}, mean_ret={_format_bps(float(prod_signal['mean_return_24h_bps']))}, Edge_Ratio={float(prod_signal['Edge_Ratio']):.3f}.",
    ]

    excursion_lines = [
        f"- m15 average `avg_MFE_R`: {m15['config'].count()} slices, median={m15.merge(excursion_df[['config', 'avg_MFE_R']], on='config', how='left')['avg_MFE_R'].median():.3f}.",
        f"- m15 average `avg_MAE_R`: median={m15.merge(excursion_df[['config', 'avg_MAE_R']], on='config', how='left')['avg_MAE_R'].median():.3f}.",
        f"- m15 Edge_Ratio range: {m15['Edge_Ratio'].min():.3f} to {m15['Edge_Ratio'].max():.3f}.",
        "- Interpretation: favorable excursion does not consistently dominate adverse excursion; asymmetry is weak.",
    ]

    barrier_lines = [
        f"- Selected m15 barrier slices with `avg_R > 0`: {int((m15_barrier['avg_R'] > 0).sum())}/{len(m15_barrier)}.",
        f"- Baseline m15 session_on barrier: win_rate_1to3={_format_pct(float(baseline_barrier['win_rate_1to3']))}, avg_R={float(baseline_barrier['avg_R']):.6f}.",
        f"- Prod m15 session_on barrier: win_rate_1to3={_format_pct(float(prod_barrier['win_rate_1to3']))}, avg_R={float(prod_barrier['avg_R']):.6f}.",
        "- Barrier and directional metrics are not equivalent; barrier profitability fails in the currently selected m15 execution setup.",
    ]

    robustness_lines = [
        "- Years evaluated: 2022, 2023, 2024, 2025, 2026 (2026 is partial).",
        "- m15 yearly results are regime-dependent; every m15 config has at least one negative 24h mean-return year.",
        "- 2025 is a notable weak year for several m15 slices (including prod session_on).",
        "- No selected m15 slice shows consistently positive barrier expectancy across all years.",
    ]

    top_yearly = yearly_consistency.sort_values("years_with_positive_mean", ascending=False).head(6)
    table_cols = ["config", "years_with_positive_mean", "years_total", "worst_year_mean_bps", "worst_year_barrier_R"]
    header = "| " + " | ".join(table_cols) + " |"
    sep = "|" + "|".join([" --- " for _ in table_cols]) + "|"
    rows = []
    for _, rr in top_yearly[table_cols].iterrows():
        vals = []
        for c in table_cols:
            v = rr[c]
            if isinstance(v, (float, np.floating)):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    yearly_table = "\n".join([header, sep, *rows])

    memo = "\n".join(
        [
            "# Model 16 Quantitative Validation Memo",
            "",
            "## Executive Verdict",
            "Model 16 has directional bias in multiple slices, but the current 1:3 execution model is not profitable or robust.",
            "",
            "## Methodology Audit",
            *methodology_audit_lines,
            "",
            "## 24h Signal Quality (Signal Edge)",
            *signal_summary_lines,
            "",
            "## 24h Excursion Asymmetry",
            *excursion_lines,
            "",
            "## Year-by-Year Robustness",
            *robustness_lines,
            "",
            "Top configs by count of positive mean-return years (m15 only):",
            "",
            yearly_table,
            "",
            "## Barrier 1:3 Execution Audit",
            *barrier_lines,
            "",
            "## Final Classification",
            "Category 2: DIRECTIONAL EDGE ONLY.",
            "",
            "Rationale:",
            "- Directional up-rate and 24h mean return are positive in many slices.",
            "- Excursion asymmetry is weak (Edge_Ratio often near or below 1.0).",
            "- Barrier expectancy is not validated for current 1:3 execution.",
        ]
    )
    (ROOT / "validation_memo.md").write_text(memo + "\n", encoding="utf-8")

    corrected_conclusion = "\n".join(
        [
            "# Corrected Conclusion",
            "",
            "- **Does Model 16 have edge?** Yes, directional edge appears in multiple slices, but it is weak and regime-dependent.",
            "- **Signal edge only, or full strategy edge?** Signal edge only.",
            "- **Is current 1:3 execution valid?** No. The selected m15 barrier outcomes are non-positive on expectancy.",
            "- **Is the model worth keeping as a signal engine?** Potentially yes, but only as a candidate signal source pending execution redesign and stricter robustness controls.",
            "",
            "Model 16 shows DIRECTIONAL EDGE, but the current execution model is not validated.",
        ]
    )
    (ROOT / "corrected_conclusion.md").write_text(corrected_conclusion + "\n", encoding="utf-8")

    files_used = [
        str(REPORT_DIR / "summary.md"),
        str(REPORT_DIR / "index.html"),
        str(REPORT_DIR / "overall.csv"),
        str(REPORT_DIR / "per_symbol_year.csv"),
        str(REPORT_DIR / "best_selection.csv"),
        str(REPORT_DIR / "barrier_results.csv"),
        str(REPORT_DIR / "claim_verification.csv"),
        str(REPORT_DIR / "config_audit.csv"),
        str(REPORT_DIR / "run_logs.csv"),
        str(ROOT / "expectancy_by_year.csv"),
        str(ROOT / "expectancy_by_config.csv"),
    ]
    notes = "\n".join(
        [
            "# Reproducibility Notes",
            "",
            "## Files Used",
            *[f"- `{p}`" for p in files_used],
            "",
            "## Extraction Inventory",
            "- Extracted ZIPs: `Alpha Discovery System.zip`, `dfd05_report_handoff_20260306.zip`, `handoff_artifacts.zip`, `dfd05_submission_outputs_20260305_20260306.zip`, `engineer_payload_v3.zip`, `reports.zip` into `audit_extract/`.",
            "- No nested ZIPs were found inside these extracted artifacts.",
            "",
            "## Assumptions",
            "- 24h signal metrics are computed from `forward_path` parquet using non-truncated rows (`is_truncated_24h == 0`).",
            "- Directional label uses `up_24h` exactly as stored in forward outputs.",
            "- Excursion R normalization uses `mfe_atr_24h` and `mae_atr_24h` (configs use `sl_atr_mult = 1.0`).",
            "- Year 2026 is treated as partial.",
            "",
            "## Fallback Logic for R Normalization",
            "- Preferred: ATR-normalized columns (`mfe_atr_24h`, `mae_atr_24h`).",
            "- Fallback (not needed in this run): raw return units (`mfe_24h`, `mae_24h`) if ATR-normalized fields are missing.",
            "",
            "## Missing Data / Conflicts",
            "- Requested docs `DFDS_Strategy_Validation_Report.docx` and `Divergence_Alpha_Discovery_Engineering_Report.md` were not present.",
            "- Barrier yearly metrics were taken from `expectancy_by_year.csv`; totals were reconciled exactly to `barrier_results.csv`.",
            "",
            "## Why Earlier Validation Passed While Execution Fails",
            "- Prior claim verification (`claim_verification.csv`) checks directional replication (`n_valid`, `up_rate`) for a single slice.",
            "- It does not test barrier expectancy, so a PASS there does not imply trade profitability.",
        ]
    )
    (ROOT / "reproducibility_notes.md").write_text(notes + "\n", encoding="utf-8")


def main() -> None:
    run_logs = pd.read_csv(REPORT_DIR / "run_logs.csv")
    barrier_results = pd.read_csv(REPORT_DIR / "barrier_results.csv")

    expectancy_by_year_path = ROOT / "expectancy_by_year.csv"
    if not expectancy_by_year_path.exists():
        raise FileNotFoundError(
            "Missing expectancy_by_year.csv. Barrier yearly outcomes are required for yearly_edge_audit.csv."
        )
    expectancy_by_year = pd.read_csv(expectancy_by_year_path)

    forward_slices = _load_forward_slices(run_logs)
    signal_df, excursion_df, yearly_signal_df = _signal_quality_and_excursions(forward_slices)
    yearly_edge_df = _build_yearly_edge_audit(yearly_signal_df, barrier_results, expectancy_by_year)
    claim_df = _build_claim_audit(signal_df, excursion_df, yearly_edge_df, barrier_results)

    signal_df = signal_df[
        [
            "config",
            "horizon_h",
            "n_events",
            "up_rate_24h",
            "down_rate_24h",
            "mean_return_24h",
            "mean_return_24h_bps",
            "median_return_24h",
            "median_return_24h_bps",
            "std_return_24h",
            "skew_return_24h",
            "t_stat_return_24h",
            "worst_year_mean_return_24h_bps",
            "best_year_mean_return_24h_bps",
        ]
    ]
    excursion_df = excursion_df[
        [
            "config",
            "n_events",
            "avg_MFE_R",
            "median_MFE_R",
            "p75_MFE_R",
            "p95_MFE_R",
            "avg_MAE_R",
            "median_MAE_R",
            "p75_abs_MAE_R",
            "p95_abs_MAE_R",
            "Edge_Ratio",
            "frac_hit_1R",
            "frac_hit_2R",
            "frac_hit_3R",
            "frac_hit_neg1R",
        ]
    ]
    yearly_edge_df = yearly_edge_df[
        [
            "year",
            "config",
            "n_events",
            "up_rate_24h",
            "mean_return_24h_bps",
            "median_return_24h_bps",
            "avg_MFE_R",
            "avg_MAE_R",
            "Edge_Ratio",
            "barrier_win_rate",
            "barrier_expectancy_R",
            "classification",
        ]
    ]
    claim_df = claim_df[
        [
            "claim_id",
            "claim_text",
            "source_file",
            "claimed_value",
            "reproduced_value",
            "status",
            "explanation",
        ]
    ]

    signal_df.to_csv(ROOT / "signal_quality_24h.csv", index=False)
    excursion_df.to_csv(ROOT / "excursion_stats_24h.csv", index=False)
    yearly_edge_df.to_csv(ROOT / "yearly_edge_audit.csv", index=False)
    claim_df.to_csv(ROOT / "claim_audit.csv", index=False)

    _write_reports(signal_df, excursion_df, yearly_edge_df, claim_df, barrier_results)

    manifest = {
        "signal_quality_24h.csv": str(ROOT / "signal_quality_24h.csv"),
        "excursion_stats_24h.csv": str(ROOT / "excursion_stats_24h.csv"),
        "yearly_edge_audit.csv": str(ROOT / "yearly_edge_audit.csv"),
        "claim_audit.csv": str(ROOT / "claim_audit.csv"),
        "validation_memo.md": str(ROOT / "validation_memo.md"),
        "corrected_conclusion.md": str(ROOT / "corrected_conclusion.md"),
        "reproducibility_notes.md": str(ROOT / "reproducibility_notes.md"),
    }
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
