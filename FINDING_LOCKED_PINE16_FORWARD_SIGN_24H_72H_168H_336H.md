# LOCKED FINDING: Pine16 Forward Sign 24h/168h/336h

Status: ACTIVE FINDING (updated 2026-03-10, supersedes old 24h/72h lock)

Truth stamp: `UNVERIFIED_PYTHON_APPROXIMATION`

## Scope and Method
- Watchlist symbols: `XAUUSD, XAGUSD, LIGHTCMDUSD, BRENTCMDUSD, EURJPY, GBPJPY, USDJPY`
- Entry timeframes validated: `m15`, `m30`
- Horizons validated: `24h`, `168h` (1 week), `336h` (2 weeks)
- Data coverage check: `2022-01-01` to `2026-01-01` complete for watchlist bars
- Study years in configs/reports: `2022, 2023, 2024, 2025`
- London+NY figures here are computed from all-sessions runs using `analysis_session_scope = london_or_newyork` (same method used in prior comparison tables)

## Hard Findings (London+NY focus)

### M30 - London+New York
| Symbol | 24h WR | n | 24h Mean Ret | 168h WR | n | 168h Mean Ret | 336h WR | n | 336h Mean Ret | Session Action |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| XAUUSD | 61.0% | 41 | +0.278% | 56.1% | 41 | +0.613% | 53.7% | 41 | +0.719% | KEEP |
| XAGUSD | 56.5% | 46 | +0.376% | 60.9% | 46 | +0.994% | 67.4% | 46 | +2.061% | KEEP |
| LIGHTCMDUSD | 55.1% | 78 | -0.080% | 56.4% | 78 | +0.629% | 48.7% | 78 | -0.242% | WATCH |
| BRENTCMDUSD | 51.6% | 64 | -0.161% | 53.1% | 64 | +0.118% | 46.9% | 64 | -0.676% | WATCH |
| EURJPY | 48.0% | 25 | -0.012% | 56.0% | 25 | +0.249% | 64.0% | 25 | +0.334% | KEEP |
| GBPJPY | 40.0% | 35 | -0.129% | 51.4% | 35 | -0.213% | 57.1% | 35 | -0.176% | WATCH |
| USDJPY | 35.7% | 28 | -0.126% | 50.0% | 28 | -0.143% | 37.0% | 27 | -0.657% | CUT |

### What changed after London+NY restriction (24h/336h action)
- m15 downgrades:
  - `BRENTCMDUSD: KEEP -> CUT` (336h WR `56.4% -> 48.2%`, mean `+0.229% -> -0.727%`)
  - `EURJPY: KEEP -> CUT` (336h WR `60.8% -> 48.1%`, mean `+0.206% -> -0.002%`)
- m15 upgrade:
  - `XAGUSD: WATCH -> KEEP` (336h WR `51.7% -> 56.3%`, mean `+0.801% -> +1.122%`)
- m30 downgrades:
  - `BRENTCMDUSD: KEEP -> WATCH`
  - `LIGHTCMDUSD: KEEP -> WATCH`
  - `USDJPY: WATCH -> CUT`

## 336h Yearly Consistency (London+NY)

### m15 (336h WR by year)
- BRENTCMDUSD: `2022 54.5% | 2023 38.9% | 2024 50.0% | 2025 47.8%`
- EURJPY: `2022 28.6% | 2023 54.5% | 2024 33.3% | 2025 73.3%`
- GBPJPY: `2022 38.5% | 2023 50.0% | 2024 43.8% | 2025 76.5%`
- LIGHTCMDUSD: `2022 57.1% | 2023 52.4% | 2024 54.5% | 2025 44.4%`
- USDJPY: `2022 25.0% | 2023 53.3% | 2024 71.4% | 2025 64.3%`
- XAGUSD: `2022 50.0% | 2023 40.0% | 2024 72.2% | 2025 72.7%`
- XAUUSD: `2022 57.9% | 2023 30.0% | 2024 63.6% | 2025 63.6%`

### m30 (336h WR by year)
- BRENTCMDUSD: `2022 0.0% | 2023 45.0% | 2024 66.7% | 2025 41.2%`
- EURJPY: `2022 71.4% | 2023 62.5% | 2024 57.1% | 2025 66.7%`
- GBPJPY: `2022 28.6% | 2023 64.3% | 2024 37.5% | 2025 100.0%`
- LIGHTCMDUSD: `2022 53.8% | 2023 45.8% | 2024 55.0% | 2025 42.9%`
- USDJPY: `2022 25.0% | 2023 55.6% | 2024 33.3% | 2025 25.0%`
- XAGUSD: `2022 60.0% | 2023 58.3% | 2024 71.4% | 2025 100.0%`
- XAUUSD: `2022 31.2% | 2023 50.0% | 2024 71.4% | 2025 87.5%`

## Final Ranked Read (London+NY)
- Best 336h cluster:
  - `XAGUSD` (strongest)
  - `XAUUSD` (strong)
  - `USDJPY` selective on `m15`
  - `EURJPY` selective on `m30` (small sample)
- Weak cluster:
  - `BRENTCMDUSD`
  - `LIGHTCMDUSD`
  - `USDJPY` on `m30`

## Execution Matrix (London+NY)
- Tier A (allowed both m15 and m30): `XAGUSD`, `XAUUSD`
- Tier B (one TF allowed): `EURJPY`, `GBPJPY`, `USDJPY`
- Tier C (not allowed): `BRENTCMDUSD`, `LIGHTCMDUSD`

## Top Performance Summary

### 336h (2-week) top performers, London+NY
- m15:
  - `USDJPY` -> WR `56.9%` (n=`51`), mean `+0.279%`
  - `XAGUSD` -> WR `56.3%` (n=`71`), mean `+1.122%`
  - `XAUUSD` -> WR `54.9%` (n=`51`), mean `+0.623%`
  - `GBPJPY` -> WR `53.4%` (n=`58`), mean `-0.106%` (hit-rate positive, returns weak)
- m30:
  - `XAGUSD` -> WR `67.4%` (n=`46`), mean `+2.061%`
  - `EURJPY` -> WR `64.0%` (n=`25`), mean `+0.334%` (sample-size caution)
  - `XAUUSD` -> WR `53.7%` (n=`41`), mean `+0.719%`
  - `GBPJPY` -> WR `57.1%` (n=`35`), mean `-0.176%` (hit-rate positive, returns weak)

### 168h (1-week) highlights, London+NY
- m30 strongest mean profiles: `XAGUSD +0.994%`, `LIGHTCMDUSD +0.629%`, `XAUUSD +0.613%`, `EURJPY +0.249%`
- m15 strongest mean profiles: `XAGUSD +0.136%`, `XAUUSD +0.109%`, `USDJPY +0.173%`
- 168h helps detect transition quality but final edge quality is still dominated by `336h` outcomes.

### 24h highlights, London+NY
- m30 leaders: `XAUUSD 61.0%`, `XAGUSD 56.5%`, `LIGHTCMDUSD 55.1%`
- m15 leaders: `XAUUSD 56.9%`, `GBPJPY 53.4%`, `XAGUSD 52.1%`
- 24h alone is not the strongest deployment horizon in this watchlist.

## Recommendations
- Primary deployment core (London+NY, 336h bias):
  - `XAGUSD` (m15 + m30)
  - `XAUUSD` (m15 + m30)
- Conditional adds:
  - `USDJPY` on `m15` only (keep)
  - `EURJPY` on `m30` only, reduced confidence until larger sample (`n=25`)
- Keep on watch, do not promote yet:
  - `LIGHTCMDUSD` (both TFs)
  - `BRENTCMDUSD` on `m30`
  - `GBPJPY` on `m30`
- Cut under London+NY:
  - `BRENTCMDUSD` on `m15`
  - `EURJPY` on `m15`
  - `USDJPY` on `m30`
- Horizon policy:
  - Treat `336h` as the primary decision horizon.
  - Use `168h` as intermediate confirmation, not as the final ranking criterion.
- Promotion rule for conditional names:
  - Require at least one new cycle with `n >= 40` and positive `336h` mean return before upgrading confidence tier.

## Canonical Current Files
- outputs/reports/pine16_watchlist_london_ny_validation_24h_168h_336h.md
- outputs/reports/pine16_watchlist_london_ny_summary_24h_168h_336h.csv
- outputs/reports/pine16_watchlist_london_ny_action_changes_vs_all_sessions_24h_168h_336h.csv
- outputs/reports/pine16_watchlist_london_ny_336h_yearly_consistency.csv
- outputs/reports/pine16_watchlist_execution_matrix_london_ny.csv
- outputs/reports/watchlist_data_completeness_2022_2026.csv
- outputs/reports_watchlist_m15_allsessions_24h_168h_336h/*
- outputs/reports_watchlist_m30_allsessions_24h_168h_336h/*
- data/derived/pine16_exact/forward_sign_24h_168h_336h_master_watchlist_m15_allsessions.parquet
- data/derived/pine16_exact/forward_sign_24h_168h_336h_master_watchlist_m30_allsessions.parquet

## Operational Note
This file now supersedes the previous `24h/72h` lock values. Keep this as the active engineer reference for watchlist execution decisions.
