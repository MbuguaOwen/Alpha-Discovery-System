# LOCKED FINDING: Pine16 Forward Sign 24h/72h

Status: ACTIVE FINDING (do not replace with older reports)
Generated from:
- outputs/reports/pine16_forward_sign_24h_72h_*.csv
- data/derived/pine16_exact/forward_sign_24h_72h_master.parquet

## Hard Findings (Current Output)
- Truth stamp: `UNVERIFIED_PYTHON_APPROXIMATION`
- 24h overall win rate: `0.666667` (n=42)
- 72h overall win rate: `0.666667` (n=42)
- Strongest symbol: `XAGUSD` at `72h` (win_rate=0.842105, n=19)
- Strongest session: `all_sessions` at `72h` (win_rate=0.666667, n=42)
- London-only vs London+NY at 72h: london_only=0.578947, london_or_newyork=0.666667
- Directional edge over 4 years: present in this run's aggregate, but not deployable-grade proof yet because truth is unverified and sample is small.

## Scope Confirmation
- Timeframe used for entries: `m30`
- Years present in master: `2022,2023,2024,2025`
- Master rows (unique signals): `42`

## Canonical Files To Keep
- outputs/audit_pine16_forward_sign_24h_72h.md
- outputs/reports/FINDING_LOCKED_PINE16_FORWARD_SIGN_24H_72H.md
- outputs/reports/pine16_forward_sign_24h_72h_report.md
- outputs/reports/pine16_forward_sign_24h_72h_report.html
- outputs/reports/pine16_forward_sign_24h_72h_overall.csv
- outputs/reports/pine16_forward_sign_24h_72h_by_symbol.csv
- outputs/reports/pine16_forward_sign_24h_72h_by_session.csv
- outputs/reports/pine16_forward_sign_24h_72h_by_year.csv
- outputs/reports/pine16_forward_sign_24h_72h_by_symbol_year.csv
- outputs/reports/pine16_forward_sign_24h_72h_by_symbol_session.csv
- outputs/reports/pine16_forward_sign_24h_72h_by_year_session.csv
- outputs/reports/pine16_forward_sign_24h_72h_by_symbol_year_session.csv
- outputs/reports/pine16_forward_sign_24h_72h_keep_watch_cut.csv
- data/derived/pine16_exact/forward_sign_24h_72h_master.parquet
- outputs/submission/pine16_forward_sign_LOCKED_FINDING_20260308_105447.zip

## M15 Validation Log (2026-03-09)
- Truth stamp for these m15 runs: `UNVERIFIED_PYTHON_APPROXIMATION`

### Benchmark still on top
- Current top benchmark: `XAUUSD, London+NY config, m15, 24h win_rate=0.631579 (n=38)`

### All-sessions check
- Config: `XAUUSD, all_sessions, m15`
- 24h: `win_rate=0.582192 (n=146)`
- 72h: `win_rate=0.606897 (n=145)`
- Verdict: the benchmark `24h win_rate=0.631579` remains higher.

### Run 1
- Config: `XAUUSD, London+NY, m15`
- 24h: `win_rate=0.631579 (n=38)`
- 72h: `win_rate=0.578947 (n=38)`
- Report: `outputs/reports_m15_xau/pine16_forward_sign_24h_72h_report.md`
- Overall: `outputs/reports_m15_xau/pine16_forward_sign_24h_72h_overall.csv`

### Run 2
- Config: `XAU+XAG+EUR, London, m15`
- 24h: `win_rate=0.495798 (n=119)`
- 72h: `win_rate=0.462185 (n=119)`
- Report: `outputs/reports_m15_xau_xag_eur_london/pine16_forward_sign_24h_72h_report.md`
- Overall: `outputs/reports_m15_xau_xag_eur_london/pine16_forward_sign_24h_72h_overall.csv`

### Sharpe for highest-performance config
- Definition used: `Sharpe = mean_forward_return_pct / std_forward_return_pct` (risk-free assumed `0`).
- Highest-performance config by current benchmark: `XAUUSD London+NY m15 24h`.
- 24h Sharpe (signal-level): `0.136298`
- 24h Sharpe annualized proxy: `0.420099` using `sqrt(signals_per_year)` with `signals_per_year = 38 / 4 = 9.5`.
- 72h Sharpe (same config, reference): `0.100771`
- 72h Sharpe annualized proxy: `0.310598`
