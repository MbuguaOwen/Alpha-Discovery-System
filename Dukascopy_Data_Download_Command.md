# 1) Download/restore more Dukascopy pairs (m1 + derived m15/m30/h1/h4)
python -m scripts.restore_dukascopy_bars `
  --symbols XAUUSD XAGUSD EURUSD GBPUSD USDJPY AUDUSD NZDUSD USDCAD USDCHF EURJPY GBPJPY EURGBP `
  --start-date 2022-01-01 --end-date 2026-01-01 --warmup-days 45 `
  --output-root data/derived/dukascopy `
  --build-timeframes m1 m15 m30 h1 h4 `
  --max-concurrency 20 --retries 3 --timeout-sec 20 `
  --fill-unexpected-gaps --max-fill-minutes 3

# 2) Retry only failed download days from manifest
python -m scripts.restore_dukascopy_bars `
  --symbols XAUUSD XAGUSD EURUSD GBPUSD USDJPY AUDUSD NZDUSD USDCAD USDCHF EURJPY GBPJPY EURGBP `
  --start-date 2022-01-01 --end-date 2026-01-01 --warmup-days 45 `
  --output-root data/derived/dukascopy `
  --build-timeframes m1 m15 m30 h1 h4 `
  --retry-manifest data/derived/dukascopy/restore_logs/download_manifest.csv `
  --max-concurrency 16 --retries 3 --timeout-sec 10 `
  --fill-unexpected-gaps --max-fill-minutes 3

# 3) Run forward-sign study on those pairs (all sessions, m15)
python -m scripts.pine16_forward_sign_study `
  --config configs/pine16_forwardsign_m30_fxmajors_metals_all_sessions.yaml `
  --truth-mode verified_python_parity `
  --timeframe m15 `
  --output-dir outputs/reports_m15_fxmajors_metals_all_sessions `
  --master-path data/derived/pine16_exact/forward_sign_24h_72h_master_m15_fxmajors_metals_all_sessions.parquet `
  --export-html

# 4) Rank top London+NY symbols at 24h (n>=30)
Import-Csv outputs/reports_m15_fxmajors_metals_all_sessions/pine16_forward_sign_24h_72h_by_symbol_session.csv |
Where-Object { $_.analysis_session_scope -eq 'london_or_newyork' -and [int]$_.horizon_h -eq 24 -and [int]$_.n_signals -ge 30 } |
Sort-Object { [double]$_.win_rate } -Descending |
Select-Object -First 10 symbol,n_signals,win_rate,mean_forward_return_pct,median_forward_return_pct

# 5) Watchlist symbol mapping (verified probe on Dukascopy endpoint)
# GOLD -> XAUUSD
# SILVER -> XAGUSD
# GBPJPY -> GBPJPY
# USDJPY -> USDJPY
# EURJPY -> EURJPY
# USOIL (WTI proxy on this feed) -> LIGHTCMDUSD
# Optional Brent -> BRENTCMDUSD
# Not available on this endpoint (404 in probe): Palladium/Platinum aliases (XPDUSD, XPTUSD, PALLCMDUSD, PLATCMDUSD)

# 6) Download watchlist data for 2022-01-01 to 2026-01-01 (with 45-day warmup)
python -m scripts.restore_dukascopy_bars `
  --symbols XAUUSD XAGUSD GBPJPY USDJPY EURJPY LIGHTCMDUSD BRENTCMDUSD `
  --start-date 2022-01-01 --end-date 2026-01-01 --warmup-days 45 `
  --output-root data/derived/dukascopy `
  --build-timeframes m1 m15 m30 h1 h4 `
  --max-concurrency 16 --retries 4 --timeout-sec 30 `
  --progress-every 25 `
  --fill-unexpected-gaps --max-fill-minutes 3

# 7) Retry only failed download days (run this repeatedly until error_days=0)
python -m scripts.restore_dukascopy_bars `
  --symbols XAUUSD XAGUSD GBPJPY USDJPY EURJPY LIGHTCMDUSD BRENTCMDUSD `
  --start-date 2022-01-01 --end-date 2026-01-01 --warmup-days 45 `
  --output-root data/derived/dukascopy `
  --build-timeframes m1 m15 m30 h1 h4 `
  --retry-manifest data/derived/dukascopy/restore_logs/download_manifest.csv `
  --max-concurrency 10 --retries 6 --timeout-sec 40 `
  --progress-every 10 `
  --fill-unexpected-gaps --max-fill-minutes 3

# 8) Check completion quickly (error_days must be 0; not_found_days should be 0)
Import-Csv data/derived/dukascopy/restore_logs/restore_summary.csv |
Sort-Object symbol |
Select-Object symbol,ok_days,not_found_days,error_days,bars_m1,unexpected_gaps_after_fill

# 9) Run watchlist forward-sign study with 24h + 1 week (168h) + 2 weeks (336h)
# Method used for London+NY validation: run all sessions, then read analysis_session_scope=london_or_newyork
python -m scripts.pine16_forward_sign_study `
  --config configs/pine16_forwardsign_m30_watchlist_all_sessions.yaml `
  --truth-mode verified_python_parity `
  --timeframe m15 `
  --horizons-hours 24 168 336 `
  --output-dir outputs/reports_watchlist_m15_allsessions_24h_168h_336h `
  --master-path data/derived/pine16_exact/forward_sign_24h_168h_336h_master_watchlist_m15_allsessions.parquet `
  --export-html

python -m scripts.pine16_forward_sign_study `
  --config configs/pine16_forwardsign_m30_watchlist_all_sessions.yaml `
  --truth-mode verified_python_parity `
  --timeframe m30 `
  --horizons-hours 24 168 336 `
  --output-dir outputs/reports_watchlist_m30_allsessions_24h_168h_336h `
  --master-path data/derived/pine16_exact/forward_sign_24h_168h_336h_master_watchlist_m30_allsessions.parquet `
  --export-html

# 10) Build London+NY validation tables and execution matrix from those runs
python -m scripts.pine16_watchlist_london_ny_validation `
  --m15-dir outputs/reports_watchlist_m15_allsessions_24h_168h_336h `
  --m30-dir outputs/reports_watchlist_m30_allsessions_24h_168h_336h `
  --output-dir outputs/reports
