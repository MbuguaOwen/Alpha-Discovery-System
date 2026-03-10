# Pine16 London+NY Validation (24h/168h/336h)

Truth stamp: `UNVERIFIED_PYTHON_APPROXIMATION`

## Core outputs
- summary_csv: `outputs/reports/pine16_watchlist_london_ny_summary_24h_168h_336h.csv`
- changes_csv: `outputs/reports/pine16_watchlist_london_ny_action_changes_vs_all_sessions_24h_168h_336h.csv`
- yearly_336_csv: `outputs/reports/pine16_watchlist_london_ny_336h_yearly_consistency.csv`
- execution_matrix_csv: `outputs/reports/pine16_watchlist_execution_matrix_london_ny.csv`

## M30 London+New York quick table (24h / 168h / 336h)
symbol,n_24h,win_rate_24h,mean_return_24h_pct,n_168h,win_rate_168h,mean_return_168h_pct,n_336h,win_rate_336h,mean_return_336h_pct,session_action_24h_336h
BRENTCMDUSD,64,51.562,-0.161,64,53.125,0.118,64,46.875,-0.676,WATCH
EURJPY,25,48.0,-0.012,25,56.0,0.249,25,64.0,0.334,KEEP
GBPJPY,35,40.0,-0.129,35,51.429,-0.213,35,57.143,-0.176,WATCH
LIGHTCMDUSD,78,55.128,-0.08,78,56.41,0.629,78,48.718,-0.242,WATCH
USDJPY,28,35.714,-0.126,28,50.0,-0.143,27,37.037,-0.657,CUT
XAGUSD,46,56.522,0.376,46,60.87,0.994,46,67.391,2.061,KEEP
XAUUSD,41,60.976,0.278,41,56.098,0.613,41,53.659,0.719,KEEP

## Downgrades/Upgrades vs all_sessions (24h/336h action)
timeframe,symbol,action_all_sessions_24h_336h,action_london_or_newyork_24h_336h,change_type,all_336h_win_rate,london_ny_336h_win_rate,all_336h_mean_return_pct,london_ny_336h_mean_return_pct
m15,BRENTCMDUSD,KEEP,CUT,DOWNGRADE,56.419,48.235,0.229,-0.727
m15,EURJPY,KEEP,CUT,DOWNGRADE,60.811,48.077,0.206,-0.002
m15,GBPJPY,KEEP,KEEP,UNCHANGED,56.552,53.448,0.048,-0.106
m15,LIGHTCMDUSD,WATCH,WATCH,UNCHANGED,50.495,51.19,-0.252,-0.048
m15,USDJPY,KEEP,KEEP,UNCHANGED,63.014,56.863,0.363,0.279
m15,XAUUSD,KEEP,KEEP,UNCHANGED,58.904,54.902,0.769,0.623
m15,XAGUSD,WATCH,KEEP,UPGRADE,51.685,56.338,0.801,1.122
m30,BRENTCMDUSD,KEEP,WATCH,DOWNGRADE,51.553,46.875,-0.037,-0.676
m30,LIGHTCMDUSD,KEEP,WATCH,DOWNGRADE,48.408,48.718,-0.023,-0.242
m30,USDJPY,WATCH,CUT,DOWNGRADE,53.279,37.037,-0.123,-0.657
m30,EURJPY,KEEP,KEEP,UNCHANGED,63.725,64.0,0.372,0.334
m30,GBPJPY,WATCH,WATCH,UNCHANGED,58.586,57.143,-0.04,-0.176
m30,XAGUSD,KEEP,KEEP,UNCHANGED,58.268,67.391,1.225,2.061
m30,XAUUSD,KEEP,KEEP,UNCHANGED,55.303,53.659,0.784,0.719
