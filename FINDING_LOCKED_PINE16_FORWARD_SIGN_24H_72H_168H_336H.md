# LOCKED FINDING: Pine16 Forward Sign 24h/168h/336h

Status: ACTIVE FINDING

Updated: March 11, 2026

Truth stamp: `UNVERIFIED_PYTHON_APPROXIMATION`

## Locked Rule

Use this rule all through the current watchlist documentation and decisions:

1. choose only one timeframe per symbol
2. keep it only if it improves the basket
3. optimize for total `actual_return_r`

The earlier duplicate-symbol top-8 basket is not the active decision rule.

## Locked Calculation

Cell-level total:

```python
actual_return_r_total = n_168h * mean_return_168h_r
```

Basket-level total:

```python
total_actual_return_r = retained_signals * retained_weighted_mean_return_r
```

This is why the current run is evaluated in total `R`, not just averages.

## Locked Risk Distance Rule

`risk_distance` should be the initial stop distance of the trade.

In the current run, the rule is:

1. If the trade already has a real `stop_price`, use it:

   ```python
   risk_distance = abs(entry_price - stop_price)
   ```

2. If `stop_price` is missing, reconstruct it from the strategy's initial ATR stop:

   ```python
   risk_distance = slAtrMult * atr_entry
   stop_price = entry_price - risk_distance
   ```

For this Pine16 setup, the default risk config is:

- `atrLen = 14`
- `slAtrMult = 1.0`

So in practice for longs:

- `atr_entry` = ATR(14) at entry
- `stop_price = entry_price - 1.0 * atr_entry`
- `risk_distance = 1.0 * atr_entry`

That is why `R` is comparable across symbols: gold, silver, FX, and oil all get normalized by their own initial stop width.

Code references:

- Risk config defaults: [pine16_config.py](C:/Users/USER/Projects/Alpha%20Discovery%20System/dfd05/pine16_config.py#L61)
- Risk-distance logic in the forward-sign study: [pine16_forward_sign_study.py](C:/Users/USER/Projects/Alpha%20Discovery%20System/dfd05/pine16_forward_sign_study.py#L362)

Short version:

- Decide `risk_distance` from the trade's original stop
- If that is unavailable, use the configured ATR stop model at entry

## Locked R Calculation

`R` is return measured in units of initial risk.

For each trade:

```python
risk_distance = abs(entry_price - stop_price)
actual_return_r = (price_at_horizon - entry_price) / risk_distance
```

For this long-only setup:

- `entry_price` = entry
- `stop_price` = initial stop
- `price_at_horizon` = close price `24h`, `168h`, `336h` later, or whatever horizon is being evaluated

Interpretation:

- `+1R` means price ended one stop-distance above entry
- `-1R` means price ended one stop-distance below entry
- `+2.5R` means profit is 2.5 times the initial risk

Example:

- entry = `100`
- stop = `95`
- risk_distance = `5`
- 7-day price = `112`

Then:

```python
actual_return_r = (112 - 100) / 5
```

So that trade returned `+2.4R`.

In this repo, if `stop_price` is missing, fall back to:

```python
risk_distance = slAtrMult * atr_entry
stop_price = entry_price - risk_distance
```

For shorts, the generic form is:

```python
actual_return_r = side_sign * (price_at_horizon - entry_price) / risk_distance
```

where:

- `side_sign = 1` for longs
- `side_sign = -1` for shorts

## Locked Best Combined Selection

This is the best unique-symbol basket for `London+NY` at `168h`.

| Symbol | Chosen TF | Signals | WR 168h | Mean R | Total actual_return_r |
| --- | --- | ---: | ---: | ---: | ---: |
| USDJPY | m15 | 52 | 57.69% | +3.028R | +157.46R |
| XAGUSD | m30 | 46 | 60.87% | +2.882R | +132.57R |
| XAUUSD | m30 | 41 | 56.10% | +2.446R | +100.29R |
| LIGHTCMDUSD | m30 | 78 | 56.41% | +1.040R | +81.14R |
| EURJPY | m30 | 25 | 56.00% | +1.226R | +30.65R |

## Locked Combined Result

- Selected symbols: `5`
- Total signals: `242`
- Weighted win rate: `57.44%`
- Weighted mean R: `+2.0748R`
- Total actual_return_r: `+502.10R`

That is the best revised combined selection under the no-duplicate-symbol rule.

## Locked Per-Symbol Timeframe Winners

| Symbol | Better TF | Why |
| --- | --- | --- |
| USDJPY | m15 | m15 = +157.46R vs m30 = -4.70R |
| XAGUSD | m30 | m30 = +132.57R vs m15 = +31.62R |
| XAUUSD | m30 | m30 = +100.29R vs m15 = +44.96R |
| LIGHTCMDUSD | m30 | m30 = +81.14R vs m15 = +14.15R |
| EURJPY | m30 | m30 = +30.65R vs m15 = -90.42R |
| BRENTCMDUSD | m30 | less bad than m15, but still negative |
| GBPJPY | m30 | less bad than m15, but still negative |

## Locked Exclusions

These symbols are excluded because even their better timeframe is still negative in total `R`.

| Symbol | Best TF | Total actual_return_r |
| --- | --- | ---: |
| BRENTCMDUSD | m30 | -19.21R |
| GBPJPY | m30 | -65.47R |

They reduce the basket and should stay out of the optimized combined selection.

## Locked Strict Quality Selection

If the hard rule is:

- weighted `168h` win rate must stay `>= 60%`

then the only keep-set is:

- `m30:XAGUSD`

with:

- `46` signals
- `60.87%` WR
- `+2.882R` mean
- `+132.57R` total actual_return_r

## Locked Forced-All-7 Comparison

If one timeframe is forced for all 7 symbols:

- `USDJPY -> m15`
- `XAGUSD -> m30`
- `XAUUSD -> m30`
- `LIGHTCMDUSD -> m30`
- `EURJPY -> m30`
- `BRENTCMDUSD -> m30`
- `GBPJPY -> m30`

Forced-all-7 result:

- Total signals: `341`
- Weighted win rate: `56.01%`
- Weighted mean R: `+1.2241R`
- Total actual_return_r: `+417.42R`

This is worse than the optimized 5-symbol basket.

## Locked Hard Conclusion

The active best combined selection is:

- `USDJPY m15`
- `XAGUSD m30`
- `XAUUSD m30`
- `LIGHTCMDUSD m30`
- `EURJPY m30`

Final locked totals:

- Weighted WR: `57.44%`
- Weighted mean R: `+2.0748R`
- Total actual_return_r: `+502.10R`

Strict-quality alternative:

- `m30:XAGUSD` only

Optimization split:

- purity / stricter edge -> `m30:XAGUSD`
- maximum cumulative actual_return_r under one-timeframe-per-symbol -> the 5-symbol basket above

## Locked Caveat

- `m30` `R` values were reconstructed, not fully rebuilt natively
- the ranking is usable
- exact `m30` `R` magnitudes should still be treated with caution until a native `m30` rebuild is completed

## Canonical Current Files

- `outputs/reports_riskr_20260311/PINE16_WATCHLIST_LONDON_NY_COMPLETE_REPORT_20260311.md`
- `outputs/reports_riskr_20260311/pine16_watchlist_london_ny_summary_24h_168h_336h.csv`
- `outputs/reports_riskr_20260311/pine16_watchlist_london_ny_168h_combined_rank.csv`
- `outputs/reports_riskr_20260311/pine16_watchlist_london_ny_168h_keep_drop_to_60.csv`
- `outputs/reports_riskr_20260311/pine16_watchlist_london_ny_336h_yearly_consistency.csv`
- `outputs/reports_riskr_20260311/pine16_watchlist_execution_matrix_london_ny.csv`
- `outputs/reports/watchlist_data_completeness_2022_2026.csv`

Operational note:

This file supersedes the older `24h/72h` and `336h-bias` lock language. Keep this as the active locked finding for current watchlist decisions.
