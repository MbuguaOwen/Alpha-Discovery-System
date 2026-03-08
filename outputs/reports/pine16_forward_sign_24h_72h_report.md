# Pine16 Forward Sign Study (24h/72h)

## 1. Executive verdict
- truth_label_block: `UNVERIFIED_PYTHON_APPROXIMATION`
- directional_edge_exists: `True`
- overall_win_rate_24h: `0.666667`
- overall_win_rate_72h: `0.666667`

## 2. Truth source used
- truth_label_block: `UNVERIFIED_PYTHON_APPROXIMATION`
- allowed_truth_labels: `EXACT_PINE_EXPORTED | VERIFIED_PYTHON_PARITY | UNVERIFIED_PYTHON_APPROXIMATION`

## 3. Study definition
- timeframe: `m30`
- horizons: `24h,72h`
- no stop/target economics
- outcome_mode: `strict_zero`
- neutral_band_pct: `0.000000`

## 4. 24h overall results
- truth_label_block: `UNVERIFIED_PYTHON_APPROXIMATION`
| horizon_h | n_signals | win_rate | loss_rate | flat_rate | mean_forward_return_pct | median_forward_return_pct | win_rate_ci95_low | win_rate_ci95_high | win_rate_band |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 24 | 42 | 0.666667 | 0.333333 | 0.000000 | 0.464389 | 0.388487 | 0.515522 | 0.789877 | strong directional edge |

## 5. 72h overall results
- truth_label_block: `UNVERIFIED_PYTHON_APPROXIMATION`
| horizon_h | n_signals | win_rate | loss_rate | flat_rate | mean_forward_return_pct | median_forward_return_pct | win_rate_ci95_low | win_rate_ci95_high | win_rate_band |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 72 | 42 | 0.666667 | 0.333333 | 0.000000 | 1.076293 | 0.511571 | 0.515522 | 0.789877 | strong directional edge |

## 6. By symbol
- truth_label_block: `UNVERIFIED_PYTHON_APPROXIMATION`
| symbol | horizon_h | n_signals | win_rate | mean_forward_return_pct | median_forward_return_pct | win_rate_band |
| --- | --- | --- | --- | --- | --- | --- |
| EURUSD | 24 | 9 | 0.666667 | 0.302867 | 0.159526 | strong directional edge |
| EURUSD | 72 | 9 | 0.666667 | 0.381494 | 0.289751 | strong directional edge |
| XAGUSD | 24 | 19 | 0.631579 | 0.596704 | 0.633278 | strong directional edge |
| XAGUSD | 72 | 19 | 0.842105 | 2.269986 | 2.991719 | strong directional edge |
| XAUUSD | 24 | 14 | 0.714286 | 0.388653 | 0.388487 | strong directional edge |
| XAUUSD | 72 | 14 | 0.428571 | -0.097062 | -0.584252 | no directional edge |

## 7. By session
- truth_label_block: `UNVERIFIED_PYTHON_APPROXIMATION`
| analysis_session_scope | horizon_h | n_signals | win_rate | mean_forward_return_pct | median_forward_return_pct | win_rate_band |
| --- | --- | --- | --- | --- | --- | --- |
| all_sessions | 24 | 42 | 0.666667 | 0.464389 | 0.388487 | strong directional edge |
| all_sessions | 72 | 42 | 0.666667 | 1.076293 | 0.511571 | strong directional edge |
| london_only | 24 | 19 | 0.684211 | 0.349335 | 0.518118 | strong directional edge |
| london_only | 72 | 19 | 0.578947 | 0.752883 | 0.646252 | strong directional edge |
| london_or_newyork | 24 | 42 | 0.666667 | 0.464389 | 0.388487 | strong directional edge |
| london_or_newyork | 72 | 42 | 0.666667 | 1.076293 | 0.511571 | strong directional edge |

## 8. By year
- truth_label_block: `UNVERIFIED_PYTHON_APPROXIMATION`
| year | horizon_h | n_signals | win_rate | mean_forward_return_pct | median_forward_return_pct |
| --- | --- | --- | --- | --- | --- |
| 2022.000000 | 24.000000 | 13.000000 | 0.692308 | 0.672264 | 0.744950 |
| 2022.000000 | 72.000000 | 13.000000 | 0.692308 | 1.343093 | 0.572004 |
| 2023.000000 | 24.000000 | 8.000000 | 0.750000 | 0.774022 | 0.593134 |
| 2023.000000 | 72.000000 | 8.000000 | 0.375000 | 0.522884 | -0.374652 |
| 2024.000000 | 24.000000 | 16.000000 | 0.625000 | 0.268528 | 0.208406 |
| 2024.000000 | 72.000000 | 16.000000 | 0.687500 | 0.596515 | 0.406143 |
| 2025.000000 | 24.000000 | 5.000000 | 0.600000 | 0.055254 | 0.446321 |
| 2025.000000 | 72.000000 | 5.000000 | 1.000000 | 2.803356 | 2.991719 |

## 9. By symbol x year
- truth_label_block: `UNVERIFIED_PYTHON_APPROXIMATION`
| symbol | year | horizon_h | n_signals | win_rate | mean_forward_return_pct |
| --- | --- | --- | --- | --- | --- |
| EURUSD | 2022 | 24 | 1 | 1.000000 | 1.454958 |
| EURUSD | 2022 | 72 | 1 | 1.000000 | 2.740699 |
| EURUSD | 2023 | 24 | 1 | 0.000000 | -0.320855 |
| EURUSD | 2023 | 72 | 1 | 1.000000 | 0.269348 |
| EURUSD | 2024 | 24 | 6 | 0.666667 | 0.176160 |
| EURUSD | 2024 | 72 | 6 | 0.500000 | -0.034226 |
| EURUSD | 2025 | 24 | 1 | 1.000000 | 0.534736 |
| EURUSD | 2025 | 72 | 1 | 1.000000 | 0.628755 |
| XAGUSD | 2022 | 24 | 5 | 0.600000 | 0.785949 |
| XAGUSD | 2022 | 72 | 5 | 1.000000 | 2.981726 |
| XAGUSD | 2023 | 24 | 4 | 0.750000 | 1.129687 |
| XAGUSD | 2023 | 72 | 4 | 0.250000 | 1.282100 |
| XAGUSD | 2024 | 24 | 7 | 0.714286 | 0.513382 |
| XAGUSD | 2024 | 72 | 7 | 1.000000 | 1.592127 |
| XAGUSD | 2025 | 24 | 3 | 0.333333 | -0.234929 |
| XAGUSD | 2025 | 72 | 3 | 1.000000 | 3.982605 |
| XAUUSD | 2022 | 24 | 7 | 0.714286 | 0.479247 |
| XAUUSD | 2022 | 72 | 7 | 0.428571 | -0.027017 |
| XAUUSD | 2023 | 24 | 3 | 1.000000 | 0.664762 |
| XAUUSD | 2023 | 72 | 3 | 0.333333 | -0.404892 |
| XAUUSD | 2024 | 24 | 3 | 0.333333 | -0.118062 |
| XAUUSD | 2024 | 72 | 3 | 0.333333 | -0.465095 |
| XAUUSD | 2025 | 24 | 1 | 1.000000 | 0.446321 |
| XAUUSD | 2025 | 72 | 1 | 1.000000 | 1.440209 |

## 10. By symbol x session
- truth_label_block: `UNVERIFIED_PYTHON_APPROXIMATION`
| symbol | analysis_session_scope | horizon_h | n_signals | win_rate | mean_forward_return_pct | median_forward_return_pct |
| --- | --- | --- | --- | --- | --- | --- |
| EURUSD | all_sessions | 24 | 9 | 0.666667 | 0.302867 | 0.159526 |
| EURUSD | all_sessions | 72 | 9 | 0.666667 | 0.381494 | 0.289751 |
| EURUSD | london_only | 24 | 3 | 0.666667 | 0.295034 | 0.257287 |
| EURUSD | london_only | 72 | 3 | 0.333333 | -0.143820 | -0.431462 |
| EURUSD | london_or_newyork | 24 | 9 | 0.666667 | 0.302867 | 0.159526 |
| EURUSD | london_or_newyork | 72 | 9 | 0.666667 | 0.381494 | 0.289751 |
| XAGUSD | all_sessions | 24 | 19 | 0.631579 | 0.596704 | 0.633278 |
| XAGUSD | all_sessions | 72 | 19 | 0.842105 | 2.269986 | 2.991719 |
| XAGUSD | london_only | 24 | 9 | 0.666667 | 0.379295 | 0.691235 |
| XAGUSD | london_only | 72 | 9 | 0.777778 | 2.057598 | 2.991719 |
| XAGUSD | london_or_newyork | 24 | 19 | 0.631579 | 0.596704 | 0.633278 |
| XAGUSD | london_or_newyork | 72 | 19 | 0.842105 | 2.269986 | 2.991719 |
| XAUUSD | all_sessions | 24 | 14 | 0.714286 | 0.388653 | 0.388487 |
| XAUUSD | all_sessions | 72 | 14 | 0.428571 | -0.097062 | -0.584252 |
| XAUUSD | london_only | 24 | 7 | 0.714286 | 0.334088 | 0.330653 |
| XAUUSD | london_only | 72 | 7 | 0.428571 | -0.540307 | -0.668406 |
| XAUUSD | london_or_newyork | 24 | 14 | 0.714286 | 0.388653 | 0.388487 |
| XAUUSD | london_or_newyork | 72 | 14 | 0.428571 | -0.097062 | -0.584252 |

## 11. 24h vs 72h comparison
- truth_label_block: `UNVERIFIED_PYTHON_APPROXIMATION`
- overall_win_rate_delta_72h_minus_24h: `0.000000`
- directional_improves_72h_vs_24h: `False`

## 12. Keep / watch / cut
- truth_label_block: `UNVERIFIED_PYTHON_APPROXIMATION`
| symbol | session_scope | n_24h | n_72h | win_rate_24h | win_rate_72h | mean_return_24h | mean_return_72h | action | rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EURUSD | all_sessions | 9 | 9 | 0.666667 | 0.666667 | 0.302867 | 0.381494 | WATCH | mixed or borderline directional profile |
| EURUSD | london_only | 3 | 3 | 0.666667 | 0.333333 | 0.295034 | -0.143820 | WATCH | mixed or borderline directional profile |
| EURUSD | london_or_newyork | 9 | 9 | 0.666667 | 0.666667 | 0.302867 | 0.381494 | WATCH | mixed or borderline directional profile |
| XAGUSD | all_sessions | 19 | 19 | 0.631579 | 0.842105 | 0.596704 | 2.269986 | WATCH | mixed or borderline directional profile |
| XAGUSD | london_only | 9 | 9 | 0.666667 | 0.777778 | 0.379295 | 2.057598 | WATCH | mixed or borderline directional profile |
| XAGUSD | london_or_newyork | 19 | 19 | 0.631579 | 0.842105 | 0.596704 | 2.269986 | WATCH | mixed or borderline directional profile |
| XAUUSD | all_sessions | 14 | 14 | 0.714286 | 0.428571 | 0.388653 | -0.097062 | WATCH | mixed or borderline directional profile |
| XAUUSD | london_only | 7 | 7 | 0.714286 | 0.428571 | 0.334088 | -0.540307 | WATCH | mixed or borderline directional profile |
| XAUUSD | london_or_newyork | 14 | 14 | 0.714286 | 0.428571 | 0.388653 | -0.097062 | WATCH | mixed or borderline directional profile |

## 13. Final answer: is there directional edge?
- truth_label_block: `UNVERIFIED_PYTHON_APPROXIMATION`
- directional_edge_exists_over_4y: `True`

## Final Questions
1. At 24h, is price above entry more than 50% of the time? `True`
2. At 72h, is price above entry more than 50% of the time? `True`
3. Which symbol is strongest? `XAGUSD` @ `72h` (win_rate=0.842105, n=19)
4. Which session is strongest? `all_sessions` @ `72h` (win_rate=0.666667, n=42)
5. Is London-only better than London+NY? `False` (72h london=0.578947, london_or_newyork=0.666667)
6. Does XAUUSD have directional edge? `True`
7. Does XAGUSD have directional edge? `True`
8. Does EURUSD deserve inclusion? `True`
9. Is there a real 4-year directional edge anywhere? `True`
