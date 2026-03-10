# Pine16 Forward Sign Study (24h/168h)

## 1. Executive verdict
- truth_label_block: `UNVERIFIED_PYTHON_APPROXIMATION`
- directional_edge_exists: `True`
- overall_win_rate_24h: `0.500000`
- overall_win_rate_168h: `0.531632`

## 2. Truth source used
- truth_label_block: `UNVERIFIED_PYTHON_APPROXIMATION`
- allowed_truth_labels: `EXACT_PINE_EXPORTED | VERIFIED_PYTHON_PARITY | UNVERIFIED_PYTHON_APPROXIMATION`

## 3. Study definition
- timeframe: `m30`
- horizons: `24h,168h`
- no stop/target economics
- outcome_mode: `strict_zero`
- neutral_band_pct: `0.000000`

## 4. 24h overall results
- truth_label_block: `UNVERIFIED_PYTHON_APPROXIMATION`
| horizon_h | n_signals | win_rate | loss_rate | flat_rate | mean_forward_return_pct | median_forward_return_pct | win_rate_ci95_low | win_rate_ci95_high | win_rate_band |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 24 | 904 | 0.500000 | 0.500000 | 0.000000 | 0.055904 | 0.000706 | 0.467475 | 0.532525 | borderline |

## 5. 168h overall results
- truth_label_block: `UNVERIFIED_PYTHON_APPROXIMATION`
| horizon_h | n_signals | win_rate | loss_rate | flat_rate | mean_forward_return_pct | median_forward_return_pct | win_rate_ci95_low | win_rate_ci95_high | win_rate_band |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 168 | 901 | 0.531632 | 0.466149 | 0.002220 | 0.243793 | 0.214681 | 0.498983 | 0.564011 | usable |

## 6. By symbol
- truth_label_block: `UNVERIFIED_PYTHON_APPROXIMATION`
| symbol | horizon_h | n_signals | win_rate | mean_forward_return_pct | median_forward_return_pct | win_rate_band |
| --- | --- | --- | --- | --- | --- | --- |
| BRENTCMDUSD | 24 | 162 | 0.555556 | 0.165957 | 0.198773 | strong directional edge |
| BRENTCMDUSD | 168 | 161 | 0.496894 | 0.251196 | -0.069517 | no directional edge |
| BRENTCMDUSD | 336 | 161 | 0.515528 | -0.037215 | 0.151927 | borderline |
| EURJPY | 24 | 102 | 0.460784 | -0.047268 | -0.103102 | no directional edge |
| EURJPY | 168 | 102 | 0.568627 | 0.211980 | 0.354610 | strong directional edge |
| EURJPY | 336 | 102 | 0.637255 | 0.372017 | 0.625867 | strong directional edge |
| GBPJPY | 24 | 99 | 0.434343 | -0.121967 | -0.123582 | no directional edge |
| GBPJPY | 168 | 99 | 0.494949 | -0.015916 | -0.010017 | no directional edge |
| GBPJPY | 336 | 99 | 0.585859 | -0.040324 | 0.441486 | strong directional edge |
| LIGHTCMDUSD | 24 | 159 | 0.534591 | 0.068774 | 0.154817 | usable |
| LIGHTCMDUSD | 168 | 157 | 0.522293 | 0.275295 | 0.342633 | usable |
| LIGHTCMDUSD | 336 | 157 | 0.484076 | -0.022529 | -0.365683 | no directional edge |
| USDJPY | 24 | 123 | 0.406504 | -0.110258 | -0.117386 | no directional edge |
| USDJPY | 168 | 123 | 0.520325 | -0.082625 | 0.086610 | usable |
| USDJPY | 336 | 122 | 0.532787 | -0.123414 | 0.115802 | usable |
| XAGUSD | 24 | 127 | 0.543307 | 0.255971 | 0.136896 | usable |
| XAGUSD | 168 | 127 | 0.574803 | 0.560879 | 0.754537 | strong directional edge |
| XAGUSD | 336 | 127 | 0.582677 | 1.224796 | 0.849112 | strong directional edge |
| XAUUSD | 24 | 132 | 0.515152 | 0.080811 | 0.086162 | borderline |
| XAUUSD | 168 | 132 | 0.553030 | 0.415747 | 0.314020 | strong directional edge |
| XAUUSD | 336 | 132 | 0.553030 | 0.784302 | 0.293524 | strong directional edge |

## 7. By session
- truth_label_block: `UNVERIFIED_PYTHON_APPROXIMATION`
| analysis_session_scope | horizon_h | n_signals | win_rate | mean_forward_return_pct | median_forward_return_pct | win_rate_band |
| --- | --- | --- | --- | --- | --- | --- |
| all_sessions | 24 | 904 | 0.500000 | 0.055904 | 0.000706 | borderline |
| all_sessions | 168 | 901 | 0.531632 | 0.243793 | 0.214681 | usable |
| all_sessions | 336 | 900 | 0.548889 | 0.298273 | 0.283138 | usable |
| london_only | 24 | 204 | 0.495098 | 0.033859 | -0.036246 | no directional edge |
| london_only | 168 | 204 | 0.544118 | 0.425278 | 0.314020 | usable |
| london_only | 336 | 204 | 0.519608 | 0.070952 | 0.144524 | borderline |
| london_or_newyork | 24 | 317 | 0.514196 | 0.012073 | 0.027114 | borderline |
| london_or_newyork | 168 | 317 | 0.552050 | 0.385754 | 0.368153 | strong directional edge |
| london_or_newyork | 336 | 316 | 0.528481 | 0.147639 | 0.179257 | usable |

## 8. By year
- truth_label_block: `UNVERIFIED_PYTHON_APPROXIMATION`
| year | horizon_h | n_signals | win_rate | mean_forward_return_pct | median_forward_return_pct |
| --- | --- | --- | --- | --- | --- |
| 2022.000000 | 24.000000 | 221.000000 | 0.452489 | -0.055831 | -0.174308 |
| 2022.000000 | 168.000000 | 221.000000 | 0.533937 | 0.561763 | 0.145083 |
| 2022.000000 | 336.000000 | 221.000000 | 0.475113 | 0.047852 | -0.087580 |
| 2023.000000 | 24.000000 | 247.000000 | 0.473684 | -0.056016 | -0.034102 |
| 2023.000000 | 168.000000 | 247.000000 | 0.522267 | 0.185821 | 0.184510 |
| 2023.000000 | 336.000000 | 247.000000 | 0.591093 | 0.570100 | 0.550249 |
| 2024.000000 | 24.000000 | 212.000000 | 0.575472 | 0.157322 | 0.114992 |
| 2024.000000 | 168.000000 | 212.000000 | 0.500000 | 0.061006 | 0.037476 |
| 2024.000000 | 336.000000 | 212.000000 | 0.547170 | -0.008198 | 0.266298 |
| 2025.000000 | 24.000000 | 224.000000 | 0.504464 | 0.193571 | 0.019003 |
| 2025.000000 | 168.000000 | 221.000000 | 0.570136 | 0.165958 | 0.507653 |
| 2025.000000 | 336.000000 | 220.000000 | 0.577273 | 0.539971 | 0.508699 |

## 9. By symbol x year
- truth_label_block: `UNVERIFIED_PYTHON_APPROXIMATION`
| symbol | year | horizon_h | n_signals | win_rate | mean_forward_return_pct |
| --- | --- | --- | --- | --- | --- |
| BRENTCMDUSD | 2022 | 24 | 22 | 0.500000 | -0.011114 |
| BRENTCMDUSD | 2022 | 168 | 22 | 0.500000 | 2.858164 |
| BRENTCMDUSD | 2022 | 336 | 22 | 0.590909 | 1.113237 |
| BRENTCMDUSD | 2023 | 24 | 39 | 0.615385 | 0.073533 |
| BRENTCMDUSD | 2023 | 168 | 39 | 0.564103 | 0.237848 |
| BRENTCMDUSD | 2023 | 336 | 39 | 0.487179 | 0.147963 |
| BRENTCMDUSD | 2024 | 24 | 47 | 0.510638 | 0.185786 |
| BRENTCMDUSD | 2024 | 168 | 47 | 0.489362 | 0.171199 |
| BRENTCMDUSD | 2024 | 336 | 47 | 0.531915 | -0.429557 |
| BRENTCMDUSD | 2025 | 24 | 54 | 0.574074 | 0.287589 |
| BRENTCMDUSD | 2025 | 168 | 53 | 0.452830 | -0.750178 |
| BRENTCMDUSD | 2025 | 336 | 53 | 0.490566 | -0.303098 |
| EURJPY | 2022 | 24 | 34 | 0.500000 | 0.058195 |
| EURJPY | 2022 | 168 | 34 | 0.588235 | 0.236253 |
| EURJPY | 2022 | 336 | 34 | 0.588235 | 0.518609 |
| EURJPY | 2023 | 24 | 23 | 0.434783 | -0.086678 |
| EURJPY | 2023 | 168 | 23 | 0.608696 | 0.416588 |
| EURJPY | 2023 | 336 | 23 | 0.739130 | 0.631004 |
| EURJPY | 2024 | 24 | 22 | 0.454545 | -0.182099 |
| EURJPY | 2024 | 168 | 22 | 0.500000 | -0.053432 |
| EURJPY | 2024 | 336 | 22 | 0.590909 | -0.051361 |
| EURJPY | 2025 | 24 | 23 | 0.434783 | -0.034790 |
| EURJPY | 2025 | 168 | 23 | 0.565217 | 0.225362 |
| EURJPY | 2025 | 336 | 23 | 0.652174 | 0.301296 |
| GBPJPY | 2022 | 24 | 30 | 0.333333 | -0.245356 |
| GBPJPY | 2022 | 168 | 30 | 0.433333 | -0.092956 |
| GBPJPY | 2022 | 336 | 30 | 0.400000 | -0.397672 |
| GBPJPY | 2023 | 24 | 25 | 0.400000 | -0.174074 |
| GBPJPY | 2023 | 168 | 25 | 0.520000 | 0.106932 |
| GBPJPY | 2023 | 336 | 25 | 0.680000 | 0.386574 |
| GBPJPY | 2024 | 24 | 17 | 0.588235 | -0.072522 |
| GBPJPY | 2024 | 168 | 17 | 0.470588 | -0.575811 |
| GBPJPY | 2024 | 336 | 17 | 0.470588 | -1.036385 |
| GBPJPY | 2025 | 24 | 27 | 0.481481 | 0.032244 |
| GBPJPY | 2025 | 168 | 27 | 0.555556 | 0.308463 |
| GBPJPY | 2025 | 336 | 27 | 0.777778 | 0.588604 |
| LIGHTCMDUSD | 2022 | 24 | 24 | 0.458333 | -0.233743 |
| LIGHTCMDUSD | 2022 | 168 | 24 | 0.625000 | 1.960865 |
| LIGHTCMDUSD | 2022 | 336 | 24 | 0.541667 | 0.677380 |
| LIGHTCMDUSD | 2023 | 24 | 40 | 0.575000 | -0.022125 |
| LIGHTCMDUSD | 2023 | 168 | 40 | 0.450000 | -0.463522 |
| LIGHTCMDUSD | 2023 | 336 | 40 | 0.475000 | -0.453298 |
| LIGHTCMDUSD | 2024 | 24 | 47 | 0.553191 | 0.080522 |
| LIGHTCMDUSD | 2024 | 168 | 47 | 0.468085 | 0.240912 |
| LIGHTCMDUSD | 2024 | 336 | 47 | 0.531915 | -0.084099 |
| LIGHTCMDUSD | 2025 | 24 | 48 | 0.520833 | 0.284278 |
| LIGHTCMDUSD | 2025 | 168 | 46 | 0.586957 | 0.073445 |
| LIGHTCMDUSD | 2025 | 336 | 46 | 0.413043 | 0.049792 |
| USDJPY | 2022 | 24 | 26 | 0.384615 | -0.099539 |
| USDJPY | 2022 | 168 | 26 | 0.576923 | -0.459149 |
| USDJPY | 2022 | 336 | 26 | 0.461538 | -0.554378 |
| USDJPY | 2023 | 24 | 43 | 0.348837 | -0.136832 |
| USDJPY | 2023 | 168 | 43 | 0.534884 | 0.151779 |
| USDJPY | 2023 | 336 | 43 | 0.627907 | 0.246917 |
| USDJPY | 2024 | 24 | 19 | 0.578947 | -0.125216 |
| USDJPY | 2024 | 168 | 19 | 0.368421 | -0.649526 |
| USDJPY | 2024 | 336 | 19 | 0.368421 | -0.990927 |
| USDJPY | 2025 | 24 | 35 | 0.400000 | -0.077454 |
| USDJPY | 2025 | 168 | 35 | 0.542857 | 0.216844 |
| USDJPY | 2025 | 336 | 34 | 0.558824 | 0.222572 |
| XAGUSD | 2022 | 24 | 35 | 0.514286 | -0.015192 |
| XAGUSD | 2022 | 168 | 35 | 0.457143 | 0.078090 |
| XAGUSD | 2022 | 336 | 35 | 0.428571 | -0.738808 |
| XAGUSD | 2023 | 24 | 36 | 0.444444 | -0.099222 |
| XAGUSD | 2023 | 168 | 36 | 0.527778 | 0.643307 |
| XAGUSD | 2023 | 336 | 36 | 0.583333 | 1.876741 |
| XAGUSD | 2024 | 24 | 36 | 0.694444 | 0.691627 |
| XAGUSD | 2024 | 168 | 36 | 0.638889 | 0.564436 |
| XAGUSD | 2024 | 336 | 36 | 0.666667 | 1.115287 |
| XAGUSD | 2025 | 24 | 20 | 0.500000 | 0.585671 |
| XAGUSD | 2025 | 168 | 20 | 0.750000 | 1.250986 |
| XAGUSD | 2025 | 336 | 20 | 0.700000 | 3.684719 |
| XAUUSD | 2022 | 24 | 50 | 0.460000 | 0.040350 |
| XAUUSD | 2022 | 168 | 50 | 0.560000 | 0.363401 |
| XAUUSD | 2022 | 336 | 50 | 0.400000 | 0.087929 |
| XAUUSD | 2023 | 24 | 41 | 0.463415 | -0.000430 |
| XAUUSD | 2023 | 168 | 41 | 0.487805 | 0.322495 |
| XAUUSD | 2023 | 336 | 41 | 0.634146 | 1.239472 |
| XAUUSD | 2024 | 24 | 24 | 0.666667 | 0.148138 |
| XAUUSD | 2024 | 168 | 24 | 0.500000 | -0.143764 |
| XAUUSD | 2024 | 336 | 24 | 0.583333 | 0.826235 |
| XAUUSD | 2025 | 24 | 17 | 0.588235 | 0.300697 |
| XAUUSD | 2025 | 168 | 17 | 0.764706 | 1.584504 |
| XAUUSD | 2025 | 336 | 17 | 0.764706 | 1.675498 |

## 10. By symbol x session
- truth_label_block: `UNVERIFIED_PYTHON_APPROXIMATION`
| symbol | analysis_session_scope | horizon_h | n_signals | win_rate | mean_forward_return_pct | median_forward_return_pct |
| --- | --- | --- | --- | --- | --- | --- |
| BRENTCMDUSD | all_sessions | 24 | 162 | 0.555556 | 0.165957 | 0.198773 |
| BRENTCMDUSD | all_sessions | 168 | 161 | 0.496894 | 0.251196 | -0.069517 |
| BRENTCMDUSD | all_sessions | 336 | 161 | 0.515528 | -0.037215 | 0.151927 |
| BRENTCMDUSD | london_only | 24 | 43 | 0.465116 | -0.192120 | -0.275435 |
| BRENTCMDUSD | london_only | 168 | 43 | 0.558140 | 0.107096 | 0.450086 |
| BRENTCMDUSD | london_only | 336 | 43 | 0.488372 | -0.850337 | -0.118606 |
| BRENTCMDUSD | london_or_newyork | 24 | 64 | 0.515625 | -0.160651 | 0.048552 |
| BRENTCMDUSD | london_or_newyork | 168 | 64 | 0.531250 | 0.118144 | 0.320825 |
| BRENTCMDUSD | london_or_newyork | 336 | 64 | 0.468750 | -0.675923 | -0.407849 |
| EURJPY | all_sessions | 24 | 102 | 0.460784 | -0.047268 | -0.103102 |
| EURJPY | all_sessions | 168 | 102 | 0.568627 | 0.211980 | 0.354610 |
| EURJPY | all_sessions | 336 | 102 | 0.637255 | 0.372017 | 0.625867 |
| EURJPY | london_only | 24 | 11 | 0.545455 | -0.022169 | 0.003644 |
| EURJPY | london_only | 168 | 11 | 0.454545 | -0.126247 | -0.502519 |
| EURJPY | london_only | 336 | 11 | 0.545455 | -0.183867 | 0.621955 |
| EURJPY | london_or_newyork | 24 | 25 | 0.480000 | -0.012052 | -0.111625 |
| EURJPY | london_or_newyork | 168 | 25 | 0.560000 | 0.248840 | 0.368153 |
| EURJPY | london_or_newyork | 336 | 25 | 0.640000 | 0.334223 | 0.646071 |
| GBPJPY | all_sessions | 24 | 99 | 0.434343 | -0.121967 | -0.123582 |
| GBPJPY | all_sessions | 168 | 99 | 0.494949 | -0.015916 | -0.010017 |
| GBPJPY | all_sessions | 336 | 99 | 0.585859 | -0.040324 | 0.441486 |
| GBPJPY | london_only | 24 | 22 | 0.363636 | -0.147145 | -0.162829 |
| GBPJPY | london_only | 168 | 22 | 0.409091 | -0.551242 | -0.786387 |
| GBPJPY | london_only | 336 | 22 | 0.454545 | -0.625741 | -0.508332 |
| GBPJPY | london_or_newyork | 24 | 35 | 0.400000 | -0.128666 | -0.140723 |
| GBPJPY | london_or_newyork | 168 | 35 | 0.514286 | -0.212856 | 0.005502 |
| GBPJPY | london_or_newyork | 336 | 35 | 0.571429 | -0.175738 | 0.138170 |
| LIGHTCMDUSD | all_sessions | 24 | 159 | 0.534591 | 0.068774 | 0.154817 |
| LIGHTCMDUSD | all_sessions | 168 | 157 | 0.522293 | 0.275295 | 0.342633 |
| LIGHTCMDUSD | all_sessions | 336 | 157 | 0.484076 | -0.022529 | -0.365683 |
| LIGHTCMDUSD | london_only | 24 | 53 | 0.528302 | 0.078339 | 0.189584 |
| LIGHTCMDUSD | london_only | 168 | 53 | 0.566038 | 0.955445 | 1.078644 |
| LIGHTCMDUSD | london_only | 336 | 53 | 0.509434 | 0.169574 | 0.008518 |
| LIGHTCMDUSD | london_or_newyork | 24 | 78 | 0.551282 | -0.080178 | 0.207744 |
| LIGHTCMDUSD | london_or_newyork | 168 | 78 | 0.564103 | 0.629487 | 0.986611 |
| LIGHTCMDUSD | london_or_newyork | 336 | 78 | 0.487179 | -0.241524 | -0.097854 |
| USDJPY | all_sessions | 24 | 123 | 0.406504 | -0.110258 | -0.117386 |
| USDJPY | all_sessions | 168 | 123 | 0.520325 | -0.082625 | 0.086610 |
| USDJPY | all_sessions | 336 | 122 | 0.532787 | -0.123414 | 0.115802 |
| USDJPY | london_only | 24 | 14 | 0.214286 | -0.080103 | -0.211526 |
| USDJPY | london_only | 168 | 14 | 0.571429 | -0.191879 | 0.145643 |
| USDJPY | london_only | 336 | 14 | 0.357143 | -0.560095 | -0.344176 |
| USDJPY | london_or_newyork | 24 | 28 | 0.357143 | -0.125994 | -0.155877 |
| USDJPY | london_or_newyork | 168 | 28 | 0.500000 | -0.142915 | 0.017761 |
| USDJPY | london_or_newyork | 336 | 27 | 0.370370 | -0.656710 | -0.407592 |
| XAGUSD | all_sessions | 24 | 127 | 0.543307 | 0.255971 | 0.136896 |
| XAGUSD | all_sessions | 168 | 127 | 0.574803 | 0.560879 | 0.754537 |
| XAGUSD | all_sessions | 336 | 127 | 0.582677 | 1.224796 | 0.849112 |
| XAGUSD | london_only | 24 | 32 | 0.562500 | 0.212264 | 0.130530 |
| XAGUSD | london_only | 168 | 32 | 0.593750 | 0.985545 | 0.980508 |
| XAGUSD | london_only | 336 | 32 | 0.718750 | 1.656698 | 1.440566 |
| XAGUSD | london_or_newyork | 24 | 46 | 0.565217 | 0.376307 | 0.229725 |
| XAGUSD | london_or_newyork | 168 | 46 | 0.608696 | 0.994012 | 0.993861 |
| XAGUSD | london_or_newyork | 336 | 46 | 0.673913 | 2.061184 | 1.440566 |
| XAUUSD | all_sessions | 24 | 132 | 0.515152 | 0.080811 | 0.086162 |
| XAUUSD | all_sessions | 168 | 132 | 0.553030 | 0.415747 | 0.314020 |
| XAUUSD | all_sessions | 336 | 132 | 0.553030 | 0.784302 | 0.293524 |
| XAUUSD | london_only | 24 | 29 | 0.620690 | 0.304358 | 0.315839 |
| XAUUSD | london_only | 168 | 29 | 0.551724 | 0.557856 | 0.312373 |
| XAUUSD | london_only | 336 | 29 | 0.482759 | 0.436798 | -0.003782 |
| XAUUSD | london_or_newyork | 24 | 41 | 0.609756 | 0.277682 | 0.315839 |
| XAUUSD | london_or_newyork | 168 | 41 | 0.560976 | 0.612901 | 0.315667 |
| XAUUSD | london_or_newyork | 336 | 41 | 0.536585 | 0.718627 | 0.381621 |

## 11. 24h vs 168h comparison
- truth_label_block: `UNVERIFIED_PYTHON_APPROXIMATION`
- overall_win_rate_delta_168h_minus_24h: `0.031632`
- directional_improves_168h_vs_24h: `True`

## 12. Keep / watch / cut
- truth_label_block: `UNVERIFIED_PYTHON_APPROXIMATION`
| symbol | session_scope | n_24h | n_168h | win_rate_24h | win_rate_168h | mean_return_24h | mean_return_168h | action | rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BRENTCMDUSD | all_sessions | 162 | 161 | 0.555556 | 0.496894 | 0.165957 | 0.251196 | KEEP | win-rate >52% with positive forward returns |
| BRENTCMDUSD | london_only | 43 | 43 | 0.465116 | 0.558140 | -0.192120 | 0.107096 | KEEP | win-rate >52% with positive forward returns |
| BRENTCMDUSD | london_or_newyork | 64 | 64 | 0.515625 | 0.531250 | -0.160651 | 0.118144 | KEEP | win-rate >52% with positive forward returns |
| EURJPY | all_sessions | 102 | 102 | 0.460784 | 0.568627 | -0.047268 | 0.211980 | KEEP | win-rate >52% with positive forward returns |
| EURJPY | london_only | 11 | 11 | 0.545455 | 0.454545 | -0.022169 | -0.126247 | WATCH | mixed or borderline directional profile |
| EURJPY | london_or_newyork | 25 | 25 | 0.480000 | 0.560000 | -0.012052 | 0.248840 | KEEP | win-rate >52% with positive forward returns |
| GBPJPY | all_sessions | 99 | 99 | 0.434343 | 0.494949 | -0.121967 | -0.015916 | CUT | sub-50% directional profile and weak forward returns |
| GBPJPY | london_only | 22 | 22 | 0.363636 | 0.409091 | -0.147145 | -0.551242 | CUT | sub-50% directional profile and weak forward returns |
| GBPJPY | london_or_newyork | 35 | 35 | 0.400000 | 0.514286 | -0.128666 | -0.212856 | WATCH | mixed or borderline directional profile |
| LIGHTCMDUSD | all_sessions | 159 | 157 | 0.534591 | 0.522293 | 0.068774 | 0.275295 | KEEP | win-rate >52% with positive forward returns |
| LIGHTCMDUSD | london_only | 53 | 53 | 0.528302 | 0.566038 | 0.078339 | 0.955445 | KEEP | win-rate >52% with positive forward returns |
| LIGHTCMDUSD | london_or_newyork | 78 | 78 | 0.551282 | 0.564103 | -0.080178 | 0.629487 | KEEP | win-rate >52% with positive forward returns |
| USDJPY | all_sessions | 123 | 123 | 0.406504 | 0.520325 | -0.110258 | -0.082625 | WATCH | mixed or borderline directional profile |
| USDJPY | london_only | 14 | 14 | 0.214286 | 0.571429 | -0.080103 | -0.191879 | WATCH | mixed or borderline directional profile |
| USDJPY | london_or_newyork | 28 | 28 | 0.357143 | 0.500000 | -0.125994 | -0.142915 | WATCH | mixed or borderline directional profile |
| XAGUSD | all_sessions | 127 | 127 | 0.543307 | 0.574803 | 0.255971 | 0.560879 | KEEP | win-rate >52% with positive forward returns |
| XAGUSD | london_only | 32 | 32 | 0.562500 | 0.593750 | 0.212264 | 0.985545 | KEEP | win-rate >52% with positive forward returns |
| XAGUSD | london_or_newyork | 46 | 46 | 0.565217 | 0.608696 | 0.376307 | 0.994012 | KEEP | win-rate >52% with positive forward returns |
| XAUUSD | all_sessions | 132 | 132 | 0.515152 | 0.553030 | 0.080811 | 0.415747 | KEEP | win-rate >52% with positive forward returns |
| XAUUSD | london_only | 29 | 29 | 0.620690 | 0.551724 | 0.304358 | 0.557856 | KEEP | win-rate >52% with positive forward returns |
| XAUUSD | london_or_newyork | 41 | 41 | 0.609756 | 0.560976 | 0.277682 | 0.612901 | KEEP | win-rate >52% with positive forward returns |

## 13. Final answer: is there directional edge?
- truth_label_block: `UNVERIFIED_PYTHON_APPROXIMATION`
- directional_edge_exists_over_4y: `True`

## Final Questions
1. At 24h, is price above entry more than 50% of the time? `False`
2. At 168h, is price above entry more than 50% of the time? `True`
3. Which symbol is strongest? `EURJPY` @ `336h` (win_rate=0.637255, n=102)
4. Which session is strongest? `london_or_newyork` @ `168h` (win_rate=0.552050, n=317)
5. Is London-only better than London+NY? `False` (168h london=0.544118, london_or_newyork=0.552050)
6. Does XAUUSD have directional edge? `True`
7. Does XAGUSD have directional edge? `True`
8. Does EURUSD deserve inclusion? `False`
9. Is there a real 4-year directional edge anywhere? `True`
