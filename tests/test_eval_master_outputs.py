from pathlib import Path

import pandas as pd

import scripts.dfd05_eval_master as em


def test_write_master_outputs_schema(tmp_path: Path):
    overall = pd.DataFrame(
        [
            {
                "variant": "baseline",
                "timeframe": "m15",
                "session_mode": "session_on",
                "level": "signal",
                "horizon_h": 24,
                "n_total": 100,
                "n_valid": 98,
                "trunc_rate": 0.02,
                "up_rate": 0.55,
                "ge_10bps_rate": 0.50,
                "ge_25bps_rate": 0.40,
                "ge_50bps_rate": 0.30,
                "ge_100bps_rate": 0.20,
                "worked_top_q80_rate": 0.20,
                "worked_mfeatr_top_q80_rate": 0.20,
                "good_rate": 0.25,
                "mean_ret": 0.001,
                "median_ret": 0.0008,
                "p25_ret": -0.001,
                "p50_ret": 0.0008,
                "p75_ret": 0.002,
                "mean_mfe": 0.005,
                "mean_mae": -0.004,
                "mean_ret_atr": 0.2,
                "mean_mfe_atr": 1.1,
                "mean_mae_atr": -0.9,
                "worst_year_mean_ret": -0.0002,
                "worst_year_good_rate": 0.18,
                "worst_year_up_rate": 0.45,
            }
        ]
    )
    per_sy = pd.DataFrame(
        [
            {
                "variant": "baseline",
                "timeframe": "m15",
                "session_mode": "session_on",
                "level": "signal",
                "horizon_h": 24,
                "symbol": "EURUSD",
                "year": 2024,
                "n_total": 50,
                "n_valid": 49,
                "trunc_rate": 0.02,
                "up_rate": 0.57,
                "ge_10bps_rate": 0.53,
                "ge_25bps_rate": 0.43,
                "ge_50bps_rate": 0.31,
                "ge_100bps_rate": 0.22,
                "worked_top_q80_rate": 0.20,
                "worked_mfeatr_top_q80_rate": 0.20,
                "good_rate": 0.27,
                "mean_ret": 0.0012,
                "median_ret": 0.0009,
                "p25_ret": -0.0011,
                "p50_ret": 0.0009,
                "p75_ret": 0.0021,
                "mean_mfe": 0.0052,
                "mean_mae": -0.0041,
                "mean_ret_atr": 0.22,
                "mean_mfe_atr": 1.15,
                "mean_mae_atr": -0.92,
            }
        ]
    )
    best = pd.DataFrame(
        [
            {
                "variant": "baseline",
                "timeframe": "m15",
                "session_mode": "session_on",
                "level": "signal",
                "selected_horizon_h": 24,
                "worst_year_mean_ret": -0.0002,
                "worst_year_good_rate": 0.18,
                "worst_year_up_rate": 0.45,
                "overall_mean_ret": 0.001,
                "overall_good_rate": 0.25,
                "overall_up_rate": 0.55,
                "selection_rank": 1,
                "selection_primary": "max worst_year_mean_ret",
                "selection_secondary": "max worst_year_good_rate",
                "selection_tertiary": "max overall_mean_ret",
                "selection_rule": "rule",
            }
        ]
    )
    run_logs = pd.DataFrame(
        [
            {
                "variant": "baseline",
                "timeframe": "m15",
                "session_mode": "session_on",
                "seconds": 1.0,
                "events_n": 100,
                "forward_n": 100,
                "labeled_n": 100,
            }
        ]
    )

    outs = em.write_master_outputs(overall, per_sy, best, run_logs, tmp_path)
    assert outs["overall"].exists()
    assert outs["per_symbol_year"].exists()
    assert outs["best_horizon_selection"].exists()
    assert outs["summary"].exists()

    got_overall_cols = pd.read_csv(outs["overall"]).columns.tolist()
    got_per_sy_cols = pd.read_csv(outs["per_symbol_year"]).columns.tolist()
    assert got_overall_cols == em.MASTER_OVERALL_COLUMNS
    assert got_per_sy_cols == em.MASTER_PER_SY_COLUMNS
