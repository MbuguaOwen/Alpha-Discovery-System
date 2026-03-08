from dfd05.config import load_config


def test_forward_atr_targets_loaded_from_yaml():
    cfg = load_config("configs/dfd05_smoke.yaml")
    fwd = cfg.forward
    assert fwd.normalized_mfe_atr_thresholds() == [0.5, 1.0, 1.5, 2.0]
    assert fwd.normalized_mae_atr_thresholds() == [0.5, 1.0, 1.5, 2.0]
    assert fwd.normalized_mfeatr_percentile_targets() == [0.7, 0.8, 0.9]
    assert fwd.normalized_quality_mfe_threshold() == 1.0
    assert fwd.normalized_quality_mae_threshold() == 1.0
