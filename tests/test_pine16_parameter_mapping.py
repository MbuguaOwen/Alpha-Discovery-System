from dfd05.pine16_config import canonical_parameter_map, load_pine16_exact_config, to_legacy_run_config


def test_pine16_exact_parameter_mapping_baseline_defaults():
    cfg = load_pine16_exact_config("configs/pine16_exact_baseline.yaml")
    cmap = canonical_parameter_map(cfg)

    assert cfg.metadata.source_truth == "exact_pine_definition"
    assert cfg.metadata.execution_truth == "exact_pine_or_verified_parity_only"
    assert cfg.risk.rrMult == 3.0
    assert cmap["core"]["donLen"] == 120
    assert cmap["risk"]["rrMult"] == 3.0

    legacy = to_legacy_run_config(cfg)
    assert legacy.risk.rr_mult == 3.0
    assert legacy.strategy.pivot_len == 5
    assert legacy.strategy.session_gate.enabled is True


def test_prod_screenshot_maps_to_confirm_trade_mode():
    cfg = load_pine16_exact_config("configs/pine16_exact_prod_screenshot.yaml")
    legacy = to_legacy_run_config(cfg)

    assert cfg.trading.mode == "TRADE"
    assert cfg.trading.entryMode == "Confirm(BOS)"
    assert cfg.features.useBOSConfirm is True
    assert legacy.strategy.mode == "CONFIRM"
    assert legacy.strategy.trade_mode == "GATED"
    assert legacy.strategy.use_bos_confirm is True
    assert legacy.strategy.one_trade_at_a_time is True

