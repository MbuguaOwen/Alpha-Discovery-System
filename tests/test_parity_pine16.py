from dfd05.config import load_config, validate_pine16_strategy_parity


def test_pine16_parity_passes_for_preset_config():
    cfg = load_config("configs/dfd05_pine16_baseline.yaml")
    diffs = validate_pine16_strategy_parity(cfg)
    assert diffs == []


def test_pine16_parity_reports_mismatch():
    cfg = load_config("configs/dfd05_pine16_baseline.yaml")
    cfg.strategy.don_len = 20
    cfg.strategy.toggles.enable_daily_adx_gate = True
    diffs = validate_pine16_strategy_parity(cfg)
    paths = {d[0] for d in diffs}
    assert "strategy.don_len" in paths
    assert "strategy.toggles.enable_daily_adx_gate" in paths


def test_prod_bos_vol_config_loads_with_expected_strategy_settings():
    cfg = load_config("configs/dfd05_pine16_prod_bos_vol.yaml")
    s = cfg.strategy
    t = s.toggles

    assert s.mode == "CONFIRM"
    assert s.trade_mode == "GATED"
    assert bool(s.use_bos_confirm) is True
    assert float(s.bos_atr_buffer) == 0.10
    assert int(s.max_wait_bars) == 15
    assert bool(s.one_trade_at_a_time) is True
    assert int(s.normalized_cooldown_bars()) == 0
    assert s.normalized_gate_vol_entry_at() == "trigger"
    assert bool(t.enable_vol_ratio_entry_gate) is True
    assert float(t.entry_vol_ratio_min) == 1.0
