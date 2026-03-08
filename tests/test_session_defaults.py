from dfd05.config import load_config


def test_session_defaults_match_pine_baseline():
    cfg = load_config("configs/dfd05.yaml")
    sg = cfg.strategy.session_gate
    assert sg.enabled is True
    assert sg.tz == "Etc/GMT-3"
    assert sg.ny is True
    assert sg.london is False
    assert sg.tokyo is False
    assert sg.sydney is False
