from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import json


@dataclass
class DataConfig:
    parquet_roots: List[str] = field(
        default_factory=lambda: ["data/derived/dukascopy", "data/dukascopy"]
    )
    csv_inputs: Dict[str, str] = field(default_factory=dict)
    years: List[int] = field(default_factory=lambda: [2022, 2023, 2024, 2025, 2026])


@dataclass
class ToggleConfig:
    enable_divergence_strength: bool = False
    min_osc_change_pct: float = 0.0

    enable_min_pivot_gap: bool = False
    min_pivot_gap_bars: int = 0

    enable_classic_only: bool = False

    enable_daily_ema_gate: bool = False
    daily_ema_len: int = 200
    daily_ema_slope_min: float = 0.0

    enable_daily_adx_gate: bool = False
    enable_daily_di_gate: bool = False
    daily_adx_len: int = 14
    daily_adx_min: float = 20.0

    enable_vol_ratio_pivot_gate: bool = False
    enable_vol_ratio_entry_gate: bool = False
    vol_ratio_len: int = 20
    pivot_vol_ratio_min: float = 0.0
    entry_vol_ratio_min: float = 0.0

    enable_atr_ratio_cap: bool = False
    atr_ratio_fast_len: int = 14
    atr_ratio_slow_len: int = 50
    atr_ratio_cap: float = 2.0

    enable_volume_behavior_gate: bool = False
    volume_behavior_len: int = 20
    up_down_ratio_min: float = 1.0
    pullback_contraction_max: float = 1.0

    enable_session_gate: bool = False
    session_start_hour: int = 0
    session_end_hour: int = 24
    session_tz_offset_hours: int = 0

    enable_cvd_proxy_gate: bool = False
    cvd_len: int = 30
    cvd_min: float = 0.0
    cvd_norm_len: int = 5000
    enable_cvd_z_gate: bool = False
    cvd_z_min: float = 0.0
    enable_cvd_pct_gate: bool = False
    cvd_pct_min: float = 50.0

    enable_vol_spike_gate: bool = False
    vol_mult: float = 1.5
    vol_sma_len: int = 20

    enable_wv70_gate: bool = False
    wv70_len: int = 70
    wv70_min: float = 0.0


@dataclass
class SessionGateConfig:
    enabled: bool = True
    tz: str = "Etc/GMT-3"
    ny: bool = True
    london: bool = False
    tokyo: bool = False
    sydney: bool = False


@dataclass
class StrategyConfig:
    don_len: int = 20
    pivot_len: int = 3
    osc_len: int = 14
    ext_band_pct: float = 0.25
    warmup_bars: int = 0

    mode: str = "RAW"  # RAW or CONFIRM
    use_bos_confirm: bool = True
    bos_atr_buffer: float = 0.0
    max_wait_bars: int = 6

    trade_mode: str = "BASELINE_ALL"  # BASELINE_ALL or GATED
    one_trade_at_a_time: bool = False
    cooldown_bars: int = 0
    gate_vol_entry_at: str = "signal"  # signal or trigger
    session_gate: SessionGateConfig = field(default_factory=SessionGateConfig)
    session_gate_source: str = "new"
    toggles: ToggleConfig = field(default_factory=ToggleConfig)

    def normalized_gate_vol_entry_at(self) -> str:
        token = str(self.gate_vol_entry_at).strip().lower()
        if token == "trigger":
            return "trigger"
        return "signal"

    def normalized_cooldown_bars(self) -> int:
        return max(0, int(self.cooldown_bars))


@dataclass
class RiskConfig:
    atr_len: int = 14
    sl_atr_mult: float = 1.0
    rr_mult: float = 2.0


@dataclass
class AtrTargetConfig:
    mfe_atr_thresholds: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0])
    mae_atr_thresholds: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0])
    pct_targets: List[float] = field(default_factory=lambda: [0.7, 0.8, 0.9])
    quality_mfe_threshold: float = 1.0
    quality_mae_threshold: float = 1.0


@dataclass
class ForwardConfig:
    mode: str = "time_only"  # time_only or barrier
    # Preferred keys (time-only research schema)
    horizons_hours: List[int] = field(default_factory=lambda: [1, 2, 4, 6, 12, 24, 48, 72])
    ret_thresholds_bps: List[int] = field(default_factory=lambda: [0, 10, 25, 50, 100, 200])
    percentile_targets: List[float] = field(default_factory=lambda: [0.6, 0.7, 0.8, 0.9])
    mfe_thresholds_bps: List[int] = field(default_factory=lambda: [25, 50, 100, 200])
    mae_thresholds_bps: List[int] = field(default_factory=lambda: [25, 50, 100, 200])
    mfe_percentile_targets: List[float] = field(default_factory=lambda: [0.8, 0.9])
    atr_targets: AtrTargetConfig = field(default_factory=AtrTargetConfig)
    truncate_policy: str = "nan"  # currently only nan

    # Barrier-compatible keys
    forward_horizons_hours: List[int] = field(default_factory=list)
    emit_barrier: bool = False
    tie_break: str = "sl"  # sl or tp
    emit_resolved: bool = True

    # Backward-compatible aliases
    horizons_hours_legacy: Optional[List[int]] = None
    tie_break_rule: Optional[str] = None  # sl_first or tp_first

    def normalized_mode(self) -> str:
        m = (self.mode or "").strip().lower()
        if m in {"time_only", "barrier"}:
            return m
        return "time_only"

    def normalized_horizons_hours(self) -> List[int]:
        hrs = self.horizons_hours
        if self.horizons_hours_legacy is not None:
            hrs = self.horizons_hours_legacy
        elif self.forward_horizons_hours:
            hrs = self.forward_horizons_hours
        cleaned = []
        for h in hrs:
            hh = int(h)
            if hh > 0:
                cleaned.append(hh)
        if not cleaned:
            cleaned = [24]
        return sorted(set(cleaned))

    def normalized_tie_break(self) -> str:
        alias = (self.tie_break_rule or "").strip().lower()
        if alias in {"sl_first", "sl"}:
            return "sl"
        if alias in {"tp_first", "tp"}:
            return "tp"
        tb = (self.tie_break or "").strip().lower()
        if tb in {"sl", "tp"}:
            return tb
        return "sl"

    def normalized_ret_thresholds_bps(self) -> List[int]:
        out: List[int] = []
        for v in self.ret_thresholds_bps:
            out.append(int(v))
        return sorted(set(out))

    def normalized_percentile_targets(self) -> List[float]:
        out: List[float] = []
        for q in self.percentile_targets:
            qq = float(q)
            if 0.0 < qq <= 1.0:
                out.append(qq)
        return sorted(set(out))

    def normalized_mfe_thresholds_bps(self) -> List[int]:
        out: List[int] = []
        for v in self.mfe_thresholds_bps:
            out.append(int(v))
        return sorted(set(out))

    def normalized_mae_thresholds_bps(self) -> List[int]:
        out: List[int] = []
        for v in self.mae_thresholds_bps:
            out.append(int(v))
        return sorted(set(out))

    def normalized_mfe_percentile_targets(self) -> List[float]:
        out: List[float] = []
        for q in self.mfe_percentile_targets:
            qq = float(q)
            if 0.0 < qq <= 1.0:
                out.append(qq)
        return sorted(set(out))

    def normalized_mfe_atr_thresholds(self) -> List[float]:
        out: List[float] = []
        for v in self.atr_targets.mfe_atr_thresholds:
            vv = float(v)
            if vv > 0.0:
                out.append(vv)
        return sorted(set(out))

    def normalized_mae_atr_thresholds(self) -> List[float]:
        out: List[float] = []
        for v in self.atr_targets.mae_atr_thresholds:
            vv = float(v)
            if vv > 0.0:
                out.append(vv)
        return sorted(set(out))

    def normalized_mfeatr_percentile_targets(self) -> List[float]:
        out: List[float] = []
        for q in self.atr_targets.pct_targets:
            qq = float(q)
            if 0.0 < qq <= 1.0:
                out.append(qq)
        return sorted(set(out))

    def normalized_quality_mfe_threshold(self) -> float:
        v = float(self.atr_targets.quality_mfe_threshold)
        return v if v > 0.0 else 1.0

    def normalized_quality_mae_threshold(self) -> float:
        v = float(self.atr_targets.quality_mae_threshold)
        return v if v > 0.0 else 1.0


@dataclass
class LabelConfig:
    write_labeled: bool = False
    label_horizon_hours: Optional[int] = 24
    label_column: str = "worked_24h"


@dataclass
class OutputConfig:
    output_dir: str = "data/derived"
    run_id: Optional[str] = None


@dataclass
class RunConfig:
    symbols: List[str] = field(default_factory=lambda: ["XAUUSD"])
    timeframe: str = "m15"
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    forward: ForwardConfig = field(default_factory=ForwardConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def toggles_json(self) -> str:
        return json.dumps(asdict(self.strategy.toggles), sort_keys=True, separators=(",", ":"))


def pine16_baseline_strategy_spec() -> dict[str, Any]:
    return {
        "don_len": 120,
        "pivot_len": 5,
        "osc_len": 14,
        "ext_band_pct": 0.15,
        "warmup_bars": 500,
        "mode": "RAW",
        "trade_mode": "BASELINE_ALL",
        "use_bos_confirm": False,
        "bos_atr_buffer": 0.10,
        "max_wait_bars": 30,
        "one_trade_at_a_time": False,
        "cooldown_bars": 0,
        "gate_vol_entry_at": "signal",
        "session_gate": {
            "enabled": True,
            "tz": "Etc/GMT-3",
            "ny": True,
            "london": False,
            "tokyo": False,
            "sydney": False,
        },
    }


def validate_pine16_strategy_parity(config: RunConfig) -> list[tuple[str, Any, Any]]:
    expected = pine16_baseline_strategy_spec()
    got_s = config.strategy
    got_gate = effective_session_gate(config.strategy)
    diffs: list[tuple[str, Any, Any]] = []

    def _check(path: str, actual: Any, exp: Any) -> None:
        if actual != exp:
            diffs.append((path, actual, exp))

    _check("strategy.don_len", int(got_s.don_len), int(expected["don_len"]))
    _check("strategy.pivot_len", int(got_s.pivot_len), int(expected["pivot_len"]))
    _check("strategy.osc_len", int(got_s.osc_len), int(expected["osc_len"]))
    _check("strategy.ext_band_pct", float(got_s.ext_band_pct), float(expected["ext_band_pct"]))
    _check("strategy.warmup_bars", int(got_s.warmup_bars), int(expected["warmup_bars"]))
    _check("strategy.mode", str(got_s.mode), str(expected["mode"]))
    _check("strategy.trade_mode", str(got_s.trade_mode), str(expected["trade_mode"]))
    _check("strategy.use_bos_confirm", bool(got_s.use_bos_confirm), bool(expected["use_bos_confirm"]))
    _check("strategy.bos_atr_buffer", float(got_s.bos_atr_buffer), float(expected["bos_atr_buffer"]))
    _check("strategy.max_wait_bars", int(got_s.max_wait_bars), int(expected["max_wait_bars"]))
    _check(
        "strategy.one_trade_at_a_time",
        bool(got_s.one_trade_at_a_time),
        bool(expected["one_trade_at_a_time"]),
    )
    _check("strategy.cooldown_bars", int(got_s.cooldown_bars), int(expected["cooldown_bars"]))
    _check(
        "strategy.gate_vol_entry_at",
        got_s.normalized_gate_vol_entry_at(),
        str(expected["gate_vol_entry_at"]),
    )
    _check("strategy.session_gate.enabled", bool(got_gate["enabled"]), bool(expected["session_gate"]["enabled"]))
    _check("strategy.session_gate.tz", str(got_gate["tz"]), str(expected["session_gate"]["tz"]))
    _check("strategy.session_gate.ny", bool(got_gate["ny"]), bool(expected["session_gate"]["ny"]))
    _check(
        "strategy.session_gate.london",
        bool(got_gate["london"]),
        bool(expected["session_gate"]["london"]),
    )
    _check("strategy.session_gate.tokyo", bool(got_gate["tokyo"]), bool(expected["session_gate"]["tokyo"]))
    _check(
        "strategy.session_gate.sydney",
        bool(got_gate["sydney"]),
        bool(expected["session_gate"]["sydney"]),
    )

    for name, value in asdict(got_s.toggles).items():
        if isinstance(value, bool) and value:
            diffs.append((f"strategy.toggles.{name}", True, False))

    return diffs


def normalize_session_tz(raw_tz: str | None) -> str:
    token = (raw_tz or "").strip().upper().replace(" ", "")
    mapping = {
        "UTC+3": "Etc/GMT-3",
        "UTC+03:00": "Etc/GMT-3",
        "+03:00": "Etc/GMT-3",
        "GMT+3": "Etc/GMT-3",
        "ETC/GMT-3": "Etc/GMT-3",
    }
    return mapping.get(token, "Etc/GMT-3")


def effective_session_gate(strategy: StrategyConfig) -> Dict[str, Any]:
    source = (strategy.session_gate_source or "new").strip().lower()
    sg = strategy.session_gate
    toggles = strategy.toggles

    if source != "legacy":
        return {
            "mode": "multi",
            "enabled": bool(sg.enabled),
            "tz": normalize_session_tz(sg.tz),
            "ny": bool(sg.ny),
            "london": bool(sg.london),
            "tokyo": bool(sg.tokyo),
            "sydney": bool(sg.sydney),
            "legacy_start_hour": None,
            "legacy_end_hour": None,
            "legacy_tz_offset_hours": None,
            "selected_count": int(bool(sg.ny)) + int(bool(sg.london)) + int(bool(sg.tokyo)) + int(bool(sg.sydney)),
        }

    if bool(toggles.enable_session_gate):
        return {
            "mode": "legacy_hour_range",
            "enabled": True,
            "tz": f"UTC{int(toggles.session_tz_offset_hours):+d}",
            "ny": None,
            "london": None,
            "tokyo": None,
            "sydney": None,
            "legacy_start_hour": int(toggles.session_start_hour),
            "legacy_end_hour": int(toggles.session_end_hour),
            "legacy_tz_offset_hours": int(toggles.session_tz_offset_hours),
            "selected_count": None,
        }

    return {
        "mode": "off",
        "enabled": False,
        "tz": normalize_session_tz(sg.tz),
        "ny": bool(sg.ny),
        "london": bool(sg.london),
        "tokyo": bool(sg.tokyo),
        "sydney": bool(sg.sydney),
        "legacy_start_hour": int(toggles.session_start_hour),
        "legacy_end_hour": int(toggles.session_end_hour),
        "legacy_tz_offset_hours": int(toggles.session_tz_offset_hours),
        "selected_count": int(bool(sg.ny)) + int(bool(sg.london)) + int(bool(sg.tokyo)) + int(bool(sg.sydney)),
    }


def _merge_dataclass(dc_cls: Any, src: Optional[Dict[str, Any]]) -> Any:
    src = src or {}
    values: Dict[str, Any] = {}
    for field_name, field_def in dc_cls.__dataclass_fields__.items():
        if field_name in src:
            value = src[field_name]
            target_type = field_def.type
            if hasattr(target_type, "__dataclass_fields__") and isinstance(value, dict):
                values[field_name] = _merge_dataclass(target_type, value)
            else:
                values[field_name] = value
    return dc_cls(**values)


def _normalize_toggle_aliases(raw_toggles: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    src = dict(raw_toggles or {})
    alias_map = {
        "useCvdZGate": "enable_cvd_z_gate",
        "cvdZMin": "cvd_z_min",
        "useCvdPctGate": "enable_cvd_pct_gate",
        "cvdPctMin": "cvd_pct_min",
        "cvdNormLen": "cvd_norm_len",
        "cvdThreshold": "cvd_min",
    }
    for old, new in alias_map.items():
        if old in src and new not in src:
            src[new] = src[old]
    return src


def _strip_yaml_comment(line: str) -> str:
    in_single = False
    in_double = False
    out = []
    for ch in line:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            break
        out.append(ch)
    return "".join(out).rstrip()


def _parse_scalar(token: str) -> Any:
    token = token.strip()
    if token == "":
        return ""
    if token == "{}":
        return {}
    if token in {"null", "Null", "NULL", "~"}:
        return None
    if token.lower() == "true":
        return True
    if token.lower() == "false":
        return False
    if token.startswith('"') and token.endswith('"'):
        return token[1:-1]
    if token.startswith("'") and token.endswith("'"):
        return token[1:-1]
    if token.startswith("[") and token.endswith("]"):
        body = token[1:-1].strip()
        if body == "":
            return []
        return [_parse_scalar(x.strip()) for x in body.split(",")]
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


def _split_key_value(content: str) -> tuple[str, str]:
    if ":" not in content:
        raise ValueError(f"Invalid YAML line (expected key:value): {content}")
    key, value = content.split(":", 1)
    return key.strip(), value.strip()


def _minimal_yaml_load(text: str) -> Dict[str, Any]:
    rows = []
    for raw in text.splitlines():
        cleaned = _strip_yaml_comment(raw)
        if not cleaned.strip():
            continue
        indent = len(cleaned) - len(cleaned.lstrip(" "))
        rows.append((indent, cleaned.lstrip(" ")))

    if not rows:
        return {}

    def parse_block(i: int, level: int) -> tuple[Any, int]:
        if i >= len(rows):
            return {}, i
        is_list = rows[i][1].startswith("- ")
        if is_list:
            arr = []
            while i < len(rows):
                indent, content = rows[i]
                if indent < level:
                    break
                if indent != level or not content.startswith("- "):
                    break
                item_text = content[2:].strip()
                i += 1
                if item_text == "":
                    child, i = parse_block(i, level + 2)
                    arr.append(child)
                    continue
                if ":" in item_text and not item_text.startswith(("'", '"')):
                    k, v = _split_key_value(item_text)
                    item: Dict[str, Any] = {}
                    if v == "":
                        if i < len(rows) and rows[i][0] > level:
                            child, i = parse_block(i, rows[i][0])
                            item[k] = child
                        else:
                            item[k] = None
                    else:
                        item[k] = _parse_scalar(v)
                    if i < len(rows) and rows[i][0] > level:
                        extra, i = parse_block(i, rows[i][0])
                        if isinstance(extra, dict):
                            item.update(extra)
                    arr.append(item)
                else:
                    arr.append(_parse_scalar(item_text))
            return arr, i

        obj: Dict[str, Any] = {}
        while i < len(rows):
            indent, content = rows[i]
            if indent < level:
                break
            if indent != level or content.startswith("- "):
                break
            k, v = _split_key_value(content)
            i += 1
            if v == "":
                if i < len(rows) and rows[i][0] > level:
                    child, i = parse_block(i, rows[i][0])
                    obj[k] = child
                else:
                    obj[k] = None
            else:
                obj[k] = _parse_scalar(v)
        return obj, i

    parsed, _ = parse_block(0, rows[0][0])
    if not isinstance(parsed, dict):
        raise ValueError("Top-level YAML must be a mapping.")
    return parsed


def _load_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        return _minimal_yaml_load(text)


def load_config(path: str) -> RunConfig:
    cfg_path = Path(path)
    raw = _load_yaml(cfg_path)
    f_raw = raw.get("forward") or {}
    # Accept flat top-level forward keys as fallback.
    for top_key in [
        "mode",
        "horizons_hours",
        "ret_thresholds_bps",
        "percentile_targets",
        "mfe_thresholds_bps",
        "mae_thresholds_bps",
        "mfe_percentile_targets",
        "atr_targets",
        "truncate_policy",
        "forward_horizons_hours",
        "emit_barrier",
        "tie_break",
        "tie_break_rule",
        "emit_resolved",
    ]:
        if top_key in raw and top_key not in f_raw:
            f_raw[top_key] = raw[top_key]
    # Legacy alias: horizons_hours -> horizons_hours_legacy when explicit forward_horizons_hours also present.
    if "horizons_hours" in f_raw and "horizons_hours_legacy" not in f_raw and "forward_horizons_hours" in f_raw:
        f_raw["horizons_hours_legacy"] = f_raw["horizons_hours"]

    s_raw = raw.get("strategy") or {}
    s_defaults = asdict(StrategyConfig())
    s_allowed = set(StrategyConfig.__dataclass_fields__.keys()) - {"toggles"}
    s_values = {k: s_raw[k] for k in s_raw.keys() if k in s_allowed}
    f_defaults = asdict(ForwardConfig())
    f_allowed = set(ForwardConfig.__dataclass_fields__.keys()) - {"atr_targets"}
    f_values = {k: f_raw[k] for k in f_raw.keys() if k in f_allowed}
    has_new_session_gate = isinstance(s_raw.get("session_gate"), dict)
    if has_new_session_gate:
        s_values["session_gate_source"] = "new"
    elif isinstance(s_raw.get("toggles"), dict):
        legacy_keys = {"enable_session_gate", "session_start_hour", "session_end_hour", "session_tz_offset_hours"}
        if any(k in s_raw["toggles"] for k in legacy_keys):
            s_values["session_gate_source"] = "legacy"
    return RunConfig(
        symbols=raw.get("symbols", RunConfig().symbols),
        timeframe=str(raw.get("timeframe", RunConfig().timeframe)).lower(),
        data=_merge_dataclass(DataConfig, raw.get("data")),
        strategy=StrategyConfig(
            **{
                **s_defaults,
                **s_values,
                "session_gate": _merge_dataclass(SessionGateConfig, s_raw.get("session_gate")),
                "toggles": _merge_dataclass(
                    ToggleConfig,
                    _normalize_toggle_aliases(s_raw.get("toggles")),
                ),
            }
        ),
        risk=_merge_dataclass(RiskConfig, raw.get("risk")),
        forward=ForwardConfig(
            **{
                **f_defaults,
                **f_values,
                "atr_targets": _merge_dataclass(AtrTargetConfig, f_raw.get("atr_targets")),
            }
        ),
        labels=_merge_dataclass(LabelConfig, raw.get("labels")),
        output=_merge_dataclass(OutputConfig, raw.get("output")),
    )
