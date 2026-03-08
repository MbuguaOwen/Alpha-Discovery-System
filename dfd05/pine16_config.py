from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .config import (
    DataConfig,
    RiskConfig,
    RunConfig,
    SessionGateConfig,
    StrategyConfig,
    ToggleConfig,
)


@dataclass
class Pine16MetaConfig:
    source_truth: str = "exact_pine_definition"
    execution_truth: str = "exact_pine_or_verified_parity_only"
    config_pack: str = "pine16_exact_baseline"
    tv_strategy_name: str = (
        "16. Divergence Feature Discovery System - LONG ONLY (Baseline + Toggles) [v6 FIXED + Session Gate]"
    )


@dataclass
class Pine16DataConfig:
    parquet_roots: List[str] = field(
        default_factory=lambda: ["data/derived/dukascopy", "data/dukascopy"]
    )
    csv_inputs: Dict[str, str] = field(default_factory=dict)
    years: List[int] = field(default_factory=lambda: [2022, 2023, 2024, 2025, 2026])


@dataclass
class Pine16CoreConfig:
    donLen: int = 120
    pivotLen: int = 5
    oscLen: int = 14
    extBandPct: float = 0.15


@dataclass
class Pine16SafetyConfig:
    useConfirmedOnly: bool = True
    warmupBars: int = 500
    cooldownBars: int = 0


@dataclass
class Pine16TradingConfig:
    mode: str = "ANALYZE"  # ANALYZE | TRADE
    tradeMode: str = "BASELINE_ALL"  # BASELINE_ALL | GATED
    entryMode: str = "Raw"  # Raw | Confirm(BOS)
    oneTradeAtATime: bool = True


@dataclass
class Pine16RiskConfig:
    atrLen: int = 14
    slAtrMult: float = 1.0
    rrMult: float = 3.0


@dataclass
class Pine16FeatureConfig:
    useMinOscStrength: bool = False
    minOscChangePct: float = 15.0
    useMinPivotGap: bool = False
    minPivotGapBars: int = 30
    classicOnly: bool = False

    useDailyEma200Gate: bool = False
    useDailyEmaSlopeUp: bool = False
    useDailyAdxGate: bool = False
    dailyDmiLen: int = 14
    dailyAdxMin: float = 18.0
    useDailyDiGate: bool = False

    useVolPivotGate: bool = False
    minVolRatioPivot: float = 1.0
    useVolEntryGate: bool = False
    minVolRatioEntry: float = 1.0

    useAtrRatioGate: bool = False
    atrRatioMax: float = 1.2

    useVolBehaviorGate: bool = False
    volLookback: int = 40
    minUpDownVolRatio: float = 1.25
    requireContraction: bool = True
    maxPullbackVsImpulse: float = 0.85

    useBOSConfirm: bool = False
    bosAtrBuffer: float = 0.10
    maxWaitBars: int = 30

    useCvdGate: bool = False
    cvdLenMin: int = 60
    cvdThreshold: float = 0.0

    useVolSpikeGate: bool = False
    volSmaLen: int = 50
    volMult: float = 1.5
    requireBullCandle: bool = True

    useWv70Gate: bool = False
    w_analyzeBars: int = 70
    w_rowCount: int = 50
    w_swingLen: int = 10
    w_volumeWeight: str = "Recent"
    w_bullThreshPct: float = 70.0


@dataclass
class Pine16SessionConfig:
    useSessionGate: bool = True
    useNY: bool = True
    useLondon: bool = False
    useTokyo: bool = False
    useSydney: bool = False
    tz: str = "Etc/GMT-3"


@dataclass
class Pine16ExactConfig:
    symbols: List[str] = field(default_factory=lambda: ["XAUUSD", "XAGUSD", "EURUSD", "GBPUSD", "USDCHF"])
    timeframe: str = "m15"
    metadata: Pine16MetaConfig = field(default_factory=Pine16MetaConfig)
    data: Pine16DataConfig = field(default_factory=Pine16DataConfig)
    core: Pine16CoreConfig = field(default_factory=Pine16CoreConfig)
    safety: Pine16SafetyConfig = field(default_factory=Pine16SafetyConfig)
    trading: Pine16TradingConfig = field(default_factory=Pine16TradingConfig)
    risk: Pine16RiskConfig = field(default_factory=Pine16RiskConfig)
    features: Pine16FeatureConfig = field(default_factory=Pine16FeatureConfig)
    session: Pine16SessionConfig = field(default_factory=Pine16SessionConfig)


def _load_yaml(path: Path) -> Dict[str, Any]:
    # Reuse legacy loader because it already supports both PyYAML and a minimal fallback parser.
    from .config import _load_yaml as _legacy_yaml_loader  # type: ignore

    obj = _legacy_yaml_loader(path) or {}
    if not isinstance(obj, dict):
        raise ValueError("Top-level config must be a mapping")
    return obj


def _merge(dc_cls: Any, payload: Dict[str, Any] | None) -> Any:
    payload = dict(payload or {})
    values: Dict[str, Any] = {}
    for name in dc_cls.__dataclass_fields__.keys():
        if name in payload:
            values[name] = payload[name]
    return dc_cls(**values)


def load_pine16_exact_config(path: str | Path) -> Pine16ExactConfig:
    raw = _load_yaml(Path(path))
    strategy = raw.get("strategy") or {}
    return Pine16ExactConfig(
        symbols=list(raw.get("symbols") or Pine16ExactConfig().symbols),
        timeframe=str(raw.get("timeframe") or Pine16ExactConfig().timeframe).lower(),
        metadata=_merge(Pine16MetaConfig, raw.get("metadata")),
        data=_merge(Pine16DataConfig, raw.get("data")),
        core=_merge(Pine16CoreConfig, strategy.get("core")),
        safety=_merge(Pine16SafetyConfig, strategy.get("safety")),
        trading=_merge(Pine16TradingConfig, strategy.get("trading")),
        risk=_merge(Pine16RiskConfig, strategy.get("risk")),
        features=_merge(Pine16FeatureConfig, strategy.get("features")),
        session=_merge(Pine16SessionConfig, strategy.get("session")),
    )


def canonical_parameter_map(cfg: Pine16ExactConfig) -> Dict[str, Any]:
    return {
        "core": cfg.core.__dict__.copy(),
        "safety": cfg.safety.__dict__.copy(),
        "trading": cfg.trading.__dict__.copy(),
        "risk": cfg.risk.__dict__.copy(),
        "features": cfg.features.__dict__.copy(),
        "session": cfg.session.__dict__.copy(),
    }


def to_legacy_run_config(cfg: Pine16ExactConfig) -> RunConfig:
    entry_mode = str(cfg.trading.entryMode).strip().lower()
    strategy_mode = "CONFIRM" if entry_mode == "confirm(bos)" else "RAW"
    pullback_max = (
        float(cfg.features.maxPullbackVsImpulse)
        if bool(cfg.features.requireContraction)
        else 1_000_000_000.0
    )

    toggles = ToggleConfig(
        enable_divergence_strength=bool(cfg.features.useMinOscStrength),
        min_osc_change_pct=float(cfg.features.minOscChangePct),
        enable_min_pivot_gap=bool(cfg.features.useMinPivotGap),
        min_pivot_gap_bars=int(cfg.features.minPivotGapBars),
        enable_classic_only=bool(cfg.features.classicOnly),
        enable_daily_ema_gate=bool(cfg.features.useDailyEma200Gate),
        daily_ema_len=200,
        daily_ema_slope_min=0.0,
        enable_daily_adx_gate=bool(cfg.features.useDailyAdxGate),
        enable_daily_di_gate=bool(cfg.features.useDailyDiGate),
        daily_adx_len=int(cfg.features.dailyDmiLen),
        daily_adx_min=float(cfg.features.dailyAdxMin),
        enable_vol_ratio_pivot_gate=bool(cfg.features.useVolPivotGate),
        enable_vol_ratio_entry_gate=bool(cfg.features.useVolEntryGate),
        vol_ratio_len=20,
        pivot_vol_ratio_min=float(cfg.features.minVolRatioPivot),
        entry_vol_ratio_min=float(cfg.features.minVolRatioEntry),
        enable_atr_ratio_cap=bool(cfg.features.useAtrRatioGate),
        atr_ratio_fast_len=14,
        atr_ratio_slow_len=50,
        atr_ratio_cap=float(cfg.features.atrRatioMax),
        enable_volume_behavior_gate=bool(cfg.features.useVolBehaviorGate),
        volume_behavior_len=int(cfg.features.volLookback),
        up_down_ratio_min=float(cfg.features.minUpDownVolRatio),
        pullback_contraction_max=float(pullback_max),
        enable_session_gate=False,
        session_start_hour=0,
        session_end_hour=24,
        session_tz_offset_hours=0,
        enable_cvd_proxy_gate=bool(cfg.features.useCvdGate),
        cvd_len=int(cfg.features.cvdLenMin),
        cvd_min=float(cfg.features.cvdThreshold),
        cvd_norm_len=5000,
        enable_cvd_z_gate=False,
        cvd_z_min=0.0,
        enable_cvd_pct_gate=False,
        cvd_pct_min=50.0,
        enable_vol_spike_gate=bool(cfg.features.useVolSpikeGate),
        vol_mult=float(cfg.features.volMult),
        vol_sma_len=int(cfg.features.volSmaLen),
        enable_wv70_gate=bool(cfg.features.useWv70Gate),
        wv70_len=int(cfg.features.w_analyzeBars),
        wv70_min=float(cfg.features.w_bullThreshPct),
    )

    strategy = StrategyConfig(
        don_len=int(cfg.core.donLen),
        pivot_len=int(cfg.core.pivotLen),
        osc_len=int(cfg.core.oscLen),
        ext_band_pct=float(cfg.core.extBandPct),
        warmup_bars=int(cfg.safety.warmupBars),
        mode=strategy_mode,
        use_bos_confirm=bool(cfg.features.useBOSConfirm),
        bos_atr_buffer=float(cfg.features.bosAtrBuffer),
        max_wait_bars=int(cfg.features.maxWaitBars),
        trade_mode=str(cfg.trading.tradeMode).upper(),
        one_trade_at_a_time=bool(cfg.trading.oneTradeAtATime),
        cooldown_bars=int(cfg.safety.cooldownBars),
        gate_vol_entry_at="trigger" if strategy_mode == "CONFIRM" else "signal",
        session_gate=SessionGateConfig(
            enabled=bool(cfg.session.useSessionGate),
            tz=str(cfg.session.tz),
            ny=bool(cfg.session.useNY),
            london=bool(cfg.session.useLondon),
            tokyo=bool(cfg.session.useTokyo),
            sydney=bool(cfg.session.useSydney),
        ),
        session_gate_source="new",
        toggles=toggles,
    )

    return RunConfig(
        symbols=list(cfg.symbols),
        timeframe=str(cfg.timeframe).lower(),
        data=DataConfig(
            parquet_roots=list(cfg.data.parquet_roots),
            csv_inputs=dict(cfg.data.csv_inputs),
            years=list(cfg.data.years),
        ),
        strategy=strategy,
        risk=RiskConfig(
            atr_len=int(cfg.risk.atrLen),
            sl_atr_mult=float(cfg.risk.slAtrMult),
            rr_mult=float(cfg.risk.rrMult),
        ),
    )


