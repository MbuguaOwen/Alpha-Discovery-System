from __future__ import annotations

from enum import Enum


class TruthLabel(str, Enum):
    EXACT_PINE_EXPORTED = "EXACT_PINE_EXPORTED"
    VERIFIED_PYTHON_PARITY = "VERIFIED_PYTHON_PARITY"
    UNVERIFIED_PYTHON_APPROXIMATION = "UNVERIFIED_PYTHON_APPROXIMATION"


class TruthMode(str, Enum):
    EXACT_PINE_EXPORTED = "exact_pine_exported"
    VERIFIED_PYTHON_PARITY = "verified_python_parity"
    UNVERIFIED_PYTHON_APPROX = "unverified_python_approx"


VALID_TRUTH_MODES = tuple(m.value for m in TruthMode)


def normalize_truth_mode(raw: str) -> TruthMode:
    token = str(raw or "").strip().lower()
    for mode in TruthMode:
        if token == mode.value:
            return mode
    raise ValueError(f"Unsupported truth mode: {raw}. Expected one of: {VALID_TRUTH_MODES}")

