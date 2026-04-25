"""Centralized paths and constants for the operational-distinctness pipeline.

This is the standalone paper repository. It does not regenerate the
ARSPI-Net features; it consumes the pickles produced by the main
``dissoAdventureExperiments`` repository.

Resolution order for input files
--------------------------------
1. Explicit env vars (highest priority):
       ARSPI_CH5_FILE, ARSPI_CH67_FILE, ARSPI_CLINICAL_FILE
2. ``data/`` inside this repository.
3. The sibling ``dissoAdventureExperiments`` checkout if it lives next
   to this repo (the conventional layout on the author's machine).

Outputs always land inside this repository under
``outputs/``, ``figures/``, and ``tables/``.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Output locations (always inside this repo)
OUT_DIR = ROOT / "outputs" / "operational_distinctness"
FIG_DIR = ROOT / "figures" / "operational_distinctness"
TABLE_DIR = ROOT / "tables" / "operational_distinctness"

for _d in (OUT_DIR, FIG_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _resolve(env_var: str, candidates: list[Path]) -> Path:
    explicit = os.environ.get(env_var)
    if explicit:
        return Path(explicit).expanduser().resolve()
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # default (may not exist; loaders will raise)


_LOCAL_DATA = ROOT / "data"
_SIBLING_ABLATION = ROOT.parent / "dissoAdventureExperiments" / "experiments" / "ablation"
_SIBLING_DATA = ROOT.parent / "dissoAdventureExperiments" / "data"

CH5_FILE = _resolve(
    "ARSPI_CH5_FILE",
    [
        _LOCAL_DATA / "shape_features_211.pkl",
        _SIBLING_ABLATION / "shape_features_211.pkl",
    ],
)
CH67_FILE = _resolve(
    "ARSPI_CH67_FILE",
    [
        _LOCAL_DATA / "ch6_ch7_3class_features.pkl",
        _SIBLING_ABLATION / "ch6_ch7_3class_features.pkl",
    ],
)
CLINICAL_FILE = _resolve(
    "ARSPI_CLINICAL_FILE",
    [
        _LOCAL_DATA / "clinical_profile.csv",
        _SIBLING_ABLATION / "clinical_profile.csv",
        _SIBLING_DATA / "clinical_profile.csv",
    ],
)

RANDOM_STATE = 42

N_FOLDS_AFFECTIVE = 10
N_FOLDS_CLINICAL = 5

DIAGNOSES = ["SUD", "MDD", "PTSD", "GAD", "ADHD"]

AFFECTIVE_LABEL_NAMES = {
    0: "Negative",
    1: "Neutral",
    2: "Pleasant",
}

AFFECTIVE_CONFIG_ORDER = [
    "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9",
]

CLINICAL_CONFIG_ORDER = ["C1", "C2", "C3", "C4", "C5", "C6"]

EXPECTED_DIMS = {
    "BandPower": 170,
    "E": 2176,
    "D": 238,
    "T": 68,
    "C": 3,
}
