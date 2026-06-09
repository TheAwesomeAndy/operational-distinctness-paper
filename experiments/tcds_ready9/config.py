"""Centralized configuration for the Ready-9 hardening package.

This module extends the resolution pattern of
``experiments/operational_distinctness/config.py`` with Ready-9 specifics:
run profiles, random seeds, perturbation levels, policy names, and output
directories. It never writes private data; all output directories live inside
this repository under ``outputs/``, ``figures/``, and ``tables/``.

Input-file resolution order (highest priority first)
----------------------------------------------------
1. Explicit env vars: ARSPI_SHAPE_FEATURES, ARSPI_CH67_FEATURES,
   ARSPI_CLINICAL_FILE.
2. ``data/`` inside this repository.
3. ``data/private/`` inside this repository.
4. ``experiments/ablation/`` inside this repository.
5. The sibling ``dissoAdventureExperiments`` checkout (env ARSPI_DISSO_REPO
   or ``../dissoAdventureExperiments``).

Run profiles
------------
``ARSPI_READY9_PROFILE`` selects ``pilot`` (engineering validation; reduced
seeds/episodes) or ``submission`` (final seeds/episodes for manuscript-facing
artifacts). The active profile is recorded in every manifest.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Repository roots ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]

DISSO_REPO = Path(
    os.environ.get("ARSPI_DISSO_REPO", str(ROOT.parent / "dissoAdventureExperiments"))
).expanduser()

UPSTREAM_FEATURE_SCRIPT = (
    DISSO_REPO / "experiments" / "ch6_ch7_3class" / "ch6_ch7_01_feature_extraction.py"
)
LOCAL_FEATURE_SCRIPT = ROOT / "prepare_inputs" / "extract_ch67_features.py"

# ── Output locations (always inside this repo) ──────────────────────────
OUT_DIR = ROOT / "outputs" / "tcds_ready9"
ANALYSIS_DIR = OUT_DIR / "analysis"
OBS_OUT_DIR = OUT_DIR / "observations"
FIG_DIR = ROOT / "figures" / "tcds_ready9"
FIG_OBS_DIR = FIG_DIR / "observations"
FIG_ANA_DIR = FIG_DIR / "analysis"
TABLE_DIR = ROOT / "tables" / "tcds_ready9"

for _d in (OUT_DIR, ANALYSIS_DIR, OBS_OUT_DIR, FIG_DIR, FIG_OBS_DIR, FIG_ANA_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ── Private-input resolution ────────────────────────────────────────────
def _resolve(env_var: str, candidates: list[Path]) -> Path:
    explicit = os.environ.get(env_var)
    if explicit:
        return Path(explicit).expanduser().resolve()
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # default (may not exist; loaders raise/report)


_LOCAL_DATA = ROOT / "data"
_LOCAL_PRIVATE = ROOT / "data" / "private"
_LOCAL_ABLATION = ROOT / "experiments" / "ablation"
_SIBLING_ABLATION = DISSO_REPO / "experiments" / "ablation"
_SIBLING_DATA = DISSO_REPO / "data"

SHAPE_FILE = _resolve(
    "ARSPI_SHAPE_FEATURES",
    [
        _LOCAL_DATA / "shape_features_211.pkl",
        _LOCAL_PRIVATE / "shape_features_211.pkl",
        _LOCAL_ABLATION / "shape_features_211.pkl",
        _SIBLING_ABLATION / "shape_features_211.pkl",
    ],
)
CH67_FILE = _resolve(
    "ARSPI_CH67_FEATURES",
    [
        _LOCAL_DATA / "ch6_ch7_3class_features.pkl",
        _LOCAL_PRIVATE / "ch6_ch7_3class_features.pkl",
        _LOCAL_ABLATION / "ch6_ch7_3class_features.pkl",
        _SIBLING_ABLATION / "ch6_ch7_3class_features.pkl",
    ],
)
CLINICAL_FILE = _resolve(
    "ARSPI_CLINICAL_FILE",
    [
        _LOCAL_DATA / "clinical_profile.csv",
        _LOCAL_PRIVATE / "clinical_profile.csv",
        _LOCAL_ABLATION / "clinical_profile.csv",
        _SIBLING_DATA / "clinical_profile.csv",
    ],
)
RAW_EEG_DIR = _resolve(
    "ARSPI_RAW_EEG_DIR",
    [
        _LOCAL_DATA / "batch_data_full",
        _LOCAL_DATA / "batch_data",
        DISSO_REPO / "data" / "batch_data_full",
    ],
)

# Destination for a *locally regenerated* ch6/ch7 pickle. Lives under the
# gitignored ``data/`` dir and must never be committed.
CH67_REGEN_OUTPUT = _LOCAL_DATA / "ch6_ch7_3class_features.pkl"

# ── Run profiles ────────────────────────────────────────────────────────
PROFILE = os.environ.get("ARSPI_READY9_PROFILE", "pilot").strip().lower()
if PROFILE not in ("pilot", "submission"):
    PROFILE = "pilot"

_PROFILE_SETTINGS = {
    "pilot": {
        "seeds": [42, 43],
        "closed_loop_episodes": 1000,
        "raw_signal_subset": 60,          # raw-signal robustness obs subset
        "kappa_n_perm": 1000,             # electrode-shuffle null permutations
    },
    "submission": {
        "seeds": [42, 43, 44, 45, 46],
        "closed_loop_episodes": 1500,
        # Raw-signal perturbation may be bounded for runtime feasibility. The
        # hard submission requirement is FULL representation-level robustness
        # across all 10 configurations (enforced as a core/fatal phase). The
        # raw-signal subset is a bounded *diagnostic* and its scope is recorded
        # in the manifest and stated in the manuscript.
        "raw_signal_subset": 150,
        "kappa_n_perm": 5000,
    },
}
SEEDS = _PROFILE_SETTINGS[PROFILE]["seeds"]
CLOSED_LOOP_EPISODES = _PROFILE_SETTINGS[PROFILE]["closed_loop_episodes"]
RAW_SIGNAL_SUBSET = _PROFILE_SETTINGS[PROFILE]["raw_signal_subset"]
KAPPA_N_PERM = _PROFILE_SETTINGS[PROFILE]["kappa_n_perm"]

RANDOM_STATE = 42

# ── Cross-validation ────────────────────────────────────────────────────
N_FOLDS_AFFECTIVE = 5      # subject-grouped; kept modest for the seed sweep
N_FOLDS_CLINICAL = 5

# ── Labels / configs ────────────────────────────────────────────────────
AFFECTIVE_LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Pleasant"}
DIAGNOSES = ["SUD", "MDD", "PTSD", "GAD", "ADHD"]

EXPECTED_SHAPE_KEYS = ["lsm_bsc6_pca", "conv_feats", "y", "subjects"]
EXPECTED_CH67_KEYS = ["D", "D_extra", "T_topo", "tPLV_mats", "pop_rate_ts", "y", "subjects", "cond_names"]

EXPECTED_DIMS = {"BandPower": 170, "E": 2176, "D": 238, "T": 68, "C": 3}

# Per-channel block structure used for channel-dropout masking.
# (n_channels, n_features_per_channel)
CHANNEL_BLOCK_SHAPE = {"BandPower": (34, 5), "E": (34, 64), "D": (34, 7), "T": (34, 2)}
N_CHANNELS = 34

# Mechanism-ablation configuration order.
ABLATION_CONFIG_ORDER = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"]

# ── Perturbation levels ─────────────────────────────────────────────────
TEMPORAL_JITTER_MS = [0, 10, 25, 50]            # ± applied
AMPLITUDE_SNR_DB = [None, 20, 10, 5]            # None == clean
CHANNEL_DROPOUT_FRAC = [0.0, 0.10, 0.20, 0.30]
GRAPH_PERTURB_FRAC = [0.0, 0.10, 0.20, 0.30]
CLOSED_LOOP_EPSILON = [0.0, 0.1, 0.2, 0.3, 0.4]

# Raw-EEG timing geometry (matches prepare_inputs/extract_ch67_features.py).
FS_RAW = 1024            # Hz, original sampling rate of the *_BC.txt files
FS_DS = 256             # Hz, downsampled rate of X_ds in shape_features_211.pkl

# ── Closed-loop policies ────────────────────────────────────────────────
# "epistemic_only" is included conditionally by the closed-loop script only if
# it is formally separable from the expected-free-energy objective.
CLOSED_LOOP_POLICIES = ["passive", "random", "pragmatic_only", "efe", "oracle"]

# ── Privacy ─────────────────────────────────────────────────────────────
HASH_SUBJECTS = True
PRIVACY_STATUS = (
    "restricted (private clinical/neurophysiological data; "
    "released outputs are aggregate or deidentified)"
)

# Patterns that must never appear inside committed artifacts.
PRIVATE_PATH_TOKENS = ["batch_data", "shape_features_211", "clinical_profile",
                       "ch6_ch7_3class_features", "/home/", "C:\\\\Users"]


def profile_summary() -> dict:
    """Lightweight dict describing the active run profile (for manifests)."""
    return {
        "profile": PROFILE,
        "seeds": list(SEEDS),
        "closed_loop_episodes": CLOSED_LOOP_EPISODES,
        "raw_signal_subset": RAW_SIGNAL_SUBSET,
        "kappa_n_perm": KAPPA_N_PERM,
        "n_folds_affective": N_FOLDS_AFFECTIVE,
        "n_folds_clinical": N_FOLDS_CLINICAL,
    }
