"""Reusable functions for the operational-distinctness reporting pipeline.

The feature definitions, the coupling computation, and the cross-validation
protocol mirror the existing ``experiments/ablation/layer_ablation.py``.
This module factors them out so the reporting scripts can import them
without re-running the printed ablation pipeline.
"""
from __future__ import annotations

import hashlib
import pickle
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config as cfg


# ─── PHI / privacy ──────────────────────────────────────────────────────

def hash_subject_id(subject_id) -> str:
    """16-char SHA-256 prefix of a stringified subject id.

    Used to keep raw subject identifiers out of any committed artifact.
    """
    return hashlib.sha256(str(subject_id).encode("utf-8")).hexdigest()[:16]


def hash_subject_array(subjects: Sequence) -> np.ndarray:
    return np.array([hash_subject_id(s) for s in subjects], dtype=object)


# ─── Data loading ───────────────────────────────────────────────────────

def load_inputs() -> Tuple[dict, dict, pd.DataFrame | None]:
    """Load the two ARSPI-Net feature pickles and (optionally) the clinical CSV.

    Returns
    -------
    ch5, ch67, clinical_df_or_none
    """
    if not cfg.CH5_FILE.exists():
        raise FileNotFoundError(
            f"Missing Chapter 5 features at {cfg.CH5_FILE}. "
            "Run chapter5Experiments/run_chapter5_experiments.py first."
        )
    if not cfg.CH67_FILE.exists():
        raise FileNotFoundError(
            f"Missing Chapter 6/7 features at {cfg.CH67_FILE}. "
            "Run experiments/ch6_ch7_3class/ch6_ch7_01_feature_extraction.py first."
        )

    with open(cfg.CH5_FILE, "rb") as f:
        ch5 = pickle.load(f)
    with open(cfg.CH67_FILE, "rb") as f:
        ch67 = pickle.load(f)

    if not np.array_equal(ch5["subjects"], ch67["subjects"]):
        raise ValueError("Subject arrays disagree between Ch5 and Ch6/7 pickles.")
    if not np.array_equal(ch5["y"], ch67["y"]):
        raise ValueError("Label arrays disagree between Ch5 and Ch6/7 pickles.")

    clinical_df: pd.DataFrame | None = None
    if cfg.CLINICAL_FILE.exists():
        clinical_df = pd.read_csv(cfg.CLINICAL_FILE)
        clinical_df = clinical_df.drop_duplicates(subset="ID", keep="first")

    return ch5, ch67, clinical_df


# ─── Feature blocks ─────────────────────────────────────────────────────

def _rankdata(x: np.ndarray) -> np.ndarray:
    sorter = np.argsort(x)
    ranks = np.empty_like(sorter, dtype=float)
    ranks[sorter] = np.arange(1, len(x) + 1, dtype=float)
    return ranks


def _spearman_coupling(D_obs: np.ndarray, T_obs: np.ndarray) -> np.ndarray:
    """7x2 Spearman coupling matrix across electrodes (matches layer_ablation.py)."""
    D_ranks = np.zeros_like(D_obs, dtype=float)
    T_ranks = np.zeros_like(T_obs, dtype=float)
    for j in range(D_obs.shape[1]):
        D_ranks[:, j] = _rankdata(D_obs[:, j])
    for k in range(T_obs.shape[1]):
        T_ranks[:, k] = _rankdata(T_obs[:, k])
    Dc = D_ranks - D_ranks.mean(axis=0)
    Tc = T_ranks - T_ranks.mean(axis=0)
    Dn = np.sqrt((Dc ** 2).sum(axis=0))
    Tn = np.sqrt((Tc ** 2).sum(axis=0))
    C_mat = (Dc.T @ Tc) / (Dn[:, None] * Tn[None, :] + 1e-12)
    return np.where(np.isfinite(C_mat), C_mat, 0.0)


def compute_coupling_block(D_perchannel: np.ndarray,
                           T_perchannel: np.ndarray) -> np.ndarray:
    """Per-observation 3-vector: (kappa, mean signed strength, mean signed clustering).

    kappa is Frobenius-norm-normalized as in the dissertation's formal definition.
    """
    n_obs = D_perchannel.shape[0]
    C_block = np.zeros((n_obs, 3))
    for i in range(n_obs):
        C_mat = _spearman_coupling(D_perchannel[i], T_perchannel[i])
        p, q = C_mat.shape
        C_block[i, 0] = np.linalg.norm(C_mat, "fro") / np.sqrt(p * q)
        C_block[i, 1] = C_mat[:, 0].mean()
        C_block[i, 2] = C_mat[:, 1].mean()
    return C_block


def build_feature_blocks(ch5: dict, ch67: dict) -> Dict[str, np.ndarray]:
    """Return the five canonical feature blocks {BandPower, E, D, T, C}."""
    n_obs = len(ch5["y"])
    E = ch5["lsm_bsc6_pca"].reshape(n_obs, -1)
    D = ch67["D"].reshape(n_obs, -1)
    T = ch67["T_topo"].reshape(n_obs, -1)
    BP = ch5["conv_feats"].reshape(n_obs, -1)
    C = compute_coupling_block(ch67["D"], ch67["T_topo"])
    return {"BandPower": BP, "E": E, "D": D, "T": T, "C": C}


# ─── Configuration matrices ─────────────────────────────────────────────

def get_affective_configs(blocks: Dict[str, np.ndarray]) -> List[Tuple[str, str, np.ndarray]]:
    BP, E, D, T, C = (blocks[k] for k in ("BandPower", "E", "D", "T", "C"))
    return [
        ("A0", "BandPower",   BP),
        ("A1", "E",           E),
        ("A2", "D",           D),
        ("A3", "T",           T),
        ("A4", "C",           C),
        ("A5", "D + T",       np.hstack([D, T])),
        ("A6", "E + D",       np.hstack([E, D])),
        ("A7", "E + T",       np.hstack([E, T])),
        ("A8", "E + D + T",   np.hstack([E, D, T])),
        ("A9", "E + D + T + C", np.hstack([E, D, T, C])),
    ]


def get_clinical_configs(blocks: Dict[str, np.ndarray]) -> List[Tuple[str, str, np.ndarray]]:
    E, D, T, C = (blocks[k] for k in ("E", "D", "T", "C"))
    return [
        ("C1", "E",           E),
        ("C2", "D",           D),
        ("C3", "T",           T),
        ("C4", "D + T",       np.hstack([D, T])),
        ("C5", "E + D + T",   np.hstack([E, D, T])),
        ("C6", "C",           C),
    ]


# ─── Subject-level utilities ────────────────────────────────────────────

def subject_average_features(X: np.ndarray,
                             subjects: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Average observation-level features across affective conditions per subject.

    Returns (X_subject, unique_subjects) with X_subject row order matching unique_subjects.
    Subjects are returned sorted to ensure deterministic alignment.
    """
    unique_subjects = np.array(sorted(set(subjects.tolist())))
    X_subj = np.vstack([X[subjects == s].mean(axis=0) for s in unique_subjects])
    return X_subj, unique_subjects


def build_clinical_labels(clinical_df: pd.DataFrame,
                          subject_ids: Sequence,
                          diagnosis: str,
                          min_per_class: int = 15
                          ) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
    """Align a binary diagnosis column with the given subject id list.

    Returns (mask_of_usable_subjects, y) where ``mask`` is a boolean array
    over ``subject_ids``. Subjects whose label is NaN are dropped. Returns
    (None, None) if either class has fewer than ``min_per_class`` subjects.
    """
    if diagnosis not in clinical_df.columns:
        return None, None
    label_map = dict(zip(clinical_df["ID"].values, clinical_df[diagnosis].values))
    mask = np.zeros(len(subject_ids), dtype=bool)
    y = np.full(len(subject_ids), -1, dtype=int)
    for i, sid in enumerate(subject_ids):
        v = label_map.get(sid, np.nan)
        if pd.isna(v):
            continue
        v_int = int(v)
        if v_int not in (0, 1):
            continue
        mask[i] = True
        y[i] = v_int
    y_usable = y[mask]
    if (y_usable == 1).sum() < min_per_class or (y_usable == 0).sum() < min_per_class:
        return None, None
    return mask, y_usable


# ─── Bootstrap CI ───────────────────────────────────────────────────────

def bootstrap_ci(values: Sequence[float],
                 n_boot: int = 5000,
                 ci: float = 95,
                 random_state: int = 42) -> Tuple[float, float]:
    """Percentile bootstrap CI of the mean of ``values``."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = (100 - ci) / 2.0
    lo = float(np.percentile(means, alpha))
    hi = float(np.percentile(means, 100 - alpha))
    return lo, hi


__all__ = [
    "hash_subject_id",
    "hash_subject_array",
    "load_inputs",
    "compute_coupling_block",
    "build_feature_blocks",
    "get_affective_configs",
    "get_clinical_configs",
    "subject_average_features",
    "build_clinical_labels",
    "bootstrap_ci",
]
