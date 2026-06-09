"""Shared utilities for the Ready-9 hardening package.

Re-exports the validated loaders / feature builders from
``experiments.operational_distinctness.common`` (single source of truth for
feature definitions and the coupling computation) and adds Ready-9 specifics:
safe file fingerprinting, JSON/CSV writers, Wilson and bootstrap CIs, metric
wrappers, posterior entropy, Brier score, a subject-grouped CV factory,
per-channel dropout masks, and privacy-safe manifest helpers.

All helpers keep raw subject identifiers and private absolute paths out of any
returned/serialized artifact.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

# ── Bootstrap import so the module works both as a script dependency and as a
#    package member (scripts are invoked from the repo root). ───────────────
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.tcds_ready9 import config as cfg  # noqa: E402
from experiments.operational_distinctness.common import (  # noqa: E402
    hash_subject_id,
    hash_subject_array,
    load_inputs,
    build_feature_blocks,
    compute_coupling_block,
    get_affective_configs,
    subject_average_features,
    build_clinical_labels,
    bootstrap_ci,
)


# ════════════════════════════════════════════════════════════════════════
# Provenance / fingerprints
# ════════════════════════════════════════════════════════════════════════
def _git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def repo_commit_hash() -> str | None:
    return _git(cfg.ROOT, "rev-parse", "HEAD")


def repo_branch() -> str | None:
    return _git(cfg.ROOT, "rev-parse", "--abbrev-ref", "HEAD")


def repo_is_dirty() -> bool | None:
    status = _git(cfg.ROOT, "status", "--porcelain")
    if status is None:
        return None
    return len(status.strip()) > 0


def upstream_commit_hash() -> str | None:
    if cfg.DISSO_REPO.exists():
        return _git(cfg.DISSO_REPO, "rev-parse", "HEAD")
    return None


def file_fingerprint(path: Path) -> dict:
    """Privacy-safe fingerprint of a file: existence, size, sha256 of contents.

    Deliberately does NOT include the absolute path; only the basename so that
    private directory layouts (and usernames) stay out of committed artifacts.
    """
    path = Path(path)
    info = {"name": path.name, "exists": path.exists()}
    if not path.exists():
        return info
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    info["size_bytes"] = size
    info["sha256"] = h.hexdigest()
    return info


def base_provenance() -> dict:
    """Common provenance block embedded in every Ready-9 manifest/metadata."""
    import datetime as _dt
    return {
        "paper_repo_commit": repo_commit_hash(),
        "paper_repo_branch": repo_branch(),
        "upstream_repo_commit": upstream_commit_hash(),
        "run_profile": cfg.profile_summary(),
        "privacy_status": cfg.PRIVACY_STATUS,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


def package_versions() -> dict:
    vers = {}
    for name in ("numpy", "scipy", "sklearn", "pandas", "matplotlib"):
        try:
            mod = __import__(name)
            vers[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            vers[name] = "not-installed"
    return vers


# ════════════════════════════════════════════════════════════════════════
# Serialization
# ════════════════════════════════════════════════════════════════════════
class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


def write_json(path: Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=_NpEncoder)


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        # Write header-only (or empty) file so downstream existence checks pass.
        with open(path, "w", newline="") as f:
            if fieldnames:
                csv.DictWriter(f, fieldnames=list(fieldnames)).writeheader()
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ════════════════════════════════════════════════════════════════════════
# Statistics / metrics
# ════════════════════════════════════════════════════════════════════════
def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (float(center - half), float(center + half))


def posterior_entropy(probs: np.ndarray) -> float:
    """Shannon entropy (nats) of a categorical distribution."""
    p = np.asarray(probs, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def brier_score(probs: np.ndarray, y_true: int, n_classes: int) -> float:
    """Multiclass Brier score for a single prediction vector."""
    onehot = np.zeros(n_classes, dtype=float)
    onehot[int(y_true)] = 1.0
    p = np.asarray(probs, dtype=float)
    return float(np.sum((p - onehot) ** 2))


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray,
                  proba: np.ndarray | None = None,
                  labels: Sequence[int] | None = None) -> dict:
    """Balanced accuracy, macro-F1, ROC-AUC (when defined), confusion matrix."""
    from sklearn.metrics import (
        balanced_accuracy_score, f1_score, confusion_matrix, roc_auc_score,
    )
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    out = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n": int(len(y_true)),
    }
    auc = float("nan")
    if proba is not None:
        try:
            proba = np.asarray(proba)
            if len(labels) == 2:
                auc = float(roc_auc_score(y_true, proba[:, 1]))
            else:
                auc = float(roc_auc_score(y_true, proba, multi_class="ovr",
                                          average="macro", labels=labels))
        except Exception:
            auc = float("nan")
    out["roc_auc"] = auc
    return out


# ════════════════════════════════════════════════════════════════════════
# Cross-validation
# ════════════════════════════════════════════════════════════════════════
def subject_grouped_cv(n_splits: int, seed: int):
    """StratifiedGroupKFold respecting subject grouping, seeded deterministically."""
    from sklearn.model_selection import StratifiedGroupKFold
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def fit_logreg_fold(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray,
                    seed: int = 42):
    """Standardize (fit on train), fit an L2 logistic readout, return (pred, proba, classes).

    The scaler is fit on the training fold only (no test leakage). The same
    fitted scaler+classifier is applied to ``X_te`` (which the caller may have
    perturbed for robustness experiments).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xte = sc.transform(X_te)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(Xtr, y_tr)
    proba = clf.predict_proba(Xte)
    pred = clf.classes_[np.argmax(proba, axis=1)]
    return pred, proba, clf.classes_


# ════════════════════════════════════════════════════════════════════════
# Channel-block dropout masks
# ════════════════════════════════════════════════════════════════════════
def channel_dropout_feature_mask(block_name: str, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean feature mask (True == keep) for a flattened per-channel block.

    Drops whole channels consistently across the per-channel feature columns,
    matching the (n_channels, n_features_per_channel) layout in config.
    Blocks without per-channel structure (e.g. C) are returned all-keep.
    """
    if block_name not in cfg.CHANNEL_BLOCK_SHAPE or frac <= 0:
        # Caller supplies the flat dimension via apply_channel_dropout.
        return None
    n_ch, n_feat = cfg.CHANNEL_BLOCK_SHAPE[block_name]
    n_drop = int(round(frac * n_ch))
    drop_ch = rng.choice(n_ch, size=n_drop, replace=False) if n_drop > 0 else np.array([], dtype=int)
    keep = np.ones((n_ch, n_feat), dtype=bool)
    keep[drop_ch, :] = False
    return keep.reshape(-1)


def apply_channel_dropout(X_block: np.ndarray, block_name: str, frac: float,
                          rng: np.random.Generator) -> np.ndarray:
    """Zero out dropped-channel columns of a flattened per-channel block."""
    mask = channel_dropout_feature_mask(block_name, frac, rng)
    if mask is None or X_block.shape[1] != mask.size:
        return X_block
    out = X_block.copy()
    out[:, ~mask] = 0.0
    return out


# ════════════════════════════════════════════════════════════════════════
# Aligned loading
# ════════════════════════════════════════════════════════════════════════
def load_ready9() -> dict:
    """Load aligned inputs and the five canonical feature blocks.

    Returns a dict with: ch5, ch67, clinical_df, blocks, y, subjects (raw, kept
    in-memory only), subj_hash (committable), cond_names.
    Raises on missing inputs / alignment failure (callers convert to reports).
    """
    ch5, ch67, clinical_df = load_inputs()
    blocks = build_feature_blocks(ch5, ch67)
    y = np.asarray(ch5["y"])
    subjects = np.asarray(ch5["subjects"])
    subj_hash = hash_subject_array(subjects)
    cond_names = ch67.get("cond_names", cfg.AFFECTIVE_LABEL_NAMES)
    return {
        "ch5": ch5, "ch67": ch67, "clinical_df": clinical_df,
        "blocks": blocks, "y": y, "subjects": subjects,
        "subj_hash": subj_hash, "cond_names": cond_names,
    }


# ════════════════════════════════════════════════════════════════════════
# Privacy guards
# ════════════════════════════════════════════════════════════════════════
def scrub_path(path) -> str:
    """Return only the basename of a path for safe inclusion in artifacts."""
    return Path(str(path)).name


__all__ = [
    "cfg",
    # re-exports
    "hash_subject_id", "hash_subject_array", "load_inputs", "build_feature_blocks",
    "compute_coupling_block", "get_affective_configs", "subject_average_features",
    "build_clinical_labels", "bootstrap_ci",
    # provenance
    "repo_commit_hash", "repo_branch", "repo_is_dirty", "upstream_commit_hash",
    "file_fingerprint", "base_provenance", "package_versions",
    # io
    "write_json", "write_csv",
    # stats
    "wilson_ci", "posterior_entropy", "brier_score", "metric_bundle",
    # cv / perturbation
    "subject_grouped_cv", "channel_dropout_feature_mask", "apply_channel_dropout",
    # loading / privacy
    "load_ready9", "scrub_path",
]
