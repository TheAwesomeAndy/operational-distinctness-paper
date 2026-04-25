#!/usr/bin/env python3
"""Layer-redundancy diagnostics for (E, D, T, C).

For each ordered pair of feature blocks we compute four
representation-similarity / cross-predictability statistics:

* linear CKA on subject-averaged features
* CCA top-1 canonical correlation
* mean of the top-5 CCA canonical correlations
* ridge cross-predictability: out-of-fold R^2 of A predicting B
  (subject-level 5-fold CV, columnwise mean R^2)

Operational interpretation: if all four numbers are small for a pair,
the two layers occupy non-equivalent feature spaces. If all four are
large, the two layers are largely redundant.

Output:
    outputs/operational_distinctness/layer_redundancy.csv
"""
from __future__ import annotations

import sys
import warnings
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from . import config as cfg
    from .common import build_feature_blocks, load_inputs, subject_average_features
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from experiments.operational_distinctness import config as cfg
    from experiments.operational_distinctness.common import (
        build_feature_blocks,
        load_inputs,
        subject_average_features,
    )


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(Yc.T @ Xc, "fro") ** 2
    den = (np.linalg.norm(Xc.T @ Xc, "fro") *
           np.linalg.norm(Yc.T @ Yc, "fro"))
    if den == 0:
        return float("nan")
    return float(num / den)


def cca_top_corrs(X: np.ndarray, Y: np.ndarray, k: int = 5) -> list[float]:
    """Top-k canonical correlations via SVD of whitened cross-covariance."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    n = X.shape[0]
    # Reduce rank if features outnumber samples to keep CCA stable.
    Ux, sx, _ = np.linalg.svd(Xc, full_matrices=False)
    Uy, sy, _ = np.linalg.svd(Yc, full_matrices=False)
    rx = min(int((sx > 1e-8 * sx[0]).sum()), n - 1) if sx.size else 0
    ry = min(int((sy > 1e-8 * sy[0]).sum()), n - 1) if sy.size else 0
    if rx == 0 or ry == 0:
        return [float("nan")] * k
    Xw = Ux[:, :rx]
    Yw = Uy[:, :ry]
    M = Xw.T @ Yw
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    out = list(s[:k])
    while len(out) < k:
        out.append(0.0)
    return [float(v) for v in out]


def ridge_cross_predict_r2(X: np.ndarray, Y: np.ndarray, alpha: float = 1.0,
                           n_splits: int = 5) -> float:
    """Mean out-of-fold R^2 across columns of Y when predicted from X."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.RANDOM_STATE)
    fold_r2 = []
    for tr, te in cv.split(X):
        sx = StandardScaler().fit(X[tr])
        sy = StandardScaler().fit(Y[tr])
        Xtr = sx.transform(X[tr])
        Xte = sx.transform(X[te])
        Ytr = sy.transform(Y[tr])
        Yte = sy.transform(Y[te])
        clf = Ridge(alpha=alpha)
        clf.fit(Xtr, Ytr)
        pred = clf.predict(Xte)
        ss_res = ((Yte - pred) ** 2).sum(axis=0)
        ss_tot = ((Yte - Yte.mean(axis=0)) ** 2).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = 1.0 - ss_res / ss_tot
        r2 = np.where(np.isfinite(r2), r2, 0.0)
        fold_r2.append(np.mean(r2))
    return float(np.mean(fold_r2))


def main() -> int:
    print("=" * 70)
    print("07 (optional): Layer redundancy diagnostics")
    print("=" * 70)

    ch5, ch67, _ = load_inputs()
    subjects = np.asarray(ch5["subjects"])
    blocks = build_feature_blocks(ch5, ch67)
    targets = ["E", "D", "T", "C"]

    subj_blocks: dict[str, np.ndarray] = {}
    for name in targets:
        X_subj, _ = subject_average_features(blocks[name], subjects)
        subj_blocks[name] = X_subj

    rows = []
    for a, b in permutations(targets, 2):
        Xa = subj_blocks[a]
        Xb = subj_blocks[b]
        cka = linear_cka(Xa, Xb)
        cca = cca_top_corrs(Xa, Xb, k=5)
        cca_top1 = cca[0]
        cca_mean5 = float(np.mean(cca))
        try:
            r2 = ridge_cross_predict_r2(Xa, Xb)
        except Exception:
            r2 = float("nan")
        rows.append({
            "source": a,
            "target": b,
            "linear_cka": cka,
            "cca_top1": cca_top1,
            "cca_mean_top5": cca_mean5,
            "ridge_cross_predict_r2": r2,
        })
        print(f"  {a} -> {b}: CKA={cka:.3f}  CCA1={cca_top1:.3f}  "
              f"CCA_top5_mean={cca_mean5:.3f}  R2={r2:.3f}")

    df = pd.DataFrame(rows)
    out = cfg.OUT_DIR / "layer_redundancy.csv"
    df.to_csv(out, index=False)
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
