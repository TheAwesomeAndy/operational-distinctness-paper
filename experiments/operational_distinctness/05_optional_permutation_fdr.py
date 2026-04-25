#!/usr/bin/env python3
"""Permutation tests + Benjamini-Hochberg FDR for the ablation results.

Tests whether the observed balanced accuracy of each affective and
clinical configuration exceeds the empirical chance distribution
obtained by permuting class labels within the same CV protocol.

For the clinical analysis, BH-FDR is applied across the
5 diagnoses x 6 configs = 30 tests.

Outputs:
    outputs/operational_distinctness/affective_inference.csv
    outputs/operational_distinctness/clinical_inference.csv
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from . import config as cfg
    from .common import (
        build_clinical_labels,
        build_feature_blocks,
        get_affective_configs,
        get_clinical_configs,
        load_inputs,
        subject_average_features,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from experiments.operational_distinctness import config as cfg
    from experiments.operational_distinctness.common import (
        build_clinical_labels,
        build_feature_blocks,
        get_affective_configs,
        get_clinical_configs,
        load_inputs,
        subject_average_features,
    )


def _cv_balanced_accuracy_3class(X, y, subjects):
    cv = StratifiedGroupKFold(
        n_splits=cfg.N_FOLDS_AFFECTIVE,
        shuffle=True,
        random_state=cfg.RANDOM_STATE,
    )
    accs = []
    for tr, te in cv.split(X, y, groups=subjects):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                                 solver="lbfgs", multi_class="multinomial",
                                 random_state=cfg.RANDOM_STATE)
        clf.fit(Xtr, y[tr])
        accs.append(balanced_accuracy_score(y[te], clf.predict(Xte)))
    return float(np.mean(accs))


def _cv_balanced_accuracy_binary(X, y):
    cv = StratifiedKFold(
        n_splits=cfg.N_FOLDS_CLINICAL,
        shuffle=True,
        random_state=cfg.RANDOM_STATE,
    )
    accs = []
    for tr, te in cv.split(X, y):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                                 random_state=cfg.RANDOM_STATE)
        clf.fit(Xtr, y[tr])
        accs.append(balanced_accuracy_score(y[te], clf.predict(Xte)))
    return float(np.mean(accs))


def _bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg adjusted p-values and reject mask."""
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out_adj = np.empty_like(adj)
    out_adj[order] = adj
    return out_adj, out_adj < alpha


def affective_inference(n_perm: int) -> Path:
    print(f"  Affective permutation test (n_perm={n_perm}) ...")
    ch5, ch67, _ = load_inputs()
    y = np.asarray(ch5["y"]).astype(int)
    subjects = np.asarray(ch5["subjects"])
    blocks = build_feature_blocks(ch5, ch67)
    configs = get_affective_configs(blocks)

    rng = np.random.default_rng(cfg.RANDOM_STATE)
    rows = []
    for cid, label, X in configs:
        observed = _cv_balanced_accuracy_3class(X, y, subjects)
        # Permute condition labels within subject groups to preserve subject structure.
        null = np.empty(n_perm)
        for k in range(n_perm):
            y_perm = y.copy()
            for sid in np.unique(subjects):
                m = subjects == sid
                y_perm[m] = rng.permutation(y_perm[m])
            null[k] = _cv_balanced_accuracy_3class(X, y_perm, subjects)
        pval = (np.sum(null >= observed) + 1) / (n_perm + 1)
        rows.append({
            "config_id": cid,
            "feature_set": label,
            "observed_balanced_accuracy": observed,
            "permutation_null_mean": float(null.mean()),
            "permutation_null_sd": float(null.std(ddof=1)),
            "permutation_null_p95": float(np.percentile(null, 95)),
            "p_value": float(pval),
            "n_permutations": n_perm,
        })
        print(f"    {cid:3s} obs={observed:.3f}  null={null.mean():.3f}  p={pval:.4f}")

    df = pd.DataFrame(rows)
    out = cfg.OUT_DIR / "affective_inference.csv"
    df.to_csv(out, index=False)
    return out


def clinical_inference(n_perm: int) -> Path | None:
    print(f"  Clinical permutation test + BH-FDR (n_perm={n_perm}) ...")
    ch5, ch67, clinical_df = load_inputs()
    if clinical_df is None:
        print("    Clinical metadata not found; skipping.")
        return None

    subjects = np.asarray(ch5["subjects"])
    blocks = build_feature_blocks(ch5, ch67)
    configs = get_clinical_configs(blocks)
    subject_level = {
        cid: subject_average_features(X, subjects) for cid, _label, X in configs
    }
    _, all_subjects = subject_level[configs[0][0]]

    rng = np.random.default_rng(cfg.RANDOM_STATE)
    rows = []
    for diagnosis in cfg.DIAGNOSES:
        mask, y_subj = build_clinical_labels(clinical_df, all_subjects.tolist(), diagnosis)
        if mask is None:
            print(f"    {diagnosis}: insufficient labels (skipped).")
            continue
        for cid, label, _X in configs:
            X_full, _ = subject_level[cid]
            X_subj = X_full[mask]
            observed = _cv_balanced_accuracy_binary(X_subj, y_subj)
            null = np.empty(n_perm)
            for k in range(n_perm):
                y_perm = rng.permutation(y_subj)
                null[k] = _cv_balanced_accuracy_binary(X_subj, y_perm)
            pval = (np.sum(null >= observed) + 1) / (n_perm + 1)
            rows.append({
                "diagnosis": diagnosis,
                "config_id": cid,
                "feature_set": label,
                "observed_balanced_accuracy": observed,
                "permutation_null_mean": float(null.mean()),
                "permutation_null_sd": float(null.std(ddof=1)),
                "p_value": float(pval),
                "n_permutations": n_perm,
            })
            print(f"    {diagnosis} {cid:3s}  obs={observed:.3f}  p={pval:.4f}")

    if not rows:
        return None
    df = pd.DataFrame(rows)
    p_adj, reject = _bh_fdr(df["p_value"].values)
    df["p_value_bh_fdr"] = p_adj
    df["reject_at_fdr_0p05"] = reject
    out = cfg.OUT_DIR / "clinical_inference.csv"
    df.to_csv(out, index=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-perm", type=int, default=1000,
                        help="Number of permutations (default: 1000).")
    parser.add_argument("--task", choices=["affective", "clinical", "both"],
                        default="both")
    args = parser.parse_args()

    print("=" * 70)
    print("05 (optional): Permutation tests + BH-FDR")
    print("=" * 70)

    if args.task in ("affective", "both"):
        out = affective_inference(args.n_perm)
        print(f"  -> {out}")
    if args.task in ("clinical", "both"):
        out = clinical_inference(args.n_perm)
        if out is not None:
            print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
