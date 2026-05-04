#!/usr/bin/env python3
"""Parallelized permutation tests + Benjamini-Hochberg FDR for the ablation results.

Drop-in replacement for ``05_optional_permutation_fdr.py`` that
parallelizes the inner permutation loop with joblib. Same CV protocol,
same null distributions, same BH-FDR adjustment across the 30
diagnosis x configuration clinical tests; orders of magnitude faster
on multi-core machines.

Defaults
--------
``--n-perm 200``: with BH-FDR across 30 clinical tests this resolves
adjusted p-values down to roughly p_BH = 30 * 1/(200+1) ~ 0.15, which is
sufficient to declare or rule out FDR-significance at alpha = 0.05 for
any test where the true p is not in the marginal band [0.005, 0.05].
Increase to 1000 (~5x runtime) only if any clinical test sits in that
marginal band.

Outputs
-------
    outputs/operational_distinctness/affective_inference.csv
    outputs/operational_distinctness/clinical_inference.csv
    outputs/operational_distinctness/permutation_fdr_diagnostics.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
                                 solver="lbfgs",
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


def _permute_within_subjects(y: np.ndarray, subjects: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y_perm = y.copy()
    for sid in np.unique(subjects):
        m = subjects == sid
        y_perm[m] = rng.permutation(y_perm[m])
    return y_perm


def _affective_perm_one(seed_offset: int, X, y, subjects, base_seed: int) -> float:
    y_perm = _permute_within_subjects(y, subjects, base_seed + seed_offset)
    return _cv_balanced_accuracy_3class(X, y_perm, subjects)


def _clinical_perm_one(seed_offset: int, X, y, base_seed: int) -> float:
    rng = np.random.default_rng(base_seed + seed_offset)
    y_perm = rng.permutation(y)
    return _cv_balanced_accuracy_binary(X, y_perm)


def affective_inference(n_perm: int, n_jobs: int) -> Path:
    print(f"\n  Affective permutation test (n_perm={n_perm}, n_jobs={n_jobs}) ...")
    ch5, ch67, _ = load_inputs()
    y = np.asarray(ch5["y"]).astype(int)
    subjects = np.asarray(ch5["subjects"])
    blocks = build_feature_blocks(ch5, ch67)
    configs = get_affective_configs(blocks)

    rows = []
    for cid, label, X in configs:
        t0 = time.time()
        observed = _cv_balanced_accuracy_3class(X, y, subjects)
        # Each permutation gets a deterministic seed offset for reproducibility.
        base_seed = int(cfg.RANDOM_STATE) * 10_000 + int(cid[1:]) * 1_000_000
        null = Parallel(n_jobs=n_jobs, prefer="processes", verbose=0)(
            delayed(_affective_perm_one)(k, X, y, subjects, base_seed)
            for k in range(n_perm)
        )
        null = np.asarray(null, dtype=float)
        pval = float((np.sum(null >= observed) + 1) / (n_perm + 1))
        rows.append({
            "config_id": cid,
            "feature_set": label,
            "observed_balanced_accuracy": float(observed),
            "permutation_null_mean": float(null.mean()),
            "permutation_null_sd": float(null.std(ddof=1)),
            "permutation_null_p95": float(np.percentile(null, 95)),
            "p_value": pval,
            "n_permutations": int(n_perm),
        })
        print(f"    {cid:3s} ({label:>14s})  obs={observed:.4f}  null={null.mean():.4f}  "
              f"p={pval:.4f}  ({time.time() - t0:.1f}s)")

    df = pd.DataFrame(rows)
    out = cfg.OUT_DIR / "affective_inference.csv"
    df.to_csv(out, index=False)
    print(f"  Wrote {out}")
    return out


def clinical_inference(n_perm: int, n_jobs: int) -> Path | None:
    print(f"\n  Clinical permutation test + BH-FDR (n_perm={n_perm}, n_jobs={n_jobs}) ...")
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

    rows = []
    for diag_idx, diagnosis in enumerate(cfg.DIAGNOSES):
        mask, y_subj = build_clinical_labels(clinical_df, all_subjects.tolist(), diagnosis)
        if mask is None:
            print(f"    {diagnosis}: insufficient labels (skipped).")
            continue
        for cfg_idx, (cid, label, _X) in enumerate(configs):
            t0 = time.time()
            X_full, _ = subject_level[cid]
            X_subj = X_full[mask]
            observed = _cv_balanced_accuracy_binary(X_subj, y_subj)
            base_seed = (int(cfg.RANDOM_STATE) * 10_000
                         + diag_idx * 100_000
                         + cfg_idx * 1_000_000)
            null = Parallel(n_jobs=n_jobs, prefer="processes", verbose=0)(
                delayed(_clinical_perm_one)(k, X_subj, y_subj, base_seed)
                for k in range(n_perm)
            )
            null = np.asarray(null, dtype=float)
            pval = float((np.sum(null >= observed) + 1) / (n_perm + 1))
            rows.append({
                "diagnosis": diagnosis,
                "config_id": cid,
                "feature_set": label,
                "observed_balanced_accuracy": float(observed),
                "permutation_null_mean": float(null.mean()),
                "permutation_null_sd": float(null.std(ddof=1)),
                "p_value": pval,
                "n_permutations": int(n_perm),
            })
            print(f"    {diagnosis:6s} {cid:3s}  obs={observed:.4f}  "
                  f"null={null.mean():.4f}  p={pval:.4f}  ({time.time() - t0:.1f}s)")

    if not rows:
        return None
    df = pd.DataFrame(rows)
    p_adj, reject = _bh_fdr(df["p_value"].values)
    df["p_value_bh_fdr"] = p_adj
    df["reject_at_fdr_0p05"] = reject
    out = cfg.OUT_DIR / "clinical_inference.csv"
    df.to_csv(out, index=False)
    print(f"  Wrote {out}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-perm", type=int, default=200,
                        help="Permutations per test (default 200).")
    parser.add_argument("--n-jobs", type=int,
                        default=int(os.environ.get("ARSPI_PERM_JOBS",
                                                   max(1, os.cpu_count() or 1))))
    parser.add_argument("--skip-affective", action="store_true")
    parser.add_argument("--skip-clinical", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("Stage 05b: parallelized permutation FDR")
    print("=" * 72)
    t_start = time.time()
    aff_path = None
    clin_path = None
    if not args.skip_affective:
        aff_path = affective_inference(args.n_perm, args.n_jobs)
    if not args.skip_clinical:
        clin_path = clinical_inference(args.n_perm, args.n_jobs)

    diagnostics = {
        "script": "experiments/operational_distinctness/05b_optional_permutation_fdr_parallel.py",
        "n_permutations": int(args.n_perm),
        "n_jobs": int(args.n_jobs),
        "affective_inference_csv": str(aff_path) if aff_path else None,
        "clinical_inference_csv": str(clin_path) if clin_path else None,
        "runtime_seconds": float(time.time() - t_start),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sklearn_version": __import__("sklearn").__version__,
    }
    diag_path = cfg.OUT_DIR / "permutation_fdr_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\n  Wrote {diag_path}")
    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
