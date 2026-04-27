#!/usr/bin/env python3
"""Paper-ready A0-A9 results for three-class affective classification.

Wraps the existing layer-ablation logic but emits clean CSV/JSON artifacts
suitable for the manuscript. No new science.

Outputs:
    outputs/operational_distinctness/affective_ablation_metrics.csv
    outputs/operational_distinctness/affective_predictions.csv
    outputs/operational_distinctness/affective_confusion_matrices.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Allow running as a script (``python 01_affective_ablation_metrics.py``)
# or as a module (``python -m experiments.operational_distinctness.01_...``).
try:
    from . import config as cfg
    from .common import (
        build_feature_blocks,
        bootstrap_ci,
        get_affective_configs,
        hash_subject_array,
        load_inputs,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from experiments.operational_distinctness import config as cfg
    from experiments.operational_distinctness.common import (
        build_feature_blocks,
        bootstrap_ci,
        get_affective_configs,
        hash_subject_array,
        load_inputs,
    )

LABEL_NAMES = [cfg.AFFECTIVE_LABEL_NAMES[k] for k in sorted(cfg.AFFECTIVE_LABEL_NAMES)]


def _make_classifier() -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=cfg.RANDOM_STATE,
    )


def evaluate_config(config_id: str,
                    feature_set: str,
                    X: np.ndarray,
                    y: np.ndarray,
                    subjects: np.ndarray,
                    subject_hash: np.ndarray) -> dict:
    cv = StratifiedGroupKFold(
        n_splits=cfg.N_FOLDS_AFFECTIVE,
        shuffle=True,
        random_state=cfg.RANDOM_STATE,
    )

    fold_bal_acc, fold_acc, fold_f1, fold_prec, fold_rec, fold_auc = [], [], [], [], [], []
    pred_rows: list[dict] = []
    cm_total = np.zeros((3, 3), dtype=int)

    for fold_idx, (tr, te) in enumerate(cv.split(X, y, groups=subjects)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])

        clf = _make_classifier()
        clf.fit(Xtr, y[tr])
        y_pred = clf.predict(Xte)
        proba = clf.predict_proba(Xte)
        # Align proba columns with [0,1,2] ordering even if a class is missing.
        proba_full = np.zeros((Xte.shape[0], 3))
        for ci, cls in enumerate(clf.classes_):
            proba_full[:, int(cls)] = proba[:, ci]

        fold_bal_acc.append(balanced_accuracy_score(y[te], y_pred))
        fold_acc.append(accuracy_score(y[te], y_pred))
        fold_f1.append(f1_score(y[te], y_pred, average="macro", zero_division=0))
        fold_prec.append(precision_score(y[te], y_pred, average="macro", zero_division=0))
        fold_rec.append(recall_score(y[te], y_pred, average="macro", zero_division=0))
        try:
            fold_auc.append(
                roc_auc_score(y[te], proba_full, multi_class="ovr", average="macro",
                              labels=[0, 1, 2])
            )
        except ValueError:
            fold_auc.append(float("nan"))

        cm_total += confusion_matrix(y[te], y_pred, labels=[0, 1, 2])

        for local_i, global_i in enumerate(te):
            pred_rows.append({
                "subject_hash": subject_hash[global_i],
                "condition_true": cfg.AFFECTIVE_LABEL_NAMES[int(y[global_i])],
                "fold": fold_idx,
                "config_id": config_id,
                "y_true": int(y[global_i]),
                "y_pred": int(y_pred[local_i]),
                "prob_negative": float(proba_full[local_i, 0]),
                "prob_neutral":  float(proba_full[local_i, 1]),
                "prob_pleasant": float(proba_full[local_i, 2]),
            })

    bal = np.array(fold_bal_acc)
    ci_lo, ci_hi = bootstrap_ci(bal, random_state=cfg.RANDOM_STATE)

    metrics = {
        "config_id": config_id,
        "feature_set": feature_set,
        "dimension": int(X.shape[1]),
        "n_observations": int(X.shape[0]),
        "n_subjects": int(np.unique(subjects).size),
        "n_folds": cfg.N_FOLDS_AFFECTIVE,
        "balanced_accuracy_mean": float(np.mean(bal)),
        "balanced_accuracy_sd": float(np.std(bal, ddof=1)) if bal.size > 1 else 0.0,
        "balanced_accuracy_ci95_low": ci_lo,
        "balanced_accuracy_ci95_high": ci_hi,
        "accuracy_mean": float(np.mean(fold_acc)),
        "accuracy_sd": float(np.std(fold_acc, ddof=1)) if len(fold_acc) > 1 else 0.0,
        "macro_f1_mean": float(np.mean(fold_f1)),
        "macro_f1_sd": float(np.std(fold_f1, ddof=1)) if len(fold_f1) > 1 else 0.0,
        "macro_precision_mean": float(np.mean(fold_prec)),
        "macro_precision_sd": float(np.std(fold_prec, ddof=1)) if len(fold_prec) > 1 else 0.0,
        "macro_recall_mean": float(np.mean(fold_rec)),
        "macro_recall_sd": float(np.std(fold_rec, ddof=1)) if len(fold_rec) > 1 else 0.0,
        "macro_roc_auc_ovr_mean": float(np.nanmean(fold_auc)),
        "macro_roc_auc_ovr_sd": float(np.nanstd(fold_auc, ddof=1)) if len(fold_auc) > 1 else 0.0,
        "fold_balanced_accuracy_values": json.dumps([float(x) for x in fold_bal_acc]),
        "fold_macro_f1_values": json.dumps([float(x) for x in fold_f1]),
        "fold_macro_auc_values": json.dumps([float(x) for x in fold_auc]),
    }
    cm_payload = {"labels": LABEL_NAMES, "matrix": cm_total.tolist()}
    return {"metrics": metrics, "predictions": pred_rows, "confusion": cm_payload}


def main() -> int:
    print("=" * 70)
    print("01: Affective ablation metrics (A0-A9)")
    print("=" * 70)
    ch5, ch67, _ = load_inputs()
    y = np.asarray(ch5["y"]).astype(int)
    subjects = np.asarray(ch5["subjects"])
    subject_hash = hash_subject_array(subjects)

    blocks = build_feature_blocks(ch5, ch67)
    configs = get_affective_configs(blocks)

    metrics_rows = []
    pred_rows: list[dict] = []
    cm_dict: dict = {}

    for cid, label, X in configs:
        print(f"  {cid:3s}  {label:18s}  dim={X.shape[1]:5d}", flush=True)
        result = evaluate_config(cid, label, X, y, subjects, subject_hash)
        metrics_rows.append(result["metrics"])
        pred_rows.extend(result["predictions"])
        cm_dict[cid] = result["confusion"]
        m = result["metrics"]
        print(f"       balanced_acc = {m['balanced_accuracy_mean']*100:5.2f}% "
              f"+/- {m['balanced_accuracy_sd']*100:4.2f}  "
              f"[{m['balanced_accuracy_ci95_low']*100:.2f}, "
              f"{m['balanced_accuracy_ci95_high']*100:.2f}]")

    metrics_df = pd.DataFrame(metrics_rows)
    pred_df = pd.DataFrame(pred_rows)

    metrics_path = cfg.OUT_DIR / "affective_ablation_metrics.csv"
    pred_path = cfg.OUT_DIR / "affective_predictions.csv"
    cm_path = cfg.OUT_DIR / "affective_confusion_matrices.json"

    metrics_df.to_csv(metrics_path, index=False)
    pred_df.to_csv(pred_path, index=False)
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump(cm_dict, f, indent=2)

    print(f"\n  -> {metrics_path}")
    print(f"  -> {pred_path}")
    print(f"  -> {cm_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
