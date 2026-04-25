#!/usr/bin/env python3
"""Paper-ready C1-C6 clinical-label sensitivity results.

This is *clinical-label sensitivity*, not diagnostic validation. The
language in this module and its outputs is deliberately conservative:
a positive result indicates that a given layer carries weak organization
that tracks a clinical-label assignment, not that the layer detects
the disorder.

Outputs:
    outputs/operational_distinctness/clinical_sensitivity_metrics.csv
    outputs/operational_distinctness/clinical_predictions.csv
    outputs/operational_distinctness/clinical_confusion_matrices.json
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
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from . import config as cfg
    from .common import (
        bootstrap_ci,
        build_clinical_labels,
        build_feature_blocks,
        get_clinical_configs,
        hash_subject_array,
        load_inputs,
        subject_average_features,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from experiments.operational_distinctness import config as cfg
    from experiments.operational_distinctness.common import (
        bootstrap_ci,
        build_clinical_labels,
        build_feature_blocks,
        get_clinical_configs,
        hash_subject_array,
        load_inputs,
        subject_average_features,
    )


def _make_classifier() -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=cfg.RANDOM_STATE,
    )


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    pos_prec = tp / (tp + fp) if (tp + fp) else 0.0
    pos_rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    return {
        "positive_precision": float(pos_prec),
        "positive_recall": float(pos_rec),
        "specificity": float(spec),
        "negative_predictive_value": float(npv),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "_cm": cm,
    }


def evaluate_clinical_config(diagnosis: str,
                             config_id: str,
                             feature_set: str,
                             X_subj: np.ndarray,
                             y_subj: np.ndarray,
                             subject_hash_subj: np.ndarray,
                             n_pos: int,
                             n_neg: int) -> dict:
    cv = StratifiedKFold(
        n_splits=cfg.N_FOLDS_CLINICAL,
        shuffle=True,
        random_state=cfg.RANDOM_STATE,
    )
    fold_bal_acc, fold_auc, fold_f1, fold_prec, fold_rec, fold_spec, fold_npv = (
        [], [], [], [], [], [], []
    )
    pred_rows: list[dict] = []
    cm_total = np.zeros((2, 2), dtype=int)

    for fold_idx, (tr, te) in enumerate(cv.split(X_subj, y_subj)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_subj[tr])
        Xte = scaler.transform(X_subj[te])

        clf = _make_classifier()
        clf.fit(Xtr, y_subj[tr])
        y_pred = clf.predict(Xte)
        proba = clf.predict_proba(Xte)
        # Probability of the positive class (1).
        if 1 in clf.classes_:
            prob_pos = proba[:, list(clf.classes_).index(1)]
        else:
            prob_pos = np.zeros(Xte.shape[0])

        bm = _binary_metrics(y_subj[te], y_pred)
        fold_bal_acc.append(bm["balanced_accuracy"])
        fold_f1.append(bm["macro_f1"])
        fold_prec.append(bm["positive_precision"])
        fold_rec.append(bm["positive_recall"])
        fold_spec.append(bm["specificity"])
        fold_npv.append(bm["negative_predictive_value"])
        cm_total += bm["_cm"]
        try:
            fold_auc.append(roc_auc_score(y_subj[te], prob_pos))
        except ValueError:
            fold_auc.append(float("nan"))

        for local_i, global_i in enumerate(te):
            pred_rows.append({
                "subject_hash": subject_hash_subj[global_i],
                "diagnosis": diagnosis,
                "config_id": config_id,
                "fold": fold_idx,
                "y_true": int(y_subj[global_i]),
                "y_pred": int(y_pred[local_i]),
                "prob_positive": float(prob_pos[local_i]),
            })

    bal = np.array(fold_bal_acc)
    auc = np.array(fold_auc)
    bal_lo, bal_hi = bootstrap_ci(bal, random_state=cfg.RANDOM_STATE)
    auc_lo, auc_hi = bootstrap_ci(auc[~np.isnan(auc)], random_state=cfg.RANDOM_STATE)

    n_total = n_pos + n_neg
    metrics = {
        "diagnosis": diagnosis,
        "config_id": config_id,
        "feature_set": feature_set,
        "dimension": int(X_subj.shape[1]),
        "n_subjects": int(n_total),
        "n_positive": int(n_pos),
        "n_negative": int(n_neg),
        "prevalence": float(n_pos / n_total) if n_total else 0.0,
        "balanced_accuracy_mean": float(np.mean(bal)),
        "balanced_accuracy_sd": float(np.std(bal, ddof=1)) if bal.size > 1 else 0.0,
        "balanced_accuracy_ci95_low": bal_lo,
        "balanced_accuracy_ci95_high": bal_hi,
        "roc_auc_mean": float(np.nanmean(auc)),
        "roc_auc_sd": float(np.nanstd(auc, ddof=1)) if auc.size > 1 else 0.0,
        "roc_auc_ci95_low": auc_lo,
        "roc_auc_ci95_high": auc_hi,
        "macro_f1_mean": float(np.mean(fold_f1)),
        "macro_f1_sd": float(np.std(fold_f1, ddof=1)) if len(fold_f1) > 1 else 0.0,
        "positive_precision_mean": float(np.mean(fold_prec)),
        "positive_recall_mean": float(np.mean(fold_rec)),
        "specificity_mean": float(np.mean(fold_spec)),
        "negative_predictive_value_mean": float(np.mean(fold_npv)),
        "fold_balanced_accuracy_values": json.dumps([float(x) for x in fold_bal_acc]),
        "fold_roc_auc_values": json.dumps([float(x) for x in fold_auc]),
    }
    cm_payload = {"labels": [0, 1], "matrix": cm_total.tolist()}
    return {"metrics": metrics, "predictions": pred_rows, "confusion": cm_payload}


def main() -> int:
    print("=" * 70)
    print("02: Clinical-label sensitivity metrics (C1-C6)")
    print("=" * 70)
    ch5, ch67, clinical_df = load_inputs()
    if clinical_df is None:
        print(f"  Clinical metadata not found at {cfg.CLINICAL_FILE}; nothing to do.")
        return 0

    y = np.asarray(ch5["y"]).astype(int)  # noqa: F841 -- alignment check above
    subjects = np.asarray(ch5["subjects"])
    blocks = build_feature_blocks(ch5, ch67)
    configs = get_clinical_configs(blocks)

    # Subject-level features once per config (averaging over conditions).
    subject_level: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cid, _label, X in configs:
        X_subj, unique_subjects = subject_average_features(X, subjects)
        subject_level[cid] = (X_subj, unique_subjects)
    # All configs share the same unique_subjects ordering.
    _, all_subjects = subject_level[configs[0][0]]
    all_subjects_hash = hash_subject_array(all_subjects)

    metrics_rows: list[dict] = []
    pred_rows: list[dict] = []
    cm_dict: dict = {}

    for diagnosis in cfg.DIAGNOSES:
        mask, y_subj = build_clinical_labels(clinical_df, all_subjects.tolist(), diagnosis)
        if mask is None:
            print(f"  {diagnosis}: insufficient labels (skipped).")
            continue
        n_pos = int((y_subj == 1).sum())
        n_neg = int((y_subj == 0).sum())
        print(f"  {diagnosis} (n_pos={n_pos}, n_neg={n_neg}):")

        cm_dict.setdefault(diagnosis, {})
        for cid, label, _X in configs:
            X_subj_full, _ = subject_level[cid]
            X_subj = X_subj_full[mask]
            subject_hash_subj = all_subjects_hash[mask]
            result = evaluate_clinical_config(
                diagnosis=diagnosis,
                config_id=cid,
                feature_set=label,
                X_subj=X_subj,
                y_subj=y_subj,
                subject_hash_subj=subject_hash_subj,
                n_pos=n_pos,
                n_neg=n_neg,
            )
            metrics_rows.append(result["metrics"])
            pred_rows.extend(result["predictions"])
            cm_dict[diagnosis][cid] = result["confusion"]
            m = result["metrics"]
            print(f"    {cid:3s}  {label:14s}  "
                  f"bal_acc={m['balanced_accuracy_mean']*100:5.2f}%  "
                  f"AUC={m['roc_auc_mean']:.3f}")

    metrics_df = pd.DataFrame(metrics_rows)
    pred_df = pd.DataFrame(pred_rows)

    metrics_path = cfg.OUT_DIR / "clinical_sensitivity_metrics.csv"
    pred_path = cfg.OUT_DIR / "clinical_predictions.csv"
    cm_path = cfg.OUT_DIR / "clinical_confusion_matrices.json"

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
