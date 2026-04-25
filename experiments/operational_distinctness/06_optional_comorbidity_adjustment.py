#!/usr/bin/env python3
"""Comorbidity-adjusted clinical-label sensitivity.

For each (diagnosis, config) we already have an out-of-fold positive-class
probability in ``clinical_predictions.csv``. Here we treat that score as a
single covariate (``layer_score``) and ask whether it remains associated
with the target diagnosis after adjusting for the four other diagnoses
in the panel and the comorbidity count.

Model:
    target_diagnosis ~ layer_score
                       + other_diagnoses (binary indicators)
                       + comorbidity_count
                       + available_covariates (age, sex, if present)

The fit uses logistic regression with no penalty (regularization off) so
the layer-score coefficient is interpretable on its own. We report the
coefficient, its standard error, the Wald p-value, and the AUROC of the
adjusted model.

Output:
    outputs/operational_distinctness/comorbidity_adjusted.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from . import config as cfg
    from .common import hash_subject_array, load_inputs
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from experiments.operational_distinctness import config as cfg
    from experiments.operational_distinctness.common import (
        hash_subject_array,
        load_inputs,
    )


COVARIATE_CANDIDATES = ["age", "sex", "Assigned_Sex"]


def _fit_logit(X: np.ndarray, y: np.ndarray) -> dict:
    """Fit unpenalized logistic regression and return Wald inference for col 0."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    # Add intercept column manually so we can build the Hessian below.
    Xd = np.column_stack([np.ones(X.shape[0]), X])

    clf = LogisticRegression(
        penalty=None, solver="lbfgs", max_iter=2000, random_state=cfg.RANDOM_STATE,
    )
    clf.fit(X, y)
    beta = np.concatenate([clf.intercept_, clf.coef_.ravel()])
    p = 1.0 / (1.0 + np.exp(-Xd @ beta))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    W = p * (1 - p)
    H = (Xd.T * W) @ Xd
    try:
        cov = np.linalg.pinv(H)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        se = np.full(beta.size, np.nan)

    # Index 1 corresponds to layer_score (column 0 of X).
    coef = float(beta[1])
    se_layer = float(se[1]) if se.size > 1 else float("nan")
    z = coef / se_layer if se_layer and not np.isnan(se_layer) and se_layer > 0 else np.nan
    from math import erf, sqrt
    pval = (1 - 0.5 * (1 + erf(abs(z) / sqrt(2)))) * 2 if not np.isnan(z) else float("nan")
    proba = clf.predict_proba(X)[:, list(clf.classes_).index(1)] if 1 in clf.classes_ else np.zeros(X.shape[0])
    try:
        auc = float(roc_auc_score(y, proba))
    except ValueError:
        auc = float("nan")
    return {
        "layer_score_coef": coef,
        "layer_score_se": se_layer,
        "layer_score_p_value": float(pval) if not np.isnan(pval) else float("nan"),
        "model_auc": auc,
    }


def _comorbidity_count(row: pd.Series, diagnoses: list[str]) -> int:
    return int(sum(int(row[d]) for d in diagnoses if d in row.index and not pd.isna(row[d])))


def main() -> int:
    print("=" * 70)
    print("06 (optional): Comorbidity-adjusted clinical-label sensitivity")
    print("=" * 70)

    pred_path = cfg.OUT_DIR / "clinical_predictions.csv"
    if not pred_path.exists():
        print(f"  Need {pred_path}; run 02_clinical_sensitivity_metrics.py first.")
        return 1

    ch5, _ch67, clinical_df = load_inputs()
    if clinical_df is None:
        print("  Clinical metadata not found; skipping.")
        return 0

    subjects_obs = np.asarray(ch5["subjects"])
    subject_hash_lookup = dict(zip(
        hash_subject_array(np.array(sorted(set(subjects_obs.tolist())))).tolist(),
        sorted(set(subjects_obs.tolist())),
    ))

    pred_df = pd.read_csv(pred_path)

    rows = []
    for diagnosis in cfg.DIAGNOSES:
        if diagnosis not in clinical_df.columns:
            continue
        other_diagnoses = [d for d in cfg.DIAGNOSES if d != diagnosis]

        for config_id in pred_df[pred_df["diagnosis"] == diagnosis]["config_id"].unique():
            sub = pred_df[(pred_df["diagnosis"] == diagnosis) &
                          (pred_df["config_id"] == config_id)].copy()
            sub["subject_id"] = sub["subject_hash"].map(subject_hash_lookup)
            sub = sub.dropna(subset=["subject_id"])
            sub["subject_id"] = sub["subject_id"].astype(int)

            merged = sub.merge(clinical_df, left_on="subject_id", right_on="ID", how="left")

            features = [merged["prob_positive"].astype(float).values]
            feature_names = ["layer_score"]
            for d in other_diagnoses:
                if d in merged.columns:
                    v = merged[d].fillna(0).astype(float).values
                    features.append(v)
                    feature_names.append(d)
            features.append(merged.apply(
                lambda r: _comorbidity_count(r, other_diagnoses), axis=1
            ).astype(float).values)
            feature_names.append("comorbidity_count")
            for cov in COVARIATE_CANDIDATES:
                if cov in merged.columns:
                    col = pd.to_numeric(merged[cov], errors="coerce")
                    if col.notna().sum() >= max(10, len(merged) // 2):
                        features.append(col.fillna(col.median()).astype(float).values)
                        feature_names.append(cov)

            X = np.column_stack(features)
            y = merged["y_true"].astype(int).values
            if (y == 1).sum() < 5 or (y == 0).sum() < 5:
                continue
            try:
                fit = _fit_logit(X, y)
            except Exception as exc:  # pragma: no cover -- diagnostic only
                print(f"    {diagnosis} {config_id}: fit failed ({exc!r})")
                continue
            rows.append({
                "diagnosis": diagnosis,
                "config_id": config_id,
                "n_subjects": int(len(merged)),
                "n_positive": int((y == 1).sum()),
                "covariates": ",".join(feature_names[1:]),
                **fit,
            })
            print(f"    {diagnosis:5s} {config_id}  beta={fit['layer_score_coef']:+.3f} "
                  f"SE={fit['layer_score_se']:.3f}  p={fit['layer_score_p_value']:.4f} "
                  f"AUC={fit['model_auc']:.3f}")

    out = cfg.OUT_DIR / "comorbidity_adjusted.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
