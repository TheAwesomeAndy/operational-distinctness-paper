#!/usr/bin/env python3
"""Phase 7 — Mechanism ablation and functional role (CORE phase).

Evaluates the A0-A9 feature configurations under subject-grouped cross-
validation across the configured seeds, plus negative controls (shuffled label,
shuffled subject), and an exploratory clinical-label sensitivity pass.

The question is whether E, D, T, C are operationally distinct, redundant, or
subsumed -- not which is the "best classifier". Clinical labels are treated as
exploratory validation/context structure, never as diagnostic biomarker
evidence.

Outputs:
    outputs/tcds_ready9/analysis/mechanism_ablation_metrics.csv
    outputs/tcds_ready9/analysis/mechanism_ablation_summary.csv
    outputs/tcds_ready9/analysis/mechanism_ablation_confusion_matrices.json
    tables/tcds_ready9/table_mechanism_ablation.tex

Run:
    python experiments/tcds_ready9/09_mechanism_ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.tcds_ready9 import config as cfg  # noqa: E402
from experiments.tcds_ready9 import common_ready9 as cr  # noqa: E402

# Which named neural mechanism each block carries (for interpretation columns).
MECHANISM = {
    "A0": "spectral band-power (non-neuromorphic baseline)",
    "A1": "LIF reservoir spike embedding (E)",
    "A2": "reservoir dynamical descriptors (D)",
    "A3": "tPLV graph topology descriptors (T)",
    "A4": "structure-function coupling (C)",
    "A5": "D+T (dynamics + topology)",
    "A6": "E+D (embedding + dynamics)",
    "A7": "E+T (embedding + topology)",
    "A8": "E+D+T (full reservoir-graph substrate)",
    "A9": "E+D+T+C (substrate + coupling readout)",
}


def _evaluate_config(X, y, groups, label_set):
    """Return per-(seed,fold) metric rows and a pooled confusion matrix."""
    rows = []
    pooled_true, pooled_pred = [], []
    for seed in cfg.SEEDS:
        cv = cr.subject_grouped_cv(cfg.N_FOLDS_AFFECTIVE, seed)
        for fold, (tr, te) in enumerate(cv.split(X, y, groups=groups)):
            pred, proba, classes = cr.fit_logreg_fold(X[tr], y[tr], X[te], seed=seed)
            mb = cr.metric_bundle(y[te], pred, proba, labels=label_set)
            rows.append({
                "seed": seed, "fold": fold,
                "balanced_accuracy": mb["balanced_accuracy"],
                "macro_f1": mb["macro_f1"], "roc_auc": mb["roc_auc"],
                "n_test": mb["n"],
            })
            pooled_true.extend(y[te].tolist())
            pooled_pred.extend(pred.tolist())
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(pooled_true, pooled_pred, labels=label_set).tolist()
    return rows, cm


def _shuffle_label_control(X, y, groups, label_set, rng):
    y_sh = rng.permutation(y)
    return _evaluate_config(X, y_sh, groups, label_set)


def _shuffle_subject_control(X, y, groups, label_set, rng):
    # Break subject grouping: assign each observation a random group id.
    g_sh = rng.permutation(groups)
    return _evaluate_config(X, y, g_sh, label_set)


def _summ(rows, key):
    vals = [r[key] for r in rows if np.isfinite(r[key])]
    if not vals:
        return float("nan"), (float("nan"), float("nan"))
    lo, hi = cr.bootstrap_ci(vals)
    return float(np.mean(vals)), (lo, hi)


def main() -> int:
    try:
        data = cr.load_ready9()
    except Exception as e:
        (cfg.OUT_DIR / "MECHANISM_ABLATION_ERROR.md").write_text(
            f"# Mechanism ablation could not run\n\nInput load failed: {e}\n")
        print(f"[ablation] FAILED to load inputs: {e}", file=sys.stderr)
        return 1

    blocks, y = data["blocks"], data["y"]
    groups = data["subjects"]
    label_set = sorted(set(y.tolist()))
    chance = 1.0 / len(label_set)

    metric_rows, summary_rows, conf_mats = [], [], {}
    configs = cr.get_affective_configs(blocks)

    for cfg_id, name, X in configs:
        rows, cm = _evaluate_config(X, y, groups, label_set)
        for r in rows:
            r.update({"config": cfg_id, "config_name": name,
                      "feature_dim": int(X.shape[1])})
            metric_rows.append(r)
        conf_mats[cfg_id] = cm
        ba, (ba_lo, ba_hi) = _summ(rows, "balanced_accuracy")
        f1, _ = _summ(rows, "macro_f1")
        auc, _ = _summ(rows, "roc_auc")
        summary_rows.append({
            "config": cfg_id, "config_name": name,
            "neural_mechanism": MECHANISM.get(cfg_id, name),
            "feature_dim": int(X.shape[1]),
            "balanced_accuracy_mean": round(ba, 4),
            "balanced_accuracy_ci_lo": round(ba_lo, 4),
            "balanced_accuracy_ci_hi": round(ba_hi, 4),
            "macro_f1_mean": round(f1, 4),
            "roc_auc_mean": round(auc, 4),
            "chance": round(chance, 4),
            "above_chance": bool(ba_lo > chance),
        })

    # Negative controls on the full substrate (A8).
    X_full = dict((c, X) for c, n, X in configs)["A8"]
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    for ctrl_name, fn in (("CTRL_shuffled_label", _shuffle_label_control),
                          ("CTRL_shuffled_subject", _shuffle_subject_control)):
        rows, cm = fn(X_full, y, groups, label_set, rng)
        for r in rows:
            r.update({"config": ctrl_name, "config_name": ctrl_name,
                      "feature_dim": int(X_full.shape[1])})
            metric_rows.append(r)
        conf_mats[ctrl_name] = cm
        ba, (ba_lo, ba_hi) = _summ(rows, "balanced_accuracy")
        summary_rows.append({
            "config": ctrl_name, "config_name": ctrl_name,
            "neural_mechanism": "negative control",
            "feature_dim": int(X_full.shape[1]),
            "balanced_accuracy_mean": round(ba, 4),
            "balanced_accuracy_ci_lo": round(ba_lo, 4),
            "balanced_accuracy_ci_hi": round(ba_hi, 4),
            "macro_f1_mean": round(_summ(rows, "macro_f1")[0], 4),
            "roc_auc_mean": round(_summ(rows, "roc_auc")[0], 4),
            "chance": round(chance, 4),
            "above_chance": bool(ba_lo > chance),
        })

    # Exploratory clinical-label sensitivity (validation/context only).
    clinical_rows = _clinical_sensitivity(blocks, data["subjects"], data["clinical_df"])

    # ── Write outputs ───────────────────────────────────────────────────
    cr.write_csv(cfg.ANALYSIS_DIR / "mechanism_ablation_metrics.csv", metric_rows)
    cr.write_csv(cfg.ANALYSIS_DIR / "mechanism_ablation_summary.csv", summary_rows)
    cr.write_csv(cfg.ANALYSIS_DIR / "clinical_label_sensitivity.csv", clinical_rows)
    cr.write_json(cfg.ANALYSIS_DIR / "mechanism_ablation_confusion_matrices.json", {
        "provenance": cr.base_provenance(),
        "labels": label_set,
        "label_names": cfg.AFFECTIVE_LABEL_NAMES,
        "confusion_matrices": conf_mats,
    })
    _write_table(summary_rows, chance)
    print(f"[ablation] {len(configs)} configs + 2 controls, "
          f"{len(cfg.SEEDS)} seeds x {cfg.N_FOLDS_AFFECTIVE} folds. "
          f"clinical labels evaluated: {len(clinical_rows)}")
    return 0


def _clinical_sensitivity(blocks, subjects, clinical_df):
    rows = []
    if clinical_df is None:
        return rows
    # Subject-averaged E+D+T for an exploratory sensitivity readout.
    X = np.hstack([blocks["E"], blocks["D"], blocks["T"]])
    X_subj, uniq = cr.subject_average_features(X, subjects)
    for dx in cfg.DIAGNOSES:
        mask, yb = cr.build_clinical_labels(clinical_df, uniq.tolist(), dx, min_per_class=15)
        if mask is None:
            rows.append({"diagnosis": dx, "status": "insufficient_class_support",
                         "balanced_accuracy_mean": float("nan"),
                         "roc_auc_mean": float("nan"), "n_pos": None, "n_neg": None})
            continue
        Xc = X_subj[mask]
        accs, aucs = [], []
        for seed in cfg.SEEDS:
            cv = cr.subject_grouped_cv(min(cfg.N_FOLDS_CLINICAL, int(min((yb == 0).sum(), (yb == 1).sum()))),
                                       seed)
            groups_c = np.arange(len(yb))  # one subject per row already
            for tr, te in cv.split(Xc, yb, groups=groups_c):
                pred, proba, classes = cr.fit_logreg_fold(Xc[tr], yb[tr], Xc[te], seed=seed)
                mb = cr.metric_bundle(yb[te], pred, proba, labels=[0, 1])
                accs.append(mb["balanced_accuracy"])
                if np.isfinite(mb["roc_auc"]):
                    aucs.append(mb["roc_auc"])
        rows.append({
            "diagnosis": dx, "status": "evaluated",
            "balanced_accuracy_mean": round(float(np.mean(accs)), 4) if accs else float("nan"),
            "roc_auc_mean": round(float(np.mean(aucs)), 4) if aucs else float("nan"),
            "n_pos": int((yb == 1).sum()), "n_neg": int((yb == 0).sum()),
            "interpretation": "exploratory clinical-label sensitivity; not diagnostic validation",
        })
    return rows


def _write_table(summary_rows, chance):
    lines = [
        r"% Auto-generated. Mechanism ablation: A0-A9 + negative controls.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Mechanism ablation under subject-grouped cross-validation. "
        r"Balanced accuracy (BA) and macro-F1 are reported as mean with a "
        r"bootstrap 95\% interval. Chance BA "
        rf"$=\,{chance:.3f}$. Each row reports the feature configuration, the "
        r"neural mechanism it carries, and its feature dimension.}",
        r"\label{tab:mechanism_ablation}",
        r"\begin{tabular}{l p{6.2cm} r r c r}",
        r"\toprule",
        r"Cfg & Mechanism & Dim & BA & 95\% CI & macro-F1 \\",
        r"\midrule",
    ]
    for r in summary_rows:
        mech = r["neural_mechanism"].replace("&", r"\&").replace("_", r"\_")
        cfg_id = r["config"].replace("_", r"\_")
        lines.append(
            f"{cfg_id} & {mech} & {r['feature_dim']} & "
            f"{r['balanced_accuracy_mean']:.3f} & "
            f"[{r['balanced_accuracy_ci_lo']:.3f}, {r['balanced_accuracy_ci_hi']:.3f}] & "
            f"{r['macro_f1_mean']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (cfg.TABLE_DIR / "table_mechanism_ablation.tex").write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
