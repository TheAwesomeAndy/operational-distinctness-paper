#!/usr/bin/env python3
"""TCDS hardening: classical ERP-feature baseline + neutral-calibration experiment.

Computes a conventional ERP-amplitude feature representation directly from
the raw, baseline-corrected SHAPE EEG files (1024 Hz, 1 s post-stimulus),
then evaluates three classification configurations under subject-grouped
5-fold CV:

  raw_3class         : 3-class affective classification (Negative / Neutral / Pleasant).
  raw_2class         : 2-class arousal contrast (Negative vs. Pleasant), neutral excluded.
  calibrated_2class  : 2-class arousal contrast where each subject's neutral
                       trial vector is subtracted from their Negative and Pleasant
                       feature vectors before scaling. This implements the
                       conventional "neutral as anchor" calibration in IAPS ERP work.

Inputs:
    data/batch_data_full/SHAPE_Community_<sid>_IAPS{Neg|Neu|Pos}_BC.txt
    1229 timepoints x 34 channels, baseline-corrected microvolts, 1024 Hz,
    rows 0..204 = pre-stimulus baseline (already subtracted).

Outputs (in outputs/tcds_hardening/):
    erp_baseline_neutral_calibration_results.csv
    diagnostics.json
    table_erp_neutral_calibration.tex

Privacy: nothing identifying is written. The diagnostics record only:
  * the observation count and label distribution,
  * a SHA-256 of the sorted file-list (so file-set integrity is verifiable
    without exposing per-subject IDs),
  * per-feature-block descriptive statistics (no per-subject values).

Usage:
    python -u experiments/tcds_hardening/run_erp_baseline.py \
        [--data-dir data/batch_data_full] \
        [--out-dir  outputs/tcds_hardening]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Acquisition + epoching constants (match data spec) ────────────────────
FS_HZ = 1024
N_CHANNELS = 34
N_SAMPLES_TOTAL = 1229
PRE_STIM_SAMPLES = 205          # rows 0..204, 200 ms baseline (already subtracted)
POST_STIM_SAMPLES = N_SAMPLES_TOTAL - PRE_STIM_SAMPLES   # 1024 samples = 1000 ms
EXCLUDED_SUBJECTS = {127}

CONDITION_MAP = {"IAPSNeg": 0, "IAPSNeu": 1, "IAPSPos": 2}
CONDITION_NAMES = {0: "Negative", 1: "Neutral", 2: "Pleasant"}

# ── ERP windows (ms post-stimulus) ────────────────────────────────────────
ERP_WINDOWS_MS = [
    ("P1",        80,  130),
    ("N1",       130,  200),
    ("P2",       200,  300),
    ("P3",       300,  450),
    ("LPP_early", 450, 700),
    ("LPP_late",  700, 1000),
]

# ── CV / classifier ───────────────────────────────────────────────────────
N_FOLDS = 5
RANDOM_STATE = 42

FILENAME_RE = re.compile(
    r"SHAPE_Community_(?P<sid>\d+)_(?P<cond>IAPSNeg|IAPSNeu|IAPSPos)_BC\.txt$"
)


def discover_files(data_dir: Path) -> list[tuple[Path, int, int]]:
    """Return [(path, subject_index, condition_label), ...] excluding EXCLUDED_SUBJECTS."""
    rows: list[tuple[Path, int, int]] = []
    for p in sorted(data_dir.iterdir()):
        m = FILENAME_RE.match(p.name)
        if not m:
            continue
        sid = int(m.group("sid"))
        if sid in EXCLUDED_SUBJECTS:
            continue
        rows.append((p, sid, CONDITION_MAP[m.group("cond")]))
    return rows


def file_list_sha256(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def ms_to_postsample(t_ms: float) -> int:
    """Convert a time in ms post-stimulus to an absolute sample index in the
    1229-sample recording, where post-stim begins at PRE_STIM_SAMPLES."""
    return int(round(PRE_STIM_SAMPLES + t_ms * FS_HZ / 1000.0))


def compute_window_indices() -> list[tuple[str, int, int, int, int]]:
    """Return [(name, t_start_ms, t_end_ms, sample_start, sample_end), ...]."""
    out: list[tuple[str, int, int, int, int]] = []
    for name, t0, t1 in ERP_WINDOWS_MS:
        s0 = ms_to_postsample(t0)
        s1 = ms_to_postsample(t1)
        s1 = min(s1, N_SAMPLES_TOTAL)
        out.append((name, t0, t1, s0, s1))
    return out


def extract_erp_features(files: list[tuple[Path, int, int]],
                         windows: list[tuple[str, int, int, int, int]]
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """For each observation, compute per-channel mean amplitude in each ERP window.

    Returns (X, y, subjects, feature_names).
    X has shape (N_obs, N_CHANNELS * len(windows)).
    """
    n_obs = len(files)
    n_feat = N_CHANNELS * len(windows)
    X = np.zeros((n_obs, n_feat), dtype=np.float64)
    y = np.zeros(n_obs, dtype=np.int64)
    subjects = np.zeros(n_obs, dtype=np.int64)
    feature_names = [f"ch{ch:02d}_{w[0]}" for w in windows for ch in range(N_CHANNELS)]

    t0 = time.time()
    for i, (path, sid, cond) in enumerate(files):
        raw = np.loadtxt(path)  # (1229, 34) microvolts, baseline-corrected
        if raw.shape != (N_SAMPLES_TOTAL, N_CHANNELS):
            raise ValueError(f"Unexpected shape {raw.shape} in {path.name}")
        feats = []
        for _name, _t0_ms, _t1_ms, s0, s1 in windows:
            feats.append(raw[s0:s1].mean(axis=0))   # (34,)
        X[i] = np.concatenate(feats)
        y[i] = cond
        subjects[i] = sid
        if (i + 1) % 100 == 0 or i == n_obs - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            print(f"  loaded {i + 1:4d}/{n_obs}  ({rate:.1f} files/s, "
                  f"elapsed {elapsed:.0f}s)")
    return X, y, subjects, feature_names


def neutral_calibrate(X: np.ndarray, y: np.ndarray, subjects: np.ndarray
                      ) -> np.ndarray:
    """Return X with each subject's neutral-trial feature vector subtracted from
    every observation belonging to that subject. If a subject has no neutral
    trial, no calibration is applied for that subject."""
    X_out = X.copy()
    for sid in np.unique(subjects):
        sub_mask = subjects == sid
        neu_mask = sub_mask & (y == CONDITION_MAP["IAPSNeu"])
        if not neu_mask.any():
            continue
        ref = X[neu_mask].mean(axis=0)
        X_out[sub_mask] = X_out[sub_mask] - ref
    return X_out


def _make_classifier() -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )


def evaluate(config_id: str, X: np.ndarray, y: np.ndarray,
             subjects: np.ndarray, n_classes: int, label_names: list[str]) -> dict:
    cv = StratifiedGroupKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    fold_bal_acc, fold_acc, fold_auc, fold_f1 = [], [], [], []
    n_eval = 0
    t0 = time.time()
    for fold_idx, (tr, te) in enumerate(cv.split(X, y, groups=subjects)):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        clf = _make_classifier()
        clf.fit(Xtr, y[tr])
        y_pred = clf.predict(Xte)
        proba = clf.predict_proba(Xte)
        proba_full = np.zeros((Xte.shape[0], n_classes))
        for ci, cls in enumerate(clf.classes_):
            proba_full[:, int(cls)] = proba[:, ci]

        fold_bal_acc.append(balanced_accuracy_score(y[te], y_pred))
        fold_acc.append(accuracy_score(y[te], y_pred))
        fold_f1.append(f1_score(y[te], y_pred, average="macro", zero_division=0))
        try:
            if n_classes == 2:
                fold_auc.append(roc_auc_score(y[te], proba_full[:, 1]))
            else:
                fold_auc.append(roc_auc_score(
                    y[te], proba_full, multi_class="ovr",
                    average="macro", labels=list(range(n_classes))))
        except ValueError:
            fold_auc.append(float("nan"))
        n_eval += int(np.sum(np.isin(np.arange(len(y)), te)))

    elapsed = time.time() - t0
    bal = np.array(fold_bal_acc)
    acc = np.array(fold_acc)
    auc = np.array(fold_auc)
    f1 = np.array(fold_f1)
    return {
        "config_id": config_id,
        "n_classes": n_classes,
        "labels": ",".join(label_names),
        "n_observations": int(X.shape[0]),
        "n_subjects": int(np.unique(subjects).size),
        "n_features": int(X.shape[1]),
        "n_folds": N_FOLDS,
        "balanced_accuracy_mean": float(np.mean(bal)),
        "balanced_accuracy_sd": float(np.std(bal, ddof=1)) if bal.size > 1 else 0.0,
        "accuracy_mean": float(np.mean(acc)),
        "accuracy_sd": float(np.std(acc, ddof=1)) if acc.size > 1 else 0.0,
        "macro_f1_mean": float(np.mean(f1)),
        "macro_f1_sd": float(np.std(f1, ddof=1)) if f1.size > 1 else 0.0,
        "roc_auc_mean": float(np.nanmean(auc)),
        "roc_auc_sd": float(np.nanstd(auc, ddof=1)) if auc.size > 1 else 0.0,
        "fold_balanced_accuracy_values": json.dumps([float(v) for v in bal]),
        "fold_roc_auc_values": json.dumps([float(v) for v in auc]),
        "elapsed_seconds": float(round(elapsed, 3)),
    }


def feature_block_stats(name: str, X: np.ndarray) -> dict:
    finite = X[np.isfinite(X)]
    return {
        "block": name,
        "shape": list(X.shape),
        "finite_fraction": float(np.isfinite(X).mean()),
        "mean": float(finite.mean()) if finite.size else None,
        "std": float(finite.std()) if finite.size else None,
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "row_norm_min": float(np.linalg.norm(X, axis=1).min()) if X.size else None,
        "row_norm_median": float(np.median(np.linalg.norm(X, axis=1))) if X.size else None,
        "row_norm_max": float(np.linalg.norm(X, axis=1).max()) if X.size else None,
    }


def emit_latex_table(results: list[dict], path: Path) -> None:
    rows = []
    for r in results:
        cid = r["config_id"].replace("_", "\\_")
        rows.append(
            f"      {cid:20s} & "
            f"{r['n_classes']} & "
            f"{r['n_observations']:5d} & "
            f"{r['balanced_accuracy_mean']*100:6.2f} $\\pm$ {r['balanced_accuracy_sd']*100:.2f} & "
            f"{r['roc_auc_mean']:6.3f} $\\pm$ {r['roc_auc_sd']:.3f} & "
            f"{r['accuracy_mean']*100:6.2f} $\\pm$ {r['accuracy_sd']*100:.2f} \\\\\n"
        )
    body = "".join(rows)
    tex = (
        "\\begin{table}[t]\n"
        "  \\centering\n"
        "  \\caption{Classical ERP-feature baseline and neutral-calibration "
        "experiment. Subject-grouped 5-fold CV; logistic regression on per-channel "
        "mean amplitudes in six standard windows (P1, N1, P2, P3, early-LPP, late-LPP).}\n"
        "  \\label{tab:erp_neutral_calibration}\n"
        "  \\begin{tabular}{lrrlll}\n"
        "    \\toprule\n"
        "      Configuration & K & N & Balanced acc.\\ (\\%) & ROC-AUC & Accuracy (\\%) \\\\\n"
        "    \\midrule\n"
        f"{body}"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )
    path.write_text(tex, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/batch_data_full"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/tcds_hardening"))
    args = parser.parse_args()

    data_dir: Path = args.data_dir.resolve()
    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TCDS hardening: ERP baseline + neutral calibration")
    print("=" * 70)
    print(f"Data dir: {data_dir}")
    print(f"Out dir:  {out_dir}")

    files = discover_files(data_dir)
    if not files:
        raise SystemExit(f"No matching .txt files in {data_dir}")
    paths = [p for p, _, _ in files]
    file_hash = file_list_sha256(paths)
    n_obs = len(files)
    n_subj = len({sid for _, sid, _ in files})
    label_counts = {int(c): sum(1 for _, _, lab in files if lab == c)
                    for c in sorted(CONDITION_NAMES)}
    print(f"\nDiscovered {n_obs} files, {n_subj} subjects "
          f"(excluded={sorted(EXCLUDED_SUBJECTS)})")
    print(f"  label counts: {label_counts}")
    print(f"  filename-list sha256: {file_hash}")

    windows = compute_window_indices()
    print(f"\nERP windows ({len(windows)}):")
    for name, t0, t1, s0, s1 in windows:
        print(f"  {name:9s}  {t0:4d}-{t1:4d} ms   samples [{s0}..{s1})  "
              f"width={s1 - s0} samples")

    print(f"\nLoading + extracting features from {n_obs} files ...")
    X, y, subjects, feat_names = extract_erp_features(files, windows)
    print(f"  X shape: {X.shape}   y shape: {y.shape}   subjects: {len(np.unique(subjects))}")

    # Calibrated features (neutral-anchor)
    X_calib = neutral_calibrate(X, y, subjects)

    # ── Run the three configurations ─────────────────────────────────────
    print(f"\nRunning {N_FOLDS}-fold CV (StratifiedGroupKFold over subjects) ...")
    results: list[dict] = []

    print("\n[1/3] raw_3class ...")
    results.append(evaluate(
        "raw_3class", X, y, subjects, n_classes=3,
        label_names=[CONDITION_NAMES[k] for k in sorted(CONDITION_NAMES)]))

    pos_neg_mask = y != CONDITION_MAP["IAPSNeu"]
    X_pn = X[pos_neg_mask]
    y_pn = (y[pos_neg_mask] == CONDITION_MAP["IAPSPos"]).astype(int)  # 0=Neg, 1=Pos
    subj_pn = subjects[pos_neg_mask]

    print("\n[2/3] raw_2class (Negative vs Pleasant, no calibration) ...")
    results.append(evaluate(
        "raw_2class_NegVsPos", X_pn, y_pn, subj_pn, n_classes=2,
        label_names=["Negative", "Pleasant"]))

    X_calib_pn = X_calib[pos_neg_mask]
    print("\n[3/3] calibrated_2class (per-subject neutral subtracted) ...")
    results.append(evaluate(
        "calibrated_2class_NegVsPos", X_calib_pn, y_pn, subj_pn, n_classes=2,
        label_names=["Negative", "Pleasant"]))

    # ── Outputs ──────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    csv_path = out_dir / "erp_baseline_neutral_calibration_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n-> {csv_path}")

    diagnostics = {
        "data_dir": str(data_dir),
        "n_files": n_obs,
        "n_subjects": n_subj,
        "excluded_subjects": sorted(EXCLUDED_SUBJECTS),
        "label_counts": label_counts,
        "filename_list_sha256": file_hash,
        "fs_hz": FS_HZ,
        "n_channels": N_CHANNELS,
        "n_samples_total": N_SAMPLES_TOTAL,
        "pre_stim_samples": PRE_STIM_SAMPLES,
        "post_stim_samples": POST_STIM_SAMPLES,
        "erp_windows_ms": [
            {"name": n, "t_start_ms": a, "t_end_ms": b,
             "sample_start": s0, "sample_end": s1}
            for n, a, b, s0, s1 in windows
        ],
        "n_features_per_observation": int(X.shape[1]),
        "feature_blocks": {
            "raw": feature_block_stats("raw", X),
            "calibrated": feature_block_stats("calibrated", X_calib),
            "raw_2class_subset": feature_block_stats("raw_2class_subset", X_pn),
            "calibrated_2class_subset": feature_block_stats(
                "calibrated_2class_subset", X_calib_pn),
        },
        "cv": {
            "scheme": "StratifiedGroupKFold",
            "n_splits": N_FOLDS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
            "group": "subject_index",
        },
        "classifier": {
            "estimator": "LogisticRegression",
            "C": 1.0,
            "solver": "lbfgs",
            "class_weight": "balanced",
            "max_iter": 2000,
            "random_state": RANDOM_STATE,
        },
        "preprocessing": "StandardScaler (per-fold fit on train, applied to test)",
        "results": results,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
        },
    }
    json_path = out_dir / "diagnostics.json"
    json_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(f"-> {json_path}")

    tex_path = out_dir / "table_erp_neutral_calibration.tex"
    emit_latex_table(results, tex_path)
    print(f"-> {tex_path}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for r in results:
        print(f"  {r['config_id']:30s}  K={r['n_classes']}  N={r['n_observations']:4d}  "
              f"BA={r['balanced_accuracy_mean']*100:5.2f}% +/- {r['balanced_accuracy_sd']*100:4.2f}  "
              f"AUC={r['roc_auc_mean']:.3f} +/- {r['roc_auc_sd']:.3f}  "
              f"({r['elapsed_seconds']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
