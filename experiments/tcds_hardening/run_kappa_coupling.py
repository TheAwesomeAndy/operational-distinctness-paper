#!/usr/bin/env python3
"""TCDS hardening: structure-function coupling kappa = ||C||_F / sqrt(p*q).

Computes the per-observation 7x2 Spearman coupling matrix between the
dynamical descriptor block D (per-electrode, q=7 features) and the
topological block T (per-electrode, p=2 features), then the Frobenius-
norm-normalized scalar coupling

    kappa_{s,c} = || C_{s,c} ||_F / sqrt(p * q)        (q=7, p=2  -> sqrt(14))

This realizes the dissertation's structure-function coupling readout
(Chapter 7, Eq. 7.4) as a single per-observation interpretable scalar
without re-deriving any feature blocks.

Three statistical tests:

  (1) Per-observation electrode-permutation null (B=5,000): shuffles
      electrode order in T while holding D fixed, computes kappa_perm,
      reports a one-sided p-value (kappa_obs >= kappa_perm) per
      observation.

  (2) Per-condition aggregate: paired Wilcoxon signed-rank tests across
      subjects (Negative vs Neutral, Negative vs Pleasant, Neutral vs
      Pleasant), with subject-mean kappa.

  (3) Per-diagnosis aggregate: Mann-Whitney U positive vs negative for
      each of the five clinical labels, BH-FDR adjusted across the five
      tests.

A fourth descriptive analysis correlates per-subject mean kappa with
the corresponding subject's correctness on ARSPI-Net configuration A6
(E + D), using the per-fold predictions in
``outputs/operational_distinctness/affective_predictions.csv``.

Privacy: all per-observation rows use ``subject_hash`` (16-char SHA-256
prefix) only. No raw subject IDs are written. Aggregate diagnostic
counts only.

Inputs:
    data/shape_features_211.pkl       (X_ds, lsm_bsc6_pca, y, subjects)
    data/ch6_ch7_3class_features.pkl  (D, T_topo, ...)
    data/clinical_profile.csv         (5 binary diagnosis columns)

Outputs (in outputs/tcds_hardening/ and figures/tcds_hardening/):
    kappa_per_observation.csv      observation-level kappa + permutation p
    kappa_summary.csv              per-condition + per-diagnosis stats
    kappa_diagnostics.json         input-hash, runtime, parameter trail
    fig_kappa_distributions.pdf    4-panel figure
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.operational_distinctness import config as cfg
from experiments.operational_distinctness.common import (
    _spearman_coupling,
    build_clinical_labels,
    hash_subject_array,
    load_inputs,
)

OUT_DIR = REPO_ROOT / "outputs" / "tcds_hardening"
FIG_DIR = REPO_ROOT / "figures" / "tcds_hardening"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CONDITION_NAMES = {0: "Negative", 1: "Neutral", 2: "Pleasant"}


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty_like(adj)
    out[order] = adj
    return out


def kappa_with_permutation_null(D_obs: np.ndarray,
                                T_obs: np.ndarray,
                                n_perm: int,
                                rng: np.random.Generator
                                ) -> tuple[float, float, float]:
    """Returns (kappa_obs, p_value, kappa_null_mean)."""
    n_elec, q = D_obs.shape
    _, p = T_obs.shape
    norm = float(np.sqrt(p * q))

    C_obs = _spearman_coupling(D_obs, T_obs)
    kappa_obs = float(np.linalg.norm(C_obs, "fro") / norm)

    null = np.empty(n_perm)
    for k in range(n_perm):
        idx = rng.permutation(n_elec)
        C_perm = _spearman_coupling(D_obs, T_obs[idx])
        null[k] = np.linalg.norm(C_perm, "fro") / norm

    p_value = float((np.sum(null >= kappa_obs) + 1) / (n_perm + 1))
    return kappa_obs, p_value, float(null.mean())


def per_subject_a6_correctness() -> pd.DataFrame | None:
    pred_path = REPO_ROOT / "outputs" / "operational_distinctness" / "affective_predictions.csv"
    if not pred_path.exists():
        return None
    df = pd.read_csv(pred_path)
    a6 = df[df["config_id"] == "A6"].copy()
    if a6.empty:
        return None
    a6["correct"] = (a6["y_true"] == a6["y_pred"]).astype(int)
    return a6.groupby("subject_hash", as_index=False)["correct"].mean().rename(
        columns={"correct": "a6_per_subject_acc"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-perm", type=int, default=5000,
                        help="Electrode-permutations per observation (default 5000).")
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_STATE)
    args = parser.parse_args()

    print("=" * 72)
    print("Tier-1.4: structure-function coupling kappa")
    print("=" * 72)
    t0 = time.time()

    ch5, ch67, clinical_df = load_inputs()
    y = np.asarray(ch5["y"]).astype(int)
    subjects = np.asarray(ch5["subjects"])
    subj_hash = hash_subject_array(subjects)
    D_perch = np.asarray(ch67["D"])         # (N, 34, 7)
    T_perch = np.asarray(ch67["T_topo"])    # (N, 34, 2)
    n_obs, n_elec, q = D_perch.shape
    p = T_perch.shape[2]
    norm = float(np.sqrt(p * q))
    print(f"  N_obs={n_obs}  N_electrodes={n_elec}  q(D)={q}  p(T)={p}  "
          f"normalizer sqrt(p*q)={norm:.4f}")
    print(f"  permutations per observation: {args.n_perm}")

    # ── (1) per-observation kappa + permutation null ───────────────────
    rng = np.random.default_rng(args.seed)
    kappas = np.empty(n_obs)
    pvals = np.empty(n_obs)
    null_means = np.empty(n_obs)
    print("\n  Computing per-observation kappa + electrode-shuffle null ...")
    for i in range(n_obs):
        kappas[i], pvals[i], null_means[i] = kappa_with_permutation_null(
            D_perch[i], T_perch[i], args.n_perm, rng,
        )
        if (i + 1) % 50 == 0 or i == n_obs - 1:
            print(f"    [{i + 1:4d}/{n_obs}]  kappa_mean_so_far={kappas[:i + 1].mean():.4f}")

    obs_df = pd.DataFrame({
        "subject_hash": subj_hash,
        "condition": [CONDITION_NAMES[int(c)] for c in y],
        "y": y,
        "kappa": kappas,
        "kappa_null_mean": null_means,
        "p_value": pvals,
    })
    obs_df["reject_at_0p05"] = obs_df["p_value"] < 0.05
    obs_path = OUT_DIR / "kappa_per_observation.csv"
    obs_df.to_csv(obs_path, index=False)
    print(f"  Wrote {obs_path}  ({len(obs_df)} rows)")
    print(f"  per-observation kappa: mean={kappas.mean():.4f}  "
          f"sd={kappas.std(ddof=1):.4f}  range=[{kappas.min():.4f}, {kappas.max():.4f}]")
    print(f"  null-rejection rate at p<0.05: {obs_df['reject_at_0p05'].mean():.3f}")

    # ── (2) per-condition aggregate (paired Wilcoxon over subjects) ────
    summary_rows = []
    pivot = (obs_df.assign(subject_hash=subj_hash)
             .pivot_table(index="subject_hash", columns="condition", values="kappa",
                          aggfunc="mean"))
    for c in ("Negative", "Neutral", "Pleasant"):
        if c in pivot.columns:
            v = pivot[c].dropna().values
            summary_rows.append({
                "scope": "condition_summary",
                "group": c,
                "n": int(v.size),
                "kappa_mean": float(v.mean()),
                "kappa_sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                "p_value": np.nan,
                "p_value_bh_fdr": np.nan,
                "test": "descriptive",
            })

    pairs = [("Negative", "Neutral"),
             ("Negative", "Pleasant"),
             ("Neutral", "Pleasant")]
    for a, b in pairs:
        if a in pivot.columns and b in pivot.columns:
            both = pivot[[a, b]].dropna()
            try:
                stat, pv = stats.wilcoxon(both[a].values, both[b].values,
                                          zero_method="wilcox", alternative="two-sided")
                pv = float(pv)
            except Exception:
                pv = np.nan
            summary_rows.append({
                "scope": "condition_paired_wilcoxon",
                "group": f"{a} vs {b}",
                "n": int(len(both)),
                "kappa_mean": float(both[a].mean() - both[b].mean()),
                "kappa_sd": float((both[a] - both[b]).std(ddof=1))
                              if len(both) > 1 else 0.0,
                "p_value": pv,
                "p_value_bh_fdr": np.nan,
                "test": "wilcoxon_signed_rank",
            })

    # ── (3) per-diagnosis aggregate (Mann-Whitney U + BH-FDR) ───────────
    if clinical_df is not None:
        unique_subjects = np.array(sorted(set(subjects.tolist())))
        subj_kappa = (pd.DataFrame({"subject": subjects, "kappa": kappas})
                      .groupby("subject", as_index=False)["kappa"]
                      .mean())
        subj_kappa = subj_kappa.set_index("subject").reindex(unique_subjects).reset_index()

        diag_rows = []
        diag_pvals = []
        diag_keys = []
        for diagnosis in cfg.DIAGNOSES:
            mask, y_diag = build_clinical_labels(
                clinical_df, unique_subjects.tolist(), diagnosis,
            )
            if mask is None:
                summary_rows.append({
                    "scope": "diagnosis_mannwhitney",
                    "group": diagnosis,
                    "n": 0,
                    "kappa_mean": np.nan,
                    "kappa_sd": np.nan,
                    "p_value": np.nan,
                    "p_value_bh_fdr": np.nan,
                    "test": "skipped_insufficient_labels",
                })
                continue
            ks = subj_kappa.loc[mask, "kappa"].values
            kp = ks[y_diag == 1]
            kn = ks[y_diag == 0]
            try:
                _, pv = stats.mannwhitneyu(kp, kn, alternative="two-sided")
                pv = float(pv)
            except Exception:
                pv = np.nan
            row = {
                "scope": "diagnosis_mannwhitney",
                "group": diagnosis,
                "n": int(kp.size + kn.size),
                "kappa_mean": float(kp.mean() - kn.mean()),
                "kappa_sd": float(np.sqrt(kp.var(ddof=1) / max(kp.size, 1)
                                          + kn.var(ddof=1) / max(kn.size, 1))),
                "p_value": pv,
                "p_value_bh_fdr": np.nan,
                "test": "mannwhitney_u",
            }
            diag_rows.append(row)
            diag_pvals.append(pv)
            diag_keys.append(diagnosis)

        if diag_pvals:
            adj = _bh_fdr(np.asarray(diag_pvals))
            for row, padj in zip(diag_rows, adj):
                row["p_value_bh_fdr"] = float(padj)
            summary_rows.extend(diag_rows)

    # ── (4) descriptive correlation kappa vs ARSPI A6 accuracy ──────────
    a6_acc = per_subject_a6_correctness()
    corr_value: float | None = None
    corr_p: float | None = None
    if a6_acc is not None:
        kappa_subj = (pd.DataFrame({"subject_hash": subj_hash, "kappa": kappas})
                      .groupby("subject_hash", as_index=False)["kappa"].mean())
        merged = kappa_subj.merge(a6_acc, on="subject_hash", how="inner")
        if len(merged) >= 10:
            corr_value, corr_p = stats.spearmanr(merged["kappa"], merged["a6_per_subject_acc"])
            corr_value = float(corr_value)
            corr_p = float(corr_p)
            summary_rows.append({
                "scope": "kappa_vs_arspi_a6_accuracy",
                "group": "all_subjects",
                "n": int(len(merged)),
                "kappa_mean": np.nan,
                "kappa_sd": np.nan,
                "p_value": corr_p,
                "p_value_bh_fdr": np.nan,
                "test": f"spearman rho={corr_value:.4f}",
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "kappa_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Wrote {summary_path}  ({len(summary_df)} rows)")

    # ── (5) figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))

    # Panel A: violin per condition
    ax = axes[0]
    cond_data = [obs_df.loc[obs_df["condition"] == c, "kappa"].values
                 for c in ("Negative", "Neutral", "Pleasant")]
    parts = ax.violinplot(cond_data, showmeans=True, showmedians=True)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Negative", "Neutral", "Pleasant"])
    ax.set_ylabel(r"$\kappa = \|C\|_F / \sqrt{p\,q}$")
    ax.set_title("(a) per-condition $\\kappa$")
    ax.grid(axis="y", alpha=0.3)

    # Panel B: per-diagnosis pos vs neg means
    ax = axes[1]
    if clinical_df is not None:
        diag_summary = summary_df[summary_df["scope"] == "diagnosis_mannwhitney"]
        diag_summary = diag_summary[diag_summary["test"] == "mannwhitney_u"]
        if not diag_summary.empty:
            xs = np.arange(len(diag_summary))
            means = diag_summary["kappa_mean"].values
            errs = diag_summary["kappa_sd"].values
            ax.bar(xs, means, yerr=errs, capsize=4,
                   color=["#d95f02" if p < 0.05 else "#1b9e77"
                          for p in diag_summary["p_value"].fillna(1).values])
            ax.set_xticks(xs)
            ax.set_xticklabels(diag_summary["group"].values, rotation=20)
            ax.axhline(0, color="black", lw=0.6)
            ax.set_ylabel(r"$\Delta\kappa$ (positive $-$ negative)")
            ax.set_title("(b) per-diagnosis $\\kappa$ contrast")
            ax.grid(axis="y", alpha=0.3)
    else:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no clinical metadata",
                ha="center", va="center", transform=ax.transAxes)

    # Panel C: kappa vs A6 accuracy scatter
    ax = axes[2]
    if a6_acc is not None and corr_value is not None:
        kappa_subj = (pd.DataFrame({"subject_hash": subj_hash, "kappa": kappas})
                      .groupby("subject_hash", as_index=False)["kappa"].mean())
        merged = kappa_subj.merge(a6_acc, on="subject_hash", how="inner")
        ax.scatter(merged["kappa"], merged["a6_per_subject_acc"],
                   s=18, alpha=0.55, color="#7570b3", edgecolor="white", linewidth=0.4)
        if len(merged) > 2:
            slope, intercept = np.polyfit(merged["kappa"], merged["a6_per_subject_acc"], 1)
            xs = np.linspace(merged["kappa"].min(), merged["kappa"].max(), 100)
            ax.plot(xs, slope * xs + intercept, "k--", lw=1, alpha=0.6)
        ax.set_xlabel(r"per-subject mean $\kappa$")
        ax.set_ylabel("A6 (E+D) per-subject accuracy")
        ax.set_title(f"(c) $\\rho={corr_value:.3f}, p={corr_p:.3g}$")
        ax.grid(alpha=0.3)
    else:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "predictions unavailable",
                ha="center", va="center", transform=ax.transAxes)

    # Panel D: observation-level kappa vs null-mean
    ax = axes[3]
    ax.scatter(null_means, kappas, s=10, alpha=0.4, color="#666666", edgecolor="none")
    diag_lo = min(null_means.min(), kappas.min())
    diag_hi = max(null_means.max(), kappas.max())
    ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], "r--", lw=1, label="$y=x$")
    sig_frac = float(obs_df["reject_at_0p05"].mean())
    ax.set_xlabel(r"electrode-shuffled null mean $\kappa$")
    ax.set_ylabel(r"observed $\kappa$")
    ax.set_title(f"(d) obs vs null  (sig@0.05: {sig_frac:.0%})")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig_path = FIG_DIR / "fig_kappa_distributions.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {fig_path}")

    # ── diagnostics ─────────────────────────────────────────────────────
    files_for_hash = [cfg.CH5_FILE, cfg.CH67_FILE]
    h = hashlib.sha256()
    for f in files_for_hash:
        h.update(str(f).encode())
        h.update(str(f.stat().st_size).encode())
    diagnostics = {
        "script": "experiments/tcds_hardening/run_kappa_coupling.py",
        "n_observations": int(n_obs),
        "n_subjects": int(np.unique(subjects).size),
        "n_electrodes": int(n_elec),
        "q_features_D": int(q),
        "p_features_T": int(p),
        "kappa_normalizer_sqrt_pq": norm,
        "n_permutations_per_observation": int(args.n_perm),
        "kappa_obs_mean": float(kappas.mean()),
        "kappa_obs_sd": float(kappas.std(ddof=1)),
        "rejection_rate_at_p0p05": float(obs_df["reject_at_0p05"].mean()),
        "kappa_vs_a6_spearman_rho": corr_value,
        "kappa_vs_a6_spearman_p": corr_p,
        "input_file_size_hash_sha256": h.hexdigest()[:32],
        "runtime_seconds": float(time.time() - t0),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": stats.__name__ and __import__("scipy").__version__,
        "pandas_version": pd.__version__,
    }
    diag_path = OUT_DIR / "kappa_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"  Wrote {diag_path}")

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
