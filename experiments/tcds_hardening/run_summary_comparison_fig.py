#!/usr/bin/env python3
"""TCDS hardening: headline summary comparison figure.

Pulls already-computed metrics from the canonical CSVs and produces a
single-panel-with-grouped-bars figure that lets the reader see at a
glance how the brain-inspired ARSPI-Net combinations compare to the
classical ERP baseline AND to the BandPower control under matched
cross-validation.

Pure post-processing: no new model fitting.

Inputs (must already exist):
    outputs/operational_distinctness/affective_ablation_metrics.csv
    outputs/tcds_hardening/erp_baseline_neutral_calibration_results.csv

Outputs:
    figures/tcds_hardening/fig_headline_comparison.pdf
    outputs/tcds_hardening/headline_comparison_table.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs" / "tcds_hardening"
FIG_DIR = REPO_ROOT / "figures" / "tcds_hardening"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    print("=" * 72)
    print("Tier-2.2: headline summary comparison figure")
    print("=" * 72)
    aff_path = REPO_ROOT / "outputs" / "operational_distinctness" / "affective_ablation_metrics.csv"
    erp_path = OUT_DIR / "erp_baseline_neutral_calibration_results.csv"
    if not aff_path.exists():
        print(f"  ERROR: missing {aff_path}", file=sys.stderr)
        return 1
    if not erp_path.exists():
        print(f"  ERROR: missing {erp_path}", file=sys.stderr)
        return 1

    aff = pd.read_csv(aff_path)
    erp = pd.read_csv(erp_path)
    print(f"  affective rows: {len(aff)}    erp rows: {len(erp)}")

    aff_subset = aff[aff["config_id"].isin(["A0", "A1", "A6", "A8", "A9"])][[
        "config_id", "feature_set", "balanced_accuracy_mean",
        "balanced_accuracy_ci95_low", "balanced_accuracy_ci95_high",
        "macro_roc_auc_ovr_mean",
    ]].copy()
    aff_subset["task"] = "3-class (Neg/Neu/Pos)"
    aff_subset["family"] = "ARSPI-Net" + np.where(aff_subset["config_id"] == "A0",
                                                  " baseline (BP)", "")
    aff_subset["label"] = aff_subset["config_id"] + ": " + aff_subset["feature_set"]

    erp = erp.rename(columns={"roc_auc_mean": "macro_roc_auc_ovr_mean"})
    # ERP CSV does not carry bootstrap CIs; approximate by mean +/- SD.
    erp["balanced_accuracy_ci95_low"] = (erp["balanced_accuracy_mean"]
                                          - erp["balanced_accuracy_sd"])
    erp["balanced_accuracy_ci95_high"] = (erp["balanced_accuracy_mean"]
                                           + erp["balanced_accuracy_sd"])
    erp_subset = erp[["config_id", "balanced_accuracy_mean",
                      "balanced_accuracy_ci95_low", "balanced_accuracy_ci95_high",
                      "macro_roc_auc_ovr_mean"]].copy()
    erp_subset["feature_set"] = erp_subset["config_id"].map({
        "raw_3class": "ERP raw (3-class)",
        "raw_2class_NegVsPos": "ERP raw (Neg vs Pos)",
        "calibrated_2class_NegVsPos": "ERP neutral-anchored (Neg vs Pos)",
    }).fillna(erp_subset["config_id"])
    erp_subset["task"] = np.where(erp_subset["config_id"].str.contains("3class"),
                                  "3-class (Neg/Neu/Pos)",
                                  "2-class (Neg vs Pos)")
    erp_subset["family"] = "Classical ERP"
    erp_subset["label"] = erp_subset["feature_set"]

    summary = pd.concat([aff_subset, erp_subset], ignore_index=True, sort=False)
    summary = summary[["task", "family", "label", "config_id",
                        "balanced_accuracy_mean",
                        "balanced_accuracy_ci95_low",
                        "balanced_accuracy_ci95_high",
                        "macro_roc_auc_ovr_mean"]]
    summary_path = OUT_DIR / "headline_comparison_table.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  Wrote {summary_path}  ({len(summary)} rows)")

    # ── figure ──────────────────────────────────────────────────────────
    family_color = {
        "ARSPI-Net": "#1b9e77",
        "ARSPI-Net baseline (BP)": "#a6cee3",
        "Classical ERP": "#d95f02",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    for ax_idx, task in enumerate(("3-class (Neg/Neu/Pos)", "2-class (Neg vs Pos)")):
        ax = axes[ax_idx]
        sub = summary[summary["task"] == task].copy()
        sub = sub.sort_values("balanced_accuracy_mean")
        if sub.empty:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"no rows for {task}", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        ys = np.arange(len(sub))
        means = sub["balanced_accuracy_mean"].values
        lo = sub["balanced_accuracy_ci95_low"].values
        hi = sub["balanced_accuracy_ci95_high"].values
        # ERP rows often lack CIs in the source CSV column order; fall back to symmetric
        lo = np.where(np.isnan(lo), means, lo)
        hi = np.where(np.isnan(hi), means, hi)
        err_lo = means - lo
        err_hi = hi - means
        colors = [family_color.get(f, "#999999") for f in sub["family"]]
        ax.barh(ys, means, xerr=[err_lo, err_hi], color=colors,
                edgecolor="black", linewidth=0.6, capsize=3)
        chance = 1.0 / (3 if "3-class" in task else 2)
        ax.axvline(chance, color="red", ls="--", lw=0.9, label=f"chance ({chance:.2f})")
        ax.set_yticks(ys)
        ax.set_yticklabels(sub["label"].values)
        ax.set_xlim(0, max(0.85, hi.max() + 0.05))
        ax.set_xlabel("balanced accuracy (5-/10-fold subject-grouped CV)")
        ax.set_title(task)
        # AUC annotations
        for y, m, auc in zip(ys, means, sub["macro_roc_auc_ovr_mean"].values):
            if not np.isnan(auc):
                ax.text(min(m + max(err_hi[int(y)], 0.0) + 0.018, 0.93), y,
                        f"AUC {auc:.3f}", va="center", fontsize=8, color="#333333")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Headline comparison: ARSPI-Net layers vs. classical ERP baseline",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    fig_path = FIG_DIR / "fig_headline_comparison.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
