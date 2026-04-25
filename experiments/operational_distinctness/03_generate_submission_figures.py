#!/usr/bin/env python3
"""Generate clean vector figures for the Operational Distinctness manuscript.

All figures are written as PDFs into ``figures/operational_distinctness/``.

Inputs (from scripts 01 and 02):
    outputs/operational_distinctness/affective_ablation_metrics.csv
    outputs/operational_distinctness/clinical_sensitivity_metrics.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from . import config as cfg
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from experiments.operational_distinctness import config as cfg


# ─── Figure 1: ARSPI-Net layer decomposition schematic ─────────────────

def fig1_layer_decomposition() -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    boxes = [
        ("EEG / ERP\ninput",          0.25, "#dfe6e9"),
        ("LIF\nreservoir",            1.85, "#a4b0be"),
        ("E\nBSC6 / PCA-64\nembedding", 3.45, "#0984e3"),
        ("D\ndynamical\ndescriptors",   5.05, "#e17055"),
        ("T\ngraph-topological\ndescriptors", 6.65, "#00b894"),
        ("C\nstructure-function\ncoupling",   8.25, "#6c5ce7"),
    ]
    for text, x, color in boxes:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 1.5), 1.4, 1.6,
            boxstyle="round,pad=0.05",
            linewidth=1, edgecolor="black", facecolor=color, alpha=0.85,
        ))
        ax.text(x + 0.7, 2.3, text, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white" if color != "#dfe6e9" else "black")

    for i in range(len(boxes) - 1):
        x0 = boxes[i][1] + 1.4
        x1 = boxes[i + 1][1]
        ax.annotate("", xy=(x1, 2.3), xytext=(x0, 2.3),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#2d3436"))

    ax.text(5.0, 3.7, "ARSPI-Net staged feature decomposition",
            ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(5.0, 0.55,
            "Each downstream block is a measurement layer of the same reservoir state.",
            ha="center", va="center", fontsize=9, style="italic", color="#636e72")

    out = cfg.FIG_DIR / "fig1_arspinet_layer_decomposition.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── Figure 2: Operational distinctness framework ──────────────────────

def fig2_framework() -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    nodes = [
        ("Predictive\nsufficiency",          2.0, 7.5, "#0984e3"),
        ("Additive\nutility",                7.5, 7.5, "#00b894"),
        ("Clinical-label\nsensitivity",      2.0, 2.5, "#e17055"),
        ("Redundancy /\nnon-equivalence",    7.5, 2.5, "#6c5ce7"),
    ]
    for text, x, y, color in nodes:
        ax.add_patch(mpatches.Circle((x, y), 1.4, facecolor=color, alpha=0.85,
                                     edgecolor="black", linewidth=1.2))
        ax.text(x, y, text, ha="center", va="center", fontsize=10,
                fontweight="bold", color="white")

    ax.add_patch(mpatches.FancyBboxPatch(
        (3.7, 4.3), 2.6, 1.4,
        boxstyle="round,pad=0.1",
        linewidth=1.2, edgecolor="black", facecolor="#dfe6e9",
    ))
    ax.text(5.0, 5.0, "Operational\ndistinctness", ha="center", va="center",
            fontsize=11, fontweight="bold")

    for x, y, _, _ in [(n[1], n[2], n[0], n[3]) for n in nodes]:
        ax.annotate("", xy=(5.0, 5.0), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-", lw=1.0, color="#636e72", alpha=0.5))

    ax.text(5.0, 9.3, "Operational Distinctness Framework",
            ha="center", va="center", fontsize=13, fontweight="bold")

    out = cfg.FIG_DIR / "fig2_operational_distinctness_framework.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── Figure 3: Affective ablation bar plot ─────────────────────────────

def fig3_affective_ablation() -> Path | None:
    metrics_path = cfg.OUT_DIR / "affective_ablation_metrics.csv"
    if not metrics_path.exists():
        print(f"  [fig3] Skipped: {metrics_path} not found.")
        return None
    df = pd.read_csv(metrics_path)
    df = df.set_index("config_id").loc[cfg.AFFECTIVE_CONFIG_ORDER].reset_index()

    fig, ax = plt.subplots(figsize=(11, 5.5))

    accs = df["balanced_accuracy_mean"].values * 100
    ci_lo = df["balanced_accuracy_ci95_low"].values * 100
    ci_hi = df["balanced_accuracy_ci95_high"].values * 100
    err_lower = accs - ci_lo
    err_upper = ci_hi - accs

    base_color = "#74b9ff"
    colors = [base_color] * len(df)
    for i, cid in enumerate(df["config_id"]):
        if cid == "A1":
            colors[i] = "#0984e3"  # embedding baseline highlight
        elif cid == "A9":
            colors[i] = "#2d3436"  # full feature set highlight

    ax.bar(np.arange(len(df)), accs,
           yerr=[err_lower, err_upper], capsize=4,
           color=colors, edgecolor="white", linewidth=0.6)
    ax.axhline(33.3, color="red", linestyle="--", alpha=0.6, label="Chance (33.3%)")
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels([f"{cid}\n{fs}" for cid, fs in zip(df["config_id"], df["feature_set"])],
                        fontsize=8)
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_title("Affective ablation: balanced accuracy with 95% CI",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(20, max(70, accs.max() + 8))

    for i, acc in enumerate(accs):
        ax.text(i, acc + err_upper[i] + 0.5, f"{acc:.1f}",
                ha="center", va="bottom", fontsize=8)

    legend_handles = [
        mpatches.Patch(color="#0984e3", label="A1 (embedding baseline)"),
        mpatches.Patch(color="#2d3436", label="A9 (full E+D+T+C)"),
        mpatches.Patch(color=base_color, label="Other configs"),
    ]
    ax.legend(handles=legend_handles + [
        plt.Line2D([0], [0], color="red", linestyle="--", label="Chance (33.3%)"),
    ], fontsize=9, loc="upper left")

    out = cfg.FIG_DIR / "fig3_affective_ablation_metrics.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── Figure 4: Clinical-label sensitivity heatmap ──────────────────────

def fig4_clinical_heatmap() -> Path | None:
    metrics_path = cfg.OUT_DIR / "clinical_sensitivity_metrics.csv"
    if not metrics_path.exists():
        print(f"  [fig4] Skipped: {metrics_path} not found.")
        return None
    df = pd.read_csv(metrics_path)
    if df.empty:
        return None

    diagnoses = [d for d in cfg.DIAGNOSES if d in df["diagnosis"].unique()]
    configs = cfg.CLINICAL_CONFIG_ORDER

    bal_mat = np.full((len(diagnoses), len(configs)), np.nan)
    auc_mat = np.full((len(diagnoses), len(configs)), np.nan)
    for i, dx in enumerate(diagnoses):
        for j, cid in enumerate(configs):
            row = df[(df["diagnosis"] == dx) & (df["config_id"] == cid)]
            if not row.empty:
                bal_mat[i, j] = row["balanced_accuracy_mean"].iloc[0] * 100
                auc_mat[i, j] = row["roc_auc_mean"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    feature_set_lookup = {
        "C1": "E", "C2": "D", "C3": "T", "C4": "D+T", "C5": "E+D+T", "C6": "C",
    }
    xticklabels = [f"{c}\n{feature_set_lookup[c]}" for c in configs]

    ax = axes[0]
    im = ax.imshow(bal_mat, aspect="auto", cmap="YlOrRd", vmin=45, vmax=65)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(xticklabels, fontsize=9)
    ax.set_yticks(range(len(diagnoses)))
    ax.set_yticklabels(diagnoses)
    for i in range(len(diagnoses)):
        for j in range(len(configs)):
            v = bal_mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="Balanced accuracy (%)")
    ax.set_title("Balanced accuracy", fontweight="bold")

    ax = axes[1]
    im2 = ax.imshow(auc_mat, aspect="auto", cmap="YlOrRd", vmin=0.45, vmax=0.65)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(xticklabels, fontsize=9)
    ax.set_yticks(range(len(diagnoses)))
    ax.set_yticklabels(diagnoses)
    for i in range(len(diagnoses)):
        for j in range(len(configs)):
            v = auc_mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im2, ax=ax, label="ROC-AUC")
    ax.set_title("ROC-AUC", fontweight="bold")

    fig.suptitle("Exploratory clinical-label sensitivity",
                 fontsize=13, fontweight="bold")
    out = cfg.FIG_DIR / "fig4_clinical_sensitivity_heatmap.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── Figure 5 (optional): best layer per diagnosis summary ─────────────

def fig5_clinical_best_layer() -> Path | None:
    metrics_path = cfg.OUT_DIR / "clinical_sensitivity_metrics.csv"
    if not metrics_path.exists():
        return None
    df = pd.read_csv(metrics_path)
    if df.empty:
        return None

    diagnoses = [d for d in cfg.DIAGNOSES if d in df["diagnosis"].unique()]
    bal_best, auc_best, bal_cfg, auc_cfg = [], [], [], []
    for dx in diagnoses:
        sub = df[df["diagnosis"] == dx]
        b_idx = sub["balanced_accuracy_mean"].idxmax()
        a_idx = sub["roc_auc_mean"].idxmax()
        bal_best.append(sub.loc[b_idx, "balanced_accuracy_mean"] * 100)
        bal_cfg.append(sub.loc[b_idx, "config_id"])
        auc_best.append(sub.loc[a_idx, "roc_auc_mean"])
        auc_cfg.append(sub.loc[a_idx, "config_id"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    bars = ax.bar(diagnoses, bal_best, color="#0984e3", edgecolor="white")
    ax.axhline(50, color="red", linestyle="--", alpha=0.5, label="Chance (50%)")
    for b, cfg_id in zip(bars, bal_cfg):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                cfg_id, ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Best balanced accuracy (%)")
    ax.set_title("Best layer (balanced accuracy)", fontweight="bold")
    ax.set_ylim(40, max(70, max(bal_best) + 5))
    ax.legend(fontsize=8)

    ax = axes[1]
    bars = ax.bar(diagnoses, auc_best, color="#00b894", edgecolor="white")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Chance (0.5)")
    for b, cfg_id in zip(bars, auc_cfg):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                cfg_id, ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Best ROC-AUC")
    ax.set_title("Best layer (ROC-AUC)", fontweight="bold")
    ax.set_ylim(0.4, max(0.7, max(auc_best) + 0.05))
    ax.legend(fontsize=8)

    fig.suptitle("Exploratory clinical-label sensitivity: best-layer summary",
                 fontsize=12, fontweight="bold")
    out = cfg.FIG_DIR / "fig5_clinical_best_layer_summary.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    print("=" * 70)
    print("03: Generate submission figures")
    print("=" * 70)
    produced = []
    for fn in (fig1_layer_decomposition, fig2_framework,
               fig3_affective_ablation, fig4_clinical_heatmap,
               fig5_clinical_best_layer):
        out = fn()
        if out is not None:
            produced.append(out)
            print(f"  -> {out}")
    if not produced:
        print("  No figures produced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
