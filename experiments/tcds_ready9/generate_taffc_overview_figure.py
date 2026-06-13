#!/usr/bin/env python3
"""Generate the early TAFFC "evidence-stream overview" figure (Figure 1).

A single, data-backed graphical abstract that gives the reader the whole
ARSPI-Net mental model on page 2-3:

    affective ERP observation  ->  LIF reservoir + BSC6 transform
        ->  four operationally distinct evidence streams (E, D, T, C)
        ->  mechanism ablation / perturbation robustness / closed-loop accumulation

Top band: the conceptual flow as labelled boxes and arrows.
Bottom band: three REAL data panels computed from the shipped features,
anchored under the first three flow stages:
    (a) trial-averaged ERPs by affective condition           (X_ds)
    (b) LIF-reservoir spike raster for one exemplar           (X_ds -> reservoir)
    (c) diagnostic BSC6 reservoir projection (PCA-2), recomputed from the
        channel-mean ERP of every observation, for visual overview only --
        NOT the production per-channel ARSPI-Net embedding E used in analyses

This script reuses the upstream reservoir spec (``LIFReservoir`` and its
constants from ``prepare_inputs/extract_ch5_features.py``) and reads ONLY
``data/shape_features_211.pkl`` via the standalone config. It does not modify
any pickle, any shared figure, or any numerical result, and writes only to
``figures/taffc/``.

    python experiments/tcds_ready9/generate_taffc_overview_figure.py
"""
from __future__ import annotations

import hashlib
import pickle
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.tcds_ready9 import config as cfg  # noqa: E402
from prepare_inputs.extract_ch5_features import (  # noqa: E402
    BETA, BSC_N_BINS, BSC_T_END, BSC_T_START, N_RES, SEED, THRESHOLD,
    LIFReservoir, extract_bsc,
)

OUT_DIR = REPO_ROOT / "figures" / "taffc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Affective-condition palette (matches the rest of the manuscript figures).
COND = {0: "Negative", 1: "Neutral", 2: "Pleasant"}
COND_COLOR = {0: "#C44E52", 1: "#7F7F7F", 2: "#4C72B0"}

# Evidence-stream palette (matches the TikZ pipeline diagram, arch_fig_pipeline_overview).
STREAM = [
    ("$E$", "spike embedding", "#2660A4", "#D6E5F7"),
    ("$D$", "dynamical",        "#1A8476", "#D1EEE7"),
    ("$T$", "topological",      "#7246A2", "#E6DDF3"),
    ("$C$", "coupling $\\kappa$", "#388C3C", "#DBEEDB"),
]
EVAL = [
    ("Mechanism ablation", "$\\Delta$ contrast, $\\Gamma$ utility"),
    ("Perturbation robustness", "stream-specific regimes"),
    ("Closed-loop accumulation", "expected-free-energy control"),
]

RASTER_CHANNEL = 15  # same channel used by run_reservoir_dynamics_fig.py

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
})


def _subject_hash(subjects: np.ndarray) -> np.ndarray:
    return np.array(
        [hashlib.sha256(str(s).encode()).hexdigest() for s in subjects], dtype=object
    )


def _box(ax, x, y, w, h, label, sub, fc, ec, fs=8.0, lw=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
        linewidth=lw, edgecolor=ec, facecolor=fc, mutation_aspect=1.0, zorder=2))
    if sub:
        ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color="#222222", zorder=3)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                fontsize=fs - 1.5, color="#444444", zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color="#222222", zorder=3)


def _arrow(ax, x0, y0, x1, y1, color="#464E5A", lw=1.3, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=10,
        linewidth=lw, color=color, shrinkA=1, shrinkB=1, zorder=4))


def main() -> int:
    t0 = time.time()
    with open(cfg.SHAPE_FILE, "rb") as f:
        d = pickle.load(f)
    X_ds = np.asarray(d["X_ds"])              # (633, 256, 34)
    y = np.asarray(d["y"]).astype(int)
    subjects = np.asarray(d["subjects"])
    n_obs, n_t, n_ch = X_ds.shape
    t_ms = np.arange(n_t) / cfg.FS_DS * 1000.0
    print(f"X_ds {X_ds.shape}; reservoir N={N_RES}, beta={BETA}, theta={THRESHOLD}, seed={SEED}")

    # Single deterministic reservoir reused for the raster and the embedding.
    res = LIFReservoir(n_input=1, n_res=N_RES, beta=BETA, threshold=THRESHOLD, seed=SEED)

    # (b) exemplar raster: deterministic Pleasant-class observation (smallest subject hash).
    subj_hash = _subject_hash(subjects)
    cls_idx = np.where(y == 2)[0]
    ex = int(cls_idx[np.argsort(subj_hash[cls_idx])[0]])
    spk_ex = res.forward(X_ds[ex, :, RASTER_CHANNEL].reshape(-1, 1))   # (256, 256)
    ts, ns = np.where(spk_ex > 0)
    print(f"exemplar obs={ex} (Pleasant) ch={RASTER_CHANNEL} total_spikes={int(spk_ex.sum())}")

    # (c) population BSC6 embedding from the channel-mean ERP of each observation.
    bsc = np.zeros((n_obs, N_RES * BSC_N_BINS))
    for i in range(n_obs):
        spk = res.forward(X_ds[i].mean(axis=1).reshape(-1, 1))
        bsc[i] = extract_bsc(spk, BSC_N_BINS, BSC_T_START, BSC_T_END)
    emb = PCA(n_components=2, random_state=SEED).fit_transform(bsc)
    pca = PCA(n_components=2, random_state=SEED).fit(bsc)
    print(f"BSC6 embedding PCA-2 explained variance: {pca.explained_variance_ratio_}")

    # ── figure ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.16, 3.45))
    canvas = fig.add_axes([0, 0, 1, 1]); canvas.set_xlim(0, 1); canvas.set_ylim(0, 1)
    canvas.axis("off")

    # Top-band flow boxes (figure fraction coords).
    by, bh = 0.62, 0.30
    _box(canvas, 0.020, by, 0.165, bh, "Affective ERP",
         r"$X^{(c)}_s\in\mathbb{R}^{34\times T}$", "#E2E9F1", "#46596F")
    _box(canvas, 0.250, by, 0.165, bh, "LIF reservoir",
         r"$+$ BSC$_6$ code", "#FFEED0", "#C48A26")

    # Evidence-stream box with four colored chips.
    sx, sw = 0.480, 0.215
    canvas.text(sx + sw / 2, by + bh + 0.045, "Evidence streams",
                ha="center", va="bottom", fontsize=8, fontweight="bold", color="#222")
    canvas.add_patch(FancyBboxPatch(
        (sx, by), sw, bh, boxstyle="round,pad=0.004,rounding_size=0.012",
        linewidth=1.0, edgecolor="#888", facecolor="#FBFBFC", zorder=2))
    cw = sw / 4
    for j, (lab, sub, ec, fc) in enumerate(STREAM):
        cx = sx + j * cw + 0.006
        canvas.add_patch(FancyBboxPatch(
            (cx, by + 0.045), cw - 0.012, bh - 0.09,
            boxstyle="round,pad=0.003,rounding_size=0.010",
            linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=3))
        canvas.text(cx + (cw - 0.012) / 2, by + bh / 2, lab, ha="center",
                    va="center", fontsize=10.0, fontweight="bold", color=ec, zorder=4)

    # Evaluation box with three stacked chips.
    ex0, ew = 0.760, 0.225
    canvas.text(ex0 + ew / 2, by + bh + 0.045, "Operational-distinctness evaluation",
                ha="center", va="bottom", fontsize=7.6, fontweight="bold", color="#222")
    eh = (bh - 0.02) / 3
    for k, (lab, sub) in enumerate(EVAL):
        ey = by + (2 - k) * eh + 0.01 * (2 - k)
        canvas.add_patch(FancyBboxPatch(
            (ex0, ey), ew, eh - 0.006, boxstyle="round,pad=0.002,rounding_size=0.008",
            linewidth=0.9, edgecolor="#5C5C62", facecolor="#ECECEE", zorder=3))
        canvas.text(ex0 + 0.012, ey + (eh - 0.006) * 0.62, lab, ha="left", va="center",
                    fontsize=6.6, fontweight="bold", color="#222", zorder=4)
        canvas.text(ex0 + 0.012, ey + (eh - 0.006) * 0.26, sub, ha="left", va="center",
                    fontsize=5.4, color="#555", zorder=4)

    # Flow arrows between stages.
    _arrow(canvas, 0.185, by + bh / 2, 0.250, by + bh / 2)
    _arrow(canvas, 0.415, by + bh / 2, 0.480, by + bh / 2)
    _arrow(canvas, 0.695, by + bh / 2, 0.760, by + bh / 2)

    # ── bottom-band REAL data panels, anchored under the first three stages ──
    pb, ph = 0.135, 0.30
    ax_erp = fig.add_axes([0.055, pb, 0.150, ph])
    for c in (0, 1, 2):
        m = X_ds[y == c].mean(axis=(0, 2))
        s = X_ds[y == c].mean(axis=2).std(axis=0)
        ax_erp.plot(t_ms, m, color=COND_COLOR[c], lw=1.0, label=COND[c])
        ax_erp.fill_between(t_ms, m - s, m + s, color=COND_COLOR[c], alpha=0.13, lw=0)
    ax_erp.set_xlabel("time (ms)", labelpad=1)
    ax_erp.set_ylabel("amplitude (z)", labelpad=1)
    ax_erp.set_title("(a) trial-averaged ERP", fontsize=7, pad=2)
    ax_erp.legend(loc="upper right", fontsize=5.2, handlelength=1.1,
                  borderpad=0.2, labelspacing=0.2, frameon=False)
    ax_erp.tick_params(length=2, pad=1)

    ax_ras = fig.add_axes([0.290, pb, 0.150, ph])
    ax_ras.scatter(ts, ns, s=0.25, color="black", marker=".", linewidths=0)
    ax_ras.axvspan(BSC_T_START, BSC_T_END, color="#1b9e77", alpha=0.10, lw=0)
    ax_ras.set_xlim(0, n_t); ax_ras.set_ylim(0, N_RES)
    ax_ras.set_xlabel("timestep", labelpad=1)
    ax_ras.set_ylabel("neuron", labelpad=1)
    ax_ras.set_title("(b) reservoir spike raster", fontsize=7, pad=2)
    ax_ras.tick_params(length=2, pad=1)

    ax_emb = fig.add_axes([0.525, pb, 0.150, ph])
    for c in (0, 1, 2):
        mm = y == c
        ax_emb.scatter(emb[mm, 0], emb[mm, 1], s=5, alpha=0.45,
                       color=COND_COLOR[c], edgecolor="white", linewidth=0.15)
    ax_emb.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})", labelpad=1)
    ax_emb.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})", labelpad=1)
    ax_emb.set_title(r"(c) BSC$_6$ projection (diagnostic)", fontsize=7, pad=2)
    ax_emb.tick_params(length=2, pad=1)

    # Thin connectors from each flow box down to its data panel.
    for cx in (0.103, 0.333, 0.560):
        _arrow(canvas, cx, by - 0.005, cx, pb + ph + 0.045,
               color="#AAAAAA", lw=0.7, style="-")

    fig_path = OUT_DIR / "fig_overview_evidence_streams.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fig_path.relative_to(REPO_ROOT)}  ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
