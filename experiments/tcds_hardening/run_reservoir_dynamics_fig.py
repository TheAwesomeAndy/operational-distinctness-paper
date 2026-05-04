#!/usr/bin/env python3
"""TCDS hardening: visualize the brain-inspired reservoir dynamics underlying ARSPI-Net.

Generates a single multi-panel figure that surfaces the LIF reservoir +
BSC6 neural-coding mechanism that the operational-distinctness pipeline
sits on top of:

  Panel A (top row, 3 cols)   spike rasters (256 neurons x 256 timesteps)
                              for one exemplar (subject, condition) per
                              affective class.
  Panel B (middle row, 3 cols) membrane-potential heatmap for the same
                              three exemplars.
  Panel C (bottom-left)        per-class average tPLV connectivity matrix
                              (34 x 34) across all 211 subjects, three
                              subplots side-by-side.
  Panel D (bottom-right)       BSC6 PCA-64 projection scatter (PC1 vs PC2)
                              colored by condition; the three exemplars
                              highlighted with star markers.

Brain-inspired grounding (TCDS CFP):
  - LIF reservoir = "spiking neural network" + "recurrent feedback".
  - BSC6 binned spike count = "neural coding scheme" + "event-driven".
  - tPLV connectivity = "hierarchical / cortical-style organization".

Privacy: subjects are referenced only by 16-char SHA-256 hash. Three
exemplars are deterministically chosen as the median-kappa subject in
each condition (using outputs/tcds_hardening/kappa_per_observation.csv
when present, otherwise the lowest-hashed subject in each class).

Inputs:
    data/shape_features_211.pkl       (X_ds, lsm_bsc6_pca, y, subjects)
    data/ch6_ch7_3class_features.pkl  (tPLV_mats)
    [optional] outputs/tcds_hardening/kappa_per_observation.csv

Outputs:
    figures/tcds_hardening/fig_reservoir_dynamics.pdf
    outputs/tcds_hardening/reservoir_dynamics_diagnostics.json
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
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.operational_distinctness import config as cfg
from experiments.operational_distinctness.common import hash_subject_array, load_inputs
from prepare_inputs.extract_ch5_features import (
    BETA, BSC_N_BINS, BSC_T_END, BSC_T_START, N_RES, SEED, THRESHOLD, LIFReservoir,
)

OUT_DIR = REPO_ROOT / "outputs" / "tcds_hardening"
FIG_DIR = REPO_ROOT / "figures" / "tcds_hardening"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CONDITION_NAMES = {0: "Negative", 1: "Neutral", 2: "Pleasant"}
CONDITION_COLORS = {0: "#d95f02", 1: "#7570b3", 2: "#1b9e77"}


class LIFReservoirWithMembrane(LIFReservoir):
    """LIF reservoir that also returns the per-step membrane potential.

    Identical dynamics to the upstream class; only the output expands.
    """

    def forward_with_membrane(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        T = X.shape[0]
        mem = np.zeros(self.n_res)
        spk_prev = np.zeros(self.n_res)
        spikes = np.zeros((T, self.n_res))
        membranes = np.zeros((T, self.n_res))
        for t in range(T):
            I_tot = self.W_in @ X[t] + self.W_rec @ spk_prev
            mem = (1.0 - self.beta) * mem * (1.0 - spk_prev) + I_tot
            membranes[t] = mem
            spk = (mem >= self.threshold).astype(float)
            mem = mem - spk * self.threshold
            mem = np.maximum(mem, 0.0)
            spikes[t] = spk
            spk_prev = spk
        return spikes, membranes


def _exemplar_indices_by_condition(y: np.ndarray,
                                   subjects: np.ndarray,
                                   subj_hash: np.ndarray) -> dict[int, int]:
    """Return one (deterministic) observation index per affective class.

    Strategy: for each class, prefer the observation whose kappa is closest
    to the per-class median (when kappa is available); otherwise the
    observation with the lexicographically smallest subject_hash.
    """
    chosen: dict[int, int] = {}
    kappa_path = OUT_DIR / "kappa_per_observation.csv"
    kappa_df = None
    if kappa_path.exists():
        try:
            kappa_df = pd.read_csv(kappa_path)
        except Exception:
            kappa_df = None
    for cls in (0, 1, 2):
        mask = y == cls
        idx_in_class = np.where(mask)[0]
        if idx_in_class.size == 0:
            continue
        if kappa_df is not None and "kappa" in kappa_df.columns:
            ks = kappa_df.loc[mask.tolist(), "kappa"].values \
                if len(kappa_df) == len(y) else None
            if ks is not None and ks.size == idx_in_class.size:
                med = float(np.median(ks))
                pick = int(idx_in_class[np.argmin(np.abs(ks - med))])
                chosen[cls] = pick
                continue
        order = np.argsort(subj_hash[idx_in_class])
        chosen[cls] = int(idx_in_class[order[0]])
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-electrode", type=int, default=15,
                        help="Channel index used for raster + V_mem (default 15).")
    args = parser.parse_args()

    print("=" * 72)
    print("Tier-1.2: reservoir-dynamics figure")
    print("=" * 72)
    t0 = time.time()

    ch5, ch67, _ = load_inputs()
    y = np.asarray(ch5["y"]).astype(int)
    subjects = np.asarray(ch5["subjects"])
    subj_hash = hash_subject_array(subjects)
    X_ds = np.asarray(ch5["X_ds"])           # (N, T_ds, 34)
    lsm = np.asarray(ch5["lsm_bsc6_pca"])    # (N, 34, 64)
    tplv = np.asarray(ch67["tPLV_mats"])     # (N, 34, 34)

    print(f"  X_ds {X_ds.shape}, lsm {lsm.shape}, tPLV {tplv.shape}")
    print(f"  reservoir params: N={N_RES}, beta={BETA}, theta={THRESHOLD}, seed={SEED}")
    print(f"  channel for raster + V_mem panels: ch={args.n_electrode}")

    exemplars = _exemplar_indices_by_condition(y, subjects, subj_hash)
    print(f"  exemplars (class -> obs_idx): {exemplars}")

    # Run reservoir on the chosen channel for each exemplar.
    res = LIFReservoirWithMembrane(n_input=1, n_res=N_RES,
                                   beta=BETA, threshold=THRESHOLD, seed=SEED)
    spikes_per_class: dict[int, np.ndarray] = {}
    membranes_per_class: dict[int, np.ndarray] = {}
    spike_counts: dict[int, int] = {}
    for cls, obs_idx in exemplars.items():
        eeg_ch = X_ds[obs_idx, :, args.n_electrode].reshape(-1, 1)
        spk, mem = res.forward_with_membrane(eeg_ch)
        spikes_per_class[cls] = spk
        membranes_per_class[cls] = mem
        spike_counts[cls] = int(spk.sum())
        print(f"    class {cls} ({CONDITION_NAMES[cls]}): "
              f"obs={obs_idx}  total_spikes={spike_counts[cls]}  "
              f"mem_range=[{mem.min():.3f}, {mem.max():.3f}]")

    # Per-class average tPLV across all subjects in that class.
    tplv_means = {cls: tplv[y == cls].mean(axis=0) for cls in (0, 1, 2)}

    # PCA on flattened lsm_bsc6_pca features for the global scatter.
    n_obs = lsm.shape[0]
    lsm_flat = lsm.reshape(n_obs, -1)
    pca = PCA(n_components=2, random_state=SEED)
    lsm_pca = pca.fit_transform(lsm_flat)
    print(f"  Global LSM-PCA explained variance: {pca.explained_variance_ratio_}")

    # ── figure ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 11))
    gs = GridSpec(3, 6, figure=fig, height_ratios=[1.0, 1.0, 1.0],
                  hspace=0.55, wspace=0.55)

    # Row 1: spike rasters
    for ci, cls in enumerate((0, 1, 2)):
        ax = fig.add_subplot(gs[0, ci * 2:(ci + 1) * 2])
        spk = spikes_per_class[cls]
        if spk.sum() > 0:
            ts, ns = np.where(spk > 0)
            ax.scatter(ts, ns, s=0.5, color="black", marker=".")
        ax.set_xlim(0, spk.shape[0])
        ax.set_ylim(0, N_RES)
        ax.set_xlabel("timestep")
        ax.set_ylabel("reservoir neuron")
        ax.set_title(f"(a{ci + 1}) raster, {CONDITION_NAMES[cls]}  "
                     f"(spikes={spike_counts[cls]})", fontsize=10)
        ax.axvspan(BSC_T_START, BSC_T_END, color="#1b9e77", alpha=0.08, lw=0)

    # Row 2: membrane heatmaps
    vmin = min(m.min() for m in membranes_per_class.values())
    vmax = max(m.max() for m in membranes_per_class.values())
    vmax = max(vmax, THRESHOLD * 2.0)
    for ci, cls in enumerate((0, 1, 2)):
        ax = fig.add_subplot(gs[1, ci * 2:(ci + 1) * 2])
        im = ax.imshow(membranes_per_class[cls].T, aspect="auto",
                       cmap="magma", vmin=vmin, vmax=vmax,
                       interpolation="nearest", origin="lower")
        ax.set_xlabel("timestep")
        ax.set_ylabel("reservoir neuron")
        ax.set_title(f"(b{ci + 1}) $V_m$, {CONDITION_NAMES[cls]}", fontsize=10)
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label(r"$V_m$ (a.u.)", fontsize=8)

    # Row 3, left half: per-class average tPLV
    vmin_t = min(m.min() for m in tplv_means.values())
    vmax_t = max(m.max() for m in tplv_means.values())
    for ci, cls in enumerate((0, 1, 2)):
        ax = fig.add_subplot(gs[2, ci])
        im = ax.imshow(tplv_means[cls], cmap="viridis",
                       vmin=vmin_t, vmax=vmax_t,
                       interpolation="nearest", aspect="equal")
        ax.set_title(f"(c{ci + 1}) mean tPLV, {CONDITION_NAMES[cls]}",
                     fontsize=10)
        ax.set_xlabel("electrode j")
        if ci == 0:
            ax.set_ylabel("electrode i")
    cax = fig.add_subplot(gs[2, 3:4])
    fig.colorbar(im, cax=cax, fraction=0.7, pad=0.02).set_label("tPLV", fontsize=8)
    cax.set_position(cax.get_position().translated(-0.018, 0))

    # Row 3, right: BSC6 PCA scatter
    ax = fig.add_subplot(gs[2, 4:6])
    for cls in (0, 1, 2):
        m = y == cls
        ax.scatter(lsm_pca[m, 0], lsm_pca[m, 1],
                   s=14, alpha=0.45, color=CONDITION_COLORS[cls],
                   label=f"{CONDITION_NAMES[cls]} (n={int(m.sum())})",
                   edgecolor="white", linewidth=0.3)
    for cls, obs_idx in exemplars.items():
        ax.scatter(lsm_pca[obs_idx, 0], lsm_pca[obs_idx, 1],
                   s=180, marker="*", color=CONDITION_COLORS[cls],
                   edgecolor="black", linewidth=1.0,
                   label=f"exemplar {CONDITION_NAMES[cls]}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("(d) BSC6 PCA-64 $\\rightarrow$ PCA-2 projection", fontsize=10)
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Brain-inspired reservoir dynamics underlying ARSPI-Net "
        f"(LIF: $N_{{res}}={N_RES}$, $\\beta={BETA}$, $\\theta={THRESHOLD}$;  "
        f"BSC{BSC_N_BINS} window $t\\in[{BSC_T_START},{BSC_T_END}]$)",
        fontsize=11, y=0.995,
    )
    fig_path = FIG_DIR / "fig_reservoir_dynamics.pdf"
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
        "script": "experiments/tcds_hardening/run_reservoir_dynamics_fig.py",
        "n_observations": int(n_obs),
        "exemplar_obs_indices": {CONDITION_NAMES[c]: int(i) for c, i in exemplars.items()},
        "exemplar_subject_hashes": {
            CONDITION_NAMES[c]: subj_hash[i] for c, i in exemplars.items()
        },
        "exemplar_total_spikes": {
            CONDITION_NAMES[c]: spike_counts[c] for c in spike_counts
        },
        "channel_for_raster_and_vmem": int(args.n_electrode),
        "reservoir_params": {
            "N_res": int(N_RES),
            "beta": float(BETA),
            "theta": float(THRESHOLD),
            "seed": int(SEED),
            "bsc_n_bins": int(BSC_N_BINS),
            "bsc_t_start": int(BSC_T_START),
            "bsc_t_end": int(BSC_T_END),
        },
        "lsm_pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "tPLV_per_class_means": {
            CONDITION_NAMES[c]: {"mean": float(m.mean()),
                                  "max": float(m.max()),
                                  "min": float(m.min())}
            for c, m in tplv_means.items()
        },
        "input_file_size_hash_sha256": h.hexdigest()[:32],
        "runtime_seconds": float(time.time() - t0),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }
    diag_path = OUT_DIR / "reservoir_dynamics_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"  Wrote {diag_path}")
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
