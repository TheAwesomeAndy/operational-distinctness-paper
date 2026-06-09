#!/usr/bin/env python3
"""Phase 5 — Observation-first figures (obs01-obs10).

Each figure shows a measured object (signal, reservoir state, graph, coupling,
belief trajectory) with a paired JSON metadata sidecar. Titles describe what is
plotted, not conclusions; no p-values appear in observation titles; no raw
subject identifiers are exposed. Figures whose inputs are unavailable write an
explicit ``objNN_SKIPPED.md`` (these are non-essential figures).

Run:
    python experiments/tcds_ready9/06_generate_observation_plots.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.tcds_ready9 import config as cfg  # noqa: E402
from experiments.tcds_ready9 import common_ready9 as cr  # noqa: E402

COND = cfg.AFFECTIVE_LABEL_NAMES
COLORS = {0: "#C44E52", 1: "#7F7F7F", 2: "#4C72B0"}


def _meta(fig_id, inputs, data_level, extra=None):
    m = {
        "source_script": "experiments/tcds_ready9/06_generate_observation_plots.py",
        "figure_type": "observation",
        "inputs": inputs,
        "provenance": cr.base_provenance(),
        "data_level": data_level,
        "privacy_status": cfg.PRIVACY_STATUS,
        "output_path": f"figures/tcds_ready9/observations/{fig_id}.pdf",
    }
    if extra:
        m.update(extra)
    cr.write_json(cfg.FIG_OBS_DIR / f"{fig_id}.json", m)


def _skip(fig_id, reason):
    (cfg.OBS_OUT_DIR / f"{fig_id}_SKIPPED.md").write_text(
        f"# {fig_id} skipped\n\n{reason}\n")
    print(f"[obs] {fig_id} SKIPPED: {reason}")


def _save(fig, fig_id):
    fig.savefig(cfg.FIG_OBS_DIR / f"{fig_id}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[obs] wrote {fig_id}.pdf")


# ── Reservoir (single observation) for raster ───────────────────────────
def _reservoir_spikes(sig_1d, seed=42, n_res=256, beta=0.05, theta=0.5):
    rng = np.random.RandomState(seed)
    limit = np.sqrt(6.0 / (1 + n_res))
    W_in = rng.uniform(-limit, limit, (n_res, 1))
    W_rec = rng.randn(n_res, n_res) / np.sqrt(n_res)
    sr = np.max(np.abs(np.linalg.eigvals(W_rec)))
    if sr > 0:
        W_rec *= 0.9 / sr
    T = len(sig_1d)
    mem = np.zeros(n_res); spk = np.zeros(n_res)
    spikes = np.zeros((n_res, T), dtype=np.int8)
    for t in range(T):
        mem = (1 - beta) * mem * (1 - spk) + W_in[:, 0] * sig_1d[t] + W_rec @ spk
        mem = np.maximum(mem, 0.0)
        spk = (mem >= theta).astype(float)
        mem = mem - spk * theta
        spikes[:, t] = spk.astype(np.int8)
    return spikes


def _exemplar_idx(y):
    return {c: int(np.where(y == c)[0][0]) for c in sorted(set(y.tolist()))}


def main() -> int:
    with open(cfg.SHAPE_FILE, "rb") as f:
        ch5 = pickle.load(f)
    y = np.asarray(ch5["y"])
    X_ds = np.asarray(ch5["X_ds"]) if "X_ds" in ch5 else None
    shp_fp = {"shape_features_211": cr.file_fingerprint(cfg.SHAPE_FILE)}

    ch67 = None
    if cfg.CH67_FILE.exists():
        with open(cfg.CH67_FILE, "rb") as f:
            ch67 = pickle.load(f)
    ch67_fp = {"ch6_ch7_3class_features": cr.file_fingerprint(cfg.CH67_FILE)}

    # obs01 — grand-average ERP by condition
    if X_ds is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        t = np.arange(X_ds.shape[1]) / cfg.FS_DS * 1000.0
        for c in sorted(set(y.tolist())):
            m = X_ds[y == c].mean(axis=(0, 2))
            s = X_ds[y == c].mean(axis=2).std(axis=0)
            ax.plot(t, m, color=COLORS[c], label=COND[c])
            ax.fill_between(t, m - s, m + s, color=COLORS[c], alpha=0.15)
        ax.set_xlabel("time since post-stimulus onset (ms)")
        ax.set_ylabel("channel-averaged amplitude (z)")
        ax.set_title("Grand-average reservoir-input ERP by condition")
        ax.legend()
        _save(fig, "obs01_erp_condition_grand_average")
        _meta("obs01_erp_condition_grand_average", shp_fp, "downsampled EEG (X_ds)")
    else:
        _skip("obs01_erp_condition_grand_average", "X_ds not present in shape pickle")

    # obs02 — channel x time variability heatmap (condition-mean contrast)
    if X_ds is not None:
        diff = np.abs(X_ds[y == 0].mean(axis=0) - X_ds[y == 2].mean(axis=0)).T  # (ch, time)
        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.imshow(diff, aspect="auto", cmap="magma",
                       extent=[0, X_ds.shape[1] / cfg.FS_DS * 1000, diff.shape[0], 0])
        ax.set_xlabel("time (ms)"); ax.set_ylabel("channel index")
        ax.set_title("Channel x time |Negative - Pleasant| mean amplitude")
        fig.colorbar(im, ax=ax, label="|mean difference|")
        _save(fig, "obs02_channel_time_variability")
        _meta("obs02_channel_time_variability", shp_fp, "downsampled EEG (X_ds)")
    else:
        _skip("obs02_channel_time_variability", "X_ds not present")

    # obs03 — clean vs perturbed signal examples
    if X_ds is not None:
        ex = _exemplar_idx(y)[0]
        sig = X_ds[ex].mean(axis=1)
        rng = np.random.default_rng(0)
        t = np.arange(len(sig)) / cfg.FS_DS * 1000.0
        fig, axes = plt.subplots(4, 1, figsize=(7, 8), sharex=True)
        axes[0].plot(t, sig, color="k"); axes[0].set_title("clean")
        jit = np.roll(sig, int(25 * cfg.FS_DS / 1000)); jit[:6] = 0
        axes[1].plot(t, jit, color="#4C72B0"); axes[1].set_title("temporal jitter (+25 ms)")
        p = np.mean(sig ** 2); noise = rng.standard_normal(len(sig)) * np.sqrt(p / 10 ** (10 / 10))
        axes[2].plot(t, sig + noise, color="#55A868"); axes[2].set_title("amplitude noise (10 dB SNR)")
        drop = X_ds[ex].copy(); drop[:, rng.choice(34, 7, replace=False)] = 0
        axes[3].plot(t, drop.mean(axis=1), color="#C44E52"); axes[3].set_title("channel dropout (~20%)")
        axes[3].set_xlabel("time (ms)")
        fig.suptitle("Clean versus perturbed representative observation")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        _save(fig, "obs03_signal_perturbation_examples")
        _meta("obs03_signal_perturbation_examples", shp_fp, "downsampled EEG (X_ds)",
              {"note": "representative observation; identifier hashed/omitted"})
    else:
        _skip("obs03_signal_perturbation_examples", "X_ds not present")

    # obs04 — reservoir spike raster
    if X_ds is not None:
        ex = _exemplar_idx(y)
        fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
        for ax, c in zip(axes, sorted(ex)):
            spk = _reservoir_spikes(X_ds[ex[c]].mean(axis=1))
            nz = np.nonzero(spk)
            ax.scatter(nz[1], nz[0], s=1, color=COLORS[c])
            ax.set_title(f"{COND[c]} (spikes={int(spk.sum())})")
            ax.set_xlabel("timestep")
        axes[0].set_ylabel("reservoir neuron")
        fig.suptitle("LIF reservoir spike raster, representative observations")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _save(fig, "obs04_reservoir_spike_raster")
        _meta("obs04_reservoir_spike_raster", shp_fp, "reservoir recomputed from X_ds",
              {"reservoir": "N=256, beta=0.05, theta=0.5, seed=42"})
    else:
        _skip("obs04_reservoir_spike_raster", "X_ds not present; spike trains not stored")

    # obs05 — population-rate traces by condition (needs ch67 pop_rate_ts)
    if ch67 is not None and "pop_rate_ts" in ch67:
        prt = np.asarray(ch67["pop_rate_ts"])  # (n,34,256)
        fig, ax = plt.subplots(figsize=(7, 4))
        for c in sorted(set(y.tolist())):
            tr = prt[y == c].mean(axis=(0, 1))
            sd = prt[y == c].mean(axis=1).std(axis=0)
            ax.plot(tr, color=COLORS[c], label=COND[c])
            ax.fill_between(np.arange(len(tr)), tr - sd, tr + sd, color=COLORS[c], alpha=0.15)
        ax.set_xlabel("reservoir timestep"); ax.set_ylabel("population rate")
        ax.set_title("Reservoir population-rate traces by condition")
        ax.legend()
        _save(fig, "obs05_population_rate_by_condition")
        _meta("obs05_population_rate_by_condition", ch67_fp, "reservoir population rate")
    else:
        _skip("obs05_population_rate_by_condition", "ch67 pop_rate_ts unavailable")

    # obs06 — BSC6 temporal-bin observation (from pop_rate_ts binned into 6)
    if ch67 is not None and "pop_rate_ts" in ch67:
        prt = np.asarray(ch67["pop_rate_ts"]).mean(axis=1)  # (n,256)
        T = prt.shape[1]; bs = T // 6
        bins = np.stack([prt[:, b * bs:(b + 1) * bs].sum(axis=1) for b in range(6)], axis=1)
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(1, 7)
        for c in sorted(set(y.tolist())):
            m = bins[y == c].mean(axis=0); s = bins[y == c].std(axis=0)
            ax.errorbar(x, m, yerr=s, marker="o", color=COLORS[c], label=COND[c], capsize=3)
        ax.set_xlabel("temporal bin (BSC$_6$)"); ax.set_ylabel("summed population activity")
        ax.set_title("BSC$_6$ temporal-bin profiles by condition")
        ax.legend()
        _save(fig, "obs06_bsc6_temporal_bins")
        _meta("obs06_bsc6_temporal_bins", ch67_fp, "reservoir activity binned into 6 windows")
    else:
        _skip("obs06_bsc6_temporal_bins", "ch67 pop_rate_ts unavailable")

    # obs07 — feature-block distribution panel
    if ch67 is not None:
        emb = np.asarray(ch5["lsm_bsc6_pca"]).reshape(len(y), -1)
        D = np.asarray(ch67["D"]).reshape(len(y), -1)
        T = np.asarray(ch67["T_topo"]).reshape(len(y), -1)
        C = cr.compute_coupling_block(np.asarray(ch67["D"]), np.asarray(ch67["T_topo"]))
        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        for ax, (name, M) in zip(axes.ravel(), [("E", emb), ("D", D), ("T", T), ("C (kappa col)", C[:, :1])]):
            ax.hist(M.ravel(), bins=60, color="#4C72B0")
            ax.set_title(f"{name} value distribution")
        fig.suptitle("Feature-block value distributions")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        _save(fig, "obs07_feature_block_distributions")
        _meta("obs07_feature_block_distributions", {**shp_fp, **ch67_fp}, "feature-level")
    else:
        _skip("obs07_feature_block_distributions", "ch67 unavailable")

    # obs08 — tPLV adjacency by condition
    if ch67 is not None and "tPLV_mats" in ch67:
        tplv = np.asarray(ch67["tPLV_mats"])
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, c in zip(axes, sorted(set(y.tolist()))):
            im = ax.imshow(tplv[y == c].mean(axis=0), cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"mean tPLV — {COND[c]}")
        fig.colorbar(im, ax=axes.tolist(), fraction=0.025, label="phase-locking")
        fig.suptitle("Theta-band tPLV adjacency by condition")
        _save(fig, "obs08_tplv_adjacency_matrices")
        _meta("obs08_tplv_adjacency_matrices", ch67_fp, "graph adjacency (tPLV)")
    else:
        _skip("obs08_tplv_adjacency_matrices", "ch67 tPLV_mats unavailable")

    # obs09 — structure-function coupling kappa observation + shuffle null
    if ch67 is not None:
        D_pc = np.asarray(ch67["D"]); T_pc = np.asarray(ch67["T_topo"])
        kappa = cr.compute_coupling_block(D_pc, T_pc)[:, 0]
        rng = np.random.default_rng(0)
        null = []
        for _ in range(200):
            perm = rng.permutation(cfg.N_CHANNELS)
            null.append(cr.compute_coupling_block(D_pc[:1], T_pc[:1][:, perm, :])[0, 0])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(kappa, bins=40, color="#4C72B0", alpha=0.8, label="observed $\\kappa$")
        ax.axvline(np.mean(null), color="k", ls="--", label="electrode-shuffle null mean")
        ax.set_xlabel("$\\kappa = \\|C\\|_F/\\sqrt{pq}$"); ax.set_ylabel("observations")
        ax.set_title("Structure-function coupling $\\kappa$ across observations")
        ax.legend()
        _save(fig, "obs09_kappa_observation")
        _meta("obs09_kappa_observation", ch67_fp, "coupling readout",
              {"kappa_mean": float(np.mean(kappa)), "kappa_sd": float(np.std(kappa))})
    else:
        _skip("obs09_kappa_observation", "ch67 unavailable")

    # obs10 — closed-loop example belief trajectories
    _obs10_trajectories(ch67, ch5, y)
    return 0


def _obs10_trajectories(ch67, ch5, y):
    fid = "obs10_closed_loop_example_trajectories"
    if ch67 is None:
        _skip(fid, "ch67 unavailable; closed-loop substrate features missing")
        return
    # Minimal illustrative rollout using the A8 substrate classifier.
    E = np.asarray(ch5["lsm_bsc6_pca"]).reshape(len(y), -1)
    D = np.asarray(ch67["D"]).reshape(len(y), -1)
    T = np.asarray(ch67["T_topo"]).reshape(len(y), -1)
    X = np.hstack([E, D, T])
    subjects = np.asarray(ch5["subjects"])
    cv = cr.subject_grouped_cv(cfg.N_FOLDS_AFFECTIVE, 42)
    tr, te = next(iter(cv.split(X, y, groups=subjects)))
    _, proba_te, _ = cr.fit_logreg_fold(X[tr], y[tr], X[te], seed=42)
    pool = {s: proba_te[y[te] == s] for s in sorted(set(y.tolist()))}
    if any(len(pool[s]) == 0 for s in pool):
        _skip(fid, "held-out class pool empty")
        return
    rng = np.random.default_rng(7)
    target = 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for eps, color in [(0.0, "#4C72B0"), (0.2, "#C44E52")]:
        b = np.full(3, 1 / 3); ents = [cr.posterior_entropy(b)]; btarget = [b[target]]
        for _ in range(12):
            true_state = int(rng.choice(3, p=((1 - eps) * np.eye(3)[target] + eps / 3)))
            L = pool[true_state][rng.integers(len(pool[true_state]))]
            b = b * np.clip(L, 1e-9, 1); b /= b.sum()
            ents.append(cr.posterior_entropy(b)); btarget.append(b[target])
            if b[target] >= 0.8:
                break
        axes[0].plot(btarget, marker="o", color=color, label=f"$\\epsilon$={eps}")
        axes[1].plot(ents, marker="o", color=color, label=f"$\\epsilon$={eps}")
    axes[0].axhline(0.8, ls="--", color="k"); axes[0].set_title("belief in target state")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("b(target)"); axes[0].legend()
    axes[1].set_title("posterior entropy"); axes[1].set_xlabel("step")
    axes[1].set_ylabel("entropy (nats)"); axes[1].legend()
    fig.suptitle("Closed-loop example belief trajectories (simulation episodes)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, fid)
    _meta(fid, {"ch6_ch7_3class_features": cr.file_fingerprint(cfg.CH67_FILE)},
          "simulated closed-loop over recorded features",
          {"note": "episode-level identifiers only; target=Pleasant", "seed": 7})


if __name__ == "__main__":
    raise SystemExit(main())
