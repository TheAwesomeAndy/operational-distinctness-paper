#!/usr/bin/env python3
"""Phase 6 — Perturbation robustness (CORE phase).

Two clearly-labelled passes test whether ARSPI-Net evidence streams retain
discriminative information under perturbation:

1. Representation-level pass (REQUIRED, all 10 configurations, full dataset):
   amplitude noise, channel dropout, and graph perturbation are applied to the
   assembled feature blocks of the held-out test fold (train stays clean -> no
   leakage).

2. Raw-signal pass (bounded diagnostic subset): temporal jitter, amplitude
   noise, and channel dropout are applied to the downsampled EEG and a reservoir
   embedding is recomputed within-fold (PCA + classifier fit on clean train).
   Its scope is recorded in the manifest and disclosed in the manuscript.

The core/fatal requirement is the representation-level pass. Operating regimes,
including streams that degrade faster than the band-power baseline, are reported
rather than hidden.

Outputs:
    outputs/tcds_ready9/analysis/robustness_metrics.csv
    outputs/tcds_ready9/analysis/robustness_summary.csv
    outputs/tcds_ready9/analysis/robustness_config.json

Run:
    python experiments/tcds_ready9/08_signal_robustness.py
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

# Configuration -> ordered component list.
CONFIG_COMPONENTS = {
    "A0": ["BandPower"], "A1": ["E"], "A2": ["D"], "A3": ["T"], "A4": ["C"],
    "A5": ["D", "T"], "A6": ["E", "D"], "A7": ["E", "T"],
    "A8": ["E", "D", "T"], "A9": ["E", "D", "T", "C"],
}


# ── Perturbation primitives (representation level) ──────────────────────
def amplitude_noise_cols(M, snr_db, rng):
    """Add Gaussian noise to each column at the given SNR (per-column power)."""
    if snr_db is None:
        return M
    out = M.copy()
    col_power = np.mean(M ** 2, axis=0) + 1e-12
    noise_power = col_power / (10 ** (snr_db / 10.0))
    noise = rng.standard_normal(M.shape) * np.sqrt(noise_power)[None, :]
    return out + noise


def perturb_tplv(tplv, frac, rng):
    """Symmetric off-diagonal edge dropout of a single tPLV matrix."""
    if frac <= 0:
        return tplv
    n = tplv.shape[0]
    iu = np.triu_indices(n, k=1)
    n_edges = len(iu[0])
    n_drop = int(round(frac * n_edges))
    if n_drop <= 0:
        return tplv
    drop = rng.choice(n_edges, size=n_drop, replace=False)
    out = tplv.copy()
    out[iu[0][drop], iu[1][drop]] = 0.0
    out[iu[1][drop], iu[0][drop]] = 0.0
    return out


def tplv_to_topo(plv):
    """Weighted strength + Onnela clustering per node (matches extraction)."""
    n = plv.shape[0]
    topo = np.zeros((n, 2))
    for i in range(n):
        topo[i, 0] = np.sum(plv[i, :]) - plv[i, i]
    for i in range(n):
        neigh = np.where((plv[i, :] > 0) & (np.arange(n) != i))[0]
        k = len(neigh)
        if k < 2:
            continue
        tri = 0.0
        for a in range(k):
            for b in range(a + 1, k):
                j, l = neigh[a], neigh[b]
                tri += (plv[i, j] * plv[i, l] * plv[j, l]) ** (1.0 / 3.0)
        topo[i, 1] = 2.0 * tri / (k * (k - 1))
    return topo


def recompute_T_C_for(indices, tplv_all, D_perchan, frac, rng):
    """Return (T_block, C_block) for given obs indices under graph perturbation."""
    T_topo = np.zeros((len(indices), cfg.N_CHANNELS, 2))
    for r, idx in enumerate(indices):
        T_topo[r] = tplv_to_topo(perturb_tplv(tplv_all[idx], frac, rng))
    T_block = T_topo.reshape(len(indices), -1)
    C_block = cr.compute_coupling_block(D_perchan[indices], T_topo)
    return T_block, C_block


# ── Component assembly ──────────────────────────────────────────────────
def assemble(components, idx, comp_clean, overrides=None):
    overrides = overrides or {}
    mats = []
    for c in components:
        if c in overrides:
            mats.append(overrides[c])
        else:
            mats.append(comp_clean[c][idx])
    return np.hstack(mats) if len(mats) > 1 else mats[0]


def evaluate_perturbation(config_id, ptype, level, comp_clean, y, groups,
                          tplv_all, D_perchan):
    """Train clean, test perturbed; return per-(seed,fold) metric rows."""
    components = CONFIG_COMPONENTS[config_id]
    label_set = sorted(set(y.tolist()))
    rows = []
    for seed in cfg.SEEDS:
        cv = cr.subject_grouped_cv(cfg.N_FOLDS_AFFECTIVE, seed)
        for fold, (tr, te) in enumerate(cv.split(comp_clean[components[0]], y, groups=groups)):
            X_tr = assemble(components, tr, comp_clean)
            rng = np.random.default_rng(1000 * seed + fold)
            overrides = {}
            if ptype == "amplitude_noise":
                for c in components:
                    overrides[c] = amplitude_noise_cols(comp_clean[c][te], level, rng)
            elif ptype == "channel_dropout":
                for c in components:
                    overrides[c] = cr.apply_channel_dropout(
                        comp_clean[c][te].copy(), c, level, rng)
            elif ptype == "graph_perturbation":
                if "T" in components or "C" in components:
                    T_blk, C_blk = recompute_T_C_for(te, tplv_all, D_perchan, level, rng)
                    if "T" in components:
                        overrides["T"] = T_blk
                    if "C" in components:
                        overrides["C"] = C_blk
                else:
                    return []  # not applicable to this config
            X_te = assemble(components, te, comp_clean, overrides)
            pred, proba, _ = cr.fit_logreg_fold(X_tr, y[tr], X_te, seed=seed)
            mb = cr.metric_bundle(y[te], pred, proba, labels=label_set)
            rows.append({
                "pass": "representation_level", "config": config_id,
                "perturbation": ptype, "level": ("clean" if level in (None, 0.0) else level),
                "seed": seed, "fold": fold,
                "balanced_accuracy": mb["balanced_accuracy"],
                "macro_f1": mb["macro_f1"], "roc_auc": mb["roc_auc"],
                "n_test": mb["n"],
            })
    return rows


# ── Raw-signal pass (bounded subset) ────────────────────────────────────
def _init_reservoir(seed=42, n_res=256):
    rng = np.random.RandomState(seed)
    limit = np.sqrt(6.0 / (1 + n_res))
    W_in = rng.uniform(-limit, limit, (n_res, 1))
    W_rec = rng.randn(n_res, n_res) / np.sqrt(n_res)
    sr = np.max(np.abs(np.linalg.eigvals(W_rec)))
    if sr > 0:
        W_rec *= 0.9 / sr
    return W_in, W_rec


def _reservoir_bsc6(X_obs, W_in, W_rec, beta=0.05, theta=0.5, n_bins=6):
    """Batched LIF reservoir over channels; return BSC6 mean-pooled embedding.

    X_obs: (T, n_ch). Returns a per-observation embedding (n_ch * n_bins,)
    of mean spike counts per bin (compact, deterministic).
    """
    T, n_ch = X_obs.shape
    n_res = W_in.shape[0]
    mem = np.zeros((n_res, n_ch))
    spk = np.zeros((n_res, n_ch))
    counts = np.zeros((n_bins, n_ch))
    bin_size = max(1, T // n_bins)
    for t in range(T):
        I_in = W_in[:, 0:1] * X_obs[t][None, :]
        I_rec = W_rec @ spk
        mem = (1 - beta) * mem * (1 - spk) + I_in + I_rec
        mem = np.maximum(mem, 0.0)
        spk = (mem >= theta).astype(float)
        mem = mem - spk * theta
        b = min(t // bin_size, n_bins - 1)
        counts[b] += spk.mean(axis=0)
    return counts.reshape(-1)


def raw_signal_pass(X_ds, y, subjects):
    """Bounded-subset raw-signal robustness on a recomputed reservoir embedding."""
    from sklearn.decomposition import PCA
    n = min(cfg.RAW_SIGNAL_SUBSET, X_ds.shape[0])
    rng0 = np.random.default_rng(cfg.RANDOM_STATE)
    # Stratified bounded subset preserving class balance.
    idx = []
    for cls in sorted(set(y.tolist())):
        cls_idx = np.where(y == cls)[0]
        take = min(len(cls_idx), n // len(set(y.tolist())))
        idx.extend(rng0.choice(cls_idx, size=take, replace=False).tolist())
    idx = np.array(sorted(idx))
    Xs, ys, gs = X_ds[idx], y[idx], subjects[idx]
    W_in, W_rec = _init_reservoir()
    label_set = sorted(set(ys.tolist()))

    def embed(batch):
        return np.vstack([_reservoir_bsc6(batch[i], W_in, W_rec) for i in range(batch.shape[0])])

    rows = []
    perturbs = (
        [("temporal_jitter", j) for j in cfg.TEMPORAL_JITTER_MS]
        + [("amplitude_noise", s) for s in cfg.AMPLITUDE_SNR_DB]
        + [("channel_dropout", f) for f in cfg.CHANNEL_DROPOUT_FRAC]
    )
    for seed in cfg.SEEDS:
        cv = cr.subject_grouped_cv(min(3, cfg.N_FOLDS_AFFECTIVE), seed)
        for fold, (tr, te) in enumerate(cv.split(Xs, ys, groups=gs)):
            E_tr = embed(Xs[tr])
            pca = PCA(n_components=min(64, E_tr.shape[0] - 1, E_tr.shape[1]))
            E_tr_p = pca.fit_transform(E_tr)
            for ptype, level in perturbs:
                rng = np.random.default_rng(7000 * seed + fold)
                Xte = Xs[te].copy()
                if ptype == "temporal_jitter" and level:
                    shift = int(round(level * cfg.FS_DS / 1000.0))
                    Xte = np.roll(Xte, shift, axis=1)
                    Xte[:, :max(shift, 0), :] = 0.0
                elif ptype == "amplitude_noise" and level is not None:
                    p = np.mean(Xte ** 2, axis=1, keepdims=True) + 1e-12
                    npow = p / (10 ** (level / 10.0))
                    Xte = Xte + rng.standard_normal(Xte.shape) * np.sqrt(npow)
                elif ptype == "channel_dropout" and level:
                    n_drop = int(round(level * Xte.shape[2]))
                    if n_drop:
                        drop = rng.choice(Xte.shape[2], size=n_drop, replace=False)
                        Xte[:, :, drop] = 0.0
                E_te_p = pca.transform(embed(Xte))
                pred, proba, _ = cr.fit_logreg_fold(E_tr_p, ys[tr], E_te_p, seed=seed)
                mb = cr.metric_bundle(ys[te], pred, proba, labels=label_set)
                lvl = "clean" if level in (None, 0, 0.0) else level
                rows.append({
                    "pass": "raw_signal", "config": "E_reservoir_recomputed",
                    "perturbation": ptype, "level": lvl, "seed": seed, "fold": fold,
                    "balanced_accuracy": mb["balanced_accuracy"],
                    "macro_f1": mb["macro_f1"], "roc_auc": mb["roc_auc"],
                    "n_test": mb["n"],
                })
    return rows, int(len(idx))


def _summarize(metric_rows):
    from collections import defaultdict
    groups = defaultdict(list)
    for r in metric_rows:
        groups[(r["pass"], r["config"], r["perturbation"], r["level"])].append(r)
    out = []
    for (pas, conf, pert, lvl), rs in groups.items():
        ba = [r["balanced_accuracy"] for r in rs]
        lo, hi = cr.bootstrap_ci(ba)
        out.append({
            "pass": pas, "config": conf, "perturbation": pert, "level": lvl,
            "balanced_accuracy_mean": round(float(np.mean(ba)), 4),
            "balanced_accuracy_ci_lo": round(lo, 4),
            "balanced_accuracy_ci_hi": round(hi, 4),
            "macro_f1_mean": round(float(np.mean([r["macro_f1"] for r in rs])), 4),
            "n_eval": len(rs),
        })
    return out


def main() -> int:
    try:
        data = cr.load_ready9()
    except Exception as e:
        (cfg.OUT_DIR / "ROBUSTNESS_ERROR.md").write_text(
            f"# Robustness phase could not run\n\nInput load failed: {e}\n")
        print(f"[robust] FAILED to load inputs: {e}", file=sys.stderr)
        return 1

    blocks, y, subjects = data["blocks"], data["y"], data["subjects"]
    ch67 = data["ch67"]
    tplv_all = np.asarray(ch67["tPLV_mats"])
    D_perchan = np.asarray(ch67["D"])
    comp_clean = {k: blocks[k] for k in ("BandPower", "E", "D", "T", "C")}

    metric_rows = []

    # Clean baseline (level 0) per config — reuse amplitude_noise with snr=None.
    for config_id in CONFIG_COMPONENTS:
        metric_rows += evaluate_perturbation(config_id, "amplitude_noise", None,
                                             comp_clean, y, subjects, tplv_all, D_perchan)
    # Representation-level perturbations.
    for config_id in CONFIG_COMPONENTS:
        for snr in [s for s in cfg.AMPLITUDE_SNR_DB if s is not None]:
            metric_rows += evaluate_perturbation(config_id, "amplitude_noise", snr,
                                                 comp_clean, y, subjects, tplv_all, D_perchan)
        for frac in [f for f in cfg.CHANNEL_DROPOUT_FRAC if f > 0]:
            metric_rows += evaluate_perturbation(config_id, "channel_dropout", frac,
                                                 comp_clean, y, subjects, tplv_all, D_perchan)
        for frac in [f for f in cfg.GRAPH_PERTURB_FRAC if f > 0]:
            metric_rows += evaluate_perturbation(config_id, "graph_perturbation", frac,
                                                 comp_clean, y, subjects, tplv_all, D_perchan)

    rep_ok = len(metric_rows) > 0
    if not rep_ok:
        (cfg.OUT_DIR / "ROBUSTNESS_ERROR.md").write_text(
            "# Robustness phase failed\n\nRepresentation-level pass produced no "
            "results.\n")
        print("[robust] FAILED: representation-level pass empty", file=sys.stderr)
        return 1

    # Raw-signal bounded-subset pass.
    raw_rows, raw_n = [], 0
    try:
        raw_rows, raw_n = raw_signal_pass(np.asarray(data["ch5"]["X_ds"]), y, subjects)
        metric_rows += raw_rows
    except Exception as e:  # raw pass is bounded/diagnostic, not fatal
        print(f"[robust] raw-signal pass skipped: {e}", file=sys.stderr)

    cr.write_csv(cfg.ANALYSIS_DIR / "robustness_metrics.csv", metric_rows)
    cr.write_csv(cfg.ANALYSIS_DIR / "robustness_summary.csv", _summarize(metric_rows))
    cr.write_json(cfg.ANALYSIS_DIR / "robustness_config.json", {
        "provenance": cr.base_provenance(),
        "representation_level": {
            "configs": list(CONFIG_COMPONENTS.keys()),
            "amplitude_snr_db": cfg.AMPLITUDE_SNR_DB,
            "channel_dropout_frac": cfg.CHANNEL_DROPOUT_FRAC,
            "graph_perturb_frac": cfg.GRAPH_PERTURB_FRAC,
            "scope": "full dataset, all 10 configurations",
        },
        "raw_signal": {
            "stream": "E_reservoir_recomputed",
            "perturbations": ["temporal_jitter", "amplitude_noise", "channel_dropout"],
            "temporal_jitter_ms": cfg.TEMPORAL_JITTER_MS,
            "subset_n_observations": raw_n,
            "scope": ("bounded diagnostic subset" if raw_n < y.shape[0] else "full dataset"),
            "note": "reservoir embedding recomputed within-fold; PCA fit on clean train",
        },
        "n_metric_rows": len(metric_rows),
    })
    print(f"[robust] representation rows + raw rows = {len(metric_rows)} "
          f"(raw subset n={raw_n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
