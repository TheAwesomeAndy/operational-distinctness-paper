#!/usr/bin/env python3
"""Regenerate the lsm_bsc6_pca embedding inside data/shape_features_211.pkl.

The pickle as shipped contains a valid X_ds (downsampled, z-scored EEG) and
valid conv_feats (BandPower) but a placeholder all-zero lsm_bsc6_pca tensor.
This script reuses the existing X_ds — no raw EEG re-loading — to compute
the BSC6+PCA-64 reservoir embedding the operational-distinctness pipeline
expects.

Algorithm and constants are the same as the upstream Chapter 5 / Chapter 4
LIF reservoir specification (N=256, beta=0.05, theta=0.5, seed=42; BSC bins
6 over t=10..70; PCA-64 on the pooled BSC vectors):
    https://github.com/TheAwesomeAndy/dissoAdventureExperiments
        chapter5Experiments/run_chapter5_experiments.py
        experiments/ch5_4class/ch5_4class_01_feature_extraction.py

Usage:
    python -u prepare_inputs/extract_ch5_features.py \
        [--input  data/shape_features_211.pkl] \
        [--output data/shape_features_211.pkl]
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

# ── reservoir + BSC + PCA constants (must match Ch4/Ch5 upstream) ─────────
N_RES = 256
BETA = 0.05
THRESHOLD = 0.5
SEED = 42
BSC_N_BINS = 6
BSC_T_START = 10
BSC_T_END = 70
PCA_N_COMPONENTS = 64


class LIFReservoir:
    """Leaky Integrate-and-Fire reservoir, byte-equivalent to the upstream Ch4/Ch5 spec.

    - Xavier-uniform W_in, W_rec
    - Spectral radius scaled to 0.9
    - Multiplicative membrane reset on spike, hard threshold subtract, floor at 0.
    """

    def __init__(self, n_input: int, n_res: int, beta: float, threshold: float, seed: int):
        rng = np.random.RandomState(seed)
        limit_in = np.sqrt(6.0 / (n_input + n_res))
        self.W_in = rng.uniform(-limit_in, limit_in, (n_res, n_input))
        limit_rec = np.sqrt(6.0 / (n_res + n_res))
        self.W_rec = rng.uniform(-limit_rec, limit_rec, (n_res, n_res))
        eigenvalues = np.abs(np.linalg.eigvals(self.W_rec))
        if eigenvalues.max() > 0:
            self.W_rec *= 0.9 / eigenvalues.max()
        self.beta = beta
        self.threshold = threshold
        self.n_res = n_res

    def forward(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        T = X.shape[0]
        mem = np.zeros(self.n_res)
        spk_prev = np.zeros(self.n_res)
        spikes = np.zeros((T, self.n_res))
        for t in range(T):
            I_tot = self.W_in @ X[t] + self.W_rec @ spk_prev
            mem = (1.0 - self.beta) * mem * (1.0 - spk_prev) + I_tot
            spk = (mem >= self.threshold).astype(float)
            mem = mem - spk * self.threshold
            mem = np.maximum(mem, 0.0)
            spikes[t] = spk
            spk_prev = spk
        return spikes


def extract_bsc(spikes: np.ndarray, n_bins: int, t_start: int, t_end: int) -> np.ndarray:
    window = spikes[t_start:t_end]
    T_w = window.shape[0]
    bin_size = T_w // n_bins
    bsc = np.zeros((spikes.shape[1], n_bins))
    for b in range(n_bins):
        bsc[:, b] = window[b * bin_size:(b + 1) * bin_size].sum(axis=0)
    return bsc.flatten()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path(os.environ.get("ARSPI_CH5_FILE",
                                                    "data/shape_features_211.pkl")))
    parser.add_argument("--output", type=Path, default=None,
                        help="Defaults to overwriting --input.")
    args = parser.parse_args()

    in_path: Path = args.input.resolve()
    out_path: Path = (args.output or args.input).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input pickle: {in_path}")

    print("=" * 70)
    print("Regenerating lsm_bsc6_pca embedding for shape_features_211.pkl")
    print("=" * 70)
    print(f"Input:  {in_path}")
    print(f"Output: {out_path}")

    print("\nLoading existing pickle (preserves X_ds, conv_feats, y, subjects) ...")
    with open(in_path, "rb") as f:
        data = pickle.load(f)
    print(f"  keys: {sorted(data.keys())}")

    X_ds = np.asarray(data["X_ds"])
    if X_ds.ndim != 3 or X_ds.shape[2] != 34:
        raise ValueError(f"Unexpected X_ds shape {X_ds.shape}; expected (N,T,34).")
    N_obs, T, N_ch = X_ds.shape
    print(f"  X_ds shape: {X_ds.shape}  (N_obs={N_obs}, T={T}, N_ch={N_ch})")
    print(f"  X_ds finite_fraction = {float(np.isfinite(X_ds).mean()):.4f}")

    print(f"\nInitializing reservoir: N={N_RES}, beta={BETA}, theta={THRESHOLD}, seed={SEED}")
    reservoir = LIFReservoir(n_input=1, n_res=N_RES,
                             beta=BETA, threshold=THRESHOLD, seed=SEED)

    print(f"\nProcessing {N_obs} observations ({N_ch} channels each) ...")
    print(f"  BSC{BSC_N_BINS} window: t=[{BSC_T_START}..{BSC_T_END}]\n")
    bsc_dim = N_RES * BSC_N_BINS
    bsc6_raw = np.zeros((N_obs, N_ch, bsc_dim), dtype=np.float64)

    t0 = time.time()
    for obs_i in range(N_obs):
        eeg = X_ds[obs_i]  # (T, 34)
        for ch in range(N_ch):
            spikes = reservoir.forward(eeg[:, ch].reshape(-1, 1))
            bsc6_raw[obs_i, ch] = extract_bsc(spikes, BSC_N_BINS, BSC_T_START, BSC_T_END)
        if (obs_i + 1) % 20 == 0 or obs_i == N_obs - 1:
            elapsed = time.time() - t0
            rate = (obs_i + 1) / elapsed if elapsed > 0 else 0.0
            eta = (N_obs - obs_i - 1) / rate if rate > 0 else 0.0
            print(f"  [{obs_i + 1:4d}/{N_obs}] elapsed={elapsed:.0f}s "
                  f"~{eta:.0f}s remaining ({rate:.2f} obs/s)")

    total_time = time.time() - t0
    print(f"\n  BSC extraction complete: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  bsc6_raw  shape={bsc6_raw.shape}  mean={bsc6_raw.mean():.3f}  "
          f"zero_fraction={(bsc6_raw == 0).mean() * 100:.1f}%")

    print(f"\nFitting PCA-{PCA_N_COMPONENTS} on pooled BSC features ...")
    pooled = bsc6_raw.reshape(-1, bsc_dim)  # (N_obs * N_ch, bsc_dim)
    n_comp = min(PCA_N_COMPONENTS, pooled.shape[0] - 1, pooled.shape[1])
    pca = PCA(n_components=n_comp, random_state=SEED)
    pooled_pca = pca.fit_transform(pooled)
    bsc6_pca = pooled_pca.reshape(N_obs, N_ch, n_comp).astype(np.float64)
    var_explained = float(pca.explained_variance_ratio_.sum() * 100)
    print(f"  PCA-{n_comp}: {var_explained:.2f}% variance explained")
    print(f"  lsm_bsc6_pca shape = {bsc6_pca.shape}")
    print(f"  np.allclose(E, 0) = {bool(np.allclose(bsc6_pca, 0))}")
    print(f"  unique value count = {int(np.unique(bsc6_pca).size)}")

    data["lsm_bsc6_pca"] = bsc6_pca

    tmp_path = out_path.with_suffix(out_path.suffix + ".new")
    print(f"\nWriting updated pickle -> {tmp_path}")
    with open(tmp_path, "wb") as f:
        pickle.dump(data, f, protocol=4)
    os.replace(tmp_path, out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved {out_path} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
