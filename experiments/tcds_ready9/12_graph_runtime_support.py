#!/usr/bin/env python3
"""Phase 9 — Bounded graph-support and runtime/resource audit.

Provides only the graph evidence needed to support the reservoir-graph substrate
claim for this paper: mean tPLV adjacency, basic graph observables (degree,
Laplacian spectrum summary, density, connectedness), Dirichlet energy, and
structure-function coupling kappa. Plus a runtime/resource summary. This phase
is deliberately bounded -- it does NOT develop graph spectral/diffusion theory,
and it makes NO energy claims.

Outputs:
    outputs/tcds_ready9/analysis/graph_support_metrics.csv
    outputs/tcds_ready9/analysis/pipeline_runtime_resource_summary.csv
    tables/tcds_ready9/table_graph_support_observables.tex
    tables/tcds_ready9/table_pipeline_runtime_resource.tex
    figures/tcds_ready9/analysis/ana11_graph_perturbation_support.pdf
    figures/tcds_ready9/analysis/ana12_pipeline_runtime_resource.pdf

Run:
    python experiments/tcds_ready9/12_graph_runtime_support.py
"""
from __future__ import annotations

import sys
import time
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


def _graph_observables(A: np.ndarray) -> dict:
    """Basic, bounded graph observables for a single adjacency matrix."""
    n = A.shape[0]
    np.fill_diagonal(A, 0.0)
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    density = float(A.sum() / (n * (n - 1)))
    # connectedness via Laplacian zero-eigenvalue multiplicity (threshold graph)
    Ab = (A > A.mean()).astype(float)
    db = Ab.sum(axis=1)
    Lb = np.diag(db) - Ab
    evals = np.linalg.eigvalsh(Lb)
    n_components = int(np.sum(evals < 1e-8))
    lap_evals = np.linalg.eigvalsh(L)
    return {
        "mean_degree": float(deg.mean()),
        "graph_density": density,
        "n_components_thresholded": n_components,
        "algebraic_connectivity": float(sorted(lap_evals)[1]) if n > 1 else 0.0,
        "laplacian_spectral_max": float(lap_evals.max()),
    }


def _dirichlet_energy(A: np.ndarray, signal: np.ndarray) -> float:
    """x^T L x for a channel-level signal x over the graph (bounded support)."""
    np.fill_diagonal(A, 0.0)
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    x = signal - signal.mean()
    return float(x @ L @ x)


def main() -> int:
    t_load0 = time.perf_counter()
    data = cr.load_ready9()
    t_load = time.perf_counter() - t_load0

    y = data["y"]; ch67 = data["ch67"]
    tplv = np.asarray(ch67["tPLV_mats"])
    D_pc = np.asarray(ch67["D"]); T_pc = np.asarray(ch67["T_topo"])

    # ── Graph observables by condition ──────────────────────────────────
    graph_rows = []
    mean_adj = {}
    for c in sorted(set(y.tolist())):
        A = tplv[y == c].mean(axis=0).copy()
        mean_adj[c] = A
        obs = _graph_observables(A.copy())
        # Dirichlet energy using mean dynamical descriptor (channel signal)
        chan_signal = D_pc[y == c].mean(axis=0)[:, 0]  # first dynamical metric per channel
        obs["dirichlet_energy_meanD"] = _dirichlet_energy(A.copy(), chan_signal)
        obs["condition"] = COND[c]
        graph_rows.append(obs)

    t_kappa0 = time.perf_counter()
    kappa = cr.compute_coupling_block(D_pc, T_pc)[:, 0]
    t_kappa = time.perf_counter() - t_kappa0
    for r in graph_rows:
        c = [k for k, v in COND.items() if v == r["condition"]][0]
        r["kappa_mean"] = float(np.mean(kappa[y == c]))
    cr.write_csv(cfg.ANALYSIS_DIR / "graph_support_metrics.csv", graph_rows)

    # ── Runtime / resource summary ──────────────────────────────────────
    blocks = data["blocks"]
    t0 = time.perf_counter(); _ = blocks["E"]; _ = blocks["D"]; _ = blocks["T"]
    t_assemble = time.perf_counter() - t0
    t0 = time.perf_counter(); _ = cr.compute_coupling_block(D_pc[:50], T_pc[:50])
    t_coupling50 = time.perf_counter() - t0
    # classifier eval timing (single fold, A8)
    X = np.hstack([blocks["E"], blocks["D"], blocks["T"]])
    cv = cr.subject_grouped_cv(cfg.N_FOLDS_AFFECTIVE, 42)
    tr, te = next(iter(cv.split(X, y, groups=data["subjects"])))
    t0 = time.perf_counter(); cr.fit_logreg_fold(X[tr], y[tr], X[te], seed=42)
    t_clf = time.perf_counter() - t0

    runtime_rows = [
        {"stage": "private_input_load", "seconds": round(t_load, 3), "note": "two pickles + clinical CSV"},
        {"stage": "feature_block_assembly", "seconds": round(t_assemble, 4), "note": "reshape/access"},
        {"stage": "coupling_block_50obs", "seconds": round(t_coupling50, 4), "note": "kappa over 50 observations"},
        {"stage": "kappa_full_dataset", "seconds": round(t_kappa, 3), "note": "all observations"},
        {"stage": "classifier_fit_eval_1fold", "seconds": round(t_clf, 3), "note": "A8 substrate, L2 logistic"},
    ]
    resource_rows = [
        {"resource": "reservoir_neurons", "value": 256, "note": "fixed LIF reservoir per channel"},
        {"resource": "reservoir_recurrent_weights", "value": 256 * 256, "note": "W_rec entries"},
        {"resource": "channels", "value": cfg.N_CHANNELS, "note": "electrodes"},
        {"resource": "embedding_dim_E", "value": cfg.EXPECTED_DIMS["E"], "note": "34 x 64 PCA"},
        {"resource": "graph_nodes", "value": cfg.N_CHANNELS, "note": "tPLV graph"},
        {"resource": "graph_edges_full", "value": cfg.N_CHANNELS * (cfg.N_CHANNELS - 1) // 2, "note": "undirected"},
    ]
    cr.write_csv(cfg.ANALYSIS_DIR / "pipeline_runtime_resource_summary.csv",
                 runtime_rows + [{"stage": r["resource"], "seconds": r["value"], "note": r["note"]}
                                 for r in resource_rows])

    _write_tables(graph_rows, runtime_rows, resource_rows)
    _figures(mean_adj, kappa, y, runtime_rows)

    cr.write_json(cfg.ANALYSIS_DIR / "graph_runtime_provenance.json", {
        "provenance": cr.base_provenance(),
        "scope": "bounded graph support for the reservoir-graph substrate claim; "
                 "no spectral/diffusion theory; no energy claim",
        "inputs": {"ch6_ch7_3class_features": cr.file_fingerprint(cfg.CH67_FILE)},
    })
    print(f"[graph] graph rows={len(graph_rows)}; runtime stages={len(runtime_rows)}")
    return 0


def _write_tables(graph_rows, runtime_rows, resource_rows):
    # graph observables table
    lines = [
        r"\begin{table}[t]\centering",
        r"\caption{Bounded graph observables of the theta-band tPLV adjacency by "
        r"condition. Reported to support the reservoir-graph substrate; a full "
        r"graph spectral analysis is out of scope for this paper.}",
        r"\label{tab:graph_support}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Condition & Density & Alg.\ conn. & Dirichlet($\bar D$) & $\bar\kappa$ \\",
        r"\midrule",
    ]
    for r in graph_rows:
        lines.append(f"{r['condition']} & {r['graph_density']:.3f} & "
                     f"{r['algebraic_connectivity']:.3f} & "
                     f"{r['dirichlet_energy_meanD']:.2f} & {r['kappa_mean']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (cfg.TABLE_DIR / "table_graph_support_observables.tex").write_text("\n".join(lines))

    # runtime/resource table
    lines = [
        r"\begin{table}[t]\centering",
        r"\caption{Pipeline runtime and resource summary. Timings are wall-clock "
        r"on the analysis host; no hardware-energy measurements are claimed.}",
        r"\label{tab:runtime_resource}",
        r"\begin{tabular}{lr}",
        r"\toprule", r"Stage / resource & Value \\", r"\midrule",
    ]
    for r in runtime_rows:
        lines.append(f"{r['stage'].replace('_', ' ')} & {r['seconds']:.3f} s \\\\")
    for r in resource_rows:
        lines.append(f"{r['resource'].replace('_', ' ')} & {int(r['value'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (cfg.TABLE_DIR / "table_pipeline_runtime_resource.tex").write_text("\n".join(lines))


def _figures(mean_adj, kappa, y, runtime_rows):
    # ana11 — graph perturbation support: mean adjacency + density under dropout
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    A = mean_adj[sorted(mean_adj)[0]]
    im = axes[0].imshow(A, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("Mean tPLV adjacency (Negative)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    fracs = cfg.GRAPH_PERTURB_FRAC
    dens = []
    rng = np.random.default_rng(0)
    for f in fracs:
        Ap = A.copy()
        n = A.shape[0]; iu = np.triu_indices(n, 1)
        nd = int(round(f * len(iu[0])))
        if nd:
            d = rng.choice(len(iu[0]), nd, replace=False)
            Ap[iu[0][d], iu[1][d]] = 0; Ap[iu[1][d], iu[0][d]] = 0
        np.fill_diagonal(Ap, 0)
        dens.append(Ap.sum() / (n * (n - 1)))
    axes[1].plot([int(f * 100) for f in fracs], dens, marker="o", color="#C44E52")
    axes[1].set_xlabel("edge dropout (%)"); axes[1].set_ylabel("graph density")
    axes[1].set_title("Graph density under edge dropout")
    fig.tight_layout()
    fig.savefig(cfg.FIG_ANA_DIR / "ana11_graph_perturbation_support.pdf", bbox_inches="tight")
    plt.close(fig)

    # ana12 — runtime/resource bar
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [r["stage"].replace("_", "\n") for r in runtime_rows]
    secs = [r["seconds"] for r in runtime_rows]
    ax.bar(names, secs, color="#4C72B0")
    ax.set_ylabel("seconds (wall-clock)")
    ax.set_title("Pipeline stage runtime (no energy claim)")
    fig.tight_layout()
    fig.savefig(cfg.FIG_ANA_DIR / "ana12_pipeline_runtime_resource.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
