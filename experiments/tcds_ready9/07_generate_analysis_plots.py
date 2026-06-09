#!/usr/bin/env python3
"""Phase 10 — Analysis figures (ana01-ana10).

Consumes the aggregate CSV/JSON produced by the mechanism-ablation, robustness,
closed-loop, and graph/runtime phases. ana10 is an *evaluation-coverage* figure
organised around ARSPI-Net components and system-level evaluation requirements
(it is NOT a named accepted-paper comparison; that comparison is kept internal).

Run after phases 6-9:
    python experiments/tcds_ready9/07_generate_analysis_plots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.tcds_ready9 import config as cfg  # noqa: E402
from experiments.tcds_ready9 import common_ready9 as cr  # noqa: E402

A = cfg.ANALYSIS_DIR


def _read(name):
    p = A / name
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None


def _save(fig, fid):
    fig.savefig(cfg.FIG_ANA_DIR / f"{fid}.pdf", bbox_inches="tight")
    plt.close(fig)
    stale = A / f"{fid}_SKIPPED.md"
    if stale.exists():
        stale.unlink()
    cr.write_json(cfg.FIG_ANA_DIR / f"{fid}.json", {
        "source_script": "experiments/tcds_ready9/07_generate_analysis_plots.py",
        "figure_type": "analysis", "provenance": cr.base_provenance(),
        "privacy_status": cfg.PRIVACY_STATUS,
        "output_path": f"figures/tcds_ready9/analysis/{fid}.pdf",
    })
    print(f"[ana] wrote {fid}.pdf")


def _skip(fid, reason):
    (A / f"{fid}_SKIPPED.md").write_text(f"# {fid} skipped\n\n{reason}\n")
    print(f"[ana] {fid} SKIPPED: {reason}")


def ana01(df):
    if df is None:
        return _skip("ana01_mechanism_ablation_performance", "mechanism_ablation_summary.csv missing")
    d = df[df["config"].str.startswith("A")].copy()
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(d))
    yerr = [d["balanced_accuracy_mean"] - d["balanced_accuracy_ci_lo"],
            d["balanced_accuracy_ci_hi"] - d["balanced_accuracy_mean"]]
    ax.bar(x - 0.2, d["balanced_accuracy_mean"], 0.4, yerr=yerr, capsize=3,
           label="balanced accuracy", color="#4C72B0")
    ax.bar(x + 0.2, d["macro_f1_mean"], 0.4, label="macro-F1", color="#55A868")
    ax.axhline(d["chance"].iloc[0], ls="--", color="k", label="chance")
    ax.set_xticks(x); ax.set_xticklabels(d["config"], rotation=0)
    ax.set_ylabel("score"); ax.set_title("Mechanism ablation performance (A0-A9)")
    ax.legend()
    _save(fig, "ana01_mechanism_ablation_performance")


def ana02(df_abl, df_rob, df_clin, df_loop):
    if df_abl is None:
        return _skip("ana02_mechanism_functional_roles", "ablation summary missing")
    streams = ["E", "D", "T", "C"]
    roles = ["affective\nclassification", "robustness", "clinical\nsensitivity",
             "topology/\ncoupling", "closed-loop\nutility"]
    M = np.zeros((len(streams), len(roles)))
    amap = {"A1": "E", "A2": "D", "A3": "T", "A4": "C"}
    for _, r in df_abl.iterrows():
        if r["config"] in amap:
            M[streams.index(amap[r["config"]]), 0] = r["balanced_accuracy_mean"]
    if df_rob is not None:
        for s in streams:
            sub = df_rob[(df_rob["config"] == {"E": "A1", "D": "A2", "T": "A3", "C": "A4"}[s])]
            if len(sub):
                M[streams.index(s), 1] = sub["balanced_accuracy_mean"].mean()
    # topology/coupling: T and C carry it by construction
    M[streams.index("T"), 3] = 1.0; M[streams.index("C"), 3] = 1.0
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(M, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(roles))); ax.set_xticklabels(roles, fontsize=8)
    ax.set_yticks(range(len(streams))); ax.set_yticklabels(streams)
    for i in range(len(streams)):
        for j in range(len(roles)):
            if M[i, j]:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Mechanism functional-role matrix (relative emphasis)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, "ana02_mechanism_functional_roles")


def ana03_04(df):
    if df is None:
        return _skip("ana03_robustness_degradation_curves", "robustness_summary.csv missing")
    rep = df[df["pass"] == "representation_level"].copy()
    perts = ["amplitude_noise", "channel_dropout", "graph_perturbation"]
    show_configs = ["A0", "A1", "A2", "A3", "A8"]
    fig, axes = plt.subplots(1, len(perts), figsize=(13, 4), sharey=True)
    for ax, pert in zip(axes, perts):
        sub = rep[rep["perturbation"].isin([pert, "amplitude_noise"]) & (rep["perturbation"] == pert)]
        for conf in show_configs:
            cc = sub[sub["config"] == conf].copy()
            if not len(cc):
                continue
            cc["lvl"] = pd.to_numeric(cc["level"], errors="coerce")
            cc = cc.sort_values("lvl")
            ax.plot(cc["lvl"], cc["balanced_accuracy_mean"], marker="o", label=conf)
        ax.set_title(pert.replace("_", " ")); ax.set_xlabel("level")
    axes[0].set_ylabel("balanced accuracy"); axes[0].legend(fontsize=8)
    fig.suptitle("Robustness degradation by perturbation (representation level)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "ana03_robustness_degradation_curves")

    # ana04 — degradation slope magnitude per stream
    fig, ax = plt.subplots(figsize=(8, 4))
    configs = sorted(rep["config"].unique())
    slopes = []
    for conf in configs:
        cc = rep[(rep["config"] == conf) & (rep["perturbation"] == "amplitude_noise")].copy()
        cc["lvl"] = pd.to_numeric(cc["level"], errors="coerce")
        cc = cc.dropna(subset=["lvl"]).sort_values("lvl")
        if len(cc) >= 2:
            slopes.append((conf, np.polyfit(cc["lvl"], cc["balanced_accuracy_mean"], 1)[0]))
    if slopes:
        ax.bar([s[0] for s in slopes], [abs(s[1]) for s in slopes], color="#C44E52")
        ax.set_ylabel("|slope| BA vs SNR"); ax.set_title("Amplitude-noise sensitivity by configuration")
        _save(fig, "ana04_robustness_summary")
    else:
        plt.close(fig); _skip("ana04_robustness_summary", "insufficient robustness levels")


def ana05(df):
    if df is None or "diagnosis" not in getattr(df, "columns", []):
        return _skip("ana05_clinical_label_sensitivity", "clinical_label_sensitivity.csv missing")
    d = df[df.get("status") == "evaluated"].copy()
    if not len(d):
        return _skip("ana05_clinical_label_sensitivity", "no diagnoses had sufficient support")
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(d))
    ax.bar(x - 0.2, d["balanced_accuracy_mean"], 0.4, label="balanced accuracy", color="#4C72B0")
    ax.bar(x + 0.2, d["roc_auc_mean"], 0.4, label="ROC-AUC", color="#8172B3")
    ax.axhline(0.5, ls="--", color="k")
    ax.set_xticks(x); ax.set_xticklabels(d["diagnosis"])
    ax.set_title("Exploratory clinical-label sensitivity (not diagnostic validation)")
    ax.set_ylabel("score"); ax.legend()
    _save(fig, "ana05_clinical_label_sensitivity")


def ana06():
    if not cfg.CH67_FILE.exists():
        return _skip("ana06_kappa_shuffle_null", "ch67 unavailable")
    import pickle
    with open(cfg.CH67_FILE, "rb") as f:
        ch67 = pickle.load(f)
    D_pc = np.asarray(ch67["D"]); T_pc = np.asarray(ch67["T_topo"])
    kappa = cr.compute_coupling_block(D_pc, T_pc)[:, 0]
    rng = np.random.default_rng(1)
    null = []
    idx = rng.choice(len(kappa), size=min(200, len(kappa)), replace=False)
    for i in idx:
        perm = rng.permutation(cfg.N_CHANNELS)
        null.append(cr.compute_coupling_block(D_pc[i:i + 1], T_pc[i:i + 1][:, perm, :])[0, 0])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(kappa, bins=40, alpha=0.7, color="#4C72B0", density=True, label="observed")
    ax.hist(null, bins=40, alpha=0.7, color="#999999", density=True, label="electrode-shuffle null")
    ax.set_xlabel("$\\kappa$"); ax.set_ylabel("density")
    ax.set_title("Structure-function coupling vs electrode-shuffle null")
    ax.legend()
    _save(fig, "ana06_kappa_shuffle_null")


def ana07_08_09(df):
    if df is None:
        for f in ("ana07_closed_loop_success_by_policy", "ana08_closed_loop_entropy_steps",
                  "ana09_closed_loop_failure_modes"):
            _skip(f, "closed_loop_policy_summary.csv missing")
        return
    policies = list(dict.fromkeys(df["policy"]))
    eps = sorted(df["epsilon"].unique())
    # ana07 success by policy and epsilon with Wilson CI
    fig, ax = plt.subplots(figsize=(8, 4))
    for pol in policies:
        d = df[df["policy"] == pol].sort_values("epsilon")
        yerr = [d["success_rate"] - d["success_ci_lo"], d["success_ci_hi"] - d["success_rate"]]
        ax.errorbar(d["epsilon"], d["success_rate"], yerr=yerr, marker="o", capsize=3, label=pol)
    ax.set_xlabel("transition noise $\\epsilon$"); ax.set_ylabel("success rate")
    ax.set_title("Closed-loop success by policy (Wilson 95% CI)")
    ax.legend(fontsize=8)
    _save(fig, "ana07_closed_loop_success_by_policy")
    # ana08 entropy & steps
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for pol in policies:
        d = df[df["policy"] == pol].sort_values("epsilon")
        axes[0].plot(d["epsilon"], d["final_entropy_mean"], marker="o", label=pol)
        axes[1].plot(d["epsilon"], d["mean_steps"], marker="o", label=pol)
    axes[0].set_title("Final posterior entropy"); axes[0].set_xlabel("$\\epsilon$")
    axes[0].set_ylabel("entropy (nats)")
    axes[1].set_title("Mean steps"); axes[1].set_xlabel("$\\epsilon$"); axes[1].set_ylabel("steps")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "ana08_closed_loop_entropy_steps")
    # ana09 failure rate by policy/epsilon
    fig, ax = plt.subplots(figsize=(8, 4))
    for pol in policies:
        d = df[df["policy"] == pol].sort_values("epsilon")
        ax.plot(d["epsilon"], d["failure_rate"], marker="o", label=pol)
    ax.set_xlabel("$\\epsilon$"); ax.set_ylabel("failure rate")
    ax.set_title("Closed-loop failure rate by policy and transition noise")
    ax.legend(fontsize=8)
    _save(fig, "ana09_closed_loop_failure_modes")


def ana10():
    """Evaluation-coverage matrix: ARSPI-Net components x evaluation requirements."""
    comps = ["E (reservoir embed)", "D (dynamical)", "T (graph topology)",
             "C (coupling)", "Closed-loop control"]
    reqs = ["measured\nsignal", "neural\nmechanism", "perturbation\nanalysis",
            "ablation", "closed-loop\neval", "runtime/\nresource"]
    M = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 1],
        [0, 1, 1, 0, 1, 1],
    ], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(M, cmap="Greens", vmin=0, vmax=1.4, aspect="auto")
    ax.set_xticks(range(len(reqs))); ax.set_xticklabels(reqs, fontsize=8)
    ax.set_yticks(range(len(comps))); ax.set_yticklabels(comps, fontsize=9)
    for i in range(len(comps)):
        for j in range(len(reqs)):
            ax.text(j, i, "✓" if M[i, j] else "·", ha="center", va="center")
    ax.set_title("Evaluation coverage of ARSPI-Net components")
    _save(fig, "ana10_evaluation_coverage")


def main() -> int:
    ana01(_read("mechanism_ablation_summary.csv"))
    ana02(_read("mechanism_ablation_summary.csv"), _read("robustness_summary.csv"),
          _read("clinical_label_sensitivity.csv"), _read("closed_loop_policy_summary.csv"))
    ana03_04(_read("robustness_summary.csv"))
    ana05(_read("clinical_label_sensitivity.csv"))
    ana06()
    ana07_08_09(_read("closed_loop_policy_summary.csv"))
    ana10()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
