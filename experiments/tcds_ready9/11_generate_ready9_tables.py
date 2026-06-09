#!/usr/bin/env python3
"""Phase 11 — LaTeX table generation (CORE phase).

Generates the manuscript tables not already emitted by their source scripts:
neural mechanisms, reproducibility panel, comparison positioning, robustness
summary, closed-loop policy, and evaluation coverage. (Dataset provenance,
mechanism ablation, graph support, and runtime/resource tables are written by
their respective phases.)

The evaluation-coverage table is organised around ARSPI-Net components and
system-level evaluation requirements -- it is NOT a named accepted-paper
comparison.

Core/fatal: exits nonzero if the robustness or closed-loop summaries (which feed
required tables) are absent.

Run:
    python experiments/tcds_ready9/11_generate_ready9_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.tcds_ready9 import config as cfg  # noqa: E402
from experiments.tcds_ready9 import common_ready9 as cr  # noqa: E402

T = cfg.TABLE_DIR
A = cfg.ANALYSIS_DIR


def _write(name, lines):
    (T / name).write_text("\n".join(lines))
    print(f"[tables] wrote {name}")


def neural_mechanisms():
    rows = [
        ("Event-driven coding", "BSC$_6$ binning", "temporal-bin spike-count vector",
         "bin evidence", "sparse perceptual observation"),
        ("Recurrent cortical dynamics", "LIF reservoir", "reservoir state trajectory",
         "embedding $E$", "nonlinear temporal transformation"),
        ("Temporal state statistics", "descriptor block $D$", "trajectory descriptors",
         "variance, autocorrelation, complexity", "dynamical evidence stream"),
        ("Functional connectivity", "tPLV graph", "adjacency / graph operator",
         "topology descriptors $T$", "spatial dependency"),
        ("Structure-function coupling", "$\\kappa$", "coupling readout",
         "shuffle-null significance", "systems-level coordination"),
        ("Embodied perception", "posterior update + policy", "belief state",
         "entropy, success, steps", "closed-loop perceptual utility"),
    ]
    out = [
        r"\begin{table*}[t]\centering",
        r"\caption{Neural mechanisms implemented by ARSPI-Net, each anchored to a "
        r"concrete computational object and a measured observable.}",
        r"\label{tab:neural_mechanisms}",
        r"\begin{tabular}{p{2.6cm}p{2.2cm}p{2.8cm}p{3.0cm}p{3.0cm}}",
        r"\toprule",
        r"Neural principle & Implementation & Computational object & Measured "
        r"observable & Relevance to embodied AI \\",
        r"\midrule",
    ]
    for r in rows:
        out.append(" & ".join(r) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    _write("table_neural_mechanisms.tex", out)


def reproducibility_panel():
    commit = (cr.repo_commit_hash() or "n/a")[:10]
    up = cr.upstream_commit_hash()
    up = (up[:10] if up else "n/a")
    rows = [
        ("Subjects", "211 (subject 127 excluded)"),
        ("Observations", "633"),
        ("Affective classes", "3 (Negative, Neutral, Pleasant)"),
        ("Channels", "34"),
        ("Split protocol", "subject-grouped cross-validation"),
        ("Feature blocks", "BandPower, $E$, $D$, $T$, $C$"),
        ("Perturbation types", "temporal jitter, amplitude noise, channel dropout, graph"),
        ("Random seeds", ", ".join(str(s) for s in cfg.SEEDS) + f" (profile: {cfg.PROFILE})"),
        ("Closed-loop episodes", f"$\\geq${cfg.CLOSED_LOOP_EPISODES} per policy/$\\epsilon$"),
        ("Policy set", "passive, random, pragmatic, epistemic, EFE, oracle"),
        ("Repo commit", f"\\texttt{{{commit}}}"),
        ("Upstream commit", f"\\texttt{{{up}}}"),
        ("Data availability", "restricted; aggregate/deidentified outputs only"),
    ]
    out = [
        r"\begin{table}[t]\centering",
        r"\caption{Reproducibility panel for the evaluation. Subject-level data are "
        r"restricted; released artifacts are aggregate or hashed.}",
        r"\label{tab:reproducibility}",
        r"\begin{tabular}{ll}", r"\toprule", r"Item & Value \\", r"\midrule",
    ]
    for k, v in rows:
        out.append(f"{k} & {v} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    _write("table_reproducibility_panel.tex", out)


def comparison_positioning():
    out = [
        r"\begin{table}[t]\centering",
        r"\caption{Positioning of the ARSPI-Net evaluation against the evaluation "
        r"practices expected of systems papers in this area. Entries describe what "
        r"is reported in this paper, not a ranking against other systems.}",
        r"\label{tab:comparison_positioning}",
        r"\begin{tabular}{p{3.2cm}p{4.4cm}}",
        r"\toprule", r"Evaluation aspect & ARSPI-Net treatment \\", r"\midrule",
        r"Signal provenance & verified source, QC, subject-level splits \\",
        r"Mechanism definition & each block anchored to a named neural mechanism \\",
        r"Ablation & A0--A9 with negative controls \\",
        r"Robustness & 4 perturbation families, representation + raw-signal \\",
        r"Closed-loop evaluation & simulated affective-control loop, defined EFE controller \\",
        r"Uncertainty & bootstrap / Wilson intervals, confusion matrices \\",
        r"Reproducibility & manifests, fingerprints, reproduction map \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ]
    _write("table_comparison_positioning.tex", out)


def robustness_summary():
    df = _maybe(A / "robustness_summary.csv")
    if df is None:
        return False
    rep = df[df["pass"] == "representation_level"].copy()
    out = [
        r"\begin{table}[t]\centering\footnotesize",
        r"\caption{Representation-level robustness summary: mean balanced accuracy "
        r"under each perturbation family for representative configurations.}",
        r"\label{tab:robustness_summary}",
        r"\begin{tabular}{llrr}", r"\toprule",
        r"Config & Perturbation & Level & BA \\", r"\midrule",
    ]
    for conf in ["A0", "A1", "A2", "A3", "A8"]:
        sub = rep[rep["config"] == conf]
        for _, r in sub.iterrows():
            if r["perturbation"] in ("amplitude_noise", "channel_dropout", "graph_perturbation"):
                out.append(f"{conf} & {r['perturbation'].replace('_',' ')} & "
                           f"{r['level']} & {r['balanced_accuracy_mean']:.3f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    _write("table_robustness_summary.tex", out)
    return True


def closed_loop_policy():
    df = _maybe(A / "closed_loop_policy_summary.csv")
    if df is None:
        return False
    out = [
        r"\begin{table}[t]\centering\footnotesize",
        r"\caption{Closed-loop policy comparison. Success rate (Wilson 95\% CI), "
        r"mean steps, and final posterior entropy by transition noise $\epsilon$.}",
        r"\label{tab:closed_loop_policy}",
        r"\begin{tabular}{llp{2.6cm}rr}", r"\toprule",
        r"Policy & $\epsilon$ & Success & Mean steps & Final $\mathcal{H}$ \\", r"\midrule",
    ]
    for _, r in df.iterrows():
        out.append(f"{r['policy'].replace('_',' ')} & {r['epsilon']} & "
                   f"{r['success_rate']:.3f} [{r['success_ci_lo']:.3f}, {r['success_ci_hi']:.3f}] & "
                   f"{r['mean_steps']:.2f} & {r['final_entropy_mean']:.3f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    _write("table_closed_loop_policy.tex", out)
    return True


def evaluation_coverage():
    comps = ["$E$ (reservoir embedding)", "$D$ (dynamical)", "$T$ (graph topology)",
             "$C$ (coupling)", "Closed-loop control"]
    cols = ["Signal", "Mechanism", "Perturb.", "Ablation", "Closed-loop", "Runtime"]
    M = [
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 1],
        [0, 1, 1, 0, 1, 1],
    ]
    out = [
        r"\begin{table}[t]\centering",
        r"\caption{Evaluation coverage of ARSPI-Net relative to system-level "
        r"requirements. Rows are ARSPI-Net components; columns are evaluation "
        r"requirements addressed in this paper.}",
        r"\label{tab:evaluation_coverage}",
        r"\begin{tabular}{l" + "c" * len(cols) + r"}", r"\toprule",
        "Component & " + " & ".join(cols) + r" \\", r"\midrule",
    ]
    for name, row in zip(comps, M):
        marks = " & ".join(r"\checkmark" if v else r"$\cdot$" for v in row)
        out.append(f"{name} & {marks} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    _write("table_evaluation_coverage.tex", out)


def _maybe(p: Path):
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None


def main() -> int:
    neural_mechanisms()
    reproducibility_panel()
    comparison_positioning()
    evaluation_coverage()
    ok_rob = robustness_summary()
    ok_loop = closed_loop_policy()
    if not (ok_rob and ok_loop):
        missing = []
        if not ok_rob:
            missing.append("robustness_summary.csv")
        if not ok_loop:
            missing.append("closed_loop_policy_summary.csv")
        (cfg.OUT_DIR / "TABLES_ERROR.md").write_text(
            "# Table generation incomplete\n\nMissing required inputs: "
            + ", ".join(missing) + "\n")
        print(f"[tables] FAILED: missing {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
