#!/usr/bin/env python3
"""TAFFC-specific figure-polish for Figures 4 and 6 (style-only).

Re-renders the two TAFFC figures
  * ana03_robustness_degradation_curves  (Figure 4)
  * ana07_closed_loop_success_by_policy   (Figure 6)
from the SAME aggregate CSVs used by
``experiments/tcds_ready9/07_generate_analysis_plots.py``, changing ONLY visual
style (line style, marker shape, grayscale-safe contrast, typography, legend) for
grayscale / colour-blind readability. It does not modify any CSV, any shared
figure, or any numerical value, and it writes only to ``figures/taffc/``.

    python experiments/tcds_ready9/taffc_figure_polish.py

The data-selection logic mirrors the shared generator exactly:
  * Figure 4: rows with pass == "representation_level"; configs A0,A1,A2,A3,A8;
    panels amplitude_noise, channel_dropout, graph_perturbation; y =
    balanced_accuracy_mean vs numeric perturbation level.
  * Figure 6: per-policy success_rate vs epsilon with Wilson 95% CI error bars.
Only marker shape, line style, colour, typography, grid, and legend differ from
the shared versions; every plotted (x, y) value is identical.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
_CSV_DIR = _REPO / "outputs" / "tcds_ready9" / "analysis"
_OUT_DIR = _REPO / "figures" / "taffc"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.8,
    "pdf.fonttype": 42,
})

# Grayscale-safe AND colour-blind-safe series styles: each series has a UNIQUE
# (line style, marker) pair, so the series remain distinguishable even when the
# (Okabe-Ito) colours collapse to similar greys. Black marker edges add contrast.
_STYLES = [
    ("#000000", "-",            "o"),  # black   solid           circle
    ("#0072B2", (0, (5, 1.5)),  "s"),  # blue    dashed          square
    ("#E69F00", (0, (1, 1.2)),  "^"),  # orange  dotted          triangle-up
    ("#009E73", (0, (4, 1.2, 1, 1.2)), "D"),  # green dash-dot   diamond
    ("#CC79A7", (0, (1.2, 1.2, 5, 1.2)), "v"),  # pink dot-dash  triangle-down
    ("#56B4E9", (0, (6, 1.2, 1, 1.2, 1, 1.2)), "X"),  # sky dash-dot-dot  X
]


def _sty(i):
    c, ls, mk = _STYLES[i % len(_STYLES)]
    return dict(color=c, linestyle=ls, marker=mk, markersize=5.5,
                markeredgecolor="black", markeredgewidth=0.5, linewidth=1.7)


def fig4_robustness():
    df = pd.read_csv(_CSV_DIR / "robustness_summary.csv")
    rep = df[df["pass"] == "representation_level"].copy()
    perts = ["amplitude_noise", "channel_dropout", "graph_perturbation"]
    titles = {"amplitude_noise": "amplitude noise",
              "channel_dropout": "channel dropout",
              "graph_perturbation": "graph perturbation"}
    configs = ["A0", "A1", "A2", "A3", "A8"]
    fig, axes = plt.subplots(1, len(perts), figsize=(13, 4), sharey=True)
    for ax, pert in zip(axes, perts):
        sub = rep[rep["perturbation"] == pert]
        for i, conf in enumerate(configs):
            cc = sub[sub["config"] == conf].copy()
            if not len(cc):
                continue
            cc["lvl"] = pd.to_numeric(cc["level"], errors="coerce")
            cc = cc.sort_values("lvl")
            ax.plot(cc["lvl"], cc["balanced_accuracy_mean"], label=conf, **_sty(i))
            _dump("fig4", pert, conf, cc["lvl"], cc["balanced_accuracy_mean"])
        ax.set_title(titles[pert])
        ax.set_xlabel("perturbation level")
        ax.grid(True, color="0.85", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("balanced accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(configs),
               frameon=False, bbox_to_anchor=(0.5, -0.01), handlelength=2.6)
    fig.suptitle("Robustness degradation by perturbation (representation level)")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    out = _OUT_DIR / "ana03_robustness_degradation_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(_REPO)}")


def fig6_closed_loop():
    df = pd.read_csv(_CSV_DIR / "closed_loop_policy_summary.csv")
    policies = list(dict.fromkeys(df["policy"]))
    pretty = {"efe": "EFE", "epistemic_only": "epistemic-only", "oracle": "oracle",
              "passive": "passive", "pragmatic_only": "pragmatic-only", "random": "random"}
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, pol in enumerate(policies):
        d = df[df["policy"] == pol].sort_values("epsilon")
        yerr = [d["success_rate"] - d["success_ci_lo"],
                d["success_ci_hi"] - d["success_rate"]]
        ax.errorbar(d["epsilon"], d["success_rate"], yerr=yerr, capsize=3,
                    label=pretty.get(pol, pol), **_sty(i))
        _dump("fig6", pol, "", d["epsilon"], d["success_rate"])
    ax.set_xlabel(r"transition noise $\epsilon$")
    ax.set_ylabel("success rate")
    ax.set_title("Closed-loop success by policy (Wilson 95% CI)")
    ax.grid(True, color="0.85", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, loc="lower left", handlelength=2.6, frameon=True,
              framealpha=0.92, facecolor="white", edgecolor="0.7")
    fig.tight_layout()
    out = _OUT_DIR / "ana07_closed_loop_success_by_policy.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(_REPO)}")


def _dump(fig, series, panel, xs, ys):
    # Print plotted values so numerical preservation can be verified against the CSV.
    pairs = ",".join(f"({x:g},{y:g})" for x, y in zip(xs, ys))
    tag = f"{series}/{panel}" if panel else series
    print(f"[values] {fig} {tag}: {pairs}")


if __name__ == "__main__":
    fig4_robustness()
    fig6_closed_loop()
