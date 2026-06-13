#!/usr/bin/env python3
"""Generate the two compact conceptual TAFFC theory/novelty figures.

These are schematic (no data); they illustrate the theoretical framing added
to the manuscript. Both are single-column width so they cost little page
budget. They write only to ``figures/taffc/``.

    fig_observation_model.pdf
        Affective ERP as the noisy, partially observed output of a latent
        biological dynamical system (latent state x_t, stimulus drive u_t,
        process noise eta_t, scalp projection H, observation noise eps_t),
        read by ARSPI-Net's fixed observable map Phi.

    fig_operational_distinctness_map.pdf
        The four evidence streams E, D, T, C evaluated by predictive
        sufficiency, redundancy, perturbation response, coupling, and
        closed-loop utility, with a novelty comparison band against endpoint,
        graph, and reservoir EEG models.

    python experiments/tcds_ready9/generate_taffc_theory_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "figures" / "taffc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Evidence-stream palette (matches Fig. 1 / the TikZ pipeline diagram).
STREAM = [
    ("$E$", "#2660A4", "#D6E5F7"),
    ("$D$", "#1A8476", "#D1EEE7"),
    ("$T$", "#7246A2", "#E6DDF3"),
    ("$C$", "#388C3C", "#DBEEDB"),
]

plt.rcParams.update({
    "font.size": 7.5,
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
})


def _box(ax, x, y, w, h, lines, fc, ec, fs=7.0, lw=1.0, bold_first=True):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.02",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    n = len(lines)
    for k, ln in enumerate(lines):
        yy = y + h * (n - k) / (n + 1)
        fw = "bold" if (k == 0 and bold_first) else "normal"
        ax.text(x + w / 2, yy, ln, ha="center", va="center",
                fontsize=fs if k == 0 else fs - 0.6, fontweight=fw,
                color="#222", zorder=3)


def _arrow(ax, x0, y0, x1, y1, color="#464E5A", lw=1.2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9,
        linewidth=lw, color=color, linestyle=ls, shrinkA=2, shrinkB=2, zorder=4))


def observation_model():
    fig = plt.figure(figsize=(3.45, 2.25))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Latent dynamics box (top), observation box (middle), ERP box (lower).
    _box(ax, 0.08, 0.66, 0.84, 0.26,
         ["Latent neural dynamics",
          r"$x_{t+1}=f_{\theta}(x_t,u_t)+\eta_t$"],
         "#FFF3DC", "#C48A26", fs=7.2)
    _box(ax, 0.20, 0.34, 0.60, 0.20,
         [r"Scalp observation   $X_t=Hx_t+\epsilon_t$"],
         "#E2E9F1", "#46596F", fs=7.0, bold_first=False)
    _box(ax, 0.20, 0.06, 0.60, 0.18,
         [r"Affective ERP  $X^{(c)}_s\in\mathbb{R}^{34\times T}$"],
         "#ECECEE", "#5C5C62", fs=7.0, bold_first=False)

    # Stimulus drive into the latent dynamics.
    ax.text(0.05, 0.90, r"$u_t$ (stimulus)", ha="left", va="center",
            fontsize=6.6, color="#B24444")
    _arrow(ax, 0.02, 0.79, 0.08, 0.79, color="#B24444")
    # Projection H (latent -> observation), with noise annotations.
    _arrow(ax, 0.5, 0.66, 0.5, 0.545)
    ax.text(0.545, 0.60, r"$H$", ha="left", va="center", fontsize=7.5, color="#46596F")
    _arrow(ax, 0.5, 0.34, 0.5, 0.245)
    ax.text(0.86, 0.79, r"$\eta_t$", ha="left", va="center", fontsize=6.8, color="#888")
    ax.text(0.815, 0.44, r"$\epsilon_t$", ha="left", va="center", fontsize=6.8, color="#888")

    # ARSPI-Net observable map reads the observation, not the latent state.
    _arrow(ax, 0.80, 0.15, 0.95, 0.15, color="#2660A4", ls=(0, (4, 1.5)))
    ax.text(0.875, 0.255, r"$\Phi$", ha="center", va="center", fontsize=9,
            color="#2660A4")
    ax.text(0.885, 0.05, "observable map", ha="center", va="center",
            fontsize=5.6, color="#2660A4")

    out = OUT_DIR / "fig_observation_model.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


def distinctness_map():
    fig = plt.figure(figsize=(3.45, 3.05))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Top: streams x measured-behavior matrix that visualizes the distinctness
    # vector Delta = (S, 1-CKA, rho, kappa, U). Rows are the four streams (small
    # row headers, not chips, to stay visually distinct from the Fig. 1 overview);
    # columns are the measured-behavior axes; each filled cell marks that the
    # stream is characterized along that axis.
    ax.text(0.5, 0.965, r"Measured behavior   $\Delta=(\,S,\,1-\mathrm{CKA},\,\rho,\,\kappa,\,U\,)$",
            ha="center", va="center", fontsize=6.8, fontweight="bold", color="#222")

    cols = [r"$S$", r"$1{-}\mathrm{CKA}$", r"$\rho$", r"$\kappa$", r"$U$"]
    col_sub = ["suffic.", "redund.", "perturb.", "coupl.", "closed-loop"]
    x0, x1 = 0.20, 0.97
    cwid = (x1 - x0) / len(cols)
    ytop, ybot = 0.83, 0.55
    rh = (ytop - ybot) / len(STREAM)

    # Column headers.
    for c, (sym, sub) in enumerate(zip(cols, col_sub)):
        cx = x0 + (c + 0.5) * cwid
        ax.text(cx, 0.885, sym, ha="center", va="center", fontsize=7.2, color="#222")
        ax.text(cx, 0.855, sub, ha="center", va="center", fontsize=4.8, color="#666")
    # Row headers + grid cells.
    for r, (lab, ec, fc) in enumerate(STREAM):
        ry = ytop - (r + 0.5) * rh
        ax.text(0.135, ry, lab, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color=ec)
        for c in range(len(cols)):
            cx = x0 + (c + 0.5) * cwid
            ax.add_patch(FancyBboxPatch(
                (x0 + c * cwid + 0.006, ry - rh / 2 + 0.006),
                cwid - 0.012, rh - 0.012,
                boxstyle="round,pad=0.001,rounding_size=0.006",
                linewidth=0.6, edgecolor="#BBB", facecolor="#F4F6F9", zorder=2))
            ax.add_patch(plt.Circle((cx, ry), 0.012, color=ec, alpha=0.85, zorder=3))
    ax.text(0.135, ytop + 0.035, "stream", ha="center", va="center",
            fontsize=4.8, color="#666")

    # Novelty comparison band (bottom) -- distinct from the matrix above.
    ax.text(0.5, 0.45, "Novelty: the reservoir-graph observable map",
            ha="center", va="center", fontsize=7.0, fontweight="bold", color="#222")
    rows = [
        ("Endpoint EEG classifiers", "collapse evidence into one score", "#ECECEE", "#888"),
        ("Graph EEG models", "topology as a latent component", "#ECECEE", "#888"),
        ("Reservoir EEG models", "judged by readout accuracy", "#ECECEE", "#888"),
        ("ARSPI-Net", r"$E,D,T,C$ kept as measured streams", "#D6E5F7", "#2660A4"),
    ]
    ry0, rbh = 0.40, 0.072
    for k, (name, desc, fc, ec) in enumerate(rows):
        y = ry0 - k * (rbh + 0.014)
        ax.add_patch(FancyBboxPatch(
            (0.05, y - rbh), 0.90, rbh, boxstyle="round,pad=0.003,rounding_size=0.012",
            linewidth=(1.2 if k == 3 else 0.8), edgecolor=ec, facecolor=fc, zorder=2))
        ax.text(0.075, y - rbh / 2, name, ha="left", va="center", fontsize=6.6,
                fontweight=("bold" if k == 3 else "normal"), color="#222", zorder=3)
        ax.text(0.93, y - rbh / 2, desc, ha="right", va="center", fontsize=6.0,
                color="#444", zorder=3)

    out = OUT_DIR / "fig_operational_distinctness_map.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


def main() -> int:
    observation_model()
    distinctness_map()
    return 0


if __name__ == "__main__":
    sys.exit(main())
