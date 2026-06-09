#!/usr/bin/env python3
"""Phase 3 - Resource and event-rate accounting for the event-driven substrate.

Strengthens the neuromorphic-substrate argument with COMPUTATIONAL resource
accounting only: reservoir size, weight counts, feature dimensions, spike
counts/sparsity, event rate, an estimated synaptic-operation count, a dense-MAC
comparison, a memory-footprint estimate, and CPU runtimes. These are structural
and timing measurements.

EXPLICITLY NOT CLAIMED: measured hardware energy, hardware power, low-power
deployment, wearable feasibility, or any Loihi/SpiNNaker deployment result.

Outputs:
  outputs/tcds_risk_reduction/resource_event_metrics.csv
  outputs/tcds_risk_reduction/resource_event_summary.json
  figures/tcds_risk_reduction/rr03_event_resource_profile.pdf
  tables/tcds_risk_reduction/table_resource_event_accounting.tex
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.tcds_ready9 import config as cfg  # noqa: E402
from experiments.tcds_ready9 import common_ready9 as cr  # noqa: E402

OUT = _REPO / "outputs" / "tcds_risk_reduction"
FIG = _REPO / "figures" / "tcds_risk_reduction"
TAB = _REPO / "tables" / "tcds_risk_reduction"
for d in (OUT, FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

N_RES = 256
N_TIMESTEPS = 256
N_CH = cfg.N_CHANNELS  # 34
N_INPUT = 1  # per-channel scalar drive


def main() -> int:
    try:
        data = cr.load_ready9()
        shape = data["ch67"]  # population-rate time series live in the ch67 pickle
    except Exception as e:
        (OUT / "RESOURCE_ERROR.md").write_text(
            f"# Resource accounting could not run\n\nInput load failed: {e}\n")
        print(f"[resource] FAILED: {e}", file=sys.stderr)
        return 1

    blocks = data["blocks"]
    n_obs = len(data["y"])

    # ---- structural counts ----
    w_rec = N_RES * N_RES
    w_in = N_RES * N_INPUT
    block_dims = {k: int(np.asarray(blocks[k]).reshape(n_obs, -1).shape[1])
                  for k in ("BandPower", "E", "D", "T", "C")}
    pca_per_channel = 64
    bsc6_dim_per_channel = 6 * N_RES
    # PCA projection weights dominate the trainable parameter count.
    pca_weights = N_CH * bsc6_dim_per_channel * pca_per_channel
    readout_weights = block_dims["E"] * 3  # 3-class linear readout on E

    # ---- spike / event-rate accounting from population-rate time series ----
    spike_metrics = {}
    if "pop_rate_ts" in shape:
        pr = np.asarray(shape["pop_rate_ts"])  # (n_obs, n_ch, T) fraction active
        sparsity = float(np.mean(pr))                       # mean fraction of neurons active/timestep
        spikes_per_obs = float(np.mean(pr) * N_RES * N_TIMESTEPS * N_CH)
        events_per_timestep = float(np.mean(pr) * N_RES * N_CH)
        spike_metrics = {
            "spike_sparsity_fraction_active": round(sparsity, 5),
            "mean_spikes_per_observation": round(spikes_per_obs, 1),
            "mean_events_per_timestep": round(events_per_timestep, 2),
        }
    else:
        spike_metrics = {"note": "pop_rate_ts unavailable; spike-rate metrics omitted (not fabricated)"}

    # ---- synaptic-operation estimate vs dense MAC (per observation, recurrent layer) ----
    # Event-driven recurrent ops = (number of spikes) x fan-out (N_RES).
    # Dense equivalent = N_RES x N_RES x T per channel x N_CH, independent of sparsity.
    dense_recurrent_macs = w_rec * N_TIMESTEPS * N_CH
    synaptic_ops = None
    sparsity_savings = None
    if "mean_spikes_per_observation" in spike_metrics:
        synaptic_ops = spike_metrics["mean_spikes_per_observation"] * N_RES
        sparsity_savings = round(synaptic_ops / dense_recurrent_macs, 4)

    # ---- memory footprint (fixed weights, fp32) ----
    fixed_weight_count = (w_rec + w_in) * 1  # reservoir shared spec across channels
    mem_bytes = {
        "reservoir_fixed_weights_fp32_bytes": (w_rec + w_in) * 4,
        "pca_weights_fp32_bytes": pca_weights * 4,
        "readout_weights_fp32_bytes": readout_weights * 4,
    }

    # ---- runtime timings (CPU wall-clock) ----
    rng = np.random.RandomState(0)
    Xsub = np.asarray(blocks["E"]).reshape(n_obs, -1)[:64]
    t0 = time.perf_counter()
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    sc = StandardScaler().fit(Xsub)
    clf = LogisticRegression(max_iter=500, C=1.0).fit(sc.transform(Xsub),
                                                      rng.randint(0, 3, size=len(Xsub)))
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(10):
        clf.predict_proba(sc.transform(Xsub))
    t_pred = (time.perf_counter() - t0) / 10.0

    # closed-loop runtime context (if Ready-9 summary present)
    cl_steps = None
    cl_csv = _REPO / "outputs" / "tcds_ready9" / "analysis" / "closed_loop_policy_summary.csv"
    if cl_csv.exists():
        import csv
        rows = list(csv.DictReader(open(cl_csv)))
        efe = [float(r["mean_steps"]) for r in rows if r.get("policy") == "efe"]
        if efe:
            cl_steps = round(float(np.mean(efe)), 2)

    metrics = [
        ("reservoir_neurons_per_channel", N_RES),
        ("reservoir_channels", N_CH),
        ("reservoir_neuron_stack_total", N_RES * N_CH),
        ("reservoir_recurrent_weights_per_channel", w_rec),
        ("reservoir_input_weights_per_channel", w_in),
        ("timesteps", N_TIMESTEPS),
        ("bandpower_dim", block_dims["BandPower"]),
        ("E_dim", block_dims["E"]),
        ("D_dim", block_dims["D"]),
        ("T_dim", block_dims["T"]),
        ("C_dim", block_dims["C"]),
        ("pca_components_per_channel", pca_per_channel),
        ("pca_projection_weights", pca_weights),
        ("readout_weights_3class", readout_weights),
        ("dense_recurrent_macs_per_obs", dense_recurrent_macs),
        ("feature_fit_runtime_s_64obs", round(t_fit, 4)),
        ("classifier_predict_runtime_s_64obs", round(t_pred, 5)),
    ]
    for k, v in spike_metrics.items():
        metrics.append((k, v))
    if synaptic_ops is not None:
        metrics.append(("estimated_recurrent_synaptic_ops_per_obs", round(synaptic_ops, 1)))
        metrics.append(("event_driven_vs_dense_recurrent_ratio", sparsity_savings))
    if cl_steps is not None:
        metrics.append(("closed_loop_mean_steps_efe", cl_steps))

    cr.write_csv(OUT / "resource_event_metrics.csv",
                 [{"metric": k, "value": v} for k, v in metrics])
    summary = {
        "provenance": cr.base_provenance(),
        "structural": dict(metrics[:15]),
        "spike_event": spike_metrics,
        "memory_footprint_bytes": mem_bytes,
        "runtime_s": {"feature_fit_64obs": round(t_fit, 4),
                      "classifier_predict_64obs": round(t_pred, 5)},
        "interpretation": ("Computational resource accounting for an event-driven "
                           "reservoir-graph substrate. These are structural counts, "
                           "spike-rate measurements, and CPU runtimes; they do NOT "
                           "constitute measured hardware energy, power, or low-power "
                           "deployment results."),
    }
    cr.write_json(OUT / "resource_event_summary.json", summary)
    _figure(spike_metrics, sparsity_savings, block_dims)
    _table(metrics, spike_metrics, sparsity_savings)
    print(f"[resource] spike_sparsity={spike_metrics.get('spike_sparsity_fraction_active')}, "
          f"event/dense ratio={sparsity_savings}, E_dim={block_dims['E']}")
    return 0


def _figure(spike_metrics, ratio, block_dims):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    names = ["BandPower", "E", "D", "T", "C"]
    axes[0].bar(names, [block_dims[k] for k in names], color="#1a73e8")
    axes[0].set_ylabel("feature dimension"); axes[0].set_title("Evidence-stream dimensions")
    axes[0].set_yscale("log")
    if ratio is not None:
        axes[1].bar(["event-driven\n(recurrent)", "dense\nequivalent"], [ratio, 1.0],
                    color=["#188038", "#9aa0a6"])
        axes[1].set_ylabel("recurrent ops (fraction of dense)")
        axes[1].set_title(f"Spike sparsity = {spike_metrics.get('spike_sparsity_fraction_active')}")
    else:
        axes[1].axis("off")
    fig.tight_layout(); fig.savefig(FIG / "rr03_event_resource_profile.pdf"); plt.close(fig)
    cr.write_json(FIG / "rr03_event_resource_profile.json",
                  {"provenance": cr.base_provenance(),
                   "block_dims": block_dims, "event_vs_dense_ratio": ratio,
                   "spike_metrics": spike_metrics})


def _table(metrics, spike_metrics, ratio):
    d = dict(metrics)
    rows = [
        ("Reservoir neurons / channel", d["reservoir_neurons_per_channel"]),
        ("Channels", d["reservoir_channels"]),
        ("Recurrent weights / channel", f"{d['reservoir_recurrent_weights_per_channel']:,}"),
        ("Embedding $E$ dimension", d["E_dim"]),
        ("PCA projection weights", f"{d['pca_projection_weights']:,}"),
        ("Spike sparsity (fraction active)", spike_metrics.get("spike_sparsity_fraction_active", "n/a")),
        ("Mean spikes / observation", spike_metrics.get("mean_spikes_per_observation", "n/a")),
        ("Recurrent ops vs dense", ratio if ratio is not None else "n/a"),
        ("Classifier predict (64 obs), s", d["classifier_predict_runtime_s_64obs"]),
    ]
    lines = [r"\begin{table}[t]\centering\footnotesize",
             r"\caption{Resource and event-rate accounting for the event-driven "
             r"substrate. Structural counts, spike-rate measurements, and CPU runtimes; "
             r"these are not measured hardware-energy results.}",
             r"\label{tab:resource_event}",
             r"\begin{tabular}{l r}", r"\toprule", r"Quantity & Value \\", r"\midrule"]
    for k, v in rows:
        lines.append(f"{k} & {v} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TAB / "table_resource_event_accounting.tex").write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
