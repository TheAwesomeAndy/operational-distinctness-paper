#!/usr/bin/env python3
"""Phase 4 — Dataset provenance and integrity audit.

Produces a privacy-safe dataset-integrity package: a provenance table, a QC
summary CSV, an integrity QC figure (obs00), and provenance metadata. Lab/source
wording is reused only because it is verifiable from the existing manuscript
data-availability section and acknowledgments.

Outputs:
    tables/tcds_ready9/table_dataset_provenance.tex
    outputs/tcds_ready9/dataset_qc_summary.csv
    figures/tcds_ready9/observations/obs00_dataset_integrity_qc.pdf
    outputs/tcds_ready9/dataset_provenance_metadata.json

Run:
    python experiments/tcds_ready9/01_dataset_provenance_qc.py
"""
from __future__ import annotations

import pickle
import re
import sys
from collections import Counter
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

RAW_PATTERN = re.compile(r"SHAPE_Community_(\d+)_IAPS(Neg|Neu|Pos)_BC\.txt$")

# Verified from the existing manuscript data-availability section + acknowledgments.
PROVENANCE_VERIFIED = True
DATA_SOURCE = ("SHAPE Community dataset, Laboratory for Clinical Affective "
               "Neuroscience, Stony Brook University (access-controlled)")


def _raw_completeness(raw_dir: Path) -> dict:
    if not raw_dir.exists():
        return {"present": False}
    by_subject: dict[int, set] = {}
    malformed = 0
    for p in raw_dir.rglob("*.txt"):
        m = RAW_PATTERN.search(p.name)
        if not m:
            malformed += 1
            continue
        by_subject.setdefault(int(m.group(1)), set()).add(m.group(2))
    complete = {s for s, c in by_subject.items() if c == {"Neg", "Neu", "Pos"}}
    return {
        "present": True,
        "n_subjects_discovered": len(by_subject),
        "n_subjects_complete": len(complete),
        "n_missing_condition_subjects": len(by_subject) - len(complete),
        "malformed_files": malformed,
        "subject_127_present": 127 in by_subject,
    }


def main() -> int:
    with open(cfg.SHAPE_FILE, "rb") as f:
        ch5 = pickle.load(f)
    y = np.asarray(ch5["y"])
    subjects = np.asarray(ch5["subjects"])
    X_ds = np.asarray(ch5["X_ds"]) if "X_ds" in ch5 else None
    conv = np.asarray(ch5["conv_feats"])
    emb = np.asarray(ch5["lsm_bsc6_pca"])

    cond_counts = {cfg.AFFECTIVE_LABEL_NAMES[k]: int(v)
                   for k, v in sorted(Counter(y.tolist()).items())}

    ch67_present = cfg.CH67_FILE.exists()
    ch67 = None
    if ch67_present:
        with open(cfg.CH67_FILE, "rb") as f:
            ch67 = pickle.load(f)

    def _nan_inf(a):
        a = np.asarray(a, dtype=float)
        return int(np.isnan(a).sum()), int(np.isinf(a).sum())

    nan_emb, inf_emb = _nan_inf(emb)
    nan_bp, inf_bp = _nan_inf(conv)
    nan_d = inf_d = nan_t = inf_t = 0
    if ch67 is not None:
        nan_d, inf_d = _nan_inf(ch67["D"])
        nan_t, inf_t = _nan_inf(ch67["T_topo"])

    raw = _raw_completeness(cfg.RAW_EEG_DIR)

    n_obs = int(len(y))
    n_subj = int(len(set(subjects.tolist())))
    n_channels = int(emb.shape[1]) if emb.ndim == 3 else cfg.N_CHANNELS

    qc_rows = [
        {"item": "subjects_included", "value": n_subj},
        {"item": "observations_total", "value": n_obs},
        {"item": "conditions", "value": len(cond_counts)},
        {"item": "obs_negative", "value": cond_counts.get("Negative")},
        {"item": "obs_neutral", "value": cond_counts.get("Neutral")},
        {"item": "obs_pleasant", "value": cond_counts.get("Pleasant")},
        {"item": "channels", "value": n_channels},
        {"item": "downsampled_timepoints", "value": (int(X_ds.shape[1]) if X_ds is not None else None)},
        {"item": "embedding_nan", "value": nan_emb},
        {"item": "embedding_inf", "value": inf_emb},
        {"item": "bandpower_nan", "value": nan_bp},
        {"item": "bandpower_inf", "value": inf_bp},
        {"item": "dynamical_nan", "value": nan_d},
        {"item": "dynamical_inf", "value": inf_d},
        {"item": "topology_nan", "value": nan_t},
        {"item": "topology_inf", "value": inf_t},
        {"item": "raw_subjects_discovered", "value": raw.get("n_subjects_discovered")},
        {"item": "raw_subjects_complete", "value": raw.get("n_subjects_complete")},
        {"item": "raw_missing_condition_subjects", "value": raw.get("n_missing_condition_subjects")},
        {"item": "excluded_subjects", "value": 1},
        {"item": "excluded_subject_note", "value": "subject 127 excluded (recording anomaly, Ch5 QC)"},
    ]
    cr.write_csv(cfg.OUT_DIR / "dataset_qc_summary.csv", qc_rows,
                 fieldnames=["item", "value"])

    # ── Provenance table ────────────────────────────────────────────────
    _write_provenance_table(n_subj, n_obs, len(cond_counts), n_channels,
                            X_ds.shape[1] if X_ds is not None else None)

    # ── obs00 integrity figure ──────────────────────────────────────────
    _obs00_figure(cond_counts, X_ds, emb, raw)

    # ── Provenance metadata ─────────────────────────────────────────────
    cr.write_json(cfg.OUT_DIR / "dataset_provenance_metadata.json", {
        "provenance": cr.base_provenance(),
        "data_source_wording": DATA_SOURCE,
        "provenance_verified": PROVENANCE_VERIFIED,
        "verification_method": "existing manuscript data-availability section and "
                               "acknowledgments (author-approved wording)",
        "local_documentation_checked": ["manuscript/main.tex", "README.md"],
        "privacy_status": cfg.PRIVACY_STATUS,
        "inputs": {
            "shape_features_211": cr.file_fingerprint(cfg.SHAPE_FILE),
            "ch6_ch7_3class_features": cr.file_fingerprint(cfg.CH67_FILE),
            "clinical_profile": cr.file_fingerprint(cfg.CLINICAL_FILE),
        },
        "n_subjects": n_subj, "n_observations": n_obs,
        "conditions": cond_counts,
    })
    print(f"[provenance] {n_subj} subjects, {n_obs} obs, conditions={cond_counts}, "
          f"ch67_present={ch67_present}")
    return 0


def _write_provenance_table(n_subj, n_obs, n_cond, n_ch, n_tp):
    rows = [
        ("Data source", DATA_SOURCE, "manuscript data-availability (verified)", "provenance"),
        ("Subjects included", str(n_subj), "feature pickle", "sample size"),
        ("Excluded subjects", "1 (subject 127, recording anomaly)", "Ch5 QC", "data quality"),
        ("Observations", str(n_obs), "feature pickle", "evaluation units"),
        ("Conditions", f"{n_cond} (Negative, Neutral, Pleasant)", "feature pickle", "affective task"),
        ("Channels", str(n_ch), "feature pickle", "spatial sampling"),
        ("Downsampled rate", "256 Hz", "extraction pipeline", "temporal sampling"),
        ("Post-stimulus timepoints", str(n_tp), "feature pickle", "epoch length"),
        ("Preprocessing", "baseline-corrected, trial-averaged ERP", "extraction pipeline", "signal conditioning"),
        ("Clinical metadata", "available (exploratory validation only)", "clinical CSV", "context labels"),
        ("Split protocol", "subject-grouped cross-validation", "analysis code", "generalisation"),
        ("Privacy rule", "aggregate/hashed outputs; subject-level data restricted", "package policy", "data governance"),
    ]
    lines = [
        r"% Auto-generated dataset provenance table.",
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Dataset provenance and integrity. Source wording is reused "
        r"from the verified manuscript data-availability statement. Subject-level "
        r"data are maintained in a restricted research environment.}",
        r"\label{tab:dataset_provenance}",
        r"\begin{tabular}{@{}p{1.5cm}p{3.0cm}p{1.8cm}p{1.2cm}@{}}",
        r"\toprule",
        r"Item & Value & Source / verification & Relevance \\",
        r"\midrule",
    ]
    for item, val, src, rel in rows:
        esc = lambda s: s.replace("&", r"\&").replace("_", r"\_")
        lines.append(f"{esc(item)} & {esc(val)} & {esc(src)} & {esc(rel)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (cfg.TABLE_DIR / "table_dataset_provenance.tex").write_text("\n".join(lines))


def _obs00_figure(cond_counts, X_ds, emb, raw):
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    # (a) condition balance
    ax = axes[0, 0]
    ax.bar(list(cond_counts.keys()), list(cond_counts.values()), color="#4C72B0")
    ax.set_title("(a) Condition balance")
    ax.set_ylabel("observations")
    # (b) amplitude distribution (downsampled EEG)
    ax = axes[0, 1]
    if X_ds is not None:
        vals = np.asarray(X_ds, dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        sub = vals[:: max(1, vals.size // 200000)]
        ax.hist(sub, bins=80, color="#55A868")
        ax.set_title("(b) Downsampled EEG amplitude (z-scored)")
        ax.set_xlabel("amplitude")
    else:
        ax.text(0.5, 0.5, "X_ds unavailable", ha="center")
        ax.set_title("(b) Amplitude distribution")
    # (c) per-channel embedding energy consistency
    ax = axes[1, 0]
    if emb.ndim == 3:
        per_ch = np.linalg.norm(emb, axis=2).mean(axis=0)
        ax.plot(per_ch, color="#C44E52")
        ax.set_title("(c) Mean per-channel embedding norm")
        ax.set_xlabel("channel index"); ax.set_ylabel("mean norm")
    # (d) raw-file completeness summary
    ax = axes[1, 1]
    ax.axis("off")
    txt = "Raw-EEG completeness\n"
    if raw.get("present"):
        txt += (f"discovered subjects: {raw['n_subjects_discovered']}\n"
                f"complete (3 conditions): {raw['n_subjects_complete']}\n"
                f"missing-condition subjects: {raw['n_missing_condition_subjects']}\n"
                f"malformed files: {raw['malformed_files']}\n"
                f"subject 127 present (excluded): {raw['subject_127_present']}")
    else:
        txt += "raw EEG directory not available"
    ax.text(0.02, 0.95, txt, va="top", fontsize=10, family="monospace")
    ax.set_title("(d) File completeness")
    fig.suptitle("Dataset integrity quality control", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = cfg.FIG_OBS_DIR / "obs00_dataset_integrity_qc.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)

    cr.write_json(cfg.FIG_OBS_DIR / "obs00_dataset_integrity_qc.json", {
        "source_script": "experiments/tcds_ready9/01_dataset_provenance_qc.py",
        "figure_type": "observation",
        "inputs": {"shape_features_211": cr.file_fingerprint(cfg.SHAPE_FILE)},
        "provenance": cr.base_provenance(),
        "data_level": "feature-level + aggregate", "privacy_status": cfg.PRIVACY_STATUS,
        "output_path": "figures/tcds_ready9/observations/obs00_dataset_integrity_qc.pdf",
    })


if __name__ == "__main__":
    raise SystemExit(main())
