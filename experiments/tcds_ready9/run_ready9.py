#!/usr/bin/env python3
"""Phase 13 — Orchestrator for the experimental-hardening package.

Runs the phases in measurement order, then generates the figure manifest, the
manuscript figure plan, and the run manifest. Core phases are fatal: if any of
feature alignment, mechanism ablation, robustness, closed-loop, or table
generation fails, the run exits nonzero and the package is marked NOT
submission-ready. Non-essential observation-figure skips do not fail the run.

Order: measured signal -> reservoir substrate -> graph evidence -> statistical
analysis -> simulated embodied control -> interpretation.

Run:
    python experiments/tcds_ready9/run_ready9.py
    ARSPI_READY9_PROFILE=submission python experiments/tcds_ready9/run_ready9.py
"""
from __future__ import annotations

import csv
import datetime as dt
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.tcds_ready9 import config as cfg  # noqa: E402
from experiments.tcds_ready9 import common_ready9 as cr  # noqa: E402

PKG = Path(__file__).resolve().parent

# (label, script, is_core)
PHASES = [
    ("preflight_audit", "preflight_audit.py", False),
    ("feature_alignment", "00_prepare_features.py", True),
    ("dataset_provenance_qc", "01_dataset_provenance_qc.py", False),
    ("observation_plots", "06_generate_observation_plots.py", False),
    ("signal_robustness", "08_signal_robustness.py", True),
    ("mechanism_ablation", "09_mechanism_ablation.py", True),
    ("closed_loop_policy", "10_closed_loop_policy_hardening.py", True),
    ("graph_runtime_support", "12_graph_runtime_support.py", False),
    ("analysis_plots", "07_generate_analysis_plots.py", False),
    ("generate_tables", "11_generate_ready9_tables.py", True),
]


def _run(script: str) -> int:
    print(f"\n{'=' * 70}\n[run_ready9] {script}\n{'=' * 70}")
    proc = subprocess.run([sys.executable, str(PKG / script)], cwd=str(_REPO_ROOT))
    return proc.returncode


# ── Figure manifest (Phase 12) ──────────────────────────────────────────
SECTION_MAP = {
    "obs00": ("Results A", "dataset integrity and provenance"),
    "obs01": ("Results A", "observed ERP morphology"),
    "obs04": ("Results A", "observed reservoir spike behavior"),
    "obs05": ("Results A", "observed population-rate behavior"),
    "obs08": ("Results A", "observed tPLV graph structure"),
    "obs09": ("Results A", "observed structure-function coupling"),
    "ana01": ("Results B", "mechanism ablation performance"),
    "ana02": ("Results B", "mechanism functional roles"),
    "obs03": ("Results C", "perturbation examples"),
    "ana03": ("Results C", "robustness degradation"),
    "ana04": ("Results C", "robustness summary"),
    "obs10": ("Results D", "closed-loop belief trajectories"),
    "ana07": ("Results D", "closed-loop success"),
    "ana08": ("Results D", "closed-loop entropy and steps"),
    "ana09": ("Results D", "closed-loop failure modes"),
    "ana10": ("Discussion", "evaluation coverage"),
    "ana11": ("Discussion", "bounded graph support"),
    "ana12": ("Discussion", "runtime/resource support"),
    "obs02": ("Results A", "channel x time variability"),
    "obs06": ("Results A", "BSC6 temporal coding"),
    "obs07": ("Results A", "feature-block distributions"),
    "ana05": ("Results B", "exploratory clinical-label sensitivity"),
    "ana06": ("Results A", "kappa shuffle-null"),
}


def generate_figure_manifest():
    rows = []
    for pdf in sorted(cfg.FIG_DIR.rglob("*.pdf")):
        fid = pdf.stem
        key = fid[:5]
        ftype = "observation" if "observations" in str(pdf) else "analysis"
        meta_path = pdf.with_suffix(".json")
        source_script = source_data = ""
        if meta_path.exists():
            import json
            try:
                m = json.loads(meta_path.read_text())
                source_script = m.get("source_script", "")
                source_data = ";".join(
                    v.get("name", "") for v in (m.get("inputs") or {}).values()
                    if isinstance(v, dict))
            except Exception:
                pass
        section, claim = SECTION_MAP.get(key, ("unassigned", ""))
        rows.append({
            "figure_id": fid,
            "figure_path": str(pdf.relative_to(cfg.ROOT)),
            "figure_type": ftype,
            "source_script": source_script,
            "source_data": source_data,
            "privacy_status": "privacy-preserving (aggregate/hashed)",
            "manuscript_section_recommended": section,
            "claim_supported": claim,
            "limitations": "bounded to the measured SHAPE ERP regime",
            "generated_timestamp": dt.datetime.fromtimestamp(pdf.stat().st_mtime).isoformat(),
        })
    fields = ["figure_id", "figure_path", "figure_type", "source_script", "source_data",
              "privacy_status", "manuscript_section_recommended", "claim_supported",
              "limitations", "generated_timestamp"]
    with open(cfg.OUT_DIR / "FIGURE_MANIFEST.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    cr.write_json(cfg.OUT_DIR / "FIGURE_MANIFEST.json",
                  {"provenance": cr.base_provenance(), "figures": rows})
    _figure_plan()
    return len(rows)


def _figure_plan():
    txt = """# Manuscript figure plan

Generated by the orchestrator. Figures are organised into the Results sections
below; all are privacy-preserving (aggregate or hashed).

## Results A -- Dataset integrity and observed computational objects
- obs00 dataset integrity QC
- obs01 ERP morphology
- obs04 reservoir spike raster
- obs05 population-rate traces
- obs08 tPLV adjacency
- obs09 kappa observation

## Results B -- Mechanism ablation
- ana01 mechanism-ablation performance
- ana02 functional-role matrix

## Results C -- Robustness
- obs03 perturbation examples
- ana03 robustness degradation curves
- ana04 robustness summary

## Results D -- Simulated embodied perceptual control
- obs10 belief trajectories
- ana07 closed-loop success
- ana08 entropy and steps
- ana09 failure modes

## Discussion
- ana10 evaluation coverage
- bounded graph/runtime support (ana11, ana12) and tables
"""
    (cfg.OUT_DIR / "MANUSCRIPT_FIGURE_PLAN.md").write_text(txt)


# ── Run manifest ────────────────────────────────────────────────────────
def main() -> int:
    results = {}
    core_failed = []
    for label, script, is_core in PHASES:
        rc = _run(script)
        results[label] = {"script": script, "returncode": rc, "core": is_core}
        if rc != 0 and is_core:
            core_failed.append(label)
            print(f"[run_ready9] CORE phase '{label}' FAILED (rc={rc}).", file=sys.stderr)
            break  # do not continue past a failed core phase

    n_figs = generate_figure_manifest()

    skipped = [p.name for p in (cfg.OBS_OUT_DIR).glob("*_SKIPPED.md")] + \
              [p.name for p in cfg.ANALYSIS_DIR.glob("*_SKIPPED.md")]

    manifest = {
        "provenance": cr.base_provenance(),
        "branch": cr.repo_branch(),
        "run_profile": cfg.profile_summary(),
        "phase_results": results,
        "core_failed": core_failed,
        "submission_ready": len(core_failed) == 0,
        "input_fingerprints": {
            "shape_features_211": cr.file_fingerprint(cfg.SHAPE_FILE),
            "ch6_ch7_3class_features": cr.file_fingerprint(cfg.CH67_FILE),
            "clinical_profile": cr.file_fingerprint(cfg.CLINICAL_FILE),
        },
        "n_figures": n_figs,
        "skipped_nonessential": skipped,
        "privacy_confirmation": "no raw EEG, feature pickles, clinical metadata, or "
                                "raw subject IDs are written to committed outputs; "
                                "subject identifiers are hashed",
        "note_on_manuscript_compilation": "compiled separately; see manuscript build "
                                          "log / quality gate",
    }
    cr.write_json(cfg.OUT_DIR / "run_manifest.json", manifest)

    print(f"\n[run_ready9] profile={cfg.PROFILE}  figures={n_figs}  "
          f"core_failed={core_failed or 'none'}  "
          f"submission_ready={manifest['submission_ready']}")
    return 1 if core_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
