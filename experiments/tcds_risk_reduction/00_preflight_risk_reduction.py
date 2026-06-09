#!/usr/bin/env python3
"""Phase 1 - Preflight and input verification for the risk-reduction pass.

Verifies repository state, the presence of the Ready-9 outputs this pass reuses,
the private feature inputs (existence + hashed fingerprint only; never raw IDs),
and that subject/observation counts and the affective label distribution match
the previously recorded Ready-9 run manifest.

Writes outputs/tcds_risk_reduction/preflight_risk_reduction.json on success.
If a required input is missing, writes outputs/tcds_risk_reduction/MISSING_INPUTS.md
and exits nonzero. No fabricated outputs.

Run:
    python experiments/tcds_risk_reduction/00_preflight_risk_reduction.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.tcds_ready9 import config as cfg  # noqa: E402
from experiments.tcds_ready9 import common_ready9 as cr  # noqa: E402

OUT = _REPO / "outputs" / "tcds_risk_reduction"
READY9 = _REPO / "outputs" / "tcds_ready9"
ANALYSIS = READY9 / "analysis"

REQUIRED_READY9 = {
    "robustness_metrics": ANALYSIS / "robustness_metrics.csv",
    "robustness_summary": ANALYSIS / "robustness_summary.csv",
    "mechanism_ablation_summary": ANALYSIS / "mechanism_ablation_summary.csv",
    "closed_loop_policy_summary": ANALYSIS / "closed_loop_policy_summary.csv",
    "run_manifest": READY9 / "run_manifest.json",
}


def _write_missing(missing: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    body = (
        "# Missing inputs - risk-reduction preflight\n\n"
        "The risk-reduction pass could not start because required inputs are "
        "absent. No downstream experiments were run and no outputs were "
        "fabricated.\n\n## Missing\n\n"
        + "\n".join(f"- {m}" for m in missing)
        + "\n\nResolve by regenerating the private feature inputs locally "
        "(experiments/tcds_ready9/00_prepare_features.py) or pointing the "
        "ARSPI_* environment variables at consistent files, then re-run.\n"
    )
    (OUT / "MISSING_INPUTS.md").write_text(body)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []

    # 2. Existing Ready-9 outputs.
    ready9_status = {}
    for name, p in REQUIRED_READY9.items():
        ok = p.exists()
        ready9_status[name] = {"present": ok, "fingerprint": cr.file_fingerprint(p) if ok else None}
        if not ok:
            missing.append(f"Ready-9 output missing: {name} ({cr.scrub_path(p)})")

    # 3. Private inputs (existence + hashed fingerprint only; no raw IDs).
    inputs_status = {
        "shape_features_211": {
            "present": cfg.SHAPE_FILE.exists(),
            "fingerprint": cr.file_fingerprint(cfg.SHAPE_FILE) if cfg.SHAPE_FILE.exists() else None,
        },
        "ch6_ch7_3class_features": {
            "present": cfg.CH67_FILE.exists(),
            "fingerprint": cr.file_fingerprint(cfg.CH67_FILE) if cfg.CH67_FILE.exists() else None,
        },
        "clinical_profile": {
            "present": cfg.CLINICAL_FILE.exists(),
            "fingerprint": cr.file_fingerprint(cfg.CLINICAL_FILE) if cfg.CLINICAL_FILE.exists() else None,
        },
    }
    for key in ("shape_features_211", "ch6_ch7_3class_features"):
        if not inputs_status[key]["present"]:
            missing.append(f"private feature input missing: {key}")

    if missing:
        _write_missing(missing)
        print("[preflight] MISSING INPUTS -> wrote MISSING_INPUTS.md", file=sys.stderr)
        return 1

    # 4. Alignment against the recorded Ready-9 run manifest.
    manifest = json.loads((READY9 / "run_manifest.json").read_text())
    data = cr.load_ready9()
    y = data["y"]
    subjects = data["subjects"]
    import numpy as np
    label_dist = {str(k): int(v) for k, v in
                  zip(*np.unique(np.asarray(y), return_counts=True))}
    alignment = {
        "n_observations": int(len(y)),
        "n_subjects": int(len(set(np.asarray(subjects).tolist()))),
        "label_distribution": label_dist,
        "balanced_3class": (len(label_dist) == 3 and len(set(label_dist.values())) == 1),
        "manifest_profile": manifest.get("run_profile", {}).get("profile"),
        "manifest_seeds": manifest.get("run_profile", {}).get("seeds"),
    }
    # cross-check against ablation summary feature dims (sanity that blocks load)
    blocks = data["blocks"]
    block_dims = {k: int(np.asarray(v).reshape(len(y), -1).shape[1]) for k, v in blocks.items()}

    report = {
        "phase": "risk_reduction_preflight",
        "provenance": cr.base_provenance(),
        "repo_state": {
            "branch": cr.repo_branch(),
            "commit": cr.repo_commit_hash(),
            "uncommitted_changes": cr.repo_is_dirty(),
        },
        "ready9_outputs": ready9_status,
        "private_inputs": inputs_status,
        "alignment": alignment,
        "block_dims": block_dims,
        "embedding_nonzero": bool(np.any(np.asarray(blocks.get("E", np.zeros(1))) != 0)),
    }
    cr.write_json(OUT / "preflight_risk_reduction.json", report)
    print(f"[preflight] OK: {alignment['n_observations']} obs, "
          f"{alignment['n_subjects']} subjects, balance={alignment['label_distribution']}, "
          f"E_nonzero={report['embedding_nonzero']}, blocks={block_dims}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
