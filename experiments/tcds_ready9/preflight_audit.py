#!/usr/bin/env python3
"""Phase 1 — Preflight audit for the Ready-9 package.

Verifies repository state, upstream dependency, private inputs, feature-pickle
schemas, clinical metadata, and raw-EEG completeness. Writes a privacy-safe
``outputs/tcds_ready9/preflight_audit.json``. Raw subject identifiers are hashed
and never written to committed logs.

If required private inputs cannot be located, writes
``outputs/tcds_ready9/MISSING_INPUTS.md`` and exits nonzero.

Run:
    python experiments/tcds_ready9/preflight_audit.py
"""
from __future__ import annotations

import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.tcds_ready9 import config as cfg  # noqa: E402
from experiments.tcds_ready9 import common_ready9 as cr  # noqa: E402

RAW_PATTERN = re.compile(r"SHAPE_Community_(\d+)_IAPS(Neg|Neu|Pos)_BC\.txt$")


def _label_dist(y) -> dict:
    return {str(k): int(v) for k, v in sorted(Counter(np.asarray(y).tolist()).items())}


def _audit_pickle(path: Path, expected_keys: list[str]) -> dict:
    rep = {"fingerprint": cr.file_fingerprint(path), "present": path.exists()}
    if not path.exists():
        rep["status"] = "missing"
        return rep
    try:
        with open(path, "rb") as f:
            d = pickle.load(f)
    except Exception as e:  # pragma: no cover
        rep["status"] = f"unreadable: {type(e).__name__}"
        return rep
    keys = list(d.keys())
    rep["keys_present"] = keys
    rep["keys_missing"] = [k for k in expected_keys if k not in keys]
    shapes = {}
    for k in expected_keys:
        if k in d and hasattr(d[k], "shape"):
            shapes[k] = list(np.asarray(d[k]).shape)
    rep["shapes"] = shapes
    if "y" in d:
        rep["label_distribution"] = _label_dist(d["y"])
        rep["n_observations"] = int(len(d["y"]))
    if "subjects" in d:
        rep["n_unique_subjects"] = int(len(set(np.asarray(d["subjects"]).tolist())))
    rep["status"] = "ok" if not rep["keys_missing"] else "missing_keys"
    return rep


def _audit_clinical(path: Path) -> dict:
    import pandas as pd
    rep = {"fingerprint": cr.file_fingerprint(path), "present": path.exists()}
    if not path.exists():
        rep["status"] = "missing"
        return rep
    df = pd.read_csv(path)
    rep["n_rows"] = int(len(df))
    rep["columns"] = list(df.columns)
    rep["id_column_present"] = "ID" in df.columns
    rep["diagnoses_present"] = [d for d in cfg.DIAGNOSES if d in df.columns]
    rep["diagnoses_missing"] = [d for d in cfg.DIAGNOSES if d not in df.columns]
    rep["status"] = "ok" if rep["id_column_present"] else "no_id_column"
    return rep


def _audit_raw_eeg(raw_dir: Path) -> dict:
    rep = {"dir_name": cr.scrub_path(raw_dir), "present": raw_dir.exists()}
    if not raw_dir.exists():
        rep["status"] = "missing"
        return rep
    by_subject: dict[int, set] = {}
    n_files = 0
    malformed = 0
    for p in raw_dir.rglob("*.txt"):
        m = RAW_PATTERN.search(p.name)
        if not m:
            malformed += 1
            continue
        n_files += 1
        sid = int(m.group(1))
        by_subject.setdefault(sid, set()).add(m.group(2))
    complete = {s for s, c in by_subject.items() if c == {"Neg", "Neu", "Pos"}}
    incomplete = {s: sorted(c) for s, c in by_subject.items() if c != {"Neg", "Neu", "Pos"}}
    rep["n_files"] = n_files
    rep["n_subjects_discovered"] = len(by_subject)
    rep["n_subjects_complete"] = len(complete)
    rep["n_subjects_incomplete"] = len(incomplete)
    rep["malformed_or_unmatched_files"] = malformed
    rep["subject_127_present"] = 127 in by_subject
    rep["note"] = "subject 127 is excluded by the feature-extraction pipeline (Ch5 QC)"
    rep["status"] = "ok" if complete else "no_complete_subjects"
    return rep


def main() -> int:
    audit = {"provenance": cr.base_provenance()}
    audit["repo"] = {
        "commit": cr.repo_commit_hash(),
        "branch": cr.repo_branch(),
        "dirty": cr.repo_is_dirty(),
    }
    audit["upstream"] = {
        "disso_repo_name": cr.scrub_path(cfg.DISSO_REPO),
        "disso_repo_present": cfg.DISSO_REPO.exists(),
        "upstream_feature_script_present": cfg.UPSTREAM_FEATURE_SCRIPT.exists(),
        "local_feature_script_present": cfg.LOCAL_FEATURE_SCRIPT.exists(),
        "upstream_commit": cr.upstream_commit_hash(),
    }
    audit["inputs"] = {
        "shape_features_211": _audit_pickle(cfg.SHAPE_FILE, cfg.EXPECTED_SHAPE_KEYS),
        "ch6_ch7_3class_features": _audit_pickle(cfg.CH67_FILE, cfg.EXPECTED_CH67_KEYS),
        "clinical_profile": _audit_clinical(cfg.CLINICAL_FILE),
        "raw_eeg": _audit_raw_eeg(cfg.RAW_EEG_DIR),
    }

    # Decide whether required inputs are available. shape_features + clinical are
    # hard requirements. ch6_ch7 is regenerable from raw EEG, so it is not fatal
    # here if raw EEG is present (Phase 2 will generate it).
    shape_ok = audit["inputs"]["shape_features_211"]["present"]
    clinical_ok = audit["inputs"]["clinical_profile"]["present"]
    ch67_ok = audit["inputs"]["ch6_ch7_3class_features"]["present"]
    raw_ok = audit["inputs"]["raw_eeg"]["present"]
    ch67_obtainable = ch67_ok or raw_ok

    missing = []
    if not shape_ok:
        missing.append("shape_features_211.pkl")
    if not clinical_ok:
        missing.append("clinical_profile.csv")
    if not ch67_obtainable:
        missing.append("ch6_ch7_3class_features.pkl (and no raw EEG to regenerate it)")

    audit["required_inputs_status"] = {
        "shape_features_present": shape_ok,
        "clinical_present": clinical_ok,
        "ch67_present": ch67_ok,
        "ch67_regenerable_from_raw": raw_ok,
        "missing": missing,
        "ready_to_proceed": len(missing) == 0,
    }

    out_path = cfg.OUT_DIR / "preflight_audit.json"
    cr.write_json(out_path, audit)
    print(f"[preflight] wrote {cr.scrub_path(out_path)}  (profile={cfg.PROFILE})")
    print(f"[preflight] shape={shape_ok} clinical={clinical_ok} "
          f"ch67={ch67_ok} raw_eeg={raw_ok}")

    if missing:
        msg = (
            "# Missing required inputs\n\n"
            "The preflight audit could not locate the following required private "
            "inputs:\n\n"
            + "\n".join(f"- `{m}`" for m in missing)
            + "\n\nResolve via the documented environment variables "
            "(`ARSPI_SHAPE_FEATURES`, `ARSPI_CH67_FEATURES`, `ARSPI_CLINICAL_FILE`, "
            "`ARSPI_RAW_EEG_DIR`) or the default `data/` locations, then re-run. "
            "No fabricated outputs are produced.\n"
        )
        (cfg.OUT_DIR / "MISSING_INPUTS.md").write_text(msg)
        print("[preflight] MISSING INPUTS — wrote MISSING_INPUTS.md", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
