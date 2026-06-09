#!/usr/bin/env python3
"""Phase 2 — Feature preparation and alignment.

Resolves the private inputs; if ``ch6_ch7_3class_features.pkl`` is missing it
regenerates it locally from the raw EEG via
``prepare_inputs/extract_ch67_features.py`` (output stays under the gitignored
``data/`` directory and is NEVER committed). Then verifies exact alignment of
the two feature pickles and the clinical metadata, and writes a privacy-safe
``outputs/tcds_ready9/feature_alignment_report.json``.

Core phase: on alignment failure writes ``ALIGNMENT_ERROR.md`` and exits
nonzero. No fabricated outputs.

Run:
    python experiments/tcds_ready9/00_prepare_features.py
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.tcds_ready9 import config as cfg  # noqa: E402
from experiments.tcds_ready9 import common_ready9 as cr  # noqa: E402


def _maybe_regenerate_embedding() -> bool:
    """Regenerate lsm_bsc6_pca in place if the shipped pickle holds the all-zero
    placeholder embedding. Reuses ``prepare_inputs/extract_ch5_features.py``
    (reservoir BSC6 + PCA from X_ds). The regenerated embedding stays local and
    must not be committed. Returns True if the embedding is usable afterwards.
    """
    try:
        with open(cfg.SHAPE_FILE, "rb") as f:
            ch5 = pickle.load(f)
    except Exception as e:
        print(f"[prepare] cannot read shape features: {e}", file=sys.stderr)
        return False
    E = np.asarray(ch5.get("lsm_bsc6_pca"))
    if E.size and not np.allclose(E, 0.0):
        return True  # already populated
    script = cfg.ROOT / "prepare_inputs" / "extract_ch5_features.py"
    if not script.exists():
        print("[prepare] embedding is a zero placeholder and the regeneration "
              "script is unavailable.", file=sys.stderr)
        return False
    print("[prepare] embedding is an all-zero placeholder; regenerating "
          "lsm_bsc6_pca locally (kept private/uncommitted)...")
    env = dict(os.environ)
    env["ARSPI_CH5_FILE"] = str(cfg.SHAPE_FILE)
    proc = subprocess.run([sys.executable, str(script),
                           "--input", str(cfg.SHAPE_FILE),
                           "--output", str(cfg.SHAPE_FILE)],
                          env=env, cwd=str(_REPO_ROOT))
    if proc.returncode != 0:
        return False
    with open(cfg.SHAPE_FILE, "rb") as f:
        E2 = np.asarray(pickle.load(f).get("lsm_bsc6_pca"))
    return bool(E2.size and not np.allclose(E2, 0.0))


def _regenerate_ch67() -> bool:
    """Invoke the local feature-extraction script to produce the ch67 pickle.

    Returns True on success. The generated pickle is written to
    ``cfg.CH67_REGEN_OUTPUT`` under the gitignored ``data/`` dir.
    """
    script = cfg.LOCAL_FEATURE_SCRIPT
    if not script.exists():
        print(f"[prepare] local feature script not found: {cr.scrub_path(script)}",
              file=sys.stderr)
        # Fall back to the upstream script location if available.
        if cfg.UPSTREAM_FEATURE_SCRIPT.exists():
            script = cfg.UPSTREAM_FEATURE_SCRIPT
        else:
            return False
    if not cfg.RAW_EEG_DIR.exists():
        print("[prepare] raw EEG directory unavailable; cannot regenerate.",
              file=sys.stderr)
        return False
    env = dict(os.environ)
    env["ARSPI_RAW_BATCH_DIR"] = str(cfg.RAW_EEG_DIR)
    env["ARSPI_CH67_OUTPUT"] = str(cfg.CH67_REGEN_OUTPUT)
    print(f"[prepare] regenerating ch6_ch7 features via {cr.scrub_path(script)} "
          f"(this may take ~10-20 min)...")
    proc = subprocess.run([sys.executable, str(script)], env=env, cwd=str(_REPO_ROOT))
    return proc.returncode == 0 and cfg.CH67_REGEN_OUTPUT.exists()


def _label_dist(y) -> dict:
    return {str(k): int(v) for k, v in sorted(Counter(np.asarray(y).tolist()).items())}


def _write_alignment_error(reasons: list[str]) -> None:
    body = (
        "# Feature alignment error\n\n"
        "Phase 2 detected a mismatch between the consumed private inputs. "
        "Downstream Ready-9 analyses were not run, and no fabricated outputs were "
        "produced.\n\n## Failures\n\n"
        + "\n".join(f"- {r}" for r in reasons)
        + "\n\nResolve the input mismatch (regenerate the feature pickle from the "
        "matching raw EEG, or point the environment variables at consistent "
        "files) and re-run.\n"
    )
    (cfg.OUT_DIR / "ALIGNMENT_ERROR.md").write_text(body)


def main() -> int:
    # 0. Ensure the reservoir embedding is populated (not the zero placeholder).
    if not _maybe_regenerate_embedding():
        _write_alignment_error([
            "lsm_bsc6_pca embedding is an all-zero placeholder and could not be "
            "regenerated locally; downstream classification would be invalid."
        ])
        print("[prepare] FAILED to obtain a valid embedding.", file=sys.stderr)
        return 1

    # 1. Ensure the ch67 pickle exists (regenerate locally if needed).
    if not cfg.CH67_FILE.exists():
        ok = _regenerate_ch67()
        if not ok or not cfg.CH67_REGEN_OUTPUT.exists():
            _write_alignment_error([
                "ch6_ch7_3class_features.pkl is missing and could not be "
                "regenerated locally from raw EEG."
            ])
            print("[prepare] FAILED to obtain ch67 features.", file=sys.stderr)
            return 1

    # 2. Load both pickles directly (avoid load_inputs' hard raise so we can
    #    produce a structured report on mismatch).
    reasons: list[str] = []
    with open(cfg.SHAPE_FILE, "rb") as f:
        ch5 = pickle.load(f)
    with open(cfg.CH67_FILE, "rb") as f:
        ch67 = pickle.load(f)

    s5 = np.asarray(ch5["subjects"])
    s67 = np.asarray(ch67["subjects"])
    y5 = np.asarray(ch5["y"])
    y67 = np.asarray(ch67["y"])

    subjects_match = s5.shape == s67.shape and bool(np.array_equal(s5, s67))
    labels_match = y5.shape == y67.shape and bool(np.array_equal(y5, y67))
    obs_match = len(y5) == len(y67)
    subj_match = len(set(s5.tolist())) == len(set(s67.tolist()))

    if not obs_match:
        reasons.append(f"observation count differs: shape={len(y5)} ch67={len(y67)}")
    if not subjects_match:
        reasons.append("subject arrays differ between shape and ch67 pickles")
    if not labels_match:
        reasons.append("label arrays differ between shape and ch67 pickles")

    # 3. Clinical alignment (by hashed ID; never expose raw IDs).
    import pandas as pd
    clinical_matched = None
    clinical_present = cfg.CLINICAL_FILE.exists()
    if clinical_present:
        cdf = pd.read_csv(cfg.CLINICAL_FILE).drop_duplicates(subset="ID", keep="first")
        clin_ids = set(cdf["ID"].tolist())
        feat_ids = set(s5.tolist())
        clinical_matched = len(feat_ids & clin_ids)
        if clinical_matched == 0:
            reasons.append("no clinical IDs match the feature subject set")

    # 4. Expected three-class balance.
    dist = _label_dist(y5)
    expected_balance = (len(dist) == 3 and len(set(dist.values())) == 1)

    report = {
        "provenance": cr.base_provenance(),
        "inputs": {
            "shape_features_211": cr.file_fingerprint(cfg.SHAPE_FILE),
            "ch6_ch7_3class_features": cr.file_fingerprint(cfg.CH67_FILE),
            "clinical_profile": cr.file_fingerprint(cfg.CLINICAL_FILE),
        },
        "alignment": {
            "subjects_match": subjects_match,
            "labels_match": labels_match,
            "observation_count_match": obs_match,
            "subject_count_match": subj_match,
            "n_observations": int(len(y5)),
            "n_unique_subjects": int(len(set(s5.tolist()))),
            "label_distribution": dist,
            "three_class_balanced": expected_balance,
        },
        "clinical": {
            "present": clinical_present,
            "n_matched_subjects": clinical_matched,
            "diagnoses_available": [d for d in cfg.DIAGNOSES
                                    if clinical_present and d in pd.read_csv(cfg.CLINICAL_FILE).columns],
        },
        "passed": len(reasons) == 0,
        "failures": reasons,
    }
    cr.write_json(cfg.OUT_DIR / "feature_alignment_report.json", report)

    if reasons:
        _write_alignment_error(reasons)
        print("[prepare] ALIGNMENT FAILED:", "; ".join(reasons), file=sys.stderr)
        return 1

    print(f"[prepare] alignment OK: {len(y5)} obs, "
          f"{len(set(s5.tolist()))} subjects, balance={dist}, "
          f"clinical_matched={clinical_matched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
