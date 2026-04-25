#!/usr/bin/env python3
"""Run the full operational-distinctness reporting pipeline.

Executes scripts 01-04 in order and writes a manifest at
``outputs/operational_distinctness/run_manifest.json`` recording
the inputs found, outputs created, and the package versions used.

Usage:
    python experiments/operational_distinctness/run_all.py
    python experiments/operational_distinctness/run_all.py --include-optional
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG_DIR.parent.parent))

from experiments.operational_distinctness import config as cfg  # noqa: E402

REQUIRED_SCRIPTS = [
    "01_affective_ablation_metrics.py",
    "02_clinical_sensitivity_metrics.py",
    "03_generate_submission_figures.py",
    "04_generate_submission_tables.py",
]

OPTIONAL_SCRIPTS = [
    "05_optional_permutation_fdr.py",
    "06_optional_comorbidity_adjustment.py",
    "07_optional_layer_redundancy.py",
]


def _run(script: Path) -> int:
    print(f"\n>>> {script.name}")
    proc = subprocess.run([sys.executable, str(script)], cwd=str(PKG_DIR.parent.parent))
    return proc.returncode


def _git_commit_hash() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PKG_DIR.parent.parent),
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except FileNotFoundError:
        pass
    return None


def _pkg_version(name: str) -> str | None:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def _input_files_found() -> dict[str, bool]:
    return {
        "shape_features_211.pkl": cfg.CH5_FILE.exists(),
        "ch6_ch7_3class_features.pkl": cfg.CH67_FILE.exists(),
        "clinical_profile.csv": cfg.CLINICAL_FILE.exists(),
    }


def _output_files_created() -> list[str]:
    found: list[str] = []
    for d in (cfg.OUT_DIR, cfg.FIG_DIR, cfg.TABLE_DIR):
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file():
                found.append(str(p.relative_to(cfg.ROOT)))
    return found


def _dataset_summary() -> dict[str, int | None]:
    """Light-weight subject/observation count for the manifest."""
    summary: dict[str, int | None] = {"n_subjects": None, "n_observations": None}
    if not cfg.CH5_FILE.exists():
        return summary
    try:
        import pickle
        with open(cfg.CH5_FILE, "rb") as f:
            ch5 = pickle.load(f)
        import numpy as np
        summary["n_observations"] = int(len(ch5["y"]))
        summary["n_subjects"] = int(np.unique(ch5["subjects"]).size)
    except Exception as exc:  # pragma: no cover -- diagnostic only
        summary["error"] = repr(exc)  # type: ignore[assignment]
    return summary


def write_manifest() -> Path:
    manifest = {
        "commit_hash": _git_commit_hash(),
        "run_datetime": dt.datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "numpy_version": _pkg_version("numpy"),
        "scipy_version": _pkg_version("scipy"),
        "sklearn_version": _pkg_version("sklearn"),
        "pandas_version": _pkg_version("pandas"),
        "random_state": cfg.RANDOM_STATE,
        "input_files_found": _input_files_found(),
        "output_files_created": _output_files_created(),
        **_dataset_summary(),
    }
    out = cfg.OUT_DIR / "run_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-optional", action="store_true",
                        help="Also run optional hardening scripts (05-07).")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Script filenames to skip.")
    args = parser.parse_args()

    scripts = list(REQUIRED_SCRIPTS)
    if args.include_optional:
        scripts += OPTIONAL_SCRIPTS

    failures: list[str] = []
    for name in scripts:
        if name in args.skip:
            print(f"\n--- Skipping {name} (per --skip)")
            continue
        path = PKG_DIR / name
        if not path.exists():
            print(f"\n--- Skipping {name} (not present)")
            continue
        rc = _run(path)
        if rc != 0:
            failures.append(f"{name} -> exit {rc}")

    manifest_path = write_manifest()
    print(f"\nManifest -> {manifest_path}")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
