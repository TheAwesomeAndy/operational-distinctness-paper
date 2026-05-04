#!/usr/bin/env python3
"""Diagnostic for the five operational-distinctness feature blocks.

Loads the same inputs as run_all.py and reports per-block descriptive
statistics, with extra detail for the E (lsm_bsc6_pca) block, plus
SHA-256 hashes of the three input files.

Outputs:
    outputs/operational_distinctness/feature_block_diagnostics.json
    outputs/operational_distinctness/feature_block_diagnostics.txt
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PKG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG_DIR.parent.parent))

from experiments.operational_distinctness import config as cfg  # noqa: E402
from experiments.operational_distinctness.common import (  # noqa: E402
    build_feature_blocks,
    load_inputs,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _approx_rank_after_standardization(X: np.ndarray, tol_ratio: float = 1e-8) -> int:
    if X.size == 0 or X.shape[0] < 2 or X.shape[1] < 1:
        return 0
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd_safe = np.where(sd > 0, sd, 1.0)
    Xs = (X - mu) / sd_safe
    Xs = np.where(np.isfinite(Xs), Xs, 0.0)
    try:
        s = np.linalg.svd(Xs, compute_uv=False)
    except np.linalg.LinAlgError:
        return -1
    if s.size == 0:
        return 0
    cutoff = s[0] * tol_ratio if s[0] > 0 else 0.0
    return int((s > cutoff).sum())


def block_stats(name: str, X: np.ndarray) -> dict:
    X = np.asarray(X)
    finite_mask = np.isfinite(X)
    finite = X[finite_mask]
    col_std = X.std(axis=0) if X.ndim == 2 else np.array([X.std()])
    row_norms = np.linalg.norm(X, axis=1) if X.ndim == 2 else np.abs(X)
    n_constant_cols = int((col_std == 0).sum()) if X.ndim == 2 else 0
    n_near_constant_cols = int((col_std < 1e-12).sum()) if X.ndim == 2 else 0
    return {
        "block": name,
        "shape": list(X.shape),
        "dtype": str(X.dtype),
        "finite_fraction": float(finite_mask.mean()) if X.size else 0.0,
        "nan_count": int(np.isnan(X).sum()) if np.issubdtype(X.dtype, np.floating) else 0,
        "inf_count": int(np.isinf(X).sum()) if np.issubdtype(X.dtype, np.floating) else 0,
        "mean": float(finite.mean()) if finite.size else None,
        "std": float(finite.std()) if finite.size else None,
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "median_abs": float(np.median(np.abs(finite))) if finite.size else None,
        "zero_fraction": float((X == 0).mean()) if X.size else 0.0,
        "row_norm_min": float(row_norms.min()) if row_norms.size else None,
        "row_norm_median": float(np.median(row_norms)) if row_norms.size else None,
        "row_norm_max": float(row_norms.max()) if row_norms.size else None,
        "n_constant_columns": n_constant_cols,
        "n_near_constant_columns_std_lt_1e_minus_12": n_near_constant_cols,
        "column_std_min": float(col_std.min()) if col_std.size else None,
        "column_std_median": float(np.median(col_std)) if col_std.size else None,
        "column_std_max": float(col_std.max()) if col_std.size else None,
        "approx_rank_after_standardization": _approx_rank_after_standardization(X),
    }


def e_block_extras(ch5: dict, E: np.ndarray, y: np.ndarray) -> dict:
    raw = ch5["lsm_bsc6_pca"]
    raw_arr = np.asarray(raw)
    extras: dict = {
        "ch5_keys": sorted(list(ch5.keys())),
        "lsm_bsc6_pca_type": type(raw).__name__,
        "lsm_bsc6_pca_dtype": str(raw_arr.dtype),
        "lsm_bsc6_pca_shape": list(raw_arr.shape),
        "E_reshaped_shape": list(E.shape),
        "np_allclose_E_zero": bool(np.allclose(E, 0)),
    }
    try:
        extras["unique_value_count"] = int(np.unique(E).size)
    except Exception as exc:
        extras["unique_value_count_error"] = repr(exc)

    row_norms = np.linalg.norm(E, axis=1)
    extras["first_5_row_norms"] = [float(v) for v in row_norms[:5].tolist()]
    extras["row_norm_zero_fraction"] = float((row_norms == 0).mean()) if row_norms.size else 0.0

    classwise = {}
    y_arr = np.asarray(y).astype(int)
    for cls in sorted(np.unique(y_arr).tolist()):
        m = y_arr == cls
        if not m.any():
            continue
        E_cls = E[m]
        col_std_cls = E_cls.std(axis=0)
        classwise[int(cls)] = {
            "n": int(m.sum()),
            "mean_row_norm": float(np.linalg.norm(E_cls, axis=1).mean()),
            "column_std_min": float(col_std_cls.min()),
            "column_std_median": float(np.median(col_std_cls)),
            "column_std_max": float(col_std_cls.max()),
            "n_constant_columns_in_class": int((col_std_cls == 0).sum()),
        }
    extras["classwise"] = classwise
    return extras


def render_text(report: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("Feature-block diagnostics — operational distinctness pipeline")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Resolved input paths and SHA-256:")
    for k, v in report["inputs"].items():
        lines.append(f"  {k}:")
        lines.append(f"    path:   {v['path']}")
        lines.append(f"    exists: {v['exists']}")
        lines.append(f"    size:   {v['size_bytes']} bytes")
        lines.append(f"    sha256: {v['sha256']}")
    lines.append("")
    lines.append(f"n_observations: {report['n_observations']}")
    lines.append(f"n_subjects:     {report['n_subjects']}")
    lines.append(f"label_counts:   {report['label_counts']}")
    lines.append("")
    for s in report["blocks"]:
        lines.append("-" * 70)
        lines.append(f"Block: {s['block']}")
        for key, val in s.items():
            if key == "block":
                continue
            lines.append(f"  {key:42s}= {val}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("E-block extra diagnostics")
    for k, v in report["e_extras"].items():
        if k == "classwise":
            lines.append(f"  classwise:")
            for cls, d in v.items():
                lines.append(f"    y={cls}: {d}")
        else:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines) + "\n"


def main() -> int:
    print("Loading inputs (this may take a moment) ...")
    ch5, ch67, _clinical = load_inputs()
    blocks = build_feature_blocks(ch5, ch67)
    y = np.asarray(ch5["y"]).astype(int)

    inputs_meta: dict = {}
    for label, path in (
        ("ARSPI_CH5_FILE", cfg.CH5_FILE),
        ("ARSPI_CH67_FILE", cfg.CH67_FILE),
        ("ARSPI_CLINICAL_FILE", cfg.CLINICAL_FILE),
    ):
        p = Path(path).resolve()
        meta = {
            "path": str(p),
            "exists": p.exists(),
            "size_bytes": p.stat().st_size if p.exists() else None,
            "sha256": sha256_file(p) if p.exists() else None,
        }
        inputs_meta[label] = meta

    block_reports = [block_stats(name, X) for name, X in blocks.items()]

    e_extras = e_block_extras(ch5, blocks["E"], y)

    label_counts = {int(c): int((y == c).sum()) for c in sorted(np.unique(y).tolist())}

    report = {
        "n_observations": int(len(y)),
        "n_subjects": int(np.unique(np.asarray(ch5["subjects"])).size),
        "label_counts": label_counts,
        "inputs": inputs_meta,
        "blocks": block_reports,
        "e_extras": e_extras,
    }

    out_dir = cfg.OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "feature_block_diagnostics.json"
    txt_path = out_dir / "feature_block_diagnostics.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    txt_path.write_text(render_text(report), encoding="utf-8")

    print(render_text(report))
    print(f"-> {json_path}")
    print(f"-> {txt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
