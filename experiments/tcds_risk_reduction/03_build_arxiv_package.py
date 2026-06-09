#!/usr/bin/env python3
"""Build the non-blind arXiv preprint package from manuscript/arxiv_preprint/main_arxiv.tex.

Creates a clean, self-contained arXiv source root (build_src/), flattens figure
and table paths, compiles, runs privacy/source-hygiene scans, zips the source,
computes SHA256 checksums, and writes the source manifest + verification report.

Does NOT submit to arXiv. Excludes blind/internal/private artifacts.

Run:
    python experiments/tcds_risk_reduction/03_build_arxiv_package.py
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
MAN = _REPO / "manuscript"
ARX = MAN / "arxiv_preprint"
BUILD = ARX / "build_src"
OUT = _REPO / "outputs" / "tcds_risk_reduction"

# Artifact/identifier tokens that must NOT leak into the public arXiv source.
# Note: bare "blind" is intentionally excluded -- it legitimately appears in the
# scientific term "belief-blind" (a control policy). The actual blind-review
# artifacts are matched by their specific filenames below.
FORBIDDEN_SUBSTRINGS = [
    "main_full_internal", "main_blind", "submission_package_blind", "ScholarOne",
    "not-for-submission", "double-blind", "double-anonymous", ".pkl",
    "clinical_profile.csv", "batch_data", "/home/", "shape_features", "ch6_ch7",
    "TODO", "FIXME",
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _pdf_pages(p: Path) -> int:
    import zlib
    data = p.read_bytes()
    t = 0
    for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S):
        try:
            t += len(re.findall(rb"/Type\s*/Page[^s]", zlib.decompress(m.group(1))))
        except Exception:
            pass
    if t == 0:
        t = len(re.findall(rb"/Type\s*/Page[^s]", data))
    return t


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    src_tex = ARX / "main_arxiv.tex"
    if not src_tex.exists():
        print("[arxiv] main_arxiv.tex missing", file=sys.stderr)
        return 1

    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)

    tex = src_tex.read_text()

    # collect figures referenced
    figs = re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", tex)
    figs = [f if f.endswith(".pdf") else f + ".pdf" for f in figs]
    inputs = re.findall(r"\\input\{([^}]+)\}", tex)

    included, missing = [], []
    # figures: search manuscript/figures and repo figures
    for f in figs:
        name = Path(f).name
        cand = list((MAN / "figures").rglob(name)) + list((_REPO / "figures").rglob(name))
        if cand:
            shutil.copy(cand[0], BUILD / name)
            included.append(name)
        else:
            missing.append(f)
    # tables: search manuscript/tables and repo tables
    for inp in inputs:
        name = Path(inp).name
        cand = [_REPO / inp, MAN / inp, _REPO / inp.lstrip("./")]
        cand += list((MAN / "tables").rglob(name)) + list((_REPO / "tables").rglob(name))
        src = next((c for c in cand if c.exists()), None)
        if src:
            shutil.copy(src, BUILD / name)
            included.append(name)
        else:
            missing.append(inp)

    # flatten paths in the tex and strip private comment lines
    flat = re.sub(r"\\graphicspath\{[^}]*\}(\{[^}]*\})*", r"\\graphicspath{{./}}", tex)
    flat = re.sub(r"\\input\{[^}]*/([^}/]+)\}", r"\\input{\1}", flat)
    flat = re.sub(r"\\input\{([^}/]+)\}", r"\\input{\1}", flat)
    cleaned_lines = []
    for line in flat.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%") and any(tok.lower() in line.lower()
                                            for tok in ("/home/", "private", "todo", "fixme")):
            continue
        cleaned_lines.append(line)
    (BUILD / "main_arxiv.tex").write_text("\n".join(cleaned_lines) + "\n")
    included.append("main_arxiv.tex")

    # compile twice
    pdflatex = shutil.which("pdflatex") or "/usr/bin/pdflatex"
    log = ""
    for _ in range(2):
        r = subprocess.run([pdflatex, "-interaction=nonstopmode", "main_arxiv.tex"],
                           cwd=BUILD, capture_output=True, text=True)
        log = r.stdout + r.stderr
    pdf = BUILD / "main_arxiv.pdf"
    compiled = pdf.exists()
    pages = _pdf_pages(pdf) if compiled else None
    undefined = len(re.findall(r"undefined (reference|citation)", log, re.I))
    overfull_big = [float(x) for x in re.findall(r"Overfull \\hbox \(([0-9.]+)pt", log) if float(x) > 50]

    # copy compiled pdf up to package root
    if compiled:
        shutil.copy(pdf, ARX / "main_arxiv.pdf")

    # privacy scan over build_src text files + PDF text
    scan_hits = {}
    for p in BUILD.iterdir():
        if p.suffix in (".tex",):
            txt = p.read_text(errors="ignore")
            hits = sorted({tok for tok in FORBIDDEN_SUBSTRINGS if tok.lower() in txt.lower()})
            if hits:
                scan_hits[p.name] = hits
    # PDF rendered-text scan (best-effort, if pdftotext available)
    pdftotext = shutil.which("pdftotext")
    pdf_hits = []
    if compiled and pdftotext:
        subprocess.run([pdftotext, str(ARX / "main_arxiv.pdf"), "/tmp/_arxiv_pdf.txt"],
                       capture_output=True)
        ptxt = Path("/tmp/_arxiv_pdf.txt").read_text(errors="ignore")
        pdf_hits = sorted({tok for tok in ["main_full_internal", "main_blind", "ScholarOne",
                                           ".pkl", "clinical_profile", "/home/", "batch_data"]
                           if tok.lower() in ptxt.lower()})

    # zip the build_src as the arxiv source
    zip_path = ARX / "ARSPI-Net_arxiv_source.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(BUILD.iterdir()):
            if p.suffix in (".aux", ".log", ".out"):
                continue
            z.write(p, p.name)

    manifest = {
        "package": "arxiv_preprint",
        "non_blind": True,
        "compiled": compiled,
        "pages": pages,
        "undefined_references": undefined,
        "overfull_boxes_gt_50pt": len(overfull_big),
        "included_files": sorted(set(included)),
        "missing_files": missing,
        "excluded_by_design": ["main_blind.tex", "main_blind.pdf", "main_full_internal.*",
                               "submission_package_blind/", "internal reports", "raw data",
                               "feature pickles", "clinical_profile.csv", ".git"],
        "privacy_scan": {"source_hits": scan_hits, "pdf_hits": pdf_hits,
                         "clean": (not scan_hits and not pdf_hits)},
        "checksums_sha256": {
            "main_arxiv.pdf": _sha(ARX / "main_arxiv.pdf") if compiled else None,
            "ARSPI-Net_arxiv_source.zip": _sha(zip_path),
        },
    }
    (ARX / "ARXIV_SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    # verification report
    report = dict(manifest)
    report["main_pdf_checksum"] = manifest["checksums_sha256"]["main_arxiv.pdf"]
    report["source_zip_checksum"] = manifest["checksums_sha256"]["ARSPI-Net_arxiv_source.zip"]
    (OUT / "arxiv_preprint_verification.json").write_text(json.dumps(report, indent=2))
    md = [
        "# arXiv preprint package - verification report", "",
        f"- compiled: {compiled}", f"- pages: {pages}",
        f"- undefined references: {undefined}",
        f"- overfull boxes > 50pt: {len(overfull_big)}",
        f"- privacy scan clean: {manifest['privacy_scan']['clean']}",
        f"- source hits: {scan_hits or 'none'}",
        f"- pdf hits: {pdf_hits or 'none'}",
        f"- main_arxiv.pdf sha256: {report['main_pdf_checksum']}",
        f"- ARSPI-Net_arxiv_source.zip sha256: {report['source_zip_checksum']}", "",
        "## Included files", *[f"- {f}" for f in sorted(set(included))],
        "", "## Missing files (must be empty)", *([f"- {m}" for m in missing] or ["- none"]),
        "", "## Author metadata placeholders still needing author input",
        "- affiliations, corresponding-author email, ORCID iDs, funding, conflict-of-interest",
        "  (see ARXIV_METADATA.md; not fabricated)",
        "", "## Caution",
        "- If the TCDS special issue is double-anonymous, public arXiv posting before review",
        "  may reduce practical anonymity. Posting timing is an author decision (see README_ARXIV.md).",
    ]
    (OUT / "arxiv_preprint_verification.md").write_text("\n".join(md) + "\n")

    print(f"[arxiv] compiled={compiled} pages={pages} undefined={undefined} "
          f"privacy_clean={manifest['privacy_scan']['clean']} missing={missing}")
    return 0 if (compiled and not missing and manifest["privacy_scan"]["clean"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
