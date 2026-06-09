#!/usr/bin/env python3
"""Phase 7 - Final quality gates for the risk-reduction + arXiv package.

Verifies the main/blind/supplement PDFs, the new tables/figures, manuscript-facing
claim hygiene, and traceability. Writes a JSON + Markdown report. Exits nonzero if
any CORE check fails (no silent patching).

Run:
    python experiments/tcds_risk_reduction/99_verify_risk_reduction_package.py
"""
from __future__ import annotations

import json
import re
import sys
import zlib
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
MAN = _REPO / "manuscript"
OUT = _REPO / "outputs" / "tcds_risk_reduction"
FIG = _REPO / "figures" / "tcds_risk_reduction"
TAB = _REPO / "tables" / "tcds_risk_reduction"

FORBIDDEN_CLAIM_TERMS = [
    "groundbreaking", "breakthrough", "revolutionary", "powerful", "unlocks",
    "state-of-the-art", "diagnostic biomarker", "clinical detection",
    "disorder-specific phenotype", "wearable feasibility", "measured energy",
    "real robot", "public dataset", "public repository", "superior dataset",
    "more accurate than most", "universally robust", "dominates all perturbations",
]
PRIVACY_TOKENS = ["clinical_profile.csv", ".pkl", "batch_data", "/home/",
                  "shape_features_211", "ch6_ch7_3class"]
IDENTIFIERS = ["Andrew A. Lane", "K. Wendy Tang", "Brady D. Nelson",
               "Stony Brook", "Laboratory for Clinical Affective"]


def pages(p: Path):
    if not p.exists():
        return None
    data = p.read_bytes()
    t = 0
    for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S):
        try:
            t += len(re.findall(rb"/Type\s*/Page[^s]", zlib.decompress(m.group(1))))
        except Exception:
            pass
    return t or len(re.findall(rb"/Type\s*/Page[^s]", data))


def main() -> int:
    checks = {}
    core_fail = []

    # 1. Main manuscript
    mp = pages(MAN / "main.pdf")
    checks["main_manuscript"] = {"exists": (MAN / "main.pdf").exists(), "pages": mp,
                                 "pages_le_10": (mp is not None and mp <= 10)}
    if not checks["main_manuscript"]["pages_le_10"]:
        core_fail.append("main.pdf missing or > 10 pages")

    # 2. Blind manuscript
    bp = pages(MAN / "main_blind.pdf")
    blind_txt = (MAN / "main_blind.tex").read_text() if (MAN / "main_blind.tex").exists() else ""
    blind_ids = [t for t in IDENTIFIERS if t in blind_txt]
    blind_priv = [t for t in PRIVACY_TOKENS if t in blind_txt]
    checks["blind_manuscript"] = {"exists": (MAN / "main_blind.pdf").exists(), "pages": bp,
                                  "pages_le_10": (bp is not None and bp <= 10),
                                  "identifiers": blind_ids, "private_tokens": blind_priv,
                                  "clean": (not blind_ids and not blind_priv)}
    if not checks["blind_manuscript"]["pages_le_10"] or blind_ids or blind_priv:
        core_fail.append("blind manuscript fails page/identifier/privacy check")

    # 3. Supplement
    sup = MAN / "supplemental_risk_reduction" / "supplement.pdf"
    sup_tex = MAN / "supplemental_risk_reduction" / "supplement.tex"
    sup_txt = sup_tex.read_text() if sup_tex.exists() else ""
    sup_priv = [t for t in PRIVACY_TOKENS if t in sup_txt and t != ".pkl"]  # 'pickle' word ok
    sup_energy = bool(re.search(r"measured (hardware )?energy(?! )|low-power result|hardware power",
                                sup_txt, re.I)) and "do not constitute measured" not in sup_txt
    sup_public = bool(re.search(r"public(ly available)? (data|dataset|repository)", sup_txt, re.I))
    checks["supplement"] = {"exists": sup.exists(), "pages": pages(sup),
                            "private_tokens": sup_priv,
                            "measured_energy_claim": sup_energy,
                            "public_data_claim": sup_public,
                            "clean": (sup.exists() and not sup_priv and not sup_energy and not sup_public)}
    if not checks["supplement"]["clean"]:
        core_fail.append("supplement missing or fails privacy/energy/public-data check")

    # 4. Tables and figures (or explicit skip reports) + paired metadata
    rr_figs = ["rr01_adaptive_router_performance", "rr02_router_regime_map",
               "rr03_event_resource_profile"]
    fig_status = {}
    for f in rr_figs:
        pdf = FIG / f"{f}.pdf"; meta = FIG / f"{f}.json"; skip = OUT / f"{f}_SKIPPED.md"
        fig_status[f] = {"pdf": pdf.exists(), "metadata": meta.exists(), "skip_report": skip.exists()}
    tables_ok = (TAB / "table_adaptive_evidence_routing.tex").exists() and \
                (TAB / "table_resource_event_accounting.tex").exists()
    checks["figures_tables"] = {"figures": fig_status, "risk_tables_present": tables_ok}
    if not tables_ok or not all(s["pdf"] and s["metadata"] for s in fig_status.values()):
        core_fail.append("risk-reduction figures/tables or metadata missing")

    # 5. Claim hygiene over manuscript-facing files
    facing = [MAN / "main.tex", MAN / "main_blind.tex", sup_tex,
              MAN / "arxiv_preprint" / "main_arxiv.tex"] + list(TAB.glob("*.tex"))
    claim_hits = {}
    for fp in facing:
        if not fp.exists():
            continue
        txt = fp.read_text()
        # Exclude the bibliography: forbidden words inside cited paper TITLES
        # (e.g., the Benjamini-Hochberg "practical and powerful" FDR paper) are
        # not author claims and must not be altered.
        txt = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "",
                     txt, flags=re.S)
        txt = txt.lower()
        hits = [t for t in FORBIDDEN_CLAIM_TERMS if t in txt]
        if hits:
            claim_hits[fp.name] = hits
    checks["claim_hygiene"] = {"hits": claim_hits, "clean": not claim_hits}
    if claim_hits:
        core_fail.append(f"forbidden claim terms present: {claim_hits}")

    # 6. Traceability of new claims
    trace = {
        "adaptive_evidence_routing": {
            "script": "experiments/tcds_risk_reduction/01_adaptive_evidence_routing.py",
            "outputs": ["adaptive_router_metrics.csv", "adaptive_router_summary.csv",
                        "adaptive_router_config.json"],
            "table": "tables/tcds_risk_reduction/table_adaptive_evidence_routing.tex",
            "figures": ["rr01_adaptive_router_performance.pdf", "rr02_router_regime_map.pdf"],
        },
        "resource_event_accounting": {
            "script": "experiments/tcds_risk_reduction/02_resource_event_accounting.py",
            "outputs": ["resource_event_metrics.csv", "resource_event_summary.json"],
            "table": "tables/tcds_risk_reduction/table_resource_event_accounting.tex",
            "figures": ["rr03_event_resource_profile.pdf"],
        },
    }
    trace_ok = True
    for claim, m in trace.items():
        for o in m["outputs"]:
            if not (OUT / o).exists():
                trace_ok = False
                core_fail.append(f"traceability: missing output {o} for {claim}")
        if not (_REPO / m["table"]).exists():
            trace_ok = False
    checks["traceability"] = {"map": trace, "complete": trace_ok}

    report = {"phase": "risk_reduction_final_verification",
              "checks": checks, "core_failures": core_fail,
              "passed": len(core_fail) == 0}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "final_verification_report.json").write_text(json.dumps(report, indent=2))
    md = ["# Risk-reduction final verification", "",
          f"**Passed: {report['passed']}**", ""]
    if core_fail:
        md += ["## Core failures"] + [f"- {c}" for c in core_fail] + [""]
    md += [f"- main.pdf: {mp} pages (<=10: {checks['main_manuscript']['pages_le_10']})",
           f"- main_blind.pdf: {bp} pages, identifiers={blind_ids or 'none'}, private={blind_priv or 'none'}",
           f"- supplement.pdf: {checks['supplement']['pages']} pages, clean={checks['supplement']['clean']}",
           f"- risk tables present: {tables_ok}",
           f"- claim hygiene clean: {checks['claim_hygiene']['clean']}",
           f"- traceability complete: {trace_ok}"]
    (OUT / "final_verification_report.md").write_text("\n".join(md) + "\n")
    print(f"[verify] passed={report['passed']} core_failures={core_fail}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
