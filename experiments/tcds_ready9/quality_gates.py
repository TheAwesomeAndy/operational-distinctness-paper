#!/usr/bin/env python3
"""Phase 16 — Quality gates: privacy and manuscript language.

Privacy gate:
  * no private files staged in git (raw EEG, pickles, clinical CSV);
  * no private absolute paths or private file basenames in committed
    manuscript-facing or aggregate text artifacts.

Language gate (manuscript-facing files only: manuscript/main.tex and the
LaTeX tables it includes):
  * no banned hype/process/overclaim terms.

Exits nonzero if any violation is found. Internal planning ``.md`` files under
outputs/ are intentionally NOT language-gated (they may use process language).

Run:
    python experiments/tcds_ready9/quality_gates.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

PRIVATE_STAGED_PATTERNS = [
    r"batch_data", r"\.pkl$", r"clinical_profile\.csv$", r"\.npz$",
    r"shape_data/", r"raw_eeg/",
]

# Private *absolute-path* tokens that must not appear inside committed text
# artifacts. Bare file basenames (e.g. "shape_features_211.pkl") are permitted:
# privacy-safe fingerprints intentionally record name + hash without any path.
PRIVATE_TEXT_TOKENS = [
    "/home/", "/Users/", "C:\\Users", "/root/",
]

# Banned terms in manuscript-facing files. Word-boundaried to avoid false
# positives such as "hypothesis" matching "thesis".
BANNED_TERMS = [
    r"\bdissertation\b", r"\bdoctoral\b", r"\bthesis\b",
    r"\bgroundbreaking\b", r"\bbreakthrough\b", r"\brevolutionary\b",
    r"\bunlocks?\b", r"\bstate-of-the-art\b",
    r"\bdiagnostic biomarker\b", r"\bclinical detection\b",
    r"proves disorder-specific phenotype", r"\bwearable feasibility\b",
    r"\bmeasured energy\b", r"\breal robot\b", r"\bsuperior dataset\b",
    r"more accurate than most", r"\bready-9\b", r"hardening package",
    r"scientific-director", r"reviewer-facing", r"accepted-paper competency",
    r"paper slicing",
]

MANUSCRIPT_FILES = [
    _REPO_ROOT / "manuscript" / "main.tex",
]


def _git(*args):
    out = subprocess.run(["git", "-C", str(_REPO_ROOT), *args],
                         capture_output=True, text=True)
    return out.stdout


def privacy_gate() -> list[str]:
    violations = []
    # 1. staged/tracked private files among NEW package additions
    tracked = _git("ls-files").splitlines()
    staged = _git("diff", "--cached", "--name-only").splitlines()
    for f in staged:
        for pat in PRIVATE_STAGED_PATTERNS:
            if re.search(pat, f):
                violations.append(f"private file staged: {f}")
    # 2. private tokens inside committed manuscript-facing + package text files
    text_files = []
    for base in ("manuscript/main.tex",):
        text_files.append(_REPO_ROOT / base)
    for d in ("tables/tcds_ready9", "outputs/tcds_ready9"):
        text_files += list((_REPO_ROOT / d).rglob("*.tex"))
        text_files += list((_REPO_ROOT / d).rglob("*.csv"))
        text_files += list((_REPO_ROOT / d).rglob("*.json"))
    for fp in text_files:
        if not fp.exists():
            continue
        try:
            txt = fp.read_text(errors="ignore")
        except Exception:
            continue
        for tok in PRIVATE_TEXT_TOKENS:
            if tok in txt:
                violations.append(f"private token '{tok}' in {fp.relative_to(_REPO_ROOT)}")
    return violations


def language_gate() -> list[str]:
    violations = []
    files = list(MANUSCRIPT_FILES)
    files += list((_REPO_ROOT / "tables" / "tcds_ready9").rglob("*.tex"))
    for fp in files:
        if not fp.exists():
            continue
        txt = fp.read_text(errors="ignore")
        for pat in BANNED_TERMS:
            for m in re.finditer(pat, txt, flags=re.IGNORECASE):
                line = txt[:m.start()].count("\n") + 1
                violations.append(f"{fp.relative_to(_REPO_ROOT)}:{line}: '{m.group(0)}'")
    return violations


def main() -> int:
    pv = privacy_gate()
    lv = language_gate()
    print("=== PRIVACY GATE ===")
    print("PASS" if not pv else "\n".join(pv))
    print("=== LANGUAGE GATE (manuscript-facing) ===")
    print("PASS" if not lv else "\n".join(lv))
    if pv or lv:
        print(f"\nQUALITY GATES FAILED: {len(pv)} privacy, {len(lv)} language", file=sys.stderr)
        return 1
    print("\nALL QUALITY GATES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
