# Repository Agent Guide

Instructions for any automated coding/writing agent operating in this repository. Read this before
making changes. (This is the vendor-neutral root agent specification; it deliberately carries no
tool/vendor brand name in its filename or body.)

## Target venue and central idea
- **Target venue:** IEEE Transactions on Affective Computing (TAFFC). Preserved fallback: IEEE TCDS.
- **Central idea (keep consistent everywhere):** ARSPI-Net is a neuromorphic reservoir–graph
  *observable map* that transforms trial-averaged affective ERP observations into four operationally
  distinct evidence streams (E spike embedding, D dynamical descriptors, T tPLV graph topology,
  C structure–function coupling κ) and evaluates them under perturbation and in an **offline**
  closed-loop evidence-accumulation simulation. It is a *measurement substrate*, not a static
  classifier.

## Manuscript files
- Primary: `manuscript/taffc/main_taffc.tex` → `main_taffc.pdf`.
- Blind twin: `manuscript/taffc/main_taffc_blind.tex` → `main_taffc_blind.pdf`.
- Shared supplement: `manuscript/supplemental_risk_reduction/supplement.pdf`.
- Cover letter: `manuscript/COVER_LETTER_TAFFC_FINAL.md`.
- Class is vendored: `manuscript/IEEEtran.cls` (IEEEtran **V1.8b**); compile from `manuscript/` so
  `./IEEEtran.cls` resolves.

## Hard rules
- **Audit before edit.** Inspect and report findings before changing manuscript text.
- **No automatic merge.** Open PRs; never merge without explicit instruction.
- **No fabrication** of experiments, baselines, metadata, arousal labels, reviewer expectations, or
  journal policies.
- **Do not change numerical results** unless a script regenerates them and the change is reported.
- **Forbidden claims:** diagnostic validation, online/real-time BCI, physical-robot embodiment,
  wearable feasibility, measured hardware energy, cognitive augmentation, and state-of-the-art /
  classifier-superiority claims. "real-time" may appear only inside an explicit not-demonstrated
  sentence.
- **Privacy:** never commit raw EEG, feature pickles (`*.pkl`), clinical CSVs, subject identifiers,
  PHI, private absolute paths, or restricted data. Keep blind files free of institution/lab/author
  identifiers.
- **Preserve the TCDS fallback** material; only keep the TAFFC bundle cleanly TAFFC-facing.
- **No vendor/AI/session footers or identifiers** in committed files, reports, commit messages, or
  PR/issue bodies. Commits carry the repository owner's git identity.

## Verifier layer (use facts, not memory)
- Base every claim on the actual repository files, the vendored class, the provided model papers, and
  the validation reports — not on recollection or guesswork.
- When TAFFC author guidelines or recent-publication/novelty fit cannot be confirmed from local
  material, report **NEEDS WEB VERIFICATION / NEEDS AUTHOR DECISION** rather than asserting.
- The blind-review model (single vs double) is a portal-time author decision; do not assume it.

## Required validation before proposing a manuscript change is "done"
Compile both variants twice from `manuscript/` and confirm:
- page counts within the 12-page free limit;
- 0 undefined references, 0 undefined citations, 0 multiply-defined labels, 0 repeated `\bibitem`
  keys;
- numeric-token multiset preserved unless a regeneration is intended and reported;
- shared TCDS package byte-identical unless explicitly in scope;
- no forbidden claims and no privacy/identity leaks introduced.
