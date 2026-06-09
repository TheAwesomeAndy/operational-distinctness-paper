# arXiv preprint package — ARSPI-Net (non-blind)

This is a **non-blind, public** preprint package, separate from the ScholarOne
review package. It is scientifically identical to the current 10-page non-blind
manuscript (`manuscript/main.pdf`), plus a preprint status note.

## Contents
- `main_arxiv.tex` — non-blind manuscript source (preprint note added).
- `main_arxiv.pdf` — compiled preprint (10 pages).
- `ARSPI-Net_arxiv_source.zip` — self-contained arXiv source (flat: `main_arxiv.tex`
  + figures + table `.tex`; compiles with pdflatex + IEEEtran.cls supplied by arXiv).
- `ARXIV_METADATA.md` — title, authors, plain-text abstract, categories, comments,
  license guidance, data/code statements.
- `ARXIV_SUBMISSION_CHECKLIST.md` — step-by-step submission checklist.
- `ARXIV_SOURCE_MANIFEST.json` — included/excluded files, page count, checksums,
  privacy-scan result.
- `build_src/` — the exact source root that produced the zip (for inspection).

## What is deliberately excluded (do not upload to arXiv)
`main_blind.*`, `submission_package_blind/`, `main_full_internal.*`, internal
reports, PR text, `.git`, raw EEG, feature pickles, `clinical_profile.csv`, and any
private/local paths. The privacy scan confirms none of these appear in the package.

## IMPORTANT — posting timing is an author decision
This package does **not** submit to arXiv. The author manually submits.

> **Anonymity caution:** evidence indicates the IEEE TCDS special issue may use a
> **double-anonymous** review process. Public arXiv posting can reduce practical
> anonymity even where technically permitted. The author must decide **when** to post:
> 1. before journal submission;
> 2. immediately after journal submission;
> 3. after the first decision;
> 4. after acceptance.
>
> This package makes no such decision automatically. Confirm the live ScholarOne /
> special-issue policy before posting.

## AI-use disclosure
No AI-use disclosure has been inserted into the manuscript. Whether to add one for
preprint posting is an author-side decision per the target policy; if added it must
not imply AI authorship. Record the decision here before posting.

## Status note used
"This manuscript is a preprint and has not been certified by peer review."
(Do not use "accepted", "in press", "published", "under review", "IEEE-approved
version", or "publisher version".)
