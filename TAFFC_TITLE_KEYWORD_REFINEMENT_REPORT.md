# TAFFC Title and Keyword Refinement Report

**Date:** 2026-06-12 · **Branch:** `submission/taffc-title-keyword-pass` (from `main` `6ecec5d`)

Scope: TAFFC manuscript **variant only**. The shared TCDS manuscript and TCDS package were not
touched. No numerical results, figures, tables, or experiment outputs changed — only the title,
the first ARSPI-Net definition in the abstract, and the keyword list.

## Title
- **Old:** `ARSPI-Net: A Neuromorphic Reservoir-Graph Substrate for Affective ERP Decoding and Closed-Loop Neural Evidence Accumulation`
- **New:** `A Neuromorphic Reservoir–Graph Framework for Affective ERP Evidence Accumulation`

The title now foregrounds the scientific object; the ARSPI-Net acronym is no longer the title's
lead. No second acronym (e.g., R-SPINET / RSPINET) was introduced — `ARSPI-Net` is used
consistently.

## ARSPI-Net retained inside the paper (confirmed)
- **Abstract:** the framework name remains defined on first use —
  *"We present ARSPI-Net (Affective Reservoir-Spike Processing and Inference Network), a neuromorphic
  event-driven reservoir–graph framework for affective event-related potential (ERP) decoding."*
  (lightly adjusted: "substrate" → "framework" to read naturally under the new title; the
  measurement-substrate framing is unchanged elsewhere in the body.)
- ARSPI-Net continues to appear throughout the methods, results, captions, and cover-letter body.

## Keywords (confirmed: ARSPI-Net added, at the end)
- **Old:** EEG, event-related potentials, brain-computer interfaces, neural decoding, neuromorphic
  computing, reservoir computing, spiking neural networks, graph signal processing, structure-function
  coupling, affective computing, closed-loop evidence accumulation, adaptive BCI simulation.
- **New:** affective computing, EEG, event-related potentials, neuromorphic computing, reservoir
  computing, spiking neural networks, graph signal processing, structure-function coupling, evidence
  accumulation, **ARSPI-Net**.

## Files changed
- `manuscript/taffc/main_taffc.tex` — title, first abstract definition, keywords.
- `manuscript/taffc/main_taffc.pdf` — recompiled.
- `manuscript/taffc/main_taffc_blind.tex` — title, first abstract definition, keywords.
- `manuscript/taffc/main_taffc_blind.pdf` — recompiled.
- `manuscript/COVER_LETTER_TAFFC_FINAL.md` — quoted manuscript title only (ARSPI-Net retained in body).
- `SUBMISSION_PROFILE_TAFFC.md` — Title field only.
- `TAFFC_TITLE_KEYWORD_REFINEMENT_REPORT.md` — this report (new).

## Page count after compilation
- `main_taffc.pdf`: **11 pages** — within TAFFC's verified 12-page free formatted limit.
- `main_taffc_blind.pdf`: **11 pages** — within the limit.
- `supplement.pdf`: 6 pages (unchanged).

## TCDS files unchanged (confirmed)
`manuscript/main.tex`, `manuscript/main.pdf`, `manuscript/main_blind.tex`, `manuscript/main_blind.pdf`,
`manuscript/supplemental_risk_reduction/supplement.pdf`, `manuscript/COVER_LETTER_TCDS_COGNEURO_BCI_FINAL.md`,
`SUBMISSION_PROFILE.md`, `CFP_ALIGNMENT_COGNEURO_BCI.md`, `READY_FOR_COGNEURO_BCI_SUBMISSION_CHECKLIST.md`
are all **byte-identical to `origin/main`**.

## No numerical results / figures / tables / experiment outputs changed (confirmed)
Numeric-token comparison of the compiled `main_taffc.pdf` before vs after this pass: the multiset of
all decimals, percentages, counts, confidence intervals, and years is **identical**. No figure or
table source was edited; shared figures were not regenerated. No forbidden claims were introduced;
the AI / session-reference scan is clean across all changed files.
