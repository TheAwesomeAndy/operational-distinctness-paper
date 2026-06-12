# TAFFC Title and Keyword Refinement Report

**Date:** 2026-06-12 · **Status:** merged into `main` via PR #21; `main` is canonical.

Scope: TAFFC manuscript **variant only**. The shared TCDS manuscript, TCDS package, figures, tables,
numerical results, and experiment outputs were not touched. Only the title, the first ARSPI-Net
definition in the abstract (plus an ERP-specificity sentence), and the keyword list changed.

> Status: this title refinement was **merged into `main` via PR #21**, which supplied the final
> title. The alternative title PR (#20, `A Neuromorphic Reservoir–Graph Framework for Affective ERP
> Evidence Accumulation`) was **closed/superseded**. `main` is now canonical.

## Title
- **Old (superseded TAFFC variant title):** `ARSPI-Net: A Neuromorphic Reservoir-Graph Substrate for Affective ERP Decoding and Closed-Loop Neural Evidence Accumulation`
- **New:** `Spatiotemporal Characterization of Affective EEG Dynamics Using a Neuromorphic Reservoir–Graph Framework`

The title now uses **EEG** for broader indexing/citation visibility and a physiological
signal-analysis tone; it does not foreground ARSPI-Net, evidence accumulation, state estimation,
online BCI, or diagnosis. No second acronym (R-SPINET/RSPINET) was introduced.

## EEG appears in the title (confirmed)
"Affective **EEG** Dynamics" is in the title; verified in the compiled PDF.

## ARSPI-Net remains defined in the abstract + ERP specificity preserved (confirmed)
- Definition (lightly adjusted to read under the new title):
  *"We present ARSPI-Net (Affective Reservoir-Spike Processing and Inference Network), a neuromorphic
  event-driven reservoir–graph framework for spatiotemporal characterization of affective
  event-related EEG dynamics."*
- ERP specificity immediately follows:
  *"In this study, the analyzed EEG observations are trial-averaged event-related potentials (ERPs)
  elicited by negative, neutral, and pleasant affective stimulus conditions."*
- ERP / event-related terminology is retained throughout the abstract, methods, and dataset
  description; ERP was **not** globally replaced with EEG, and no continuous raw-EEG decoding is
  implied (the analysis is trial-averaged ERP observations).

## Keywords (confirmed: ARSPI-Net added near the end)
- **Old:** EEG, event-related potentials, brain-computer interfaces, neural decoding, neuromorphic
  computing, reservoir computing, spiking neural networks, graph signal processing, structure-function
  coupling, affective computing, closed-loop evidence accumulation, adaptive BCI simulation.
- **New:** affective computing, EEG, event-related potentials, electrophysiological signal analysis,
  affective EEG dynamics, neuromorphic computing, reservoir computing, spiking neural networks, graph
  signal processing, structure-function coupling, **ARSPI-Net**.

## Files changed
- `manuscript/taffc/main_taffc.tex` — title, first abstract definition + ERP sentence, keywords.
- `manuscript/taffc/main_taffc.pdf` — recompiled.
- `manuscript/taffc/main_taffc_blind.tex` — title, first abstract definition + ERP sentence, keywords.
- `manuscript/taffc/main_taffc_blind.pdf` — recompiled.
- `manuscript/COVER_LETTER_TAFFC_FINAL.md` — quoted manuscript title only (ARSPI-Net retained in body).
- `SUBMISSION_PROFILE_TAFFC.md` — Title field only.
- `TAFFC_TITLE_KEYWORD_REFINEMENT_REPORT.md` — this report (new).

(`TAFFC_SCOPE_ALIGNMENT.md`, `TAFFC_AUTHOR_GUIDELINES_VERIFICATION.md`, `TAFFC_ACCEPTANCE_READINESS_REPORT.md`,
and `TAFFC_REVIEWER_OBJECTION_AUDIT.md` do not repeat the full manuscript title, so they were not changed.)

## Page count after compilation
- `main_taffc.pdf`: **11 pages** — within the verified 12-page TAFFC free formatted limit.
- `main_taffc_blind.pdf`: **11 pages** — within the limit.

## TCDS files unchanged (confirmed)
`manuscript/main.tex`, `manuscript/main.pdf`, `manuscript/main_blind.tex`, `manuscript/main_blind.pdf`,
`manuscript/supplemental_risk_reduction/supplement.pdf`, `manuscript/COVER_LETTER_TCDS_COGNEURO_BCI_FINAL.md`,
`SUBMISSION_PROFILE.md`, `CFP_ALIGNMENT_COGNEURO_BCI.md`, `READY_FOR_COGNEURO_BCI_SUBMISSION_CHECKLIST.md`
are all **byte-identical to `origin/main`**.

## No numerical results / figures / tables / experiment outputs changed (confirmed)
Numeric-token comparison of `main_taffc.pdf` (before vs after): the multiset of all decimals,
percentages, counts, confidence intervals, and years is **identical**. All figures and tables are
byte-identical to `origin/main`; shared figures were not regenerated. No forbidden claims were
introduced (the only "real-time" string is the Maass et al. citation title in the bibliography); the
AI / session-reference scan is clean across all changed files.
