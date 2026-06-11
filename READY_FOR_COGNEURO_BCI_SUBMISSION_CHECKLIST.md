# Submission Readiness Checklist — TCDS Cognitive Neuroscience + BCI Special Issue

Branch: `submission/tcds-cogneuro-bci-2026` · Preservation tag (pre-retargeting): `v1.0-tcds-embodied-control-ready`

## Compiled artifacts (pdflatex, inline bibliography)
- [x] Main manuscript: `manuscript/main.pdf` — **10 pages**, 0 undefined references.
- [x] Blind manuscript: `manuscript/main_blind.pdf` — **10 pages**, 0 undefined references.
- [x] Supplement: `manuscript/supplemental_risk_reduction/supplement.pdf` — **6 pages**, 0 undefined references.
- [x] Blind source bundle: `manuscript/submission_package_blind/main_blind.tex` → `main_blind_reference.pdf` — **10 pages**, 0 undefined references.
- [x] Figures render: pipeline overview and closed-loop figures recompiled from TikZ.
- [x] Page counts within IEEE TCDS expectations (10-page main text).

## Title / framing consistency
- [x] Title updated in main, blind, blind bundle, supplement, README, cover letter, submission profile.
- [x] Cover-letter title matches manuscript title.
- [x] Supplement title matches manuscript title.
- [x] Manuscript no longer relies on "embodied AI"; closed loop framed as adaptive BCI simulation over recorded ERP observations.

## Compile health
- [x] Zero undefined references / citations (verified from each `.log`).
- [x] No readability-breaking table/figure overflow (no overfull `\hbox` > 20pt).
- [x] Figures render (pipeline + closed-loop).

## Language / claim scans
- [x] U.S. English retained.
- [x] No state-of-the-art / breakthrough / groundbreaking / revolutionary / unlocks / powerful (except the Benjamini–Hochberg citation title).
- [x] No online-BCI-deployment, neurofeedback, cognitive-enhancement, functional-restoration, or hardware-energy claims (only as explicitly not-claimed scope).
- [x] "real-time" appears only in not-demonstrated sentences (plus the Maass 2002 citation title).
- [x] "function augmentation" absent from the manuscript; appears only inside the exact SI title in documentation.
- [x] No diagnostic-biomarker claim (negation only); documentation uses "clinical translation relevance".
- [x] Presence scan: affective ERP decoding, neural evidence, closed-loop, adaptive BCI, perturbation, reservoir, graph, structure-function coupling, subject-grouped, recorded ERP observations, "not an online BCI deployment".

## Privacy / anonymization
- [x] No raw EEG, feature pickles, or clinical metadata committed (gitignored).
- [x] No subject identifiers, local paths, or emails in submission-facing files.
- [x] Blind manuscripts omit dataset name (SHAPE) and institution; cover letter uses "an access-controlled affective ERP cohort".

## Numerical integrity
- [x] All scientific numbers unchanged: numeric-token multiset identical pre/post across the three manuscripts, the supplement, and the edited tables.

## ScholarOne
- [x] Manuscript type: "SI: Cognitive Neuroscience Meets Brain-Computer Interfaces: From Neural Decoding to Function Augmentation".
- [x] Correct title.
- [x] Upload set (initial submission): manuscript PDF (`main_blind.pdf` if double-anonymous review is required, else `main.pdf`), `supplement.pdf`, cover letter. LaTeX source bundle only if requested.
- [ ] Confirm the SI review-anonymity policy (single- vs double-anonymous) before choosing the manuscript PDF.
