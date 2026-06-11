# Submission Profile — ARSPI-Net (TCDS Cognitive Neuroscience + BCI Special Issue)

- **Target journal:** IEEE Transactions on Cognitive and Developmental Systems (TCDS)
- **Special issue:** Cognitive Neuroscience Meets Brain-Computer Interfaces: From Neural Decoding to Function Augmentation
- **ScholarOne manuscript type:** SI: Cognitive Neuroscience Meets Brain-Computer Interfaces: From Neural Decoding to Function Augmentation
- **Submission deadline:** July 31, 2026
- **Manuscript title:** ARSPI-Net: A Neuromorphic Reservoir-Graph Substrate for Affective ERP Decoding and Closed-Loop Neural Evidence Accumulation

## Primary claim
ARSPI-Net is a neuromorphic signal-processing substrate for affective ERP decoding. It transforms
each subject-condition ERP observation into spike-coded (E), dynamical (D), graph-topological (T),
and structure-function coupling (C) evidence streams, evaluated through subject-grouped validation,
mechanism ablation with negative controls, perturbation response, representational redundancy, and a
closed-loop adaptive BCI simulation that accumulates neural evidence over recorded ERP observations.

## Scope (claimed / not claimed)
- **Not** a static-classifier leaderboard result: a conventional ERP-amplitude baseline remains
  stronger for static three-class classification; ARSPI-Net is positioned as a measurement substrate.
- The closed-loop study is a **simulation over recorded ERP observations**; it is **not** an online
  BCI deployment and does not acquire EEG in real time.
- Clinical-label analyses are **exploratory** and FDR-bounded; **not** diagnostic biomarker validation.
- No claim of cognitive enhancement, neurofeedback training, functional restoration, or
  hardware-energy efficiency.

## Initial-submission upload set (PDFs + cover letter; LaTeX source only if requested)
- **Manuscript PDF:** `manuscript/main_blind.pdf` if the special issue requires double-anonymous
  review; otherwise `manuscript/main.pdf`. Both are compiled and committed. Confirm the SI
  review-anonymity policy before choosing.
- **Supplement PDF:** `manuscript/supplemental_risk_reduction/supplement.pdf`
- **Cover letter:** `manuscript/COVER_LETTER_TCDS_COGNEURO_BCI_FINAL.md`
- **LaTeX source bundle** (`manuscript/submission_package_blind/`) is retargeted and kept ready only
  if ScholarOne requests source files.

## Repository state
- **Branch:** `submission/tcds-cogneuro-bci-2026` (pull request into `main`)
- **Preservation tag:** `v1.0-tcds-embodied-control-ready` (created on the pre-retargeting commit)
- **Compile status:** `main.pdf`, `main_blind.pdf`, `supplement.pdf`, and the blind source bundle
  compile cleanly with zero undefined references.
- **Privacy / anonymization:** raw EEG, clinical metadata, and subject-level features are
  access-controlled and not committed. Blind manuscripts omit the dataset name and institution;
  non-blind materials may name the cohort. Only aggregate, deidentified outputs are included.
