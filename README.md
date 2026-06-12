# ARSPI-Net: a neuromorphic reservoir–graph substrate for affective ERP decoding

This repository holds the manuscript, analysis pipelines, figures, tables, and aggregate
results for:

**ARSPI-Net: A Neuromorphic Reservoir-Graph Substrate for Affective ERP Decoding and
Closed-Loop Neural Evidence Accumulation**

## What ARSPI-Net is

ARSPI-Net treats an affective event-related potential (ERP) as the noisy, partially observed
output of a biological dynamical system and transforms each subject-condition observation into
four operationally distinct **neural evidence streams**:

- **E** — a leaky integrate-and-fire (LIF) spiking-reservoir, spike-coded embedding
- **D** — reservoir dynamical descriptors
- **T** — temporal phase-locking (tPLV) graph-topological descriptors
- **C** — a structure–function coupling readout (κ)

These streams are evaluated through a mechanism ablation with negative controls, a
perturbation-robustness analysis, and a **closed-loop adaptive BCI simulation over recorded ERP
observations**. ARSPI-Net is positioned as a *measurement substrate*, not a static classifier: a
conventional ERP-amplitude baseline remains stronger for static endpoint classification, and the
closed-loop study is a simulation over recorded observations, not an online BCI deployment.

## Submission

Targeted at the **IEEE Transactions on Cognitive and Developmental Systems (TCDS)** special issue
*"Cognitive Neuroscience Meets Brain-Computer Interfaces: From Neural Decoding to Function
Augmentation"* (single-blind review).

| Document | Purpose |
|---|---|
| [`SUBMISSION_PROFILE.md`](SUBMISSION_PROFILE.md) | venue, manuscript type, title, scope, and upload package |
| [`CFP_ALIGNMENT_COGNEURO_BCI.md`](CFP_ALIGNMENT_COGNEURO_BCI.md) | mapping of the paper to the call topics |
| [`READY_FOR_COGNEURO_BCI_SUBMISSION_CHECKLIST.md`](READY_FOR_COGNEURO_BCI_SUBMISSION_CHECKLIST.md) | pre-submission checklist |

**Default single-blind upload package:** `manuscript/main.pdf`,
`manuscript/supplemental_risk_reduction/supplement.pdf`, and
`manuscript/COVER_LETTER_TCDS_COGNEURO_BCI_FINAL.md`. The anonymized `main_blind.pdf` and the
self-contained LaTeX bundle are kept only in case anonymized files or source are requested.

## Repository layout

```
manuscript/                        LaTeX sources, compiled PDFs, references, cover letter
  main.tex / main.pdf                submission manuscript (non-blind; default upload)
  main_blind.tex / main_blind.pdf    anonymized backup
  supplemental_risk_reduction/       supplement.tex / supplement.pdf
  submission_package_blind/          self-contained anonymized LaTeX bundle
  figures/  tables/                  figures and tables included by the manuscript
  data_public/                       aggregate, de-identified result tables behind the figures
  scripts/                           figure-generation helpers
  references.bib                     bibliography
  COVER_LETTER_TCDS_COGNEURO_BCI_FINAL.md

experiments/                       analysis pipelines (read access-controlled feature inputs)
  operational_distinctness/          original layer-ablation study
  tcds_ready9/                       primary pipeline: QC/provenance, mechanism ablation,
                                     perturbation robustness, closed-loop simulation, figures/tables
  tcds_risk_reduction/               adaptive evidence routing; resource/event-rate accounting
  tcds_hardening/                    ERP baseline, κ coupling, reservoir-dynamics figures

prepare_inputs/                    feature-extraction scripts (reservoir embedding; dynamical/topological features)
figures/  tables/  outputs/        generated figures, LaTeX tables, and aggregate CSV/JSON, grouped by analysis
data/                              access-controlled input data location (see Data availability)
prior_paper/                       the earlier layer-ablation manuscript this work builds on
```

## Data availability

Raw EEG and clinical metadata are restricted human-subject research data, governed by a data-use
agreement and available from the Laboratory for Clinical Affective Neuroscience at Stony Brook
University subject to approval. Subject-level inputs under `data/` are covered by that agreement
and should be treated as access-controlled material rather than an open-data release; the figures,
tables, and aggregate result files elsewhere in the repository are de-identified.

## Reproducing the analysis

The pipelines read the restricted feature inputs and emit the figures, tables, and result files.
Input locations resolve from environment variables (`ARSPI_SHAPE_FEATURES`, `ARSPI_CH67_FEATURES`,
`ARSPI_CLINICAL_FILE`, `ARSPI_RAW_EEG_DIR`); see `experiments/tcds_ready9/config.py`.

```
conda env create -f environment.yml        # or: pip install -r requirements.txt
python experiments/tcds_ready9/run_ready9.py
```

## Building the manuscript

The manuscript compiles with `pdflatex` (IEEEtran document class, inline bibliography — no bibtex step):

```
cd manuscript && pdflatex main.tex && pdflatex main.tex
```

## Earlier ARSPI-Net work

- A. Lane, W. Tang, and B. Nelson, "Towards ARSPI-Net: Development of an efficient hybrid deep
  learning framework," IEEE Long Island Systems, Applications and Technology Conference (LISAT), 2023.
- A. Lane, W. Tang, and B. Nelson, "Towards ARSPI-Net: Advancing EEG feature extraction with
  neuromorphic algorithms," IEEE LISAT, 2024.
