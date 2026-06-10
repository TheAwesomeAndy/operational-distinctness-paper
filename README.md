# ARSPI-Net: an event-driven reservoir–graph substrate for affective EEG

This repository holds the manuscript, analysis code, figures, tables, and aggregate
results for the paper:

**ARSPI-Net: An Event-Driven Reservoir-Graph Substrate for Affective EEG Perception
in Simulated Embodied Control**

ARSPI-Net is a staged neuromorphic pipeline that converts affective event-related
EEG into four evidence streams — a spiking-reservoir embedding (E), dynamical
descriptors (D), a temporal phase-locking graph (T), and a structure–function
coupling readout (C) — and evaluates them through a mechanism ablation, a
perturbation-robustness analysis, and a simulated embodied perceptual-control loop.

## Layout

```
manuscript/                 LaTeX source, compiled PDFs, references, cover letter
  main.tex / main.pdf         submission manuscript (non-blind)
  main_blind.tex / .pdf       anonymized version (for double-blind venues)
  supplemental_risk_reduction/  supplementary material (supplement.tex / .pdf)
  submission_package_blind/   self-contained anonymized source bundle
  COVER_LETTER_*.md           cover letter
  references.bib              bibliography
  data_public/                aggregate, de-identified result tables for the figures
  scripts/                    figure-generation helpers

experiments/                analysis pipelines (run on the restricted feature inputs)
  operational_distinctness/   layer-ablation analysis (the original study)
  tcds_ready9/                primary pipeline: provenance and QC, mechanism
                              ablation, perturbation robustness, closed-loop
                              control, and figure/table generation
  tcds_risk_reduction/        adaptive evidence routing and resource/event-rate
                              accounting
  tcds_hardening/             supporting analyses: ERP baseline, kappa coupling,
                              reservoir-dynamics and summary figures

prepare_inputs/             feature-extraction scripts (reservoir embedding and the
                            dynamical / topological features from the recordings)

figures/  tables/  outputs/  generated figures, LaTeX tables, and aggregate result
                             files (CSV / JSON), grouped by analysis

prior_paper/                the earlier layer-ablation manuscript this work builds on
```

## Data availability

The raw EEG and clinical metadata are restricted human-subject research data and
are not publicly distributed. They are available from the Laboratory for Clinical
Affective Neuroscience at Stony Brook University subject to approval and a data-use
agreement. Only aggregate, de-identified outputs are included in this repository.

## Reproducing the analysis

The pipelines read the restricted feature inputs and emit the figures, tables, and
result files. Input locations are resolved from environment variables
(`ARSPI_SHAPE_FEATURES`, `ARSPI_CH67_FEATURES`, `ARSPI_CLINICAL_FILE`,
`ARSPI_RAW_EEG_DIR`); see `experiments/tcds_ready9/config.py`.

```
conda env create -f environment.yml        # or: pip install -r requirements.txt
python experiments/tcds_ready9/run_ready9.py
```

The manuscript compiles with `pdflatex` (IEEEtran document class, inline
bibliography):

```
cd manuscript && pdflatex main.tex && pdflatex main.tex
```

## Earlier ARSPI-Net work

- A. Lane, W. Tang, and B. Nelson, "Towards ARSPI-Net: Development of an efficient
  hybrid deep learning framework," IEEE Long Island Systems, Applications and
  Technology Conference (LISAT), 2023.
- A. Lane, W. Tang, and B. Nelson, "Towards ARSPI-Net: Advancing EEG feature
  extraction with neuromorphic algorithms," IEEE LISAT, 2024.
