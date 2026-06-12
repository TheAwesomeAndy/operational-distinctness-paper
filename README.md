# ARSPI-Net: affective ERP evidence streams under a neuromorphic reservoir–graph framework

This repository holds the manuscript sources, analysis pipelines, figures, tables, and aggregate
results for a study of **ARSPI-Net** (Affective Reservoir-Spike Processing and Inference Network) —
a fixed, event-driven neuromorphic reservoir–graph framework that transforms an affective
event-related potential (ERP) into operationally distinct neural evidence streams and characterizes
how those streams behave under perturbation and closed-loop evidence accumulation.

The same study and the same numerical results are currently prepared as **two venue-specific
manuscript framings**. No metric, table, figure, or result differs between them — only the title,
abstract framing, and keywords.

| Manuscript | Path | Pages | Title | Role |
|---|---|---|---|---|
| **TAFFC variant** | `manuscript/taffc/main_taffc.pdf` (blind twin `main_taffc_blind.pdf`) | 11 | *Spatiotemporal Characterization of Affective EEG Dynamics Using a Neuromorphic Reservoir–Graph Framework* | **current / primary submission target** |
| **TCDS manuscript** | `manuscript/main.pdf` (blind twin `main_blind.pdf`) | 10 | *ARSPI-Net: A Neuromorphic Reservoir-Graph Substrate for Affective ERP Decoding and Closed-Loop Neural Evidence Accumulation* | **preserved fallback** |

Both share the same 6-page supplement (`manuscript/supplemental_risk_reduction/supplement.pdf`),
the same figures, tables, and analysis code, and the same access-controlled dataset. ARSPI-Net is
the named framework and a keyword in both. `main` is the canonical branch.

## What ARSPI-Net is

ARSPI-Net treats an affective ERP as the noisy, partially observed output of a biological dynamical
system and transforms each trial-averaged subject-condition observation into four operationally
distinct **neural evidence streams**:

- **E** — a leaky integrate-and-fire (LIF) spiking-reservoir, spike-coded embedding
- **D** — reservoir dynamical descriptors
- **T** — temporal phase-locking (tPLV) graph-topological descriptors
- **C** — a structure–function coupling readout (κ)

These streams are evaluated through a mechanism ablation with negative controls, a
representational-redundancy (CKA) analysis, a perturbation-robustness analysis (temporal, amplitude,
channel, and graph corruption), and a **closed-loop evidence-accumulation simulation over recorded
ERP observations**.

**Claim discipline (held across both framings).** ARSPI-Net is positioned as a *measurement
substrate*, not a static classifier — a conventional ERP-amplitude baseline remains stronger for
static endpoint classification. The closed-loop study is an offline simulation over recorded
observations, **not** an online BCI deployment, neurofeedback, or a human-in-the-loop system.
Clinical labels are used only as exploratory contextual variables bounded by a false-discovery-rate
result, **not** as diagnostic validation.

## Submission status

The work is submitted on a **TAFFC-first cascade** (see [`SUBMISSION_CASCADE.md`](SUBMISSION_CASCADE.md));
on rejection it is reframed and reformatted to the next venue rather than re-run — **no new
experiments** are required at any step:

1. **IEEE Transactions on Affective Computing (TAFFC)** — current target
2. IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE)
3. IEEE Journal of Biomedical and Health Informatics (JBHI)
4. IEEE Transactions on Biomedical Engineering (TBME)
5. IEEE Transactions on Cognitive and Developmental Systems (TCDS) — preserved fallback

### TAFFC package (current target)

Upload set: `manuscript/taffc/main_taffc.pdf` (single-blind) **or** `main_taffc_blind.pdf`
(double-blind), plus the shared `supplement.pdf` and the TAFFC cover letter.

| Document | Purpose |
|---|---|
| [`manuscript/COVER_LETTER_TAFFC_FINAL.md`](manuscript/COVER_LETTER_TAFFC_FINAL.md) | TAFFC cover letter (affective-computing framing; prior-work disclosure) |
| [`SUBMISSION_PROFILE_TAFFC.md`](SUBMISSION_PROFILE_TAFFC.md) | venue, title, upload set, editor, review model, open-access election |
| [`TAFFC_AUTHOR_GUIDELINES_VERIFICATION.md`](TAFFC_AUTHOR_GUIDELINES_VERIFICATION.md) | verified author guidelines (12-page free limit, APC, portal) |
| [`TAFFC_SCOPE_ALIGNMENT.md`](TAFFC_SCOPE_ALIGNMENT.md) | mapping of the paper to TAFFC scope |
| [`TAFFC_REVIEWER_OBJECTION_AUDIT.md`](TAFFC_REVIEWER_OBJECTION_AUDIT.md) | likely reviewer objections and where the manuscript answers them |
| [`TAFFC_ACCEPTANCE_READINESS_REPORT.md`](TAFFC_ACCEPTANCE_READINESS_REPORT.md) | acceptance-readiness record |
| [`TAFFC_TITLE_KEYWORD_REFINEMENT_REPORT.md`](TAFFC_TITLE_KEYWORD_REFINEMENT_REPORT.md) | title/keyword convergence record |
| [`TAFFC_FINAL_SUBMISSION_AUDIT.md`](TAFFC_FINAL_SUBMISSION_AUDIT.md) | final consistency audit + Fig. 4/6 grayscale-risk assessment |

### TCDS package (preserved fallback)

Upload set: `manuscript/main.pdf`, the shared `supplement.pdf`, and the TCDS cover letter.

| Document | Purpose |
|---|---|
| [`manuscript/COVER_LETTER_TCDS_COGNEURO_BCI_FINAL.md`](manuscript/COVER_LETTER_TCDS_COGNEURO_BCI_FINAL.md) | TCDS cover letter |
| [`SUBMISSION_PROFILE.md`](SUBMISSION_PROFILE.md) | TCDS venue, title, scope, and upload package |
| [`CFP_ALIGNMENT_COGNEURO_BCI.md`](CFP_ALIGNMENT_COGNEURO_BCI.md) | mapping of the paper to the TCDS special-issue call |
| [`READY_FOR_COGNEURO_BCI_SUBMISSION_CHECKLIST.md`](READY_FOR_COGNEURO_BCI_SUBMISSION_CHECKLIST.md) | TCDS pre-submission checklist |

A known, non-blocking polish item is tracked in the final audit: Figures 4 and 6 distinguish their
series by color with identical markers, so they degrade in grayscale; a TAFFC-specific, style-only
figure-polish pass is recommended but not required before submission.

## Repository layout

```
manuscript/
  main.tex / main.pdf                 TCDS manuscript (shared; 10 pp; fallback)
  main_blind.tex / main_blind.pdf     anonymized twin of the TCDS manuscript
  taffc/                              additive TAFFC variant (current target)
    main_taffc.tex / .pdf              TAFFC manuscript (11 pp; EEG-dynamics title)
    main_taffc_blind.tex / .pdf        anonymized twin of the TAFFC manuscript
  supplemental_risk_reduction/        supplement.tex / supplement.pdf (6 pp; shared)
  submission_package_blind/           self-contained anonymized LaTeX bundle
  figures/  tables/                   figures and tables included by the manuscript
  data_public/                        aggregate, de-identified result tables behind the figures
  scripts/                            figure-generation helpers
  references.bib                      bibliography
  COVER_LETTER_TAFFC_FINAL.md         TAFFC cover letter
  COVER_LETTER_TCDS_COGNEURO_BCI_FINAL.md   TCDS cover letter

experiments/                         analysis pipelines (read access-controlled feature inputs)
  operational_distinctness/           original layer-ablation study
  tcds_ready9/                        primary pipeline: QC/provenance, mechanism ablation,
                                      perturbation robustness, closed-loop simulation, figures/tables
  tcds_risk_reduction/                adaptive evidence routing; resource/event-rate accounting
  tcds_hardening/                     ERP baseline, κ coupling, reservoir-dynamics figures

prepare_inputs/                      feature-extraction scripts (reservoir embedding; dynamical/topological features)
figures/  tables/  outputs/          generated figures, LaTeX tables, and aggregate CSV/JSON, grouped by analysis
data/                                access-controlled input data location (see Data availability)
prior_paper/                         the earlier layer-ablation manuscript this work builds on
Dockerfile  environment.yml  requirements.txt    reproducible environment

SUBMISSION_CASCADE.md                ranked venue cascade (TAFFC → TNSRE → JBHI → TBME → TCDS)
SUBMISSION_PROFILE_TAFFC.md, TAFFC_*.md          TAFFC submission package + audits
SUBMISSION_PROFILE.md, CFP_ALIGNMENT_COGNEURO_BCI.md, READY_FOR_*.md   TCDS submission package
```

## Data availability

Raw EEG and clinical metadata are restricted human-subject research data, governed by a data-use
agreement and available from the Laboratory for Clinical Affective Neuroscience at Stony Brook
University subject to approval. Subject-level inputs under `data/` are covered by that agreement and
should be treated as access-controlled material rather than an open-data release. No protected
health information or raw subject identifiers appear in any public artifact; the figures, tables,
and aggregate result files elsewhere in the repository are de-identified.

## Reproducing the analysis

The pipelines read the restricted feature inputs and emit the figures, tables, and aggregate result
files. Input locations resolve from environment variables (`ARSPI_SHAPE_FEATURES`,
`ARSPI_CH67_FEATURES`, `ARSPI_CLINICAL_FILE`, `ARSPI_RAW_EEG_DIR`); see
`experiments/tcds_ready9/config.py`.

```
conda env create -f environment.yml        # or: pip install -r requirements.txt
python experiments/tcds_ready9/run_ready9.py
```

## Building the manuscripts

Both manuscripts use the IEEEtran document class with an inline bibliography (no bibtex step). The
shared figures/tables resolve relative to `manuscript/`, so compile from that directory:

```
# TAFFC variant (current target) -> manuscript/taffc/main_taffc.pdf
cd manuscript && pdflatex -output-directory=taffc taffc/main_taffc.tex \
                && pdflatex -output-directory=taffc taffc/main_taffc.tex

# TCDS manuscript (fallback) -> manuscript/main.pdf
cd manuscript && pdflatex main.tex && pdflatex main.tex
```

## Earlier ARSPI-Net work

- A. Lane, W. Tang, and B. Nelson, "Towards ARSPI-Net: Development of an efficient hybrid deep
  learning framework," IEEE Long Island Systems, Applications and Technology Conference (LISAT), 2023.
- A. Lane, W. Tang, and B. Nelson, "Towards ARSPI-Net: Advancing EEG feature extraction with
  neuromorphic algorithms," IEEE LISAT, 2024.
