# Operational Distinctness in Staged Neuromorphic Architectures

Support scripts for the manuscript

> **Operational Distinctness in Staged Neuromorphic Architectures: A Layer
> Ablation Analysis of Clinical EEG**

This repository **does not** run the ARSPI-Net feature extraction. It
consumes the feature pickles produced by the main
[`dissoAdventureExperiments`](../dissoAdventureExperiments) repository
and emits paper-ready metrics, figures, and LaTeX tables.

No new science is introduced. No new architectures. No new feature
definitions.

---

## Layout

```
experiments/operational_distinctness/
├── config.py                              # paths + constants
├── common.py                              # data loading, feature blocks, helpers
├── 01_affective_ablation_metrics.py       # A0–A9 metrics, predictions, confusion matrices
├── 02_clinical_sensitivity_metrics.py     # C1–C6 clinical-label sensitivity
├── 03_generate_submission_figures.py      # fig1–fig5 PDFs
├── 04_generate_submission_tables.py       # table1–table4 LaTeX
├── 05_optional_permutation_fdr.py         # permutation tests + BH-FDR
├── 06_optional_comorbidity_adjustment.py  # comorbidity-adjusted layer scores
├── 07_optional_layer_redundancy.py        # CKA, CCA, ridge cross-predictability
└── run_all.py                             # orchestrator + run_manifest.json

outputs/operational_distinctness/    # CSVs + JSONs (manifest, predictions, metrics)
figures/operational_distinctness/    # fig1–fig5 PDFs
tables/operational_distinctness/     # table1–table4 .tex
data/                                # (gitignored) optional drop point for input pickles
```

---

## Inputs

Three files, produced by the main feature pipeline:

| File | Source script | Notes |
| --- | --- | --- |
| `shape_features_211.pkl`            | `chapter5Experiments/run_chapter5_experiments.py`            | E + BandPower features |
| `ch6_ch7_3class_features.pkl`        | `experiments/ch6_ch7_3class/ch6_ch7_01_feature_extraction.py` | D + T + per-observation arrays for C |
| `clinical_profile.csv`               | source dataset                                                | binary diagnosis indicators |

`config.py` resolves these in this order:

1. The env vars `ARSPI_CH5_FILE`, `ARSPI_CH67_FILE`, `ARSPI_CLINICAL_FILE`.
2. The local `data/` directory inside this repo.
3. The sibling `dissoAdventureExperiments/` checkout, if it lives next
   to this repo (default on the author's machine).

---

## How to run

```bash
# Required pipeline (scripts 01–04 + manifest):
python experiments/operational_distinctness/run_all.py

# Required + optional hardening (scripts 05–07):
python experiments/operational_distinctness/run_all.py --include-optional
```

Or run individual stages:

```bash
python experiments/operational_distinctness/01_affective_ablation_metrics.py
python experiments/operational_distinctness/02_clinical_sensitivity_metrics.py
python experiments/operational_distinctness/03_generate_submission_figures.py
python experiments/operational_distinctness/04_generate_submission_tables.py
```

---

## Outputs (paper-ready)

| File | Produced by |
| --- | --- |
| `outputs/.../affective_ablation_metrics.csv`         | 01 |
| `outputs/.../affective_predictions.csv`              | 01 |
| `outputs/.../affective_confusion_matrices.json`      | 01 |
| `outputs/.../clinical_sensitivity_metrics.csv`       | 02 |
| `outputs/.../clinical_predictions.csv`               | 02 |
| `outputs/.../clinical_confusion_matrices.json`       | 02 |
| `figures/.../fig1_arspinet_layer_decomposition.pdf`  | 03 |
| `figures/.../fig2_operational_distinctness_framework.pdf` | 03 |
| `figures/.../fig3_affective_ablation_metrics.pdf`    | 03 |
| `figures/.../fig4_clinical_sensitivity_heatmap.pdf`  | 03 |
| `figures/.../fig5_clinical_best_layer_summary.pdf`   | 03 |
| `tables/.../table1_feature_blocks.tex`               | 04 |
| `tables/.../table2_affective_ablation.tex`           | 04 |
| `tables/.../table3_clinical_sensitivity.tex`         | 04 |
| `tables/.../table4_full_clinical_matrix.tex`         | 04 |
| `outputs/.../affective_inference.csv`                | 05 (optional) |
| `outputs/.../clinical_inference.csv`                 | 05 (optional) |
| `outputs/.../comorbidity_adjusted.csv`               | 06 (optional) |
| `outputs/.../layer_redundancy.csv`                   | 07 (optional) |
| `outputs/.../run_manifest.json`                      | run_all.py |

---

## PHI / privacy

Subject IDs are replaced with the first 16 hex chars of
`SHA-256(str(subject_id))` in every committed artifact. Raw subject
IDs, names, DOB, MRN, addresses, clinical notes, and session dates are
**never** written to any output file.

The `data/` directory is gitignored. Do **not** commit the input
pickles or `clinical_profile.csv`.

---

## Scientific framing

The clinical analysis is *clinical-label sensitivity*, not diagnostic
validation. Use:

- weak clinical-label sensitivity
- exploratory clinical structure
- layer-specific sensitivity pattern

Avoid:

- diagnostic biomarker
- clinical detection
- proves disorder-specific phenotype
