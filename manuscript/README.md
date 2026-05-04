# Operational Distinctness Overleaf Package v5

This package corrects the earlier underbuilt Overleaf draft. It includes:

- **Source figures copied from the dissertation bundle**:
  - `figures/source_from_dissertation/dissertation_fig_pipeline_overview.pdf`
  - `figures/source_from_dissertation/dissertation_fig_bsc6_real_eeg.pdf`
- **Novel figures generated specifically for this paper from corrected reporting outputs**:
  - `figures/generated/fig03_operational_distinctness_formalism.pdf`
  - `figures/generated/fig04_affective_ablation_dual_metric.pdf`
  - `figures/generated/fig05_embedding_additive_utility.pdf`
  - `figures/generated/fig06_clinical_sensitivity_heatmap.pdf`
  - `figures/generated/fig07_clinical_best_layer_dual_metric.pdf`
  - `figures/generated/fig08_layer_redundancy_cka.pdf`
- **Figure-generation code**:
  - `scripts/generate_novel_figures.py`
- **Corrected public reporting data**:
  - `data_public/affective_ablation_corrected.csv`
  - `data_public/clinical_sensitivity_corrected.csv`
  - `data_public/layer_redundancy_corrected.csv`
  - `data_public/comorbidity_selected.csv`
- **LaTeX tables** in `tables/`
- **Compiled preview**: `compiled_preview.pdf`

## Important scientific status

This draft uses corrected results from PR head commit:

`bcafc50ca544da35ee215f030849a34bfb395a4c`

The prior zero-valued E embedding artifact is not used. The clinical-label findings remain exploratory because permutation-FDR inference was not completed.

## Build instructions

Upload this ZIP to Overleaf and compile `main.tex` with pdfLaTeX. The manuscript uses a manual `thebibliography` section, so BibTeX is not required.
