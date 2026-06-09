# Submission reproduction map

Every manuscript-facing quantitative claim traces to a script, input, output
CSV/JSON, figure/table, and a manifest entry. Private inputs are consumed from a
restricted local environment and are never committed; all committed outputs are
aggregate or hashed.

Run order (measured signal → reservoir substrate → graph evidence → analysis →
simulated embodied control → interpretation):

```
python experiments/tcds_ready9/run_ready9.py                 # pilot (default)
ARSPI_READY9_PROFILE=submission \
    python experiments/tcds_ready9/run_ready9.py             # submission-grade
```

Manuscript-facing figures/tables come from the `submission` profile; the run
manifest (`outputs/tcds_ready9/run_manifest.json`) records the profile used.

## Claim → artifact map

| # | Claim | Script | Input (private) | Output CSV/JSON | Figure / Table | Committed |
|---|---|---|---|---|---|---|
| 1 | Dataset provenance & QC | `01_dataset_provenance_qc.py` | shape, ch67, clinical | `dataset_qc_summary.csv`, `dataset_provenance_metadata.json` | `obs00`, `table_dataset_provenance.tex` | yes (aggregate) |
| 2 | Observed ERP morphology | `06_generate_observation_plots.py` | shape (X_ds) | `obs01_*.json` | `obs01` | yes |
| 3 | Observed reservoir spike/rate | `06_*`, ch67 `pop_rate_ts` | shape, ch67 | `obs04/obs05 .json` | `obs04`, `obs05` | yes |
| 4 | Observed BSC6 temporal coding | `06_*` | ch67 | `obs06.json` | `obs06` | yes |
| 5 | Observed tPLV graph structure | `06_*`, `12_*` | ch67 | `graph_support_metrics.csv` | `obs08`, `table_graph_support_observables.tex` | yes |
| 6 | Observed structure-function coupling | `06_*`, `07_*` | ch67 | `obs09.json`, `graph_support_metrics.csv` | `obs09`, `ana06` | yes |
| 7 | Affective ablation (A0–A9) | `09_mechanism_ablation.py` | shape, ch67 | `mechanism_ablation_summary.csv`, `..._metrics.csv`, `..._confusion_matrices.json` | `ana01`, `table_mechanism_ablation.tex` | yes |
| 8 | Exploratory clinical-label sensitivity | `09_*` | shape, ch67, clinical | `clinical_label_sensitivity.csv` | `ana05` | yes |
| 9 | Operational distinction of E/D/T/C | `09_*`, `07_*` | shape, ch67 | `mechanism_ablation_summary.csv` | `ana02` | yes |
| 10 | Signal perturbation robustness | `08_signal_robustness.py` | shape, ch67 | `robustness_metrics.csv`, `robustness_summary.csv`, `robustness_config.json` | `ana03`, `ana04`, `obs03`, `table_robustness_summary.tex` | yes |
| 11 | Graph perturbation robustness | `08_*`, `12_*` | ch67 | `robustness_summary.csv`, `graph_support_metrics.csv` | `ana03`, `ana11` | yes |
| 12 | Simulated closed-loop policy hardening | `10_closed_loop_policy_hardening.py` | shape, ch67 | `closed_loop_policy_summary.csv`, `..._metrics.csv`, `..._config.json` | `ana07/08/09`, `obs10`, `table_closed_loop_policy.tex` | yes |
| 13 | Evaluation-coverage positioning | `11_*`, `07_*` | — | (static) | `ana10`, `table_evaluation_coverage.tex` | yes |
| 14 | Runtime/resource support | `12_graph_runtime_support.py` | shape, ch67 | `pipeline_runtime_resource_summary.csv` | `ana12`, `table_pipeline_runtime_resource.tex` | yes |
| 15 | Privacy-safe output generation | all + `run_ready9.py` | — | `run_manifest.json`, `FIGURE_MANIFEST.{csv,json}` | — | yes |

Private inputs required for claims 1–12, 14: `shape_features_211.pkl`,
`ch6_ch7_3class_features.pkl` (regenerable locally from raw EEG via
`prepare_inputs/extract_ch67_features.py`), `clinical_profile.csv`. None are
committed.

## Per-figure measured object

| Figure | Measured object | Script | Privacy |
|---|---|---|---|
| obs00 | dataset integrity QC | `01_*` | aggregate |
| obs01 | grand-average ERP by condition | `06_*` | aggregate |
| obs02 | channel×time amplitude contrast | `06_*` | aggregate |
| obs03 | clean vs perturbed example | `06_*` | hashed/representative |
| obs04 | LIF reservoir spike raster | `06_*` | representative |
| obs05 | population-rate traces | `06_*` | aggregate |
| obs06 | BSC6 temporal-bin profiles | `06_*` | aggregate |
| obs07 | feature-block distributions | `06_*` | aggregate |
| obs08 | tPLV adjacency by condition | `06_*` | aggregate |
| obs09 | κ across observations + null | `06_*` | aggregate |
| obs10 | closed-loop belief trajectories | `06_*` | episode-level |

## Per-analysis figure

| Figure | Analysis question | Script | Output | Limitation |
|---|---|---|---|---|
| ana01 | Do blocks differ in predictive sufficiency? | `07_*` | `mechanism_ablation_summary.csv` | SHAPE ERP regime |
| ana02 | What role does each block emphasise? | `07_*` | same | qualitative |
| ana03 | How does performance degrade under perturbation? | `07_*` | `robustness_summary.csv` | representation-level |
| ana04 | Which stream degrades fastest? | `07_*` | same | amplitude-noise slope |
| ana05 | Exploratory clinical-label sensitivity | `07_*` | `clinical_label_sensitivity.csv` | not diagnostic |
| ana06 | Is κ above an electrode-shuffle null? | `07_*` | recomputed | bounded null size |
| ana07 | Does the controller reach the target? | `07_*` | `closed_loop_policy_summary.csv` | simulation |
| ana08 | Entropy/steps by policy | `07_*` | same | simulation |
| ana09 | Failure modes by ε | `07_*` | same | simulation |
| ana10 | Evaluation coverage | `07_*` | static | not a ranking |
