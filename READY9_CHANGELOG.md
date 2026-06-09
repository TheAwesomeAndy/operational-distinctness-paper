# Changelog — experimental-hardening package

This package adds an experimental-hardening layer for the architecture/substrate
manuscript (IEEE TCDS special issue, Brain-Inspired Computing for Embodied AI).
It consumes private inputs from a restricted local environment and emits only
aggregate, privacy-preserving artifacts.

## Experiments added (`experiments/tcds_ready9/`)
- `preflight_audit.py` — repo/upstream/input audit (privacy-safe).
- `00_prepare_features.py` — input resolution, local feature regeneration (never
  committed), and feature/clinical alignment verification (core).
- `01_dataset_provenance_qc.py` — provenance table, QC summary, integrity figure.
- `06_generate_observation_plots.py` — observation-first figures obs00–obs10.
- `08_signal_robustness.py` — representation-level (all 10 configs) + bounded
  raw-signal perturbation robustness (core).
- `09_mechanism_ablation.py` — A0–A9 ablation, negative controls, exploratory
  clinical-label sensitivity (core).
- `10_closed_loop_policy_hardening.py` — simulated affective-control loop with a
  mathematically-defined expected-free-energy controller (core).
- `12_graph_runtime_support.py` — bounded graph observables + runtime/resource.
- `07_generate_analysis_plots.py` — analysis figures ana01–ana10.
- `11_generate_ready9_tables.py` — LaTeX tables (core).
- `run_ready9.py` — orchestrator + figure/run manifests.
- `config.py`, `common_ready9.py` — configuration and shared utilities (reuse
  `operational_distinctness` loaders/feature builders).

## Outputs generated (`outputs/tcds_ready9/`)
- Audits/manifests: `preflight_audit.json`, `feature_alignment_report.json`,
  `dataset_qc_summary.csv`, `dataset_provenance_metadata.json`, `run_manifest.json`,
  `FIGURE_MANIFEST.{csv,json}`, `MANUSCRIPT_FIGURE_PLAN.md`.
- Analysis CSV/JSON: `analysis/mechanism_ablation_*`, `analysis/robustness_*`,
  `analysis/closed_loop_policy_*`, `analysis/clinical_label_sensitivity.csv`,
  `analysis/graph_support_metrics.csv`, `analysis/pipeline_runtime_resource_summary.csv`.
- Governance: `BASE_BRANCH_DIFF_AUDIT.md`, `DATA_GOVERNANCE_REMEDIATION_NOTE.md`,
  `ACCEPTED_PAPER_COMPETENCY_MATRIX.md` (internal).
- Figures under `figures/tcds_ready9/{observations,analysis}/`; tables under
  `tables/tcds_ready9/`.

## Manuscript sections requiring update
- Title, abstract, introduction, contributions, methods framing,
  mechanism-to-implementation table, results structure, limitations, and
  restricted-data availability language.

## Claims strengthened
- Operational distinctness of E/D/T/C via ablation + functional-role evidence.
- Perturbation robustness across four families with disclosed scope.
- Simulated embodied control with a defined controller and oracle bound.
- Provenance/QC and reproducibility manifests with input fingerprints.

## Claims deliberately bounded
- Clinical labels are exploratory validation/context only — not diagnostic
  biomarker evidence.
- Graph evidence is bounded support for the reservoir–graph substrate, not a
  graph-spectral/diffusion study.
- No physical-robot embodiment; no measured hardware energy.
- Results are restricted to the measured SHAPE ERP regime.

## Known remaining risks
- The `submission` profile is compute-heavy; pilot runs validate the pipeline and
  the profile is recorded per artifact.
- Feature regeneration is slow on modest hardware; the pickle remains local/private.

## Deferred dissertation material (reserved for later papers)
- Reservoir operating-regime, graph spectral/diffusion, clinical-label
  sensitivity, structure–function coupling, interpretability/audit, and
  neuromorphic-resource studies.
