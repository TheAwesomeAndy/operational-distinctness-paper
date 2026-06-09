# Scientific-Director Report — ARSPI-Net Substrate Package

**Paper 1 (architecture/substrate), IEEE TCDS special issue "Brain-Inspired Computing for Embodied AI."**

This report is a complete index of the work: what was built, where every artifact
lives, what the results are, what decisions and issues arose, and how to reproduce
everything. All paths are relative to the `operational-distinctness-paper` repository.

---

## 1. Locations at a glance

| Item | Location |
|---|---|
| Repository | `operational-distinctness-paper` (GitHub: `TheAwesomeAndy/operational-distinctness-paper`) |
| Branch | `submission/tcds-ready9-substrate` (created from `main`; zero-diff base, see §7) |
| Pull request | **#6** → `main` : https://github.com/TheAwesomeAndy/operational-distinctness-paper/pull/6 |
| Scripts | `experiments/tcds_ready9/` (15 Python files) |
| Aggregate outputs | `outputs/tcds_ready9/` (+ `outputs/tcds_ready9/analysis/`) |
| Observation figures | `figures/tcds_ready9/observations/` (11 PDFs + JSON sidecars) |
| Analysis figures | `figures/tcds_ready9/analysis/` (12 PDFs + JSON sidecars) |
| LaTeX tables | `tables/tcds_ready9/` (10 `.tex`) |
| Manuscript | `manuscript/main.tex` → compiled `manuscript/main.pdf` (17 pp) |
| Reproduction map | `SUBMISSION_REPRODUCTION_MAP.md` (repo root) |
| Changelog | `READY9_CHANGELOG.md` (repo root) |
| This report | `outputs/tcds_ready9/SCIENTIFIC_DIRECTOR_REPORT.md` |

**Central claim supported.** ARSPI-Net defines a brain-inspired event-driven
reservoir–graph substrate for transforming affective EEG into operationally distinct
neural evidence streams, evaluated through dataset provenance, observation-level
signal objects, mechanism ablation, perturbation robustness, and simulated embodied
affective-control. No diagnostic, hardware-energy, or physical-embodiment claims.

---

## 2. Commit history (branch `submission/tcds-ready9-substrate`)

```
2569c0d  Add experimental-hardening package: scripts, governance, manuscript reframing
c713d05  Quality gates + manuscript language pass
bf3a285  Use class_weight=balanced in shared fold classifier
95affad  Wire hardening figures/tables into manuscript; fix LaTeX table widths
dff9715  Add submission-profile results: figures, tables, manifests, compiled PDF
8c88b92  Auto-regenerate placeholder embedding in feature-prep phase
1aa2ad2  Refresh feature alignment report
(+ cleanup of stale skip/error status files and this report)
```

---

## 3. Scripts (`experiments/tcds_ready9/`)

| File | Phase | Purpose | Core? |
|---|---|---|---|
| `config.py` | 3 | Paths, env overrides, run profiles, seeds, perturbation levels, policy names | — |
| `common_ready9.py` | 3 | Reused loaders/feature builders + fingerprints, JSON/CSV writers, Wilson/bootstrap CIs, metric bundle, subject-grouped CV, fold classifier, channel-dropout masks | — |
| `preflight_audit.py` | 1 | Repo/upstream/input audit; key + shape + label checks; raw-EEG completeness | — |
| `00_prepare_features.py` | 2 | Resolve inputs; **auto-regenerate** placeholder embedding and missing ch6/ch7 pickle locally; verify exact alignment | **yes** |
| `01_dataset_provenance_qc.py` | 4 | Provenance table, QC summary, integrity figure (obs00), provenance metadata | — |
| `06_generate_observation_plots.py` | 5 | Observation figures obs00–obs10 with paired metadata; skip reports if input absent | — |
| `08_signal_robustness.py` | 6 | Representation-level robustness (all 10 configs) + bounded raw-signal pass | **yes** |
| `09_mechanism_ablation.py` | 7 | A0–A9 ablation, negative controls, exploratory clinical-label sensitivity, table | **yes** |
| `10_closed_loop_policy_hardening.py` | 8 | Simulated affective-control loop; **mathematically-defined EFE controller** | **yes** |
| `12_graph_runtime_support.py` | 9 | Bounded graph observables + runtime/resource; no energy claim | — |
| `07_generate_analysis_plots.py` | 10 | Analysis figures ana01–ana12 (incl. evaluation-coverage ana10) | — |
| `11_generate_ready9_tables.py` | 11 | LaTeX tables from CSVs | **yes** |
| `run_ready9.py` | 12–13 | Orchestrator; figure manifest, manuscript figure plan, run manifest; **core-phase failure ⇒ nonzero exit** | — |
| `quality_gates.py` | 16 | Privacy gate (no staged private data / private paths) + language gate (banned terms in manuscript-facing files) | — |
| `__init__.py` | — | Package marker | — |

Upstream dependency (read-only): `prepare_inputs/extract_ch67_features.py` (ch6/ch7
features) and `prepare_inputs/extract_ch5_features.py` (reservoir embedding). The
canonical `dissoAdventureExperiments` repo was **not modified**.

---

## 4. Outputs (`outputs/tcds_ready9/`)

**Audits / manifests / governance**
- `preflight_audit.json` — repo/upstream/input audit (hashed subjects).
- `feature_alignment_report.json` — exact subjects/labels/counts alignment + clinical match.
- `dataset_qc_summary.csv`, `dataset_provenance_metadata.json` — provenance/QC.
- `run_manifest.json` — profile, seeds, episodes, per-phase return codes, input fingerprints, privacy confirmation, `submission_ready`.
- `FIGURE_MANIFEST.csv` / `.json`, `MANUSCRIPT_FIGURE_PLAN.md` — figure catalogue + section plan.
- `BASE_BRANCH_DIFF_AUDIT.md` — base-branch zero-diff audit (§7).
- `DATA_GOVERNANCE_REMEDIATION_NOTE.md` — internal record of pre-existing tracked private files (§6).
- `ACCEPTED_PAPER_COMPETENCY_MATRIX.md` — **internal only**; the manuscript uses the neutral `table_evaluation_coverage.tex` instead.

**Analysis CSV/JSON (`outputs/tcds_ready9/analysis/`)**
- `mechanism_ablation_{summary,metrics}.csv`, `mechanism_ablation_confusion_matrices.json`
- `clinical_label_sensitivity.csv`
- `robustness_{summary,metrics}.csv`, `robustness_config.json`
- `closed_loop_policy_{summary,metrics}.csv`, `closed_loop_policy_config.json`
- `graph_support_metrics.csv`, `pipeline_runtime_resource_summary.csv`, `graph_runtime_provenance.json`

---

## 5. Results (submission profile: seeds 42–46, 1500 episodes/policy/ε)

`run_manifest.json`: `profile=submission`, `core_failed=[]`, `submission_ready=true`, `n_figures=23`.

### 5.1 Mechanism ablation — balanced accuracy (bootstrap 95% CI)
| Cfg | Stream | dim | BA | macro-F1 | above chance |
|---|---|---|---|---|---|
| A0 | BandPower | 170 | 0.485 [0.471, 0.500] | 0.483 | yes |
| A1 | E | 2176 | 0.463 [0.450, 0.476] | 0.460 | yes |
| A2 | D | 238 | 0.432 [0.413, 0.450] | 0.429 | yes |
| A3 | T | 68 | 0.355 [0.343, 0.367] | 0.353 | yes |
| A4 | C | 3 | 0.368 [0.355, 0.382] | 0.352 | yes |
| A5 | D+T | 306 | 0.439 | 0.436 | yes |
| A6 | E+D | 2414 | 0.478 | 0.475 | yes |
| A7 | E+T | 2244 | 0.464 | 0.461 | yes |
| A8 | E+D+T | 2482 | 0.481 [0.468, 0.494] | 0.478 | yes |
| A9 | E+D+T+C | 2485 | 0.481 | 0.478 | yes |
| — | shuffled-label control | 2482 | **0.335 (chance)** | 0.333 | **no ✓** |
| — | shuffled-subject control | 2482 | 0.474 | 0.471 | yes (grouping control) |

Reading: embedding-containing configurations carry most affective signal; standalone
T and C are lower but carry distinct topological/coupling structure (operational
differentiation, not classifier superiority). The shuffled-label control collapses to
chance, validating the protocol.

### 5.2 Simulated closed-loop control — success rate (Wilson 95% CI)
| Policy | ε=0.0 | ε=0.4 | final entropy (ε=0) |
|---|---|---|---|
| oracle (perfect perception, upper bound) | 1.000 | 1.000 | 0.000 |
| passive (belief-blind) | 1.000 | 0.733 | 0.459 |
| pragmatic-only | 0.864 | 0.808 | 0.207 |
| expected-free-energy | 0.862 | 0.805 | 0.210 |
| random | 0.614 | 0.625 | 0.186 |
| epistemic-only | 0.506 | 0.576 | 0.122 |

Reading (honest operating-regime result): the **EFE controller shows no advantage
over pragmatic-only** under an action-determined transition; epistemic value has
limited marginal utility here. The oracle bounds the cost of single-trial perceptual
unreliability; belief-blind control degrades fastest with transition noise.

### 5.3 Exploratory clinical-label sensitivity (validation/context only — NOT diagnostic)
SUD 0.527, MDD 0.524, PTSD 0.507, GAD 0.488, ADHD 0.532 (balanced accuracy). All near
chance; reported as exploratory clinical-label sensitivity, bounded, no biomarker claim.

### 5.4 Bounded graph + runtime support
Mean tPLV graph density ≈ 0.63, single connected component; κ ≈ 0.24–0.26 by condition.
Runtime/resource summary reports per-stage wall-clock and structural counts
(256-neuron reservoir, 34 channels, 2176-dim E) with **no energy claim**.

---

## 6. Data governance & privacy

- **No private data added by this PR.** Verified: the PR diff vs `main` contains no
  raw EEG, feature pickles, clinical CSV, `.npz`, private paths, or raw subject IDs.
- Missing/placeholder inputs were regenerated **locally and kept uncommitted**:
  - `ch6_ch7_3class_features.pkl` (was absent) — regenerated from raw EEG.
  - `lsm_bsc6_pca` embedding (shipped as an all-zero placeholder) — regenerated via
    `prepare_inputs/extract_ch5_features.py`; the tracked pickle was marked
    `assume-unchanged` so the real embedding is never staged.
- Subject identifiers are SHA-256 hashed; fingerprints record file **name + hash**
  only (no paths). Privacy + language quality gates pass.
- **Pre-existing issue flagged, not extended:** raw EEG (636 files),
  `shape_features_211.pkl`, and `clinical_profile.csv` were already tracked on `main`
  before this task. Documented in `DATA_GOVERNANCE_REMEDIATION_NOTE.md`. History
  purging is a separate, owner/steward-authorized action and is **out of scope** here.

---

## 7. Branch base audit
`HEAD` was byte-identical to `origin/main` at branch creation (zero commits either
direction; details in `BASE_BRANCH_DIFF_AUDIT.md`). The submission branch therefore
starts from a `main`-equivalent state and targets `main` via PR #6.

---

## 8. Manuscript changes (`manuscript/main.tex`)
- Retitled: *"ARSPI-Net: An Event-Driven Reservoir–Graph Substrate for Embodied Affective EEG Perception."*
- Rewritten abstract, introduction, and bounded contribution list around the substrate framing.
- Added mechanism-to-implementation table (`table_neural_mechanisms`) near Methods.
- New results section §"Dataset Provenance, Robustness, and Simulated Control" wiring
  obs00, ana01, ana03, ana07, obs10 and tables for provenance, mechanism ablation,
  robustness, closed-loop, evaluation coverage, and reproducibility.
- Strict limitations paragraph (simulation not embodiment; not diagnostic; no measured
  energy; restricted dataset; SHAPE-ERP-regime scope).
- Restricted-data availability language (no public-release wording).
- Removed dissertation/doctoral framing, process language, `\todo` markers; renamed
  source figures to drop the token.
- Compiles to **17 pp**, no undefined references/citations, no visible table overflow.

---

## 9. Issues encountered and resolutions
1. **Zero-placeholder embedding (E).** Shipped `lsm_bsc6_pca` was all zeros → classifier
   collapsed to chance. Regenerated locally; pipeline now auto-detects and regenerates
   (Phase 2) so a clean checkout reproduces the results.
2. **Slow feature regeneration (~3.6 h).** Profiled to a single `lz76_count` routine using
   `np.array_equal` in a Python loop. Replaced with a **numerically-identical** bytes
   substring search (verified 240/240 sample sequences, 0 mismatches; ≈200× faster);
   regeneration dropped to ~5 min. The canonical script's numerics were not altered.
3. **Missing ch6/ch7 pickle.** Regenerated locally from the present raw EEG; uncommitted.
4. **Classifier collapse under imbalance.** Adopted `class_weight="balanced"` to match the
   validated affective-ablation protocol.
5. **LaTeX table overflow + a `Missing $`.** Fixed by full-width/`\footnotesize` tables,
   escaped underscores in control-row labels, and shorter evaluation-coverage labels.
6. **Stale skip/error status files** from incremental development were removed; figure
   scripts now self-clean stale skip reports on success.

---

## 10. Compliance with director constraints
| Constraint | Status |
|---|---|
| Repo scope = paper repo; disso = upstream only | ✓ |
| No new private data in PR; restricted-data language | ✓ |
| Pilot vs submission profiles; manuscript artifacts from submission | ✓ (manifest labels profile) |
| Core-phase failures fatal | ✓ (`run_ready9.py` exits nonzero) |
| EFE controller mathematically defined (code/metadata/manuscript) | ✓ (`closed_loop_policy_config.json`, §"Simulated…") |
| Competency matrix internal; manuscript uses evaluation-coverage table | ✓ |
| Graph/runtime bounded; no energy claim | ✓ |
| Clinical labels = validation/context, not biomarker | ✓ |
| Substantive manuscript revision | ✓ |
| Every quantitative claim traceable | ✓ (`SUBMISSION_REPRODUCTION_MAP.md`) |

---

## 11. Reproduction
```bash
# Full pipeline (pilot validates; submission produces manuscript artifacts)
python experiments/tcds_ready9/run_ready9.py
ARSPI_READY9_PROFILE=submission python experiments/tcds_ready9/run_ready9.py

# Individual gates / manuscript
python experiments/tcds_ready9/quality_gates.py
cd manuscript && pdflatex main.tex && pdflatex main.tex
```
Inputs auto-resolved from `data/` or env vars (`ARSPI_SHAPE_FEATURES`,
`ARSPI_CH67_FEATURES`, `ARSPI_CLINICAL_FILE`, `ARSPI_RAW_EEG_DIR`,
`ARSPI_DISSO_REPO`). Missing ch6/ch7 features and a placeholder embedding are
regenerated locally and never committed. The claim→artifact map is in
`SUBMISSION_REPRODUCTION_MAP.md`.

---

## 12. Open items for author/steward
- (Optional, authorized) purge pre-existing private files from git history.
- Verify the SHAPE/Stony Brook provenance wording before submission (currently reused
  from the verified manuscript data-availability statement).
- Anonymize author/acknowledgment block for double-blind review.
