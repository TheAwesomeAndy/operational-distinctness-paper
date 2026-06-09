# Structural revision report — manuscript restructuring pass

Editorial restructuring only; **no new experiments**. Goal: convert the
17-page accumulated-evidence manuscript into a coherent, compressed TCDS
special-issue paper reading as one argument: neural mechanism → reservoir–graph
substrate → measured evidence streams → perturbation robustness → simulated
embodied affective-control.

## Outcome
- **Submission candidate:** `manuscript/main.tex` → `manuscript/main.pdf`,
  **13 pages** (was 17), 8 figures, 7 tables, 0 undefined references, no visible
  overflow, privacy + language gates pass.
- **Internal full-evidence record (preserved):**
  `manuscript/main_full_internal.tex` → `manuscript/main_full_internal.pdf`,
  17 pages, all original analyses retained verbatim.

## 1. Which ablation protocol is primary
- **Primary (main text):** the multi-seed mechanism ablation (5 seeds, bootstrap
  intervals, shuffled-label and shuffled-subject negative controls) —
  `tables/tcds_ready9/table_mechanism_ablation.tex`, Fig. `ana01`. Numbers:
  A0 BandPower 0.485, E 0.463, D 0.432, T 0.355, C 0.368, E+D 0.478, E+D+T 0.481,
  E+D+T+C 0.481; shuffled-label control 0.335 (chance).
- **Secondary/corroborating:** the original 10-fold permutation–FDR affective
  ablation is reduced to **one corroboration sentence** in the main text ("eight
  of ten configurations exceed a subject-respecting permutation null; standalone
  T and C do not") and preserved in full in the internal version.
- **Abstract** now reports **only** the primary protocol, explicitly noting the
  independent protocol corroborates it. The former 10-fold figures (49.27%,
  46.96%, …) no longer appear in the abstract.

## 2. How the two embodied analyses were merged
Combined into a single pillar D ("Simulated Embodied Affective-Control"):
1. one belief-state / evidence-accumulation model statement, with the
   action-determined transition $P(s'|a)=(1-\epsilon)\mathbf{1}[s'{=}a]+\epsilon\mathcal{U}$;
2. **open-loop** sequential evidence accumulation presented as
   observation-channel characterisation (Fig. `fig_embodied_loop`; ordering
   $E{+}D{+}T<E<D<$ random $<T$, matching the offline ablation);
3. **closed-loop** policy evaluation as the embodied-control test, with the EFE
   objective given explicitly as Eq. (efe): risk = KL(P(s'|a)‖C), ambiguity =
   E[Ĥ(s')], action = argmin EFE, stop at b(target)≥0.8 (Fig. `ana07`,
   `table_closed_loop_policy`);
4. EFE-vs-pragmatic reported honestly as an **operating-regime** result (no
   advantage under action-determined transitions; oracle gap = cost of
   single-trial perceptual unreliability).
The two analyses now read as one open-loop→closed-loop narrative, not two
unrelated experiments.

## 3. Results rebuilt around four pillars
- **A. Dataset provenance and observed computational objects** — provenance/QC
  (Fig. `obs00` + reproducibility panel), observed objects (Fig.
  `fig_reservoir_dynamics`: raster, membrane, tPLV, BSC6), κ summary sentence.
- **B. Reservoir–graph mechanism ablation** — primary ablation table + Fig.
  `ana01`; operational-differentiation reading; CKA and clinical sensitivity
  condensed to sentences.
- **C. Perturbation robustness and operating regimes** — Fig. `ana03`; ERP
  baseline reframed as a static discrimination anchor (substrate, not static
  classifier).
- **D. Simulated embodied affective-control** — merged open-loop + closed-loop
  (above).
- Plus a short **Reproducibility** subsection (reproducibility panel).

## 4. Figures/tables retained in the main paper
**Figures (8):** architecture overview + BSC6 source (Methods); `obs00`,
`fig_reservoir_dynamics`, `ana01`, `ana03`, `fig_embodied_loop`, `ana07`
(Results).
**Tables (7):** protocol, feature blocks, neural mechanisms (Methods);
mechanism ablation, closed-loop policy, reproducibility panel (Results).

## 5. Material moved out of the main flow (now only in the internal full version)
- 10-fold affective-ablation table (`table2`) and Figs `fig04`, `fig05`.
- Permutation-FDR and power-bounded clinical subsections.
- Clinical-label sensitivity tables (`table3`, `table4`) and Figs `fig06`,
  `fig07`.
- Comorbidity-adjusted table (`table6`).
- Layer-redundancy/CKA table (`table5`) and Fig `fig08`.
- Standalone κ figure (`fig_kappa_distributions`).
- Headline ERP-baseline figure (`fig_headline_comparison`).
- Closed-loop example-trajectory figure (`obs10`), dataset-provenance,
  evaluation-coverage, graph-support and runtime tables.
The reproduction map and changelog continue to index all of these.

## 6. Remaining page-count risk
- The submission candidate is **13 pages**. This is comfortably below the 17-page
  full version, but the **current IEEE TCDS / special-issue page policy is not
  yet verified**. If the regular-paper limit (and overlength-charge threshold) is
  below 13 pages, the next compression levers are: trim the Discussion (currently
  multiple subsections — hardware, plasticity, positioning, clinical), move the
  BSC6 source figure to the supplement, and tighten Related Work.
- No factual inconsistencies were introduced; numbers in the compressed version
  are the submission-profile (5-seed) results, with the 10-fold protocol cited as
  corroboration.

## Not done by design
- No new experiments. No change to claim discipline (clinical = exploratory; no
  diagnostic/biomarker, physical-robot, measured-energy, SOTA, or public-data
  claims; restricted-data language retained).
- Final page-policy verification and double-blind anonymization remain for the
  packaging step.
