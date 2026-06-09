# Submission-Readiness Review — `manuscript/main.pdf`

Read as a paper, not as a code package. Target: IEEE TCDS special issue,
"Brain-Inspired Computing for Embodied AI." Current state: 17 pp, ~12.2k words,
16 figures, ~13 tables, 27 results/discussion subsections. **Not yet ready for
ScholarOne.** The science is sound and the framing is correct; the blocker is
**structural coherence and float burden** created by layering the new
substrate-hardening results onto an already-complete manuscript.

---

## 1. Title / abstract fit to the special issue
- **Title** — strong and on-target: *"ARSPI-Net: An Event-Driven Reservoir–Graph
  Substrate for Embodied Affective EEG Perception."* Keep verbatim.
- **Abstract** — foregrounds the substrate, the four evidence streams, the
  mechanisms, ablation, robustness, and simulated control, with a bounded-scope
  sentence. Good direction. **Two fixes:**
  1. It reports the **existing 10-fold** numbers (BandPower 49.27%, E 46.96%,
     E+D 49.34%, E+D+T+C 49.33%) and `n_perm`/FDR detail, but the body now also
     reports the **5-seed** mechanism-ablation numbers (BandPower 0.485, E 0.463,
     E+D+T 0.481). The abstract must quote **one** primary protocol's numbers.
  2. It is dense (single 250-word block). For a substrate paper, lead with the
     substrate + embodied-control contribution and move the permutation/FDR
     mechanics to the body.

## 2. Introduction and contribution clarity
- The introduction is reframed correctly around *event-driven neural evidence
  extraction under uncertainty* and the four bounded contributions are crisp.
- **Residual dual identity:** the intro still carries the legacy
  "operational distinctness is empirical non-equivalence of layers" definitional
  paragraph alongside the new substrate framing. Keep operational distinctness as
  a *property the substrate exposes*, not as the paper's organizing question, or a
  reviewer will read it as the older operational-distinctness paper relabeled.
- Add one sentence making the **delta vs prior ARSPI-Net work** explicit (what is
  new here: robustness + defined closed-loop controller + provenance), to preempt
  a self-overlap/novelty objection.

## 3. Mechanism-to-implementation adequacy
- **Strong.** A dedicated mechanisms section (LIF reservoir, BSC6, tPLV, κ) plus
  the mechanism-to-implementation table and the layer-formulation equations give a
  clean mechanism → object → observable chain. This is the paper's best asset for
  the special issue. No change required beyond trimming length (§5).

## 4. Results coherence — **PRIMARY BLOCKER**
The results section reads as two papers spliced together. Two pairs of competing
primary narratives must be reconciled:

- **Ablation (duplicate):** §"Affective Ablation: Embedding-Containing
  Combinations Are Strongest" (existing, 10-fold, permutation+FDR) **and**
  §"Mechanism Ablation and Operational Distinctness" (new, 5-seed, bootstrap CIs +
  negative controls). Both report A0–A9 with slightly different values.
  → **Designate one primary.** Recommended: keep the new multi-seed mechanism
  ablation (negative controls + CIs are stronger evidence of operational
  distinctness) as primary; fold the 10-fold permutation/FDR result into a single
  table with an explicit **protocol-label column**, or demote it to corroboration.
- **Embodied/closed-loop (duplicate):** §"Sequential Evidence Accumulation
  Simulation" (existing, open-loop policy ranking by information gain) **and**
  §"Simulated Embodied Affective-Control Loop" (new, closed-loop, defined EFE
  controller, ε noise). → **Merge into one narrative:** sequential evidence
  accumulation as the *open-loop precursor*, then the closed-loop EFE controller
  as the *main* embodied result. One figure each, one combined message.
- **Subsection sprawl:** 16 results subsections. Regroup strictly under the four
  pillars — (A) observed objects + provenance, (B) mechanism ablation, (C)
  robustness, (D) simulated embodied control — and move QC, comorbidity-adjusted
  models, and layer-redundancy (CKA) to a supplement.

## 5. Figure/table burden and page compression
- **16 figures + ~13 tables in 17 pp is far too heavy** for a TCDS regular paper
  and is the main length driver.
- **Suggested main-text budget: ≤ 8 figures, ≤ 6 tables.** Candidate cuts /
  moves to supplement: the QC integrity panel (obs00), feature-block distributions
  (obs07), BSC6-bin panel (obs06), comorbidity table, layer-redundancy (CKA)
  table+figure, and one of the two ablation tables and one of the two embodied
  figures (per §4 consolidation).
- Consolidate the dataset-provenance, reproducibility, evaluation-coverage, and
  comparison-positioning tables into **one or two** compact tables.

## 6. Claim discipline
- **Excellent overall.** Clinical results explicitly bounded (near chance, no
  biomarker), no measured-energy claim (hardware section is a feasibility
  statement), no physical-robot claim, EFE-shows-no-advantage handled as an
  honest operating-regime result, restricted-data language correct.
- **One exposure:** the "Headline Comparison vs Classical ERP Baseline" shows the
  classical ERP baseline (67.2% BA) **beating** ARSPI-Net (~49%) on the affective
  task. The text correctly says the contribution is not endpoint supremacy, but a
  reviewer will still read "the substrate is a worse classifier." Reframe this so
  the substrate's value (operationally distinct, robust, controllable evidence
  streams) is the headline and the ERP-accuracy gap is context, not the finish.

## 7. Likely reviewer objections (and current defensibility)
1. *"Why are there two different A0–A9 ablation results?"* — currently
   **undefended**; fixed by §4 consolidation.
2. *"The proposed substrate underperforms a classical ERP baseline on accuracy."*
   — partially defended; strengthen per §6.
3. *"~49% on a 3-class task (chance 33%) — is the signal real?"* — defended by
   subject-respecting permutation tests; keep that prominent.
4. *"The EFE controller shows no advantage over pragmatic control — what is its
   contribution?"* — defended as an operating-regime result; make the oracle gap
   (cost of single-trial perceptual unreliability) the positive takeaway.
5. *"Single private dataset; no public benchmark; closed-loop is simulation only."*
   — acknowledged in limitations; keep, and state the restricted-data
   reproducibility posture clearly.
6. *"Novelty vs the authors' prior ARSPI-Net work."* — add the explicit delta
   (§2).

## 8. Required edits before ScholarOne submission
1. **Reconcile the two ablations** into one primary result (+ protocol-labeled
   corroboration). *(blocker)*
2. **Merge the two embodied analyses** into a single open-loop→closed-loop
   narrative. *(blocker)*
3. **Reconcile abstract numbers** to the chosen primary protocol. *(blocker)*
4. **Compress to the TCDS page limit** (verify current policy); cut floats to
   ≤ 8 figures / ≤ 6 tables; move QC/redundancy/comorbidity to a supplement.
   *(blocker once policy known)*
5. **Reframe the ERP headline comparison** so the substrate contribution leads.
6. **Tighten the introduction** to a single organizing question (substrate), with
   operational distinctness as a property, and add the prior-work delta.
7. **Author-ready / anonymization:** prepare the double-blind version (author
   block, acknowledgments, repo-link references).
8. Keep the **17-page version as the internal full-evidence technical record**;
   produce the compressed submission version separately.

---

### Verdict
Scientifically defensible and correctly scoped for the special issue, but **not
submission-ready** until the duplicate ablation and duplicate embodied narratives
are consolidated and the float/page burden is cut. These are **editorial
restructuring** tasks, not new experiments — the evidence already exists in the
merged package. Recommend one focused revision pass addressing §8 items 1–6, then
a final page-policy check.
