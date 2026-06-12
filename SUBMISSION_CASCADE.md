# Submission Cascade — ARSPI-Net

Aim high and do not undersell. Submit to the top choice first; on rejection, incorporate the
reviewers' feedback, reformat to the next venue's class file and reference style, and resubmit.
Submit to exactly one venue at a time.

**Paper identity (held constant across the cascade):** a neuromorphic reservoir-graph
**measurement substrate** for affective ERP decoding and **closed-loop adaptive BCI simulation
over recorded ERP observations** — a rigorous, claim-bounded methodology (subject-grouped
validation, negative controls, FDR control, perturbation/operating-regime characterization,
operational-distinctness analysis, expected-free-energy evidence accumulation). It is explicitly
**not** a static-accuracy leaderboard result, and it makes **no** claim of online BCI deployment,
real-time EEG operation, neurofeedback, cognitive augmentation, diagnostic validation, or
hardware-energy efficiency.

**No new experiments are required for any venue in this cascade.** Each retargeting step is a
reframing and reformatting task, not a re-measurement. The reserved extensions (public-benchmark
generalization, online/single-trial decoding, hardware-energy accounting) are deliberately out of
scope and must not be added merely to chase a venue.

## Ranked target list

### 1. IEEE Trans. Cognitive and Developmental Systems (TCDS) — SI "Cognitive Neuroscience Meets BCIs" — **current target**
- **Scope fit:** direct theme match — neural decoding at the cognitive-neuroscience/BCI interface; neuromorphic, spiking, and graph precedents already published in TCDS (Cai et al., 2024; Kim et al., 2025).
- **Main strength here:** claim-bounded methodology and the closed-loop expected-free-energy evidence-accumulation simulation sit squarely on the "neural decoding" side of *From Neural Decoding to Function Augmentation*.
- **Main reviewer risk:** a conventional ERP-amplitude baseline beats ARSPI-Net on static three-class accuracy; a reviewer may expect a stronger endpoint or an online/deployment result.
- **Retargeting action if rejected:** move to TNSRE; shift the lead framing from "cognitive/developmental" to "neural-engineering / BCI decoding"; sharpen the measurement-substrate argument against the endpoint-accuracy expectation.
- **New experiments needed:** No.
- **Logistics:** 10 free pages, $200/overlength page, 15 max. Deadline 2026-07-31. Re-verify the current Editor-in-Chief and guest editors before submitting.

### 2. IEEE Trans. Neural Systems and Rehabilitation Engineering (TNSRE)
- **Scope fit:** core EEG/BCI decoding and neural-engineering venue; our graph-EEG (Klepl et al.) and spiking-graph BCI (Gong et al.) anchors are TNSRE papers.
- **Main strength here:** rigorous cross-subject protocol, perturbation robustness, and reproducible evaluation; adaptive evidence accumulation is relevant to assistive/rehabilitation BCI.
- **Main reviewer risk:** expectation of an online or patient-facing evaluation, or a rehabilitation endpoint; the static-accuracy gap resurfaces.
- **Retargeting action if rejected:** move to JBHI; foreground the clinically annotated cohort and a health-informatics framing while keeping all claim boundaries.
- **New experiments needed:** No (reframe only).

### 3. IEEE Journal of Biomedical and Health Informatics (JBHI)
- **Scope fit:** biomedical and health-informatics venue; home of several cited works (EEG-Deformer; Guo et al., 2025; Chiang et al., 2023).
- **Main strength here:** clinically annotated 211-subject cohort, FDR-bounded exploratory clinical-label sensitivity, deidentified aggregate reporting, and methodological rigor.
- **Main reviewer risk:** reviewers may push for diagnostic or clinical-utility claims that we deliberately do not make; near-chance clinical-label accuracy invites a "so what" question.
- **Retargeting action if rejected:** move to TAFFC; lead with affective ERP decoding and operational distinctness; de-emphasize the clinical-label and BCI-loop framing.
- **New experiments needed:** No — and the exploratory, non-diagnostic framing must be preserved rather than upgraded to fit the venue.

### 4. IEEE Trans. Affective Computing (TAFFC)
- **Scope fit:** affective-computing home of TSception, RGNN, and Fourati et al. (all cited here).
- **Main strength here:** affective ERP decoding into operationally distinct streams; CKA redundancy and perturbation analyses are uncommon in affective-EEG work.
- **Main reviewer risk:** affective-computing reviewers typically expect higher endpoint emotion-recognition accuracy and a public-benchmark (SEED/DEAP) comparison.
- **Retargeting action if rejected:** move to TBME or Neural Networks; broaden the methodological framing beyond affect.
- **New experiments needed:** No — a public-benchmark comparison is a reserved scope expansion and must not be bolted on solely to satisfy this venue.

### 5. IEEE Trans. Biomedical Engineering (TBME) *or* Neural Networks (Elsevier) — later fallback
- **Scope fit:** broad biomedical-signal / neural-computation venues.
- **Main strength here:** a general neuromorphic measurement-substrate methodology with a reproducible evaluation protocol.
- **Main reviewer risk:** breadth and positioning — must justify general interest beyond the specific cohort.
- **Retargeting action if rejected:** revisit framing; consider a methods-forward venue (e.g., Journal of Neural Engineering) or a focused workshop.
- **New experiments needed:** No for the initial submission; a public-benchmark or generalization extension is the reserved option, used only if a venue explicitly requires demonstrated generality.

## Per-resubmission checklist (if rejected)
1. Read every review; separate **must-fix** (validity, clarity) from **scope** comments.
2. Incorporate fixable feedback; for out-of-scope asks, prepare a brief rebuttal note for the next cover letter.
3. Reformat to the next venue's class file, page limit, and reference style; re-run the page / undefined-reference / numeric-preservation / forbidden-claim checks.
4. Rewrite the cover letter for the new venue: name its current editor, restate scope fit, and cite 1–2 recent papers that venue published **as scope analogues, not technical ancestors**.
5. Keep all claim boundaries intact across venues (no online-deployment / real-time / neurofeedback / diagnostic / hardware-energy / cognitive-augmentation / full-active-inference claims).

## Notes
- Editors-in-chief and editorial boards rotate; re-verify the target venue's current Editor-in-Chief and any guest editors before each submission.
- Page-charge thresholds differ by venue; confirm the free-page limit before finalizing length.
