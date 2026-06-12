# TAFFC Reviewer-Objection & Acceptance-Readiness Audit

**Date:** 2026-06-12 · **Branch:** `main` (merged via PR #19) · **Manuscript:** ARSPI-Net (10 pages,
`manuscript/main.pdf`; blind twin `main_blind.pdf`) · **Supplement:** 6 pages.

**Method.** This is a desk-rejection-prevention and reviewer-risk audit. No numerical result, figure,
table, metric, confidence interval, subject/observation count, or perturbation level was changed; no
experiment was added. Line/section references are to `manuscript/main.tex`. The shared manuscript is
part of the preserved TCDS package, so any manuscript wording change is recorded here as a
**recommendation only** (marked *Optional — not applied*), never silently edited.

---

## A. Likely reviewer objections and current manuscript defenses

### 1. "ARSPI-Net does not outperform the ERP baseline."
- **Where addressed:** Abstract ("A conventional ERP-amplitude baseline remains stronger for static
  classification, so we position ARSPI-Net as a measurement substrate rather than a static
  classifier"); Intro ¶2 (L44, "does not present ARSPI-Net as a superior static affective-EEG
  classifier"); Results §IV-C (L244, the ERP baseline "reaches 67.2% balanced accuracy… exceeding
  ARSPI-Net under a static readout. This comparison anchors the endpoint-classification trade-off");
  Discussion (L275); Conclusion (L293).
- **Explicit enough?** **Yes.** The baseline gap is stated up front, quantified, and reframed as the
  design premise (measurement substrate, not leaderboard entry). The objection is pre-empted, not
  hidden.
- **Recommended edit:** None.

### 2. "The dataset is private; no public-benchmark comparison."
- **Where addressed:** Limitations §VII (L286, "the access-controlled dataset limits direct public
  benchmarking, but it provides the properties required for this study… Public emotion-EEG datasets
  such as SEED or DEAP support different benchmarking questions, but they do not replace the
  clinically annotated SHAPE ERP regime"); Protocol/Privacy §V-A (L156); Data and Code Availability
  (L295).
- **Explicit enough?** **Yes.** The limitation is acknowledged, the cohort's required properties
  (211-subject affective ERP, clinical annotation, balance, subject-grouped validation) are
  justified, and public benchmarking is named as a *different, reserved* question rather than a
  missing result.
- **Recommended edit:** None required. *Optional* (not applied): one clause in §VII naming
  public-benchmark generalization as explicit future work would harden the "reserved scope" framing.

### 3. "Clinical labels are near-chance — why are they here?"
- **Where addressed:** Abstract ("Clinical labels serve only as exploratory contextual variables,
  bounded by a false-discovery-rate result"); Protocol Table I (L172, "not diagnostic validation");
  Results §IV-B (L232, "near chance (0.49 to 0.53)… bounded by a Benjamini-Hochberg
  false-discovery-rate result… exploratory context only"); Limitations §VII (L288, "The
  false-discovery-rate result therefore bounds the clinical interpretation rather than weakening the
  engineering contribution").
- **Explicit enough?** **Yes.** Reported honestly, scoped as exploratory, FDR-bounded, and
  explicitly walled off from the engineering claim. This converts a potential "so what?" into
  disciplined scope.
- **Recommended edit:** None.

### 4. "Over-engineered hybrid — LIF + graph + coupling looks like a kitchen sink."
- **Where addressed:** §IV "Brain-Inspired Mechanisms" (L126–152) anchors each block to a named
  neural correlate with an explicit observable (LIF spiking, BSC₆ coding, tPLV connectivity, κ
  coupling); the **operational-distinctness result is the empirical answer** — under CKA the
  off-diagonal alignment is small (largest CKA(E,D)=0.31; Fig. 2, L228/L232), so the streams are
  non-redundant, not duplicative complexity; Discussion (L275, "A stream that is sub-threshold as a
  classifier can still be operationally distinct as a measurement").
- **Explicit enough?** **Largely yes, but distributed.** The defense (mechanism-anchoring +
  measured non-redundancy) is spread across §IV and §V-B rather than stated as a single
  anti-"kitchen-sink" sentence. A skeptical reviewer will find it but must assemble it.
- **Recommended edit (*Optional — not applied*; would touch the shared manuscript):** add one
  consolidating sentence to the Discussion, e.g. *"Because each stream is anchored to a named neural
  mechanism (§IV) and shown non-redundant under centered-kernel alignment (Fig. 2), the four-stream
  design reflects measured complementarity rather than uncontrolled model complexity."* Page-neutral
  candidate; apply only if you accept a shared-manuscript change.

### 5. "The closed loop is simulated, not a real online BCI."
- **Where addressed:** Abstract ("a simulation and not an online BCI deployment"); §III-E (L152);
  Results §IV-D (L247, "The loop is offline: it is not an online BCI deployment and does not acquire
  EEG in real time"); Discussion (L279); Limitations §VII (L284, "not online BCI deployment,
  neurofeedback training, or functional restoration"); Conclusion (L293).
- **Explicit enough?** **Yes — emphatically.** The "simulation over recorded ERP observations"
  qualifier is attached at every occurrence (6+ times). There is no real-time/online claim to
  retract.
- **Recommended edit:** None.

### 6. "Is the contribution a framework or just a classifier / feature set?"
- **Where addressed:** Intro ¶2 (L44, "The contribution is instead a neuromorphic reservoir-graph
  measurement substrate"); the formal **operational-distinctness criteria** §III-D (L108, eqs. 9–11)
  define the framework contribution mathematically; contributions list (L53–58) is framed as
  substrate + distinctness + perturbation + closed-loop, none as accuracy; Discussion (L275,
  "operational differentiation rather than classifier superiority"); Conclusion (L293,
  "substrate-level methodology").
- **Explicit enough?** **Yes.** The framework-vs-classifier distinction is the spine of the paper and
  is formalized, not asserted.
- **Recommended edit:** None.

### 7. "Why does this belong in affective computing rather than neural engineering?"
- **Where addressed:** The decoded signal is the **affective ERP** (negative/neutral/pleasant
  conditions are the target throughout §V–§VI); Abstract leads with "affective event-related
  potential (ERP) decoding"; Related Work cites the affective-computing canon, including TAFFC
  papers (TSception [ding2023tsception], RGNN [zhong2022rgnn], Fourati reservoir-EEG-emotion
  [fourati2018reservoir]); Discussion (L281). The **cover letter** makes the explicit
  physiological-affective-computing case for venue fit.
- **Explicit enough?** **Yes, with correct division of labor.** The manuscript is deliberately
  venue-neutral (cascade reuse), decoding affective state and citing affective-computing work; the
  TAFFC-specific positioning lives in `COVER_LETTER_TAFFC_FINAL.md`, which is the right place for
  venue-fit argument. A TAFFC reviewer has the affective signal, the affective-computing lineage,
  and the cover-letter framing.
- **Recommended edit:** None for the manuscript (venue-neutral by design). The cover letter already
  carries the explicit TAFFC fit (see §D).

---

## B. Page-one gap / contribution audit
- **Contributions:** **Explicit and forward-referenced.** L53–58 enumerate four contributions, each
  with a Section and/or Figure cross-reference (§IV mechanisms; §V-B + Fig. 2 distinctness; §V-C
  perturbation; §V-D closed-loop). No change needed.
- **Research gap:** **Stated** at L52 ("Accuracy-optimized affective-EEG pipelines report a single
  endpoint but rarely expose named evidence streams, perturbation-specific operating regimes, or
  closed-loop accumulation utility under subject-level validation. This paper closes that gap…").
  This is explicit about the gap concept but does **not name the four stream dimensions**
  (temporal, dynamical, graph-topological, coupling).
- **Recommended edit (*Optional — not applied*; shared manuscript):** sharpen L52 to name the four
  dimensions, e.g. *"…report a single endpoint but rarely measure how affective neural information
  partitions across temporal, dynamical, graph-topological, and coupling evidence streams, how those
  streams degrade under perturbation, or how they support closed-loop accumulation under
  subject-level validation."* The manuscript compiles **deterministically to 10 pages** (verified;
  byte-identical reproduction), so this edit can be applied and re-verified safely on request — but
  it would modify `main.tex`/`main_blind.tex`/`main.pdf`, which the TCDS package reuses, so it is
  deferred pending explicit approval.

## C. Abstract audit
The abstract foregrounds, in order: (1) affective ERP as a partially observed physiological signal;
(2) the neuromorphic reservoir-graph substrate; (3) the four operationally distinct streams E, D, T,
C; (4) the four perturbation families / operating regimes; then, **secondarily**, (5) the
expected-free-energy controller in a closed-loop simulation "which is a simulation and not an online
BCI deployment." **Closed-loop BCI does not dominate** — it occupies one late sentence after the
measurement-substrate framing. **All five required elements are present and correctly ordered. No
change needed.**

## D. Cover-letter audit (`manuscript/COVER_LETTER_TAFFC_FINAL.md`)
All five required statements are present:
- **Not a classifier leaderboard paper** — "We do not claim recognition-accuracy superiority: a
  conventional ERP-amplitude baseline remains stronger…" (¶ Scope and claim discipline).
- **Not diagnostic validation** — "Clinical labels enter only as exploratory contextual variables…
  not as diagnostic validation."
- **Not online BCI deployment** — "a simulation over recorded observations, not an online or
  real-time system."
- **Contributes an affective neural evidence framework** — "decodes the affective ERP… into
  structured, interpretable evidence… four named neural evidence streams" (¶ Affective-computing
  contribution).
- **TAFFC fit from physiological affective computing** — affective ERP decoding, evidence-stream
  modeling, perturbation robustness, and uncertainty-aware (expected-free-energy) evidence
  accumulation are each named. **No change needed.**

## E. Final submission scan (all submission-facing artifacts)
Scanned: `main.tex`, `main_blind.tex`, `supplement.tex`, `COVER_LETTER_TAFFC_FINAL.md`, and the TAFFC
package docs.
- **Automated tool-attribution / vendor / session-link references:** **none.**
- **Online deployment / real-time / neurofeedback / cognitive augmentation / diagnostic validation /
  hardware-energy / full active inference:** present **only as negations or scoping** (Abstract L34;
  §II L68; §IV-D L247; Discussion L277, L279; Limitations L284; Conclusion L293) or as **citation
  titles** (Maass "Real-time computing…"; the active-inference references). No affirmative forbidden
  claim.
- **Single-trial claims:** **none** (the analysis is trial-averaged subject-condition observations).
- **Unscoped clinical claims:** **none** — every clinical mention is scoped exploratory / FDR-bounded
  / "not diagnostic validation."
- **Unsupported superiority language:** **none** — "superior"/"superiority" appears only in
  explicit negations (L44, L232, L293; cover letter ¶ Scope).

---

## Verdict
The package is **acceptance-ready against the audited risks**. Every major reviewer objection is
already pre-empted in the text; the abstract, contributions, and cover letter satisfy the TAFFC
positioning requirements; and the artifact scans are clean. Two **optional, page-neutral** wording
sharpenings (objection 4 consolidation; gap-sentence dimension naming) are available but are **not
applied**, because they would modify the manuscript shared with the preserved TCDS package — apply
only on explicit request.

## Remaining reviewer risks before submission (residual, not defects)
1. **Static-accuracy reflex.** A reviewer may still reward endpoint accuracy despite the substrate
   framing; this is a framing risk inherent to the contribution, fully disclosed, not fixable by
   wording.
2. **Affective-computing vs. neural-engineering fit.** A TAFFC reviewer could prefer a SEED/DEAP
   emotion-recognition comparison; the cover letter and affective-ERP framing mitigate this, but it
   remains the most likely reason for a scope-fit critique.
3. **Hybrid-complexity skepticism.** Mitigated by mechanism-anchoring + CKA non-redundancy, but
   distributed; the optional objection-4 sentence would consolidate the rebuttal.
4. **Portal-time confirmations** (tracked in `TAFFC_AUTHOR_GUIDELINES_VERIFICATION.md`): review model
   (single- vs double-blind → which PDF to upload) and the open-access/APC election remain to be
   confirmed on ScholarOne at submission.
