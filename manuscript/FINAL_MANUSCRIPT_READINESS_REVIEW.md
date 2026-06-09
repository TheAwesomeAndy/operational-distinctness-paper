# Final manuscript-readiness review — 10-page `manuscript/main.pdf`

Read as a paper for the IEEE TCDS special issue "Brain-Inspired Computing for
Embodied AI." 10 pages, ~7.4k words, 5 figures, 6 tables, 0 undefined references.
**Overall: structurally ready; submittable after a short list of pre-submission
edits (§8).** The paper now reads as one argument; the duplication and float
overload of the prior versions are gone.

## 1. Title / abstract fit
- **Title** on-target; keep.
- **Abstract** now reports only the primary (multi-seed) protocol, notes the
  independent corroboration, and foregrounds substrate → graph evidence →
  robustness → simulated embodied control → bounded scope. Fit is good.
- *Minor:* it is a single dense ~290-word block. Acceptable, but one or two
  sentences could be shortened (e.g., the κ aside) to improve readability.

## 2. Introduction and contribution clarity
- The four contributions are crisp and bounded (substrate specification;
  operational-distinctness via ablation; perturbation robustness; embodied
  control). The central problem is correctly framed as event-driven neural
  evidence extraction under uncertainty.
- *Minor residue:* the legacy "operational distinctness = empirical
  non-equivalence" definitional paragraph still sits in the intro alongside the
  substrate framing. It no longer dominates, but one sentence could subordinate
  it ("a property the substrate exposes") rather than presenting it as a second
  organizing question.

## 3. Mechanism-to-implementation adequacy
- **Strong and the paper's best asset.** Section IV maps each mechanism (LIF
  reservoir, BSC6 code, tPLV topology, κ coupling) to a computational object and
  a measured observable, with the layer-formulation equations and the mechanism
  table. Nothing to add.

## 4. Results coherence
- **Now coherent.** Four pillars, one primary ablation, one merged
  open-loop→closed-loop embodied narrative, ERP reframed as a static anchor. The
  argument flows mechanism → substrate → evidence streams → robustness → embodied
  control.
- *Gap to fix (see §8):* the **robustness pillar is qualitative** — it states
  that embedding-containing streams retain the most information and the
  graph-derived stream degrades fastest, but cites **no numbers**. A reviewer
  will want at least one quantitative anchor (e.g., balanced accuracy at 10 dB
  SNR and at 30% channel dropout for E+D+T vs T).

## 5. Figure/table burden and page compression
- **Appropriate.** 5 figures + 6 tables in 10 pages, no overflow, at the TCDS
  no-fee limit. Each float earns its place; the secondary evidence is preserved
  internally.

## 6. Claim discipline
- **Excellent and preserved.** Clinical = exploratory (no biomarker, near-chance,
  FDR-bounded); no physical-robot, measured-energy, SOTA, or public-data claims;
  EFE-no-advantage stated honestly as an operating-regime result; restricted-data
  language present and accurate.

## 7. Likely reviewer objections (and current defensibility)
1. *"Affective accuracy is modest (~0.46–0.48 vs 0.33 chance)."* — defended by
   subject-respecting permutation inference and the substrate (not classifier)
   framing; keep both prominent.
2. *"A classical ERP baseline beats it."* — now a compressed, honest static-anchor
   statement; well handled.
3. *"The EFE controller shows no advantage."* — framed as an operating-regime
   result with the oracle gap as the positive takeaway; defensible.
4. *"Robustness claims lack numbers."* — currently **weakly defended** (§4/§8).
5. *"Single private dataset; simulation-only embodiment."* — acknowledged in
   limitations; keep.
6. *"Novelty vs prior ARSPI-Net work."* — handled in one Related-Work sentence;
   adequate but could be made slightly more explicit about the new robustness +
   defined-controller contributions.
7. *"Where is the moved material?"* — the text repeatedly cites a "supplementary
   technical version"; a supplement must actually accompany the submission (§8).

## 8. Required edits before ScholarOne submission
1. **Add 1–2 quantitative anchors to the robustness paragraph** (a balanced-
   accuracy value at a stated SNR and at a stated dropout level for a
   representative stream vs the band-power baseline). Numbers already exist in
   `robustness_summary.csv`; this is a one-sentence text edit, not a new
   experiment.
2. **Prepare the supplementary material the text refers to.** Either attach the
   17-page `main_full_internal` content as a formal Supplementary Material file,
   or reword the "supplementary technical version" references to "available on
   approved-access request." Do not leave dangling promises of a supplement.
3. **Double-blind anonymization** for the special-issue review: author block,
   acknowledgments, the repository-link sentence in Data and Code Availability,
   and any first-person prior-work attribution.
4. **Subordinate the legacy operational-distinctness definition** in the
   introduction (one-sentence edit) so the substrate is the single organizing
   frame.
5. **Optional polish:** shorten the abstract by ~15–20 words; final proofread of
   equations (LIF update, EFE Eq.) and reference list completeness.

### Verdict
The manuscript is a coherent, correctly-scoped, claim-disciplined 10-page TCDS
special-issue candidate. None of the §8 items require new experiments; they are
short text edits plus a supplementary-material packaging step. Recommend: apply
§8 items 1–4, prepare the supplement, anonymize, then submit.
