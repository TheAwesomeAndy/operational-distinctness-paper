# TAFFC Acceptance-Readiness Report

**Date:** 2026-06-12 · **Status:** merged to `main` via PR #19.

This records the TAFFC framing-and-compliance pass that incorporated the acceptance feedback into an
**additive** TAFFC manuscript variant, leaving the shared (TCDS) manuscript untouched. It was a
framing pass, not a new-experiment pass.

## 1. Changes made
All TAFFC framing edits were applied to a new manuscript variant
(`manuscript/taffc/main_taffc.tex` and its double-blind twin `main_taffc_blind.tex`), **not** to the
shared `manuscript/main.tex`. Prose only — no equation, number, figure, or table was altered.

- **A — Introduction / gap (page one).** Reframed the opening around *physiological affective
  computing* (affective ERP decoding, physiological affective evidence, discrete affective stimulus
  conditions). Replaced the gap sentence with the compound affective-computing gap: how affective
  neural information is transformed across temporal, dynamical, graph-topological, and
  structure-function coupling observables, how those streams degrade under perturbation, and how they
  support closed-loop evidence accumulation under uncertainty. One concise discrete-affect clause
  with a citation; psychology not over-centered.
- **B — Contributions.** Expanded the explicit, forward-referenced list from four to **five**, adding
  a dedicated structure-function coupling (κ) contribution that reports both its above-chance
  electrode-coordination signal and its **measured limit as a non-predictor of static classification**
  (constraint 5 respected). Each contribution points to its Section/Figure.
- **C — Static-baseline framing.** The scoped statement that conventional ERP-amplitude features are
  the stronger static endpoint classifier in this regime — with ARSPI-Net positioned as a measurement
  substrate, not a leaderboard classifier — is retained and reinforced by the new positioning
  paragraph. No superiority implied.
- **D — Closed-loop scope.** Tightened to "an algorithmic validation layer for policy behavior under
  uncertainty — not online BCI deployment, neurofeedback, or a human-in-the-loop intervention." No
  "continuous online," "live deployment," "function augmentation," or "safety gate" language.
- **E — Related Work / positioning.** Kept the compact early Related Work (four families) and added a
  **late Discussion positioning paragraph** comparing ARSPI-Net to endpoint decoders, affective
  temporal-spatial models, graph-EEG, and reservoir/neuromorphic models across the requested axes,
  explicitly marking the ERP baseline as stronger on static accuracy and claiming no across-axis win.
- **F — Figure / grayscale audit.** Audited; see §4. No figure data altered.
- **G — Data Availability and Ethics.** Replaced "Data and Code Availability" with a dedicated **Data
  Availability and Ethics** statement covering: restricted human-subject EEG/ERP data;
  access-controlled, non-redistributed clinical metadata; aggregate outputs / scripts / figure-code /
  tables / manifests provided where permissible; no PHI or raw identifiers in public artifacts;
  clinical labels exploratory, not diagnostic. No IRB/DUA numbers invented.
- **H — Abstract.** Added a one-sentence problem/motivation lead (endpoint accuracy is not enough;
  physiological affective computing also needs evidence provenance, robustness, and uncertainty
  behavior); solution and evidence follow. Closed-loop remains secondary; does not read as a
  deployment paper.
- **I — Cover letter.** `manuscript/COVER_LETTER_TAFFC_FINAL.md` updated to reflect the refined frame
  and the κ observable with its measured limit; the prior-work disclosure is kept; explicit limits
  retained (not a classifier leaderboard, not diagnostic validation, not online BCI).

## 2. Files changed
**Added (additive):**
- `manuscript/taffc/main_taffc.tex`, `manuscript/taffc/main_taffc.pdf` — 11 pp, single-blind / non-anonymized.
- `manuscript/taffc/main_taffc_blind.tex`, `manuscript/taffc/main_taffc_blind.pdf` — 11 pp, double-blind.
- `TAFFC_ACCEPTANCE_READINESS_REPORT.md` — this file.

**Modified (TAFFC-package docs only):**
- `manuscript/COVER_LETTER_TAFFC_FINAL.md`
- `SUBMISSION_PROFILE_TAFFC.md`
- `TAFFC_AUTHOR_GUIDELINES_VERIFICATION.md` (packaging-decision section)

**Untouched:** the shared manuscript (`main.tex/.pdf`, `main_blind.tex/.pdf`), `supplement.pdf`, all
figures and tables, and every TCDS package file.

## 3. Page count
`main_taffc.pdf` and `main_taffc_blind.pdf`: **11 pages** each (the shared `main.pdf` is 10). Within
TAFFC's 12-page free limit; no overlength charge. `supplement.pdf`: 6 pages (unchanged).

## 4. Figure / grayscale audit (Task F)
- **Captions / scope:** every caption describes measured objects (no promotional conclusions), and
  scope lives in captions/text. **Figure 5** (closed-loop) has **no in-figure scope note**.
- **Grayscale legibility:** Figures 4 (robustness curves) and 6 (closed-loop success) distinguish
  their series by **color/shade with identical circle markers**, so the middle series are hard to
  separate in grayscale — a legibility risk for B&W print and color-blind readers. These are
  **shared, pre-rendered figures** (also used by the TCDS manuscript); regenerating them needs the
  plotting pipeline, so they were **not altered** here (no figure-data change, no TCDS impact).
  **Recommendation:** add distinct line styles/markers at the next figure regeneration — this also
  benefits TCDS. Figures 1–3 (schematic, raster, CKA heatmap) are intensity/structure based and read
  acceptably in grayscale.

## 5. No numerical results changed (confirmation)
Confirmed by numeric-token comparison of the compiled PDFs: the multiset of all decimals,
percentages, counts, confidence intervals, and years is **identical** between `main_taffc.pdf` and the
shared `main.pdf`, and between `main_taffc_blind.pdf` and `main_blind.pdf`. The only token differences
are added citation markers and the extra page-11 footer. No metric, CI, subject/observation count, or
perturbation level changed; no experiment was added.

## 6. TCDS package preserved (confirmation)
All TCDS package files — `manuscript/main.pdf`, `manuscript/supplemental_risk_reduction/supplement.pdf`,
`manuscript/COVER_LETTER_TCDS_COGNEURO_BCI_FINAL.md`, `SUBMISSION_PROFILE.md`,
`CFP_ALIGNMENT_COGNEURO_BCI.md`, `READY_FOR_COGNEURO_BCI_SUBMISSION_CHECKLIST.md` — together with the
shared `main.tex` / `main_blind.tex` are **byte-identical to `origin/main`** (verified). The TAFFC
layer is purely additive.

## 7. Remaining reviewer risks before submission
1. **Static-accuracy reflex.** Disclosed and reframed, but a reviewer may still reward endpoint
   accuracy; inherent to the substrate contribution.
2. **Affective-computing vs. neural-engineering fit.** Strengthened by the page-one reframing and the
   positioning paragraph; a SEED/DEAP benchmark remains a reserved (out-of-scope) extension.
3. **Grayscale figure legibility (Figs 4, 6).** Color/shade-only series distinction; recommend distinct
   markers at the next figure regeneration (see §4).
4. **Portal-time confirmations** (`TAFFC_AUTHOR_GUIDELINES_VERIFICATION.md`): review model
   (single- vs double-blind → which variant PDF to upload) and the open-access / APC election.

## 8. Compile / scan status
- `main_taffc.pdf` / `main_taffc_blind.pdf`: 11 pp, 0 undefined references, clean two-pass compile.
- Forbidden-claim scan (online / real-time, neurofeedback, human-in-the-loop, cognitive augmentation,
  diagnostic validation, hardware-energy, full active inference, single-trial, superiority): present
  only as negations / scoping, or as citation titles.
- Tool-attribution / vendor / session-link scan: clean across all TAFFC artifacts.
