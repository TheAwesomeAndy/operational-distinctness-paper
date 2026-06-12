# TAFFC Final Submission Audit

**Date:** 2026-06-12 · **Branch:** `submission/taffc-final-consistency-audit` (from `main` `4987944`)

Documentation-consistency audit after title convergence (PR #21 merged, PR #20 closed/superseded)
and branch consolidation. **No scientific content changed** — no title, abstract framing, claim,
figure, table, or numerical value was altered, and **no figures were regenerated**.

## Final title (canonical on `main`)
`Spatiotemporal Characterization of Affective EEG Dynamics Using a Neuromorphic Reservoir–Graph
Framework` — the active title in the TAFFC variant (`main_taffc.tex` / `main_taffc_blind.tex`), the
cover letter, and the profile.

## Files changed (documentation only)
- `TAFFC_AUTHOR_GUIDELINES_VERIFICATION.md`
- `SUBMISSION_PROFILE_TAFFC.md`
- `TAFFC_TITLE_KEYWORD_REFINEMENT_REPORT.md`
- `TAFFC_ACCEPTANCE_READINESS_REPORT.md`
- `TAFFC_REVIEWER_OBJECTION_AUDIT.md`
- `TAFFC_FINAL_SUBMISSION_AUDIT.md` (this file, new)

## Stale documentation fixed
- `TAFFC_AUTHOR_GUIDELINES_VERIFICATION.md` — replaced the obsolete "reuse `main.pdf`; **no overlay,
  no manuscript edits**; 10 pages" implication with the current state: TAFFC uses the additive
  variant under `manuscript/taffc/`; `main_taffc.pdf` / `main_taffc_blind.pdf` are 11 pages within
  the free limit; shared TCDS files preserved. (Section 10 already reflected the variant.)
- `SUBMISSION_PROFILE_TAFFC.md` — `Branch: submission/taffc-first-target` → `Branch: main`
  (canonical; merged via PRs #19 and #21).
- `TAFFC_TITLE_KEYWORD_REFINEMENT_REPORT.md` — header branch → "merged into `main` via PR #21";
  replaced the "PR #20 open / mutually exclusive / needs reconciliation" note with "PR #21 supplied
  the final title; PR #20 closed/superseded; `main` canonical"; relabeled the old title as a
  "superseded TAFFC variant title."
- `TAFFC_ACCEPTANCE_READINESS_REPORT.md`, `TAFFC_REVIEWER_OBJECTION_AUDIT.md` — stale
  `Branch: submission/taffc-first-target` headers → `main` (merged via PR #19).

## Final title consistency status
- **Active title** in all TAFFC-facing files (variant `.tex`, cover letter, profile) is the
  EEG-dynamics title. ✓
- Old titles appear only **clearly labeled as old/superseded** in
  `TAFFC_TITLE_KEYWORD_REFINEMENT_REPORT.md` (historical record). ✓
- The "ARSPI-Net … Substrate …" title that remains in `manuscript/main.tex`, `main_blind.tex`,
  `supplement.tex`, `submission_package_blind/main_blind.tex`, `SUBMISSION_PROFILE.md`, and
  `README.md` is the **shared TCDS manuscript's own title** and is correct there — those are shared /
  TCDS artifacts (not TAFFC-facing) and were not changed.

## Page counts
`main_taffc.pdf` **11** · `main_taffc_blind.pdf` **11** (within the verified 12-page free limit) ·
shared `main.pdf` 10 · `supplement.pdf` 6.

## TCDS preservation status
All shared TCDS files are byte-identical to `main` (`main.tex/.pdf`, `main_blind`, `supplement.pdf`,
TCDS cover letter, `SUBMISSION_PROFILE.md`, `CFP_ALIGNMENT_COGNEURO_BCI.md`, readiness checklist). No
figures or tables changed.

## Figure 4 / Figure 6 grayscale-risk assessment (Task C)
1. **Exact figures.** Figure 4 = `figures/tcds_ready9/analysis/ana03_robustness_degradation_curves.pdf`
   (robustness degradation curves); Figure 6 =
   `figures/tcds_ready9/analysis/ana07_closed_loop_success_by_policy.pdf` (closed-loop success by
   policy). Both plot multiple series distinguished by **color with an identical circular marker**
   (`marker="o"`, default matplotlib color cycle), so the middle series are hard to separate in
   grayscale and for color-blind readers.
2. **Severe enough to delay submission?** **No.** Submitted PDFs are viewed and printed in color by
   the journal; the issue is grayscale / B&W-print and color-blind accessibility, not correctness or
   content. The figures are correct and legible in color — a polish/accessibility item, not a blocker.
3. **TAFFC-specific copies without touching shared TCDS figures?** **Yes.** Restyled copies can be
   written to a new `figures/taffc/` directory and that directory prepended to the `\graphicspath` in
   `main_taffc.tex` (or referenced by distinct filenames), leaving the shared
   `figures/tcds_ready9/analysis/*.pdf` byte-identical for the TCDS manuscript.
4. **Exact source scripts.** `experiments/tcds_ready9/07_generate_analysis_plots.py` — `ana03_04()`
   (Figure 4) and `ana07_08_09()` (Figure 6). Inputs are the pre-aggregated CSVs
   `robustness_summary.csv` and `closed_loop_policy_summary.csv`.
5. **Can regeneration preserve all numbers while changing only style?** **Yes.** The script *reads*
   the aggregate CSVs and only renders them; it does not recompute results. A style-only pass would
   edit the plotting calls — `marker=` (distinct shapes), `linestyle=` (solid/dashed/dotted/dash-dot),
   grayscale-safe greys/contrast, direct line labels or a clearer legend, and panel font sizes —
   while leaving the data arguments (`cc["lvl"]`, `cc["balanced_accuracy_mean"]`, `d["epsilon"]`,
   `d["success_rate"]`, `yerr`) untouched. Re-rendering the same CSV yields identical data
   coordinates, so every numerical value is preserved.

## Recommendation: figure-polish PR before submission?
**Recommended but NOT blocking.** A TAFFC-specific figure-polish pass is safe and well-scoped
(style-only; TAFFC figure copies in `figures/taffc/`; shared TCDS figures and all numbers untouched)
and would improve grayscale / color-blind legibility of Figures 4 and 6. It is **not required before
submission** because the figures are correct and legible in color. Do it as a follow-up if you want
grayscale-print robustness; otherwise the package is submission-ready as is.

## Validation (this audit)
- Both variants compile to **11 pages**; 0 undefined references.
- Shared TCDS files and all figures/tables byte-identical to `main`; numeric-token multiset of
  `main_taffc.pdf` unchanged.
- No forbidden claims introduced; tool-attribution / session-reference scan clean.
- Documentation-only changes; no figures regenerated; history not rewritten.
