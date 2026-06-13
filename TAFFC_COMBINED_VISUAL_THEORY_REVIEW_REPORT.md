# TAFFC Combined Visual + Theory Review Report

Integrated validation of PR #27 in its current state, after PR #28 was merged
into PR #27's feature branch. PR #27 is therefore no longer overview-only: it is
the combined Phase 1 (early
evidence-stream overview figure) + Phase 2 (dynamical-system theory framing,
novelty paragraph, two conceptual figures) pull request.

- Validated head: merge commit `b4f5823` (`Merge pull request #28 ...`).
- Base for review: `main` (`c73c196`).
- PR #27 state: open, unmerged, `mergeable_state: clean`.
- `main` is unchanged by either phase; both phases live only on the PR #27 head.

## 1. Combined changed-file inventory (vs `main`)

| File | Status | Category |
|---|---|---|
| `manuscript/taffc/main_taffc.tex` | modified | manuscript (non-blind) |
| `manuscript/taffc/main_taffc_blind.tex` | modified | manuscript (blind) |
| `manuscript/taffc/main_taffc.pdf` | modified | rendered artifact |
| `manuscript/taffc/main_taffc_blind.pdf` | modified | rendered artifact |
| `figures/taffc/fig_overview_evidence_streams.pdf` | added | Fig. 1 (Phase 1) |
| `figures/taffc/fig_observation_model.pdf` | added | Fig. 2 (Phase 2) |
| `figures/taffc/fig_operational_distinctness_map.pdf` | added | Fig. 3 (Phase 2) |
| `experiments/tcds_ready9/generate_taffc_overview_figure.py` | added | figure generator |
| `experiments/tcds_ready9/generate_taffc_theory_figures.py` | added | figure generator |
| `TAFFC_EARLY_OVERVIEW_FIGURE_REPORT.md` | added | validation report |
| `TAFFC_THEORY_VISUAL_NOVELTY_REPORT.md` | added | validation report |
| `TAFFC_FIGURE_POLISH_REPORT.md` | modified | validation report |
| `.gitignore` | modified | ignore `.bbl`/`.blg` build artifacts |

No CSV, no `tables/**`, no shared-TCDS manuscript source (`manuscript/main.tex`,
`manuscript/main_blind.tex`), and no data artifacts (`*.pkl`) appear in the diff.

## 2. Compile validation (clean rebuild of both variants)

Build directory `manuscript/`, output to `taffc/`, three `pdflatex` passes each.

| Check | `main_taffc` | `main_taffc_blind` |
|---|---|---|
| Compiles | yes | yes |
| Page count | 12 | 12 |
| Undefined references | 0 | 0 |
| Multiply-defined labels | 0 | 0 |
| Undefined citations | 0 | 0 |
| Repeated `\bibitem` keys | 0 (55 unique) | 0 (55 unique) |
| Overfull hboxes | 2 (cosmetic) | 1 (cosmetic) |

The two/one overfull hboxes are minor typographic warnings (no content loss, no
error) and are not introduced defects.

## 3. Figure number to page placement (from `.aux` `\newlabel`)

Identical in both variants:

| Figure | Label | Page |
|---|---|---|
| Fig. 1 — evidence-stream overview | `fig:overview` | 2 |
| Fig. 2 — observation model | `fig:obsmodel` | 3 |
| Fig. 3 — operational-distinctness map | `fig:distinctmap` | 4 |
| Fig. 4 — staged pipeline (was Fig. 2 pre-Phase-1) | `fig:source_pipeline` | 7 |
| Fig. 5 — reservoir dynamics | `fig:reservoir_dynamics` | 8 |
| Fig. 6 — layer redundancy (CKA) | `fig:cka` | 9 |
| Fig. 7 — robustness degradation | `fig:ana03` | 10 |
| Fig. 8 — closed-loop accumulation | `fig:embodied_loop` | 10 |
| Fig. 9 — closed-loop success by policy | `fig:ana07` | 10 |

The early overview figure is top of page 2; the two new conceptual figures fall
on pages 3 and 4; downstream figures renumbered automatically via `\ref`.

## 4. Content integrity

- **No numerical result changes.** The only numeric-bearing lines added to the
  manuscript source are two new *theory* equations — the latent dynamical-system
  observation model `x_{t+1}=f_theta(x_t,u_t)+eta_t, X_t=H x_t+epsilon_t` and the
  operational-distinctness distance vector `Delta_{ij}`. No reported accuracy,
  statistic, p-value, count, or table value was altered.
- **No table changes.** All result tables are `\input` from unchanged files.
- **No CSV / data changes.**
- **No shared-TCDS changes.** The shared TCDS manuscript sources and shared
  figure/table trees are untouched.
- **Cross-references intact.** The `\section{ARSPI-Net Layer Formulation}`
  carries `\label{sec:formulation}`; the new overview-figure caption and novelty
  paragraph reference it and `\ref{eq:phi}`/`\ref{fig:distinctmap}` resolve
  (0 undefined references).
- **Panel (c) wording preserved.** The overview-figure caption labels panel (c)
  a *diagnostic* BSC6 reservoir projection recomputed from the channel-mean ERP
  for visual overview only, explicitly distinguished from the production
  per-channel embedding `E` defined in the Layer Formulation. No "produces
  embedding E" phrasing exists.

## 5. Blinding integrity

The blind/non-blind source differ only in blinding substitutions: dataset name
("SHAPE" -> "measured"), originating institution -> generic phrasing, and
self-citation author names -> "Anonymous Authors". No author, affiliation, or
dataset-identity content leaks into the blind variant.

## 6. Privacy / provenance scan (changed text artifacts)

Scan of changed `.tex`, `.md`, and `.py` files for AI-assistant / vendor /
session strings, absolute home paths, and PHI / subject-ID patterns: no
introduced occurrences. Residual matches are limited to (a) report sentences
that *assert* the absence of such strings, (b) the Greek operator symbol matched
case-insensitively, and (c) the pre-existing cohort-exclusion note ("subject 127
excluded for a recording anomaly"), which is standard aggregate cohort reporting
present in `main` and is not a subject identifier. Reported outputs remain
aggregate and deidentified.

## 7. Combined-vs-split recommendation

Recommendation: **review PR #27 as a single combined PR.**

Rationale:
- Both phases are additive, visual/theoretical, and share the same constraints
  (no result, table, CSV, or shared-TCDS changes); they validate cleanly
  together at 12/12 pages with zero reference/label/citation/bibitem defects.
- PR #28 is already merged into PR #27's branch. Splitting back would require
  reverting the merge commit and reopening a deleted-branch PR — high effort,
  added history churn, and no validation benefit.

One caveat for the reviewer: PR #27's **title and description still describe
only Phase 1** (the overview figure). They do not mention the Phase 2 theory
subsection, novelty paragraph, or the two conceptual figures now included. The
PR metadata should be updated to reflect the combined scope before review so the
description matches the diff. (This report does not change PR metadata.)

## 8. Status

PR #27 is validated as a combined PR and is a merge candidate **after** a visual
review of the rendered PDFs and a decision on the metadata update above. Not
merged; no automatic merge performed.
