# TAFFC Figure-Polish Report

**Date:** 2026-06-12 · **Branch:** `submission/taffc-figure-polish` (from `main` `10c7750`)

Style-only, TAFFC-specific polish of **Figure 4** (`ana03_robustness_degradation_curves`) and
**Figure 6** (`ana07_closed_loop_success_by_policy`) for grayscale readability and colour-blind
accessibility. **No numerical value, CSV, experiment, table, claim, or shared TCDS figure was
changed.**

## What changed (style only)
Both figures previously distinguished their series **by colour with an identical circular marker**,
which collapses in grayscale and for colour-blind readers. The polished versions distinguish every
series by a **unique (line style, marker shape)** pair — so they remain separable with no colour at
all — plus:

- a colour-blind-safe (Okabe-Ito) palette with black marker edges for added contrast;
- consistent typography, line weights, and marker sizes;
- light reference grids drawn behind the data;
- clearer axis labels ("perturbation level") and prettified policy names (EFE, epistemic-only,
  pragmatic-only, …);
- Figure 4: a single shared legend below the three panels;
- Figure 6: a semi-opaque white-framed legend that reads cleanly over the data;
- vector PDF export with embedded fonts.

The plotted data — configs/policies, x-values, y-values, and Wilson 95% CI error bars — are
**identical** to the shared figures; only visual style differs.

## How it was done (shared generator untouched)
A **new, standalone TAFFC restyle script**, `experiments/tcds_ready9/taffc_figure_polish.py`, reads
the **same aggregate CSVs** the shared generator uses
(`outputs/tcds_ready9/analysis/robustness_summary.csv` and `closed_loop_policy_summary.csv`),
replicates the shared selection logic exactly (Fig. 4: `pass == representation_level`, configs
A0/A1/A2/A3/A8, panels amplitude-noise / channel-dropout / graph-perturbation; Fig. 6: per-policy
`success_rate` vs `epsilon` with Wilson CI), and writes **only** to `figures/taffc/`. The shared
`experiments/tcds_ready9/07_generate_analysis_plots.py` and the shared figure PDFs are not modified.

The TAFFC variant's `\graphicspath` was **prepended** with `{figures/taffc/}{../figures/taffc/}` in
`main_taffc.tex` and `main_taffc_blind.tex` only, so the variant resolves `ana03…` / `ana07…` to the
polished copies while every other figure — and the entire TCDS manuscript — keeps the shared
versions.

## Outputs
- `figures/taffc/ana03_robustness_degradation_curves.pdf` — Figure 4, polished
- `figures/taffc/ana07_closed_loop_success_by_policy.pdf` — Figure 6, polished
- `experiments/tcds_ready9/taffc_figure_polish.py` — restyle script (additive)
- `manuscript/taffc/main_taffc.tex`, `manuscript/taffc/main_taffc_blind.tex` — graphicspath prepend only
- recompiled `manuscript/taffc/main_taffc.pdf`, `manuscript/taffc/main_taffc_blind.pdf`

## Numerical preservation (verified)
The restyle script reads the CSVs and plots the identical columns; spot cross-checks against the CSVs
confirm exact equality — e.g. closed-loop `EFE` success rate {0.862, 0.8467, 0.8447, 0.808, 0.8047}
at ε ∈ {0, 0.1, 0.2, 0.3, 0.4}, and robustness `A0` balanced accuracy {0.3953, 0.4242, 0.4718}
(clean 0.4851) under amplitude noise — matching `closed_loop_policy_summary.csv` and
`robustness_summary.csv` exactly. The "clean" baseline maps to a non-numeric perturbation level and
is rendered identically to the shared generator (NaN x-coordinate).

## Validation
1. Both polished figure PDFs exist in `figures/taffc/`. ✓
2. Shared TCDS figure PDFs (`figures/tcds_ready9/analysis/ana03…`, `ana07…`) byte-identical to `main`. ✓
3. Source CSVs unchanged (0 `outputs/` CSVs modified). ✓
4. Plotted numerical values unchanged (CSV cross-check). ✓
5–6. `main_taffc.tex` and `main_taffc_blind.tex` compile twice, clean, 0 undefined references. ✓
7. Both variants **11 pages** — within the verified 12-page TAFFC free formatted limit. ✓
8. Shared TCDS manuscript/package files and the `main.tex` graphicspath unchanged (`main.tex` has no
   `figures/taffc` reference). ✓
9. No numerical result, table, experiment output, subject count, or observation count changed. ✓
10. No forbidden claims and no AI-assistant/vendor/session references in artifact-facing files. ✓

## Scope
TAFFC-variant-only, style-only. The change set is: two polished figure PDFs, one restyle script, a
one-line graphicspath prepend in each variant, and the recompiled variant PDFs. Shared TCDS figures,
CSVs, tables, manuscript, and package are byte-identical to `main`.
