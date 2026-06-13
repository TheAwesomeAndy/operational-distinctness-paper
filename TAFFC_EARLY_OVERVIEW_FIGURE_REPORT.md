# TAFFC Early Overview Figure — Validation Report

Scope: the early evidence-stream overview figure (Fig. 1) added to the TAFFC
manuscript variants. This report documents the change introduced on the feature
branch and the validation performed.

## Branch and commits

- **Branch:** `claude/ieee-manuscript-overview-wjiis7`
- **Base commit (merge-base with `main`):** `c73c19613311a2206b95d3815e27751515acaff0`
- **Head commit (figure caption-precision revision):** `0b9f294765fa8a82a46d0abf5cdace970b45f745`
- Intermediate commits on the branch:
  - `f6a3bfe` — add early evidence-stream overview figure (Fig. 1) and wire it into both TAFFC variants
  - `4fcc85a` — ignore LaTeX `.bbl`/`.blg` build artifacts
  - `0b9f294` — refine Fig. 1 caption precision and panel (c) labeling

## Files changed (vs. base)

- `manuscript/taffc/main_taffc.tex` — figure float + forward reference + caption; `\label{sec:formulation}` added
- `manuscript/taffc/main_taffc_blind.tex` — identical edits (blind variant)
- `manuscript/taffc/main_taffc.pdf`, `manuscript/taffc/main_taffc_blind.pdf` — recompiled
- `figures/taffc/fig_overview_evidence_streams.pdf` — new figure (Fig. 1)
- `experiments/tcds_ready9/generate_taffc_overview_figure.py` — new generator script
- `.gitignore` — ignore LaTeX `.bbl`/`.blg` artifacts

No other files were modified.

## Figure composition: data-derived, schematic, or mixed

**Mixed.**

- **Top band — schematic:** a conceptual left-to-right flow (Affective ERP →
  LIF reservoir + BSC$_6$ → evidence streams $E, D, T, C$ → ablation /
  perturbation robustness / closed-loop accumulation), drawn as labeled boxes
  and arrows. No data.
- **Bottom band — data-derived (three panels):**
  - (a) trial-averaged ERPs by affective condition — real, from `X_ds`.
  - (b) a LIF-reservoir spike raster for one exemplar observation — real;
    the reservoir is run on `X_ds` using the upstream reservoir spec.
  - (c) a **diagnostic** BSC$_6$ reservoir projection (PCA-2), recomputed from
    the **channel-mean** ERP of every observation, **for visual overview only**.
    This is explicitly *not* the production per-channel ARSPI-Net spike
    embedding $E$; the production $E$ used in all reported analyses is defined in
    the ARSPI-Net Layer Formulation (`\label{sec:formulation}`). Both the figure
    panel title ("BSC$_6$ projection (diagnostic)") and the caption state this.

## Exact input file used

- `data/shape_features_211.pkl`, keys used: `X_ds` (downsampled, z-scored EEG,
  shape `(633, 256, 34)`), `y` (condition labels), `subjects`.
- The reservoir parameters (`N_RES=256`, `beta=0.05`, `theta=0.5`, `seed=42`,
  BSC$_6$ window `t in [10, 70]`) are imported from
  `prepare_inputs/extract_ch5_features.py`.
- No other data files are read by the generator.

## Privacy status

- The generator **reads only the local pickle** above and writes only to
  `figures/taffc/`. It does not write any data file, log, or report containing
  subject-level content.
- **Generator output, script, and figure are clean:** no absolute filesystem
  paths, no raw subject IDs, and no clinical/PHI fields appear in the script,
  in the figure PDF, or in the figure PDF metadata (Producer/Creator are
  `Matplotlib`; the manuscript PDFs report `TeX`/`pdfTeX`). The exemplar for
  panel (b) is selected by a hashed subject identifier and no identifier is
  printed or embedded.
- **Repository-level note (pre-existing, not introduced by this change):** the
  repository tracks the input data under `data/` (the feature pickle, a
  per-subject clinical CSV, and raw per-subject `.txt` recordings). Per the
  repository owner, this committed data is intended and approved for public
  release. This figure change adds no new data and no new identifiers beyond
  the aggregate visual panels described above.

## Page count and figure placement

| Variant | Pages | Fig. 1 placement |
|---|---|---|
| `main_taffc.pdf` | 12 | top of page 2 |
| `main_taffc_blind.pdf` | 12 | top of page 2 |

Both are within the TAFFC 12-page free limit. (The manuscript was 11 pages
before this figure; the full-width float and its caption add one page of
reflow.)

## Validation results

- `main_taffc.tex`: compiles (pdflatex ×2 + bibtex), **0 undefined references**,
  **0 multiply-defined labels**, figure embedded, 12 pages.
- `main_taffc_blind.tex`: same — 0 undefined, 0 multiply-defined, 12 pages.
- The `sec:formulation` cross-reference in the caption resolves (no `??`).
- Figure regenerates deterministically from the committed generator and the
  committed input pickle.

## Explicit no-change statement

No numerical result, CSV, subject count, observation count, statistical
conclusion, table, shared TCDS file, or shared TCDS figure was changed by this
work. The change is limited to: one new figure and its generator, the wiring
of that figure into the two TAFFC `.tex` variants (with one forward-reference
sentence and one section label), the recompiled TAFFC PDFs, and a `.gitignore`
entry for LaTeX build artifacts. The TCDS variants (`manuscript/main.tex`,
`manuscript/main_blind.tex`), the shared tables under `tables/tcds_ready9/`,
and the shared figure trees were not touched.
