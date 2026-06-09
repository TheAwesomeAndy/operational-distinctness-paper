# Ten-page compression report

Editorial compression only; **no new experiments**. Target met:
**`manuscript/main.pdf` = 10 pages** (the IEEE TCDS no-fee Regular-paper limit),
0 undefined references, no visible overflow, privacy + language gates pass.

## Page-policy basis
IEEE TCDS Regular paper: 10-page base limit; mandatory overlength charge
$200/page beyond 10; hard ceiling under 15 pages. The author will not pay
overlength charges, so the submission candidate is compressed to 10 pages.

## Versions
- `manuscript/main.pdf` — **10-page submission candidate** (no overlength fee).
- `manuscript/main_full_internal.pdf` — 17-page full-evidence record, title-marked
  internal/not-for-submission.
- The intermediate 13-page version is preserved in git history (commit `e21ebf2`).

## Compression actions applied
1. **Observation figures reduced.** Main text keeps one observed-object figure
   (`fig_reservoir_dynamics`: raster, membrane, tPLV, BSC6). Moved out of main:
   the dataset-integrity QC figure (`obs00`, now covered by the QC summary +
   reproducibility panel) and the open-loop sequential-evidence figure
   (`fig_embodied_loop`).
2. **Related Work compressed** from four subsections (+ a novelty-boundary table)
   to two paragraphs covering spiking/reservoir computation, graph-structured
   EEG, embodied/perception–action affective AI, and the EEG/ERP baseline norm.
3. **Provenance + reproducibility merged** into the single reproducibility panel;
   no separate provenance table in the main text.
4. **Runtime/resource moved out**, reduced to one sentence ("reports
   runtime/resource estimates but does not claim measured hardware energy"); the
   detailed table remains in the internal version.
5. **Discussion tightened** from eight subsections to three paragraphs (what the
   ablation establishes; what robustness says about operating regimes; what the
   closed-loop result establishes and does not), plus one consolidated Limitations
   paragraph; removed the hardware, plasticity, embodied-comparison, positioning,
   and clinical-interpretation subsections and the repeated summary paragraphs.
6. **ERP baseline compressed** to the static-anchor statement (no full narrative).
7. **Primary ablation only.** The multi-seed mechanism ablation is the sole
   main-text ablation; the 10-fold result is one corroboration sentence.
8. **Open-loop compressed into the closed-loop setup** (one sentence); the
   closed-loop EFE result and figure are retained.

## Main-text floats after compression
- **Figures (5):** architecture overview (Methods); `fig_reservoir_dynamics`,
  `ana01` (mechanism ablation), `ana03` (robustness), `ana07` (closed-loop).
- **Tables (6):** protocol, feature blocks, neural mechanisms (Methods);
  mechanism ablation, closed-loop policy, reproducibility panel (Results).

## Verification
- 10 pages; 0 undefined references; no overflow >30 pt.
- No banned terminology, no private paths, no public-data wording (gates pass).
- Claim discipline preserved: clinical = exploratory; no diagnostic/biomarker,
  physical-robot, measured-energy, SOTA, or public-data claims; restricted-data
  language retained.
- One dangling reference created by removing the plasticity subsection was
  repaired (reworded in the reservoir-formulation text).

## Not done by design
- No new experiments. Numbers are the submission-profile (5-seed) results.
- Double-blind anonymization remains for the final packaging step.
