# TAFFC_ACCEPTANCE_STRATEGY_GATE_V2_REPORT

Acceptance-**strategy** audit (not merely formatting) for the ARSPI-Net article targeting
**IEEE Transactions on Affective Computing (TAFFC)**. Every judgment is tied to a file, line, count,
or compiled artifact. This pass is **audit-first**: no manuscript `.tex`, PDF, figure, table, output,
or class file was changed. New files are governance/process docs only.

## Provenance
| Field | Value |
|---|---|
| Repository | `TheAwesomeAndy/operational-distinctness-paper` |
| Audit branch | `submission/taffc-acceptance-strategy-gate-v2` (from `main`) |
| `main` commit | `bd29431d9ff797b5657ad848da2d7fd78e9699c4` |
| Date/time | 2026-06-16 ~17:45 UTC |
| Manuscript | `manuscript/taffc/main_taffc.tex` (+ blind twin), 12 pp, vendored IEEEtran V1.8b |

## Files inspected
`manuscript/taffc/main_taffc.tex` (abstract, intro, contributions, related work, results, discussion,
limitations, conclusion, data-availability, bibliography), `manuscript/taffc/main_taffc_blind.tex`,
`manuscript/COVER_LETTER_TAFFC_FINAL.md`, the `experiments/tcds_ready9/*.py` evaluation scripts,
and prior validation reports.

---

## Gate table

| Phase | Item | Status | Evidence |
|---|---|---|---|
| 1.1 | Compound research gap | **PASS** | Intro builds endpoint-accuracy limitation + partially-observed/noisy bottleneck + operational-distinctness + closed-loop accumulation, then "closes that gap with five contributions" (lines 42, 51, 55, 59) — not a trivial "few papers used X" gap |
| 1.2 | Single infectious idea | **PASS** | "measurement substrate" recurs in abstract, intro (51), discussion (329); not centered on classifier superiority/diagnosis/GNN-novelty/full-active-inference/online-BCI/hardware |
| 1.3 | Wardle's Law (true + so-what) | **PASS** | each result paired with interpretation: ablation→"sub-threshold classifier can still be operationally distinct" (321); perturbation→"stream-specific operating regime" + hardware/latency disclaimer (323); closed-loop→"usable evidence estimator… precursor, not full implementation" (325); explicit adaptive-BCI so-what (327) |
| 2.1 | Journal fit beyond impact factor | **PASS** | cover letter "Fit with this journal" maps DEAP/MAHNOB, continuous affect, connectivity, neural-encoding, biomarker measurement; discussion (329) frames it as extending the journal's trajectory |
| 2.2 | Reference venue mining | **PASS** | of 55 bibitems: **18** IEEE Trans. Affective Computing, **24** IEEE Trans (any), **31** IEEE, ~9 neuromorphic/reservoir, ~16 graph/connectivity — TAFFC strongly represented; no citation padding needed |
| 2.3 | Recent-publication duplication | **NEEDS WEB VERIFICATION** | no duplication evident from local material (verified TAC comparators are positioned as complementary), but a live novelty/duplication check requires author/web verification not performed here |
| 3.1 | Claims traceable to figures/tables | **PASS** | abstract/intro/conclusion claims map to Fig. 1 overview, Fig. 3 distinctness map, CKA (Fig.), ablation/perturbation/closed-loop tables and figures |
| 3.2 | Inside-out (wall-to-fish) coherence | **PASS** | figures → mechanisms → results/tables → discussion-of-limits → title/abstract aligned; no early claim left unsupported |
| 4.1 | Title (8–13 words, searchable, no overclaim) | **PASS** | "Spatiotemporal Characterization of Affective EEG Dynamics Using a Neuromorphic Reservoir–Graph Framework" = **11 words**, technical, no overclaim |
| 4.2 | Abstract four-unit structure, no overclaim | **PASS** | problem→motivation→E/D/T/C solution+protocol→"measurement substrate, not classifier; simulation not online BCI; SHAPE-regime-bounded" (line 34) |
| 4.3 | Page-1/2 hook | **PASS** | no "organized as follows"; Fig. 1 floats to page 2; first page states problem/gap/object/bounded claim |
| 4.4 | Crunchy, refutable contributions | **PASS** | 5 contributions, each section/figure-linked and measurable (lines 60–66) |
| 5.1 | Related-work placement | **PASS** | short launchpad in §2 after the object + Fig. 1 are established; deeper critical contrast in Related Work + Discussion |
| 5.2 | Critical comparison by technical axes | **PASS** | organized by compact-decoder / graph-spatial / neuromorphic-reservoir / interpretability axes, not a chronological dump |
| 5.3 | Credit generously, novelty intact | **PASS** | "canonical", "established", "closely related"-style language; F2FNet etc. positioned as comparators, not strawmen or outperformed |
| 5.4 | Weakness ownership in place (not buried) | **PASS** | baseline-stronger (×5), offline/recorded (×11), access-controlled (×5), no-online-BCI (×5), exploratory clinical (×6); repeated in intro, discussion, limitations, conclusion |
| 6.1 | Reproducibility / no leakage | **PASS** | `StratifiedGroupKFold` + `groups=` across all analysis scripts; subject-grouped; private data not committed; class pinned (vendored V1.8b) |
| 6.2 | Statistical reporting | **PASS (optional hardening)** | balanced accuracy + CIs + permutation inference + multi-seed reported; macro-F1/ROC-AUC/confusion matrices minimal — optional reviewer-anticipation additions, not fabricated |
| 6.3 | Visual polish / grayscale | **PASS (recommend proof-glance)** | Figs 4/6 grayscale-polished; Fig. 1/2/3 use color + text/shape labels (not color-alone); page 12 = clean references page; one trivial 4.8 pt overfull |
| 7.1 | Cover letter (sober, fits TAFFC) | **PASS (optional trim)** | addresses Hoey; what/why/fit + prior-work (LISAT) disclosure; no breakthrough/SOTA/diagnostic/real-time except as negations. ~587 words / ~6 labeled paragraphs — longer than the strict "three short paragraphs" ideal; optional to tighten to one page |
| 7.2 | Review-response protocol exists | **PASS** | added `REVIEW_RESPONSE_PROTOCOL.md` |
| 7.3 | Citation-integrity policy exists | **PASS** | added `CITATION_INTEGRITY_POLICY.md` |
| 8.1 | Root agent specification | **PASS (named `AGENTS.md`) / NEEDS AUTHOR DECISION on filename** | added vendor-neutral `AGENTS.md` (venue, central idea, files, data restrictions, no-fabrication, forbidden claims, validation commands, blind caution, audit-before-edit, no-auto-merge, no-raw-data/paths, no-vendor-footers). Named `AGENTS.md` rather than the tool-branded filename specified in the task, to honor the standing no-vendor-reference rule — rename only if you want the branded filename |
| 8.2 | Verifier layer (facts not memory) | **PASS** | `AGENTS.md` mandates using actual files/guidelines/model papers/validation reports, and reporting NEEDS WEB VERIFICATION when guidelines/recent-fit are unavailable |
| 8.3 | Environment layer at repo root | **PASS** | `AGENTS.md` lives at repo root for ingestion before editing |
| 9 | Post-publication amplification plan | **PASS** | added `POST_ACCEPTANCE_VISIBILITY_PLAN.md`; no promotion language placed in the manuscript |

---

## Manuscript edits recommended
**None required.** No strategy FAIL was found; the manuscript already implements the gap/idea/hook/
contribution/ownership structure. No manuscript text was edited (audit-first; nothing reached the
edit threshold).

## Documentation-only edits made on this branch
- `TAFFC_ACCEPTANCE_STRATEGY_GATE_V2_REPORT.md` (this file)
- `REVIEW_RESPONSE_PROTOCOL.md`
- `CITATION_INTEGRITY_POLICY.md`
- `AGENTS.md` (vendor-neutral root agent spec)
- `POST_ACCEPTANCE_VISIBILITY_PLAN.md`

## Optional reviewer-anticipation hardening (deferrable, never fabricated)
- Add macro-F1 / confusion matrices alongside balanced accuracy (requires a script re-run; report any change).
- Add an endpoint deep-learning baseline (EEGNet / TCN) under the identical subject-grouped split.
- Tighten the cover letter toward one page / three paragraphs.
- Grayscale proof-print of Fig. 1/2/3.
- A consolidated claim→script→output→figure reproduction map.

## Author decisions outstanding
1. **Blind model** (single vs double) on ScholarOne → upload `main_taffc.pdf` or `main_taffc_blind.pdf`.
2. **Agent-spec filename**: keep vendor-neutral `AGENTS.md` (default) or rename to the tool-branded filename named in the task spec.
3. **Recent-publication/novelty** check needs a live literature/web confirmation.

## Final verdict
**Submit after portal blind-review selection.** The article implements top-tier acceptance strategy,
not just formatting: a compound gap, a single transferable idea, Wardle-compliant results, genuine
TAFFC fit, an inside-out structure with an early hook, crunchy contributions, axis-organized related
work, and disciplined weakness ownership. The remaining pre-upload action is the portal blind-review
choice; all other items are optional post-submission hardening or governance docs, none blocking.
