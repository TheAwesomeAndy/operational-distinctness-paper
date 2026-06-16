# SUBMISSION_GATE_TAFFC_FINAL_REPORT

Strict pre-submission audit and hardening gate for the ARSPI-Net article targeting
**IEEE Transactions on Affective Computing (TAFFC)**. Every claim below is tied to a file, command,
or compiled artifact. This pass is **audit-only**: no manuscript source, figure, table, output, or
class file was modified on this branch (the report file is the only addition).

## Provenance lock (Phase 0)

| Field | Value |
|---|---|
| Repository | `TheAwesomeAndy/operational-distinctness-paper` |
| Audit branch | `submission/taffc-submission-gate-audit` (from `main`) |
| `main` commit audited | `3085434e11bf7e1c95d2be8309237283056d273a` (post-PR #31 merge) |
| Date/time | 2026-06-16 ~16:35 UTC |
| Python | 3.11.15 |
| LaTeX engine | pdfTeX 3.141592653-2.6-1.40.25 (TeX Live 2023/Debian) |
| Resolved class | `./IEEEtran.cls` (vendored), version **V1.8b (2015/08/26)** |

- PR #27 (visual-theory scaffold) merged into `main`: **confirmed** (`git log` shows the merge).
- PR #29 (IEEEtran vendoring) merged into `main`: **confirmed**.
- PR #30 (duplicate early overview) **closed, not merged** — superseded by #27.
- PR #31 (README page count 11→12) **merged** into `main` (`3085434`); the TAFFC variant now reads 12 pp.
- No open PR touches the TAFFC manuscript, figures, class, tables, or outputs. This audit report is the
  only open PR (#32) and adds no such file.

## Files inspected
`manuscript/taffc/main_taffc.tex`, `manuscript/taffc/main_taffc_blind.tex`,
`manuscript/taffc/main_taffc.pdf`, `manuscript/taffc/main_taffc_blind.pdf`,
`manuscript/COVER_LETTER_TAFFC_FINAL.md`, `manuscript/supplemental_risk_reduction/supplement.pdf`,
`README.md`, `manuscript/IEEEtran.cls`, the `experiments/tcds_ready9/*.py` evaluation/figure
scripts, and the tracked `*.md` reports.

## Commands run (representative)
- Compile (×2 passes each, vendored class): `pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp/gate taffc/main_taffc{,_blind}.tex` from `manuscript/`.
- Log scans for `undefined`, `multiply.defined`, `Citation .* undefined`; `\bibitem` duplicate-key check.
- Forbidden/claim-discipline/affective term scans over the manuscript and cover letter.
- Blind identity scan via `pdftotext` + `pdfinfo` metadata on both variant PDFs and the supplement.
- Split-logic scan over evaluation scripts (`StratifiedGroupKFold`, `groups=`).
- Bundle assembly to `/tmp/taffc_submission_bundle` (outside the working tree).

---

## Gate table

| # | Item | Status | Evidence |
|---|---|---|---|
| 0 | Provenance; PR #27/#29/#31 merged, #30 closed; no manuscript-touching open PR | **PASS** | `git log`; the only open PR is this report (#32) |
| 1 | Target is TAFFC; no stale TCDS CFP text in TAFFC-facing files | **PASS** | cover letter names TAFFC/Hoey; the only `tcds` hits are the internal `tcds_ready9` *pipeline path*, not venue text |
| 1 | Vendored IEEEtran V1.8b resolves; no silent system override | **PASS** | `kpsewhich` from `manuscript/` → `./IEEEtran.cls`; version line V1.8b |
| 1 | Both variants compile; 12 / 12 pages | **PASS** | non-blind 12 pp, blind 12 pp, exit 0 |
| 1 | 0 undefined refs / 0 undefined citations / 0 multiply-defined labels / 0 repeated bibitems | **PASS** | log scans = 0/0/0; 0 dup keys of 55 |
| 1 | No TODO/FIXME/draft/placeholder markers | **PASS** | scan clean |
| 1 | README page counts current | **PASS** | PR #31 merged; `main` README now reads **12 pp** for the TAFFC variant (lines 15 and 95); TCDS 10 pp and supplement 6 pp unchanged |
| 2 | Blind PDF free of identity (institution, lab, author, GitHub, paths, subject IDs) | **PASS** | `pdftotext` blind = 0 identity hits; self-cites anonymized ("Anonymous Authors"); PDF metadata has no Author/Title |
| 2 | Non-blind provenance truthful (not deceptively scrubbed) | **PASS** | names SHAPE (×4) + Stony Brook in Data Availability |
| 2 | Supplement blind-safe | **PASS** | shared `supplement.pdf` text = 0 identity hits |
| 2 | Single vs double blind decision | **NEEDS AUTHOR DECISION** | portal-time: pick `main_taffc.pdf` (single) or `main_taffc_blind.pdf` (double) |
| 3 | No forbidden/overclaiming terms | **PASS** | only hit is "powerful" inside the Benjamini–Hochberg reference *title* (allowed) |
| 3 | Not framed as superior classifier; baseline stronger stated; closed-loop offline; clinical exploratory; SHAPE-regime scope sentence present | **PASS** | all present (1/5/11/6/1 respectively) |
| 4 | Affective-computing framing; conditions Negative/Neutral/Pleasant; no unsupported arousal claims; E/D/T/C as observables; κ not over-interpreted | **PASS** | dynamical-system framing ×9; arousal/IAPS = 0; structure-function coupling ×14 |
| 5 | No "organized as follows" roadmap; refutable 5-contribution list; Related Work after the object | **PASS** | roadmap = 0; 5 contributions; Related Work is §2 |
| 6 | No subject leakage; subject-grouped (not falsely LOSO); uncertainty reported | **PASS** | `StratifiedGroupKFold` + `groups=` in all analysis scripts; "each subject appears in either train or test"; balanced accuracy + CIs + permutation + multi-seed; LOSO not claimed |
| 6 | macro-F1 / ROC-AUC / confusion matrices | **NEEDS AUTHOR DECISION** | balanced accuracy is primary with uncertainty; these secondary metrics are minimal — optional to add if a reviewer asks |
| 7 | Run baselines present (ERP-amplitude, band-power, ablations, negative controls) | **PASS** | all present |
| 7 | Deep baselines (EEGNet/TCN/transformer) | **NEEDS AUTHOR DECISION** | cited as Related Work, **not run**; no SOTA claim, not framed as outperformed — honest. Optional post-submission hardening; do not fabricate |
| 8 | Perturbation families (temporal/amplitude/channel/graph) reported by stream | **PASS** | all four present; reported as operating regimes |
| 8 | No unsupported EMG / latency robustness claims | **PASS** | EMG = 0; the one "latency" hit is an explicit *disclaimer* ("do not constitute hardware-energy, latency, or deployment results") |
| 9 | Exactly one early overview (`fig:overview`); Fig. 1/2/3 distinct; page 12 not cramped | **PASS** | 1 `fig:overview` per variant; labels `fig:overview`/`fig:obsmodel`/`fig:distinctmap`; page 12 = clean references page; 1 trivial 4.8 pt overfull |
| 9 | Grayscale legibility | **PASS (recommend proof-glance)** | Figs 4/6 were grayscale-polished; Fig. 1/2/3 use color accents **plus** text/shape labels (not color-alone). A final grayscale proof-print of Fig. 1/2/3 is advisable |
| 10 | Data Availability / Ethics section; restricted-data language; no PHI in artifacts | **PASS** | `\section*{Data Availability and Ethics}` present; restricted + DUA + aggregate-deidentified + exploratory clinical labels |
| 10 | No AI/vendor/session footers in tracked reports | **PASS** | scan of all tracked `*.md` = clean |
| 10 | Consolidated reproduction map (claim→script→output→figure→privacy) | **NEEDS AUTHOR DECISION** | README "Reproducing" section + scripts + referenced manifests exist; no single consolidated map doc — optional to add |
| 11 | Clean TAFFC-facing submission bundle; correct page counts; clean PDF metadata; TAFFC cover letter | **PASS** | `/tmp/taffc_submission_bundle`: `main_taffc.pdf` (12), `main_taffc_blind.pdf` (12), `supplement.pdf` (6), cover letter; metadata carries no author/path; cover letter TAFFC, zero TCDS/CFP |
| 11 | Bundle excludes TCDS CFP, private data, pickles, raw EEG, logs | **PASS** | bundle contains only the four curated files |

---

## Changes made
- **None to the manuscript, figures, tables, outputs, or class files.** This branch adds only this
  report. The audit confirmed the existing `main` artifacts already satisfy the gate.

## Changes deliberately **not** made
- Did not rewrite the abstract/intro (audited clean; no overclaim; contribution-first already in place).
- Did not run EEGNet/TCN/transformer baselines, EMG-artifact, processing-delay, or LOSO experiments
  (would be new experiments / fabricated baselines — out of scope per the absolute rules). They are
  reported as honestly absent, not claimed.
- Did not add macro-F1/ROC-AUC/confusion matrices (no script regeneration in scope).
- Did not alter the README page count in this report's diff; that fix landed via PR #31 (merged), now reflected on `main` (12 pp).
- Did not remove or alter the preserved TCDS fallback material.

## Remaining risks (all non-blocking)
1. **Blind model** must be confirmed on ScholarOne; pick the matching PDF (the one remaining pre-upload action).
2. Reviewer may request **macro-F1 / confusion matrices** or **deep-learning baselines**; both are
   defensible to defer given the observable-map framing, but are the most likely revision asks.
3. A **grayscale proof-print** of Fig. 1/2/3 is advisable before upload.
4. The PR #30 close comment on GitHub carries an auto-appended tooling footer that cannot be edited
   via the available tools — remove it in the GitHub UI if the PR trail will be shared.

## Final recommendation
**Submit after portal blind-review selection.** With PR #31 merged, the documentation is current and
the manuscript is submission-ready: it compiles cleanly against the pinned class, sits at 12 pages
within the free limit, is leakage-safe (subject-grouped), claim-disciplined, blind-safe, and
ethically scoped.

The one remaining author decision is the portal blind-review choice:
- confirm whether TAFFC ScholarOne requests **single-blind** or **double-blind** review;
- upload `main_taffc.pdf` for single-blind, or `main_taffc_blind.pdf` for double-blind.

The experimental-hardening items (deep baselines, extra metrics, EMG/latency models, LOSO) are
genuine post-submission improvements, not blockers, and must not be fabricated. (Open-access/APC
election is a portal checkbox, not a manuscript gate.)
