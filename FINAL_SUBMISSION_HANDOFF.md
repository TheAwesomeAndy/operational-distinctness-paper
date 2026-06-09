# Final submission handoff — ARSPI-Net

Engineering loop closed. No further experiments. The repository state is a
submission-ready package; remaining work is author-side submission administration.

## Venue decision
- **Primary:** IEEE Transactions on Cognitive and Developmental Systems (TCDS),
  Special Issue on *Brain-Inspired Computing for Embodied AI*.
- **Backup (scope/embodiment rejection):** Neuromorphic Computing and Engineering.
- **Backup (neural-signal / clinical-data framing rejection):** Journal of Neural
  Engineering, or IEEE Transactions on Neural Systems and Rehabilitation
  Engineering (TNSRE).

## Scientific position (preserve in all venues)
ARSPI-Net is **not** submitted as a universally superior EEG classifier. It is an
event-driven reservoir–graph signal-processing substrate that exposes operationally
distinct neural evidence streams, characterizes their perturbation-dependent
operating regimes, and evaluates their utility in simulated embodied affective
perception. Adaptive routing is a **bounded operating-regime result** (separable
streams; no deployable routing advantage). Resource/event accounting is
**computational accounting only** — no measured hardware energy.

## Exact files and checksums (commit `94b560d` on `main`)

| File | Pages | Role | SHA256 |
|---|---|---|---|
| `manuscript/main.pdf` | 10 | **Non-blind submission manuscript** | `35ab9e00f2d563ae777d07d83bb90445d07e6ac823a1d3f25d7e39571a7cae6e` |
| `manuscript/main_blind.pdf` | 10 | **Anonymized (double-blind) manuscript** | `e913752d91f089dec90899b07ce7f6e2b8f7c3e85ee56a8f8eacc5d5b3b1fe26` |
| `manuscript/supplemental_risk_reduction/supplement.pdf` | 6 | Supplemental technical appendix | `99f3c638b97c3e5615a032ecd0c3ce4b58885312569b2fd6d4030c069eb85517` |
| `manuscript/arxiv_preprint/main_arxiv.pdf` | 10 | arXiv preprint (non-blind) | `fdbe77196673271a568185ac8488a3297a56e021a4ce7a12bc98dbea3fa478e3` |
| `manuscript/arxiv_preprint/ARSPI-Net_arxiv_source.zip` | – | arXiv source package | `bc5214d127f4ff6dd936da330a17b0ff20145ae2e65b854839bd7dc30b630233` |
| `manuscript/main_full_internal.pdf` | 17 | **INTERNAL record — NOT for submission** | `f3932b202633e6011240b95ddebac6e653beb08d196fa586bf82f0ae890712ea` |

ScholarOne LaTeX source packages:
- Non-blind source: build from `manuscript/main.tex` (+ figures/tables; IEEEtran
  supplied by ScholarOne).
- Blind source: `manuscript/submission_package_blind/` (committed, self-contained).

## Which file to upload, by review mode (verify on the live ScholarOne workflow)
- **Single-anonymous or non-anonymous review →** upload `manuscript/main.pdf`
  (non-blind).
- **Double-anonymous review →** upload `manuscript/main_blind.pdf` (anonymized) and
  the blind source package; do **not** upload the non-blind PDF.
- **Supplement:** upload `supplement.pdf` as supplementary material in either mode
  (it contains no author identifiers).
- **Never upload** `manuscript/main_full_internal.pdf` (17-page internal record).
- Confirm the live portal's stated review mode before selecting; evidence suggests
  TCDS uses double-anonymous review (Oct 2024+), but the portal is controlling.

## Remaining author-side fields (enter in the portal / camera-ready; not fabricated)
- Author names (current: A. A. Lane, K. W. Tang, B. D. Nelson — confirm order).
- Affiliations.
- Corresponding author + email.
- ORCID iDs.
- Funding statement / grant numbers (none stated in the manuscript — confirm).
- Conflict-of-interest statement.
- Acknowledgments (restore on the non-blind / camera-ready only; not in the blind PDF).
- AI-use disclosure if required by the portal/publisher policy — handle in the
  non-blind/camera-ready Acknowledgment or a portal field; do **not** add
  author-identifying text to the blind PDF.

## arXiv warning
Do **not** post to arXiv before deciding how it interacts with TCDS anonymous
review. Public preprinting can reduce practical anonymity even where technically
permitted. Decide timing (before submission / after submission / after first
decision / after acceptance) per `manuscript/arxiv_preprint/README_ARXIV.md`. The
arXiv package is prepared only — not submitted.

## Status
- ScholarOne package: **prepared, not submitted.**
- arXiv package: **prepared, not submitted.**
- Cover letter draft: `manuscript/COVER_LETTER_TCDS_SPECIAL_ISSUE_DRAFT.md`
  (prepared, author to finalize/sign).
- Compliance: no raw EEG / feature pickles / clinical CSV / subject IDs / PHI /
  private paths committed; no measured-energy, physical-robot, diagnostic-biomarker,
  or universal-robustness claims; adaptive routing reported as bounded.
