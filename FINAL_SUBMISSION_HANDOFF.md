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

## Exact files and checksums (after manuscript-cleanup)

| File | Pages | Role | SHA256 |
|---|---|---|---|
| `manuscript/main.pdf` | 10 | **Non-blind submission manuscript** | `dc0bb0c8d4303a2ad31d7e0d05d99948f1e3f862bb428cac254f86b5c29efb0f` |
| `manuscript/main_blind.pdf` | 10 | **Anonymized (double-blind) manuscript** | `1954554a61eaae89741473c472d70ff767938504a9ea6e7939b347f1bd727a26` |
| `manuscript/supplemental_risk_reduction/supplement.pdf` | 6 | Supplemental technical appendix | `99f3c638b97c3e5615a032ecd0c3ce4b58885312569b2fd6d4030c069eb85517` |
| `manuscript/arxiv_preprint/main_arxiv.pdf` | 10 | arXiv preprint (non-blind) | `e41fb8cbbe7d256b910eb12f1ba127b35f8987f98567e1aa34794189b7b9fb4c` |
| `manuscript/arxiv_preprint/ARSPI-Net_arxiv_source.zip` | – | arXiv source package | `4430974f8fecdba676f4ca1da0ddda91253971233a22f0c5e64bcd18c72ecbda` |
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
