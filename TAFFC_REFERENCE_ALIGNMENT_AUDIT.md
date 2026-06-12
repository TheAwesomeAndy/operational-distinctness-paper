# TAFFC Reference-Alignment Audit

**Date:** 2026-06-12 · **Branch:** `submission/taffc-citation-alignment` (from `main` `f5962a0`)

Adds a compact, **independently verified** set of IEEE Transactions on Affective Computing (TAC)
references to the TAFFC manuscript variant and cover letter, establishing journal fit along the
physiological-affect → graph/connectivity → measurement line. **TAFFC-variant-only; no numerical
result, figure, table, or shared TCDS file changed.** Every entry was checked against the live
literature before inclusion; unverifiable entries were **refused**.

> **Citation-style note.** The manuscript's existing journal bibitems carry **no DOI field** (only
> the two LISAT conference self-citations include DOIs). To match that style and to avoid asserting
> any unverified DOI, the new journal entries are formatted **without DOIs**; verified DOIs are
> recorded in the table below for the record.

## References added (11, all verified)
| Key | Reference (TAC) | Category | Inserted in | Role | Priority | DOI (verified; omitted in bib to match style) |
|---|---|---|---|---|---|---|
| `koelstra2012deap` | DEAP, 3(1):18–31, 2012 | 1 datasets | Intro | journal fit | essential | 10.1109/T-AFFC.2011.15 |
| `soleymani2012mahnob` | MAHNOB-HCI, 3(1):42–55, 2012 | 1 datasets / implicit tagging | Intro + cover letter | journal fit | strongly rec. | 10.1109/T-AFFC.2011.25 |
| `soleymani2016continuous` | Continuous emotion detection, 7(1):17–28, 2016 | 1/3 continuous | Intro | journal fit | essential | 10.1109/TAFFC.2015.2436926 |
| `zheng2019stable` | Stable patterns over time, 10(3):417–429, 2019 | 2 stable patterns | Intro | fit + supports "spatiotemporal" framing | essential | 10.1109/TAFFC.2017.2712143 |
| `alarcao2019survey` | EEG emotion survey, 10(3):374–393, 2019 | 2 survey | Intro | reviewer awareness | strongly rec. | 10.1109/TAFFC.2017.2714671 |
| `ding2021interbrain` | Inter-brain continuous implicit tagging, 12(1):92–102, 2021 | 3 continuous dynamics | Related Work | fit | optional | not independently confirmed |
| `shen2023contrastive` | Subject-invariant (CLISA), 14(3):2496–2511, 2023 | 5 robustness | Related Work | fit | strongly rec. | not independently confirmed |
| `song2023variational` | Variational instance-adaptive graph, 14(1):343–356, 2023 | 4 graph | Related Work | fit + method contrast | essential | 10.1109/TAFFC.2021.3064940 |
| `chen2024gddn` | GDDN, 15(3):1739–1753, 2024 | 4/5 graph generalization | Related Work | fit | strongly rec. | not independently confirmed (IEEE doc 10453943) |
| `xu2024amdet` | AMDET transformer, 15(3):1067–1077, 2024 | 4/D transformer | Related Work (method contrast) | positions reservoir vs. transformer | strongly rec. | 10.1109/TAFFC.2023.3318321 |
| `jiang2025seedvii` | SEED-VII continuous labels, 16(2):969–985, 2025 | 1/3 continuous labels | Discussion | fit + dynamics framing | strongly rec. | 10.1109/TAFFC.2024.3485057 |

These join the three TAC papers already cited (RGNN, TSception, Fourati) → **14 TAC references**.

## Insertion sentences (compact, citation-dense; allowed language only)
- **Introduction:** one lineage sentence (datasets → continuous affect → stable signatures).
- **Related Work:** one sentence on graph / subject-invariant / continuous-dynamics paradigms; one
  sentence contrasting ARSPI-Net's event-driven reservoir step with graph/transformer models (no
  superiority claim).
- **Discussion:** one sentence framing the contribution as extending the journal's movement toward
  physiological characterization and continuous-label dynamics, "restricted to the measured ERP
  regime."
- **Cover letter:** one "Fit with this journal" paragraph (prose; names DEAP / MAHNOB-HCI), presenting
  the work as physiological measurement "rather than static classifier superiority or clinical
  validation."

## Recommendations REFUSED — citation integrity held
| Entry (as recommended) | Reason refused |
|---|---|
| **"F2FNet," claimed TAC 17(2):1663–1676, 2026, doi:10.1109/TAFFC.2026.3671843** | **No such paper found** in multiple targeted searches. Appears **fabricated** (the list called it "the single most important recent TAC citation"). Not added. |
| **Pan, "Toward a Realistic Encoding Model…," claimed TAC 17(2):2354–2367, 2026, doi:10.1109/TAFFC.2026.3670015** | Exists only as an **arXiv preprint (2509.21381)**, not a TAC paper. Venue misattribution; not added. |
| **Kumar & Joshi, "Nasal Dominance…," claimed TAC 17(2):2288–2304, 2026, doi:10.1109/TAFFC.2026.3673829** | **No such paper found**; same fabricated 2026/DOI pattern. Not added. |

**Held (real, but bibliographic detail unconfirmed):**
- **Wang et al., "Identifying Cortical Brain Directed Connectivity Networks…"** — a **real** TAC paper
  (IEEE doc 9139334), but exact volume/pages could not be independently confirmed, so it is **held**
  pending an IEEE Xplore check. Recommended for the structure–function/connectivity angle once
  verified.

## Validation
- Both variants compile to **11 pages** (within the 12-page free limit); **0 undefined references**;
  **no duplicate bibitems** (51 unique); all 11 new `\cite` commands resolve.
- **No numerical result changed** (result decimals/percentages identical to `main`'s variant; new
  numeric tokens are only the 11 bibliography entries' vol/pages/years).
- Shared TCDS manuscript/package, figures, and tables byte-identical to `main`.
- No forbidden language added (no "state-of-the-art," "superior," "outperforms," "diagnostic
  biomarker," "real-time," "online BCI," "wearable," "hardware energy," etc.); claim discipline and
  scope boundaries preserved. No automated tool-attribution or session-link references.

## Note on parallel branches
Independent, figure-independent change from `main`. It is a **superset** of the earlier
`submission/taffc-citations` branch (PR #25), which added four of these citations — **PR #25 should be
closed in favor of this PR**. It is also independent of figure-polish PR #24 (both touch the variant
PDF; whichever merges second needs a quick recompile/rebase).
