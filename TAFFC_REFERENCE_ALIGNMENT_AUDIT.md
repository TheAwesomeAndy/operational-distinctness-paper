# TAFFC Reference-Alignment Audit

**Date:** 2026-06-12 · **Branch:** `submission/taffc-citation-alignment` (from `main` `f5962a0`)

Adds a compact, **independently verified** set of IEEE Transactions on Affective Computing (TAC)
references to the TAFFC manuscript variant and cover letter, establishing the physiological-affect →
graph/connectivity → measurement journal-fit line. **TAFFC-variant-only; no numerical result,
figure, table, or shared TCDS file changed.**

> **Verification method.** Every entry was checked against authoritative metadata. The 2019–2025
> entries were confirmed via web index and (for the control) Crossref. The four 2022/2026 entries
> below were confirmed by **direct Crossref DOI metadata** (`https://api.crossref.org/works/<DOI>`),
> which returned matching title, journal (*IEEE Transactions on Affective Computing*), volume, issue,
> and page range.

> **Correction note (important).** An earlier draft of this audit labeled F2FNet, Pan et al., and
> Kumar & Joshi as "fabricated/misattributed" because a general web search did not return them. That
> conclusion was an overreach: 2026 articles are not yet indexed by general web search, but they are
> registered with authoritative DOIs. Direct **Crossref** verification confirms all three (and Wang)
> are **genuine IEEE TAC papers**, and they have been added. The "fabricated" characterization was
> incorrect and is retracted.

> **Citation-style note.** The manuscript's existing journal bibitems carry **no DOI field**. The new
> journal entries match that style (no DOI in the bibitem); the verified DOIs are recorded below.

## References added (15, all verified)
| Key | Reference (TAC) | Category | Inserted in | DOI (verified) |
|---|---|---|---|---|
| `koelstra2012deap` | DEAP, 3(1):18–31, 2012 | datasets | Intro + cover letter | 10.1109/T-AFFC.2011.15 |
| `soleymani2012mahnob` | MAHNOB-HCI, 3(1):42–55, 2012 | datasets / implicit tagging | Intro + cover letter | 10.1109/T-AFFC.2011.25 |
| `soleymani2016continuous` | Continuous emotion detection, 7(1):17–28, 2016 | continuous | Intro | 10.1109/TAFFC.2015.2436926 |
| `zheng2019stable` | Stable patterns over time, 10(3):417–429, 2019 | stable patterns | Intro | 10.1109/TAFFC.2017.2712143 |
| `alarcao2019survey` | EEG emotion survey, 10(3):374–393, 2019 | survey | Intro | 10.1109/TAFFC.2017.2714671 |
| `ding2021interbrain` | Inter-brain continuous implicit tagging, 12(1):92–102, 2021 | continuous dynamics | Related Work | (Crossref to confirm; web-verified) |
| `wang2022directed` | Directed connectivity networks, 13(3):1489–1500, 2022 | graph/connectivity | Related Work | 10.1109/TAFFC.2020.3006847 (Crossref) |
| `shen2023contrastive` | Subject-invariant (CLISA), 14(3):2496–2511, 2023 | robustness | Related Work | (web-verified) |
| `song2023variational` | Variational instance-adaptive graph, 14(1):343–356, 2023 | graph | Related Work | 10.1109/TAFFC.2021.3064940 |
| `chen2024gddn` | GDDN, 15(3):1739–1753, 2024 | graph generalization | Related Work | IEEE doc 10453943 |
| `xu2024amdet` | AMDET transformer, 15(3):1067–1077, 2024 | transformer | Related Work (contrast) | 10.1109/TAFFC.2023.3318321 |
| `jiang2025seedvii` | SEED-VII continuous labels, 16(2):969–985, 2025 | continuous labels | Discussion | 10.1109/TAFFC.2024.3485057 |
| `xu2026f2fnet` | F2FNet few→full-channel reconstruction, 17(2):1663–1676, 2026 | robustness / sparse sensing | Related Work | **10.1109/TAFFC.2026.3671843 (Crossref)** |
| `pan2026encoding` | Realistic auditory affective encoding model, 17(2):2354–2367, 2026 | neural encoding | Discussion | **10.1109/TAFFC.2026.3670015 (Crossref)** |
| `kumar2026nasal` | Nasal-breathing stress biomarkers, 17(2):2288–2304, 2026 | physiological biomarker | Discussion | **10.1109/TAFFC.2026.3673829 (Crossref)** |

With the three TAC papers already cited (RGNN, TSception, Fourati) → **18 TAC references**.

## Insertion sentences (compact, citation-dense; allowed language only)
- **Introduction:** one lineage sentence (datasets → continuous affect → stable signatures).
- **Related Work:** one sentence on graph / connectivity / subject-invariant / sparse-sensor
  reconstruction / continuous-dynamics paradigms (adds Wang and F2FNet); one sentence contrasting
  ARSPI-Net's event-driven reservoir step with graph/transformer models — **no superiority claim**,
  and F2FNet is positioned as a comparator, not as something ARSPI-Net outperforms.
- **Discussion:** one sentence framing the contribution as extending the journal's movement toward
  physiological characterization, **neural encoding** (Pan), **physiological-biomarker measurement**
  (Kumar & Joshi), connectivity-based analysis, and continuous-label dynamics (SEED-VII), "restricted
  to the measured ERP regime."
- **Cover letter:** one "Fit with this journal" paragraph (prose) covering the same trajectory,
  presenting the work as physiological measurement "rather than static classifier superiority or
  clinical validation."

## Held references pending direct IEEE metadata verification
None remaining. (Wang was previously held for unconfirmed volume/pages; Crossref now confirms
13(3):1489–1500, 2022, DOI 10.1109/TAFFC.2020.3006847, so it has been added.)

## Validation
- Both variants compile to **11 pages** (within the 12-page free limit); **0 undefined references**;
  **no duplicate bibitems** (55 unique); all 15 new `\cite` commands resolve.
- **No numerical result changed** (result decimals/percentages identical to `main`'s variant; new
  numeric tokens are only the 15 bibliography entries' vol/pages/years).
- Shared TCDS manuscript/package, figures, and tables byte-identical to `main`.
- No forbidden language added (no "state-of-the-art," "superior," "outperforms," "diagnostic
  biomarker," "real-time," "online BCI," "wearable," "hardware energy," etc.); claim discipline and
  scope boundaries preserved; F2FNet not framed as outperformed. No automated tool-attribution or
  session-link strings.

## Note on parallel branches
Independent, figure-independent change from `main`. It is a **superset** of the earlier
`submission/taffc-citations` branch (PR #25) — **PR #25 should be closed in favor of this PR**. It is
also independent of figure-polish PR #24 (both touch the variant PDF; whichever merges second needs a
quick recompile/rebase).
