# TAFFC Citation Integration — IEEE-TAC journal-fit references

**Date:** 2026-06-12 · **Branch:** `submission/taffc-citations` (from `main` `10c7750`)

Incorporates a deep-literature-review recommendation to strengthen the TAFFC variant's fit with the
**IEEE Transactions on Affective Computing (TAC)** physiological-affect line. **TAFFC-variant-only**;
the shared TCDS manuscript and all numerical results, figures, and tables are unchanged.

## Citations added (each independently verified against the live literature)
| Key | Reference | Placement |
|---|---|---|
| `koelstra2012deap` | Koelstra et al., "DEAP: A database for emotion analysis using physiological signals," *IEEE Trans. Affect. Comput.* 3(1):18–31, 2012 | Related Work (affective EEG) + cover letter |
| `soleymani2016continuous` | Soleymani et al., "Analysis of EEG signals and facial expressions for continuous emotion detection," *IEEE Trans. Affect. Comput.* 7(1):17–28, 2016 | Related Work |
| `zheng2019stable` | Zheng, Zhu, Lu, "Identifying stable patterns over time for emotion recognition from EEG," *IEEE Trans. Affect. Comput.* 10(3):417–429, 2019 | Related Work |
| `song2023variational` | Song et al., "Variational instance-adaptive graph for EEG emotion recognition," *IEEE Trans. Affect. Comput.* 14(1):343–356, 2023 | Related Work (graph EEG) |

These join the three TAC papers the manuscript already cited (RGNN, TSception, Fourati reservoir),
giving seven TAC references and a clear physiological-affect → graph → measurement positioning that
directly supports the "Spatiotemporal Characterization of Affective EEG Dynamics" title.

## Framing added
- **Related Work:** a sentence establishing TAC's physiological-affect line (benchmark datasets →
  continuous/time-resolved affect → stable spatiotemporal structure), and a clause adding
  adaptive/uncertainty-aware graph modelling alongside RGNN.
- **Cover letter:** a "Fit with this journal" paragraph positioning the work as extending TAC's
  established line into a neuromorphic reservoir–graph measurement direction — with no superiority or
  out-of-scope claims.

## Recommendations NOT followed — citation integrity held
The review report recommended several entries I could **not verify and therefore did not add**:
- **"F2FNet"** (claimed *IEEE Trans. Affect. Comput.* 17(2):1663–1676, 2026, doi:10.1109/TAFFC.2026.3671843)
  — the report called this "the single most important recent TAC citation," but multiple targeted
  searches found **no such paper**. The entry appears **fabricated** and was excluded.
- **Pan et al., "Toward a Realistic Encoding Model of Auditory Affective Understanding in the Brain"**
  — exists only as an **arXiv preprint (2509.21381, Sept 2025)**, not an IEEE TAC 2026 paper as the
  report claimed; excluded to avoid a mis-attributed venue.
- **Wang et al., "Identifying Cortical Brain Directed Connectivity Networks…"** — a **real** TAC paper
  (IEEE doc 9139334), but its exact volume/pages could not be independently confirmed, so it is
  **held** pending a final IEEE Xplore check rather than cited with unverified detail. (Recommended
  addition once verified, for the directed-connectivity / structure–function angle.)

## Validation
- Both variants compile to **11 pages** (within the 12-page free limit); 0 undefined references; the
  four new `\cite` commands all resolve.
- **No numerical result changed** — result decimals/percentages are identical to `main`'s variant;
  the only new numeric tokens are the four new bibliography entries' volumes/pages/years.
- Shared TCDS manuscript/package, figures, and tables are byte-identical to `main`.
- No forbidden claims introduced; no Claude/Anthropic/session references.

## Note on parallel branches
This is an additive, figure-independent change from `main`, **independent of** the open figure-polish
PR (#24). Both touch the variant PDF, so whichever merges second will need a quick recompile/rebase of
the other.
