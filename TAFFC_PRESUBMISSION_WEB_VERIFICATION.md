# TAFFC Pre-Submission Web Verification

Closes the two outstanding pre-upload items from the acceptance-strategy gate (v2), using
authoritative web sources checked 2026-06-16. Documentation only — no manuscript source, figure,
table, output, PDF, or class file changed.

## 1. Review model — single-blind vs double-blind

**Verified: IEEE Transactions on Affective Computing uses single-anonymous (single-blind) review by
default.** Double-anonymous (double-blind) review is available only if the author requests it at
submission, granted at the Editor-in-Chief's discretion, and the request must be indicated during
submission.

- IEEE Author Center (peer-review process): "most IEEE publications use the single-anonymous format."
  Single-anonymous = reviewers' identities hidden from the author, author's identity known to
  reviewers; double-anonymous = neither side knows the other.
- IEEE Computer Society policy (TAFFC is a Computer Society title): unless a double-anonymous review
  is requested, each article undergoes single-anonymous review; double-anonymous requests are
  honored at the EIC's discretion and must be clearly indicated at submission.
- Submission portal (TAC Call for Papers): `https://mc.manuscriptcentral.com/taffc-cs`.

**Decision:** submit the non-blind **`manuscript/taffc/main_taffc.pdf`** as the primary file under
the single-blind default. Use **`manuscript/taffc/main_taffc_blind.pdf`** only if the authors elect
double-anonymous review, in which case the request must be indicated in ScholarOne. Both PDFs are
compiled and validated (12 pp each).

Sources:
- https://journals.ieeeauthorcenter.ieee.org/submit-your-article-for-peer-review/about-the-peer-review-process/
- https://www.computer.org/publications/author-resources/peer-review/journals
- https://www.computer.org/digital-library/journals/ta/tac-general-call-for-papers

## 2. Recent-publication / novelty duplication check

**Method.** Fan-out web searches (June 2026) over the distinctive contribution combination —
a fixed neuromorphic LIF spiking reservoir + tPLV graph topology + structure–function coupling as
**operationally distinct evidence streams** (E, D, T, C), characterized by **perturbation operating
regimes** and an **offline expected-free-energy closed-loop accumulation**, framed as a
**measurement substrate** rather than a classifier — plus searches on the name "ARSPI-Net" and on the
exact framing terms.

**Finding: no duplicate of the integrated ARSPI-Net contribution was located.**
- Searching "ARSPI-Net" returns only the authors' **own** prior conference work (LISAT 2023/2024),
  which the manuscript already discloses (cover letter) and cites (`lane2023arspi`, `lane2024arspi`)
  and which it substantially extends. No third-party paper uses the name or the integrated method.
- The exact combination (operationally distinct evidence streams + perturbation regimes + closed-loop
  EFE accumulation + measurement-substrate framing) was not found in any single paper.

**Closest neighbors (separate, established research lines — all appropriately distinct):**
- Reservoir / echo-state EEG emotion recognition (e.g. multi-reservoir computing, ICMI 2024; deep
  echo-state networks; Fourati reservoir) — endpoint valence/arousal **recognition**, not an
  operational-distinctness / measurement framing.
- Spiking- and graph-attention EEG emotion/stress models (e.g. residual spatio-temporal graph
  attention, 2026; spiking neural networks for EEG stress, 2025) — endpoint classifiers.
- Active-inference / expected-free-energy BCI frameworks — the conceptual lineage the manuscript
  already cites and frames ARSPI-Net's controller as a *precursor* to, not a full implementation.
- Structure–function coupling in EEG via graph signal processing (DWI–EEG coupling, 2024) — an
  anatomical structure-vs-function coupling, distinct from ARSPI-Net's internal reservoir-graph
  coupling readout κ.

**Optional (citation-integrity decision, not required and not performed here).** One or two recent
(2024–2026) endpoint comparators could be cited for currency. Any addition must follow
`CITATION_INTEGRITY_POLICY.md` (relevance + verified metadata, no padding); none were added
automatically.

**Caveat.** This is a search-based check (June 2026). Absence of a duplicate in these searches is not
proof of non-existence; no exhaustiveness is claimed and nothing was fabricated. A final author-side
scan of the most recent issues remains advisable.

Sources:
- https://www.mdpi.com/2624-6120/7/1/16
- https://www.nature.com/articles/s41598-025-10270-0
- https://dl.acm.org/doi/10.1145/3678957.3688618
- https://ieeexplore.ieee.org/document/10808138/
- https://www.biorxiv.org/content/10.1101/2021.02.02.429272v1.full
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12293440/

## Net effect on the gate
Both previously-outstanding items are resolved: the review model is determined (single-blind default
→ `main_taffc.pdf`; double-anonymous is an optional author election), and the novelty check found no
duplicate. The manuscript is ready to submit via the single-blind default path.
