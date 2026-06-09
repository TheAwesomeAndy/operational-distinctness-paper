# ScholarOne submission metadata — ARSPI-Net (IEEE TCDS special issue)

Target: **IEEE Transactions on Cognitive and Developmental Systems**, special
issue *Brain-Inspired Computing for Embodied AI*.
Paper type: **Regular paper, 10 pages** (within the 10-page no-overlength-charge
base limit).
Review mode: **NOT verified — the author must confirm on the live ScholarOne
portal / special-issue instructions before upload.**
- **Default artifact:** `manuscript/main.pdf` (non-blind) — use this unless the
  portal explicitly requires an anonymous/double-blind manuscript.
- **Contingency artifact:** `manuscript/main_blind.pdf` + `ARSPI-Net_blind_source.zip`
  — use only if the portal says "anonymous manuscript," "double-blind,"
  "double-anonymous," or equivalent.

> Evidence (not a substitute for the portal check): IEEE CIS pages indicate TCDS
> moved to a **double-anonymous** review process from October 2024
> (https://cis.ieee.org/publications/t-cognitive-and-developmental-systems).
> If that applies to this special issue, the contingency (blind) artifact is the
> correct upload. **Confirm on the actual ScholarOne file-upload instructions.**

System name "ARSPI-Net" is retained as a technical object (not masked); mask only
if the portal/chairs explicitly require method/system-name masking.

**IEEE AI-use disclosure (author action):** IEEE policy requires that AI-generated
text be disclosed in the Acknowledgment and that AI-assisted sections cite the AI
system used. The author must decide whether the manuscript's preparation triggers
this disclosure and, if so, add it to the (non-blind) camera-ready.

## Files to upload to ScholarOne
- **If non-blind (default):** `manuscript/main.pdf` (10 pp).
- **If anonymous/double-blind (contingency):** `manuscript/main_blind.pdf` (10 pp)
  + source `manuscript/submission_package_blind/` (`main_blind.tex` + five figure
  PDFs + five table `.tex`; `\graphicspath{{./}}`). IEEEtran.cls is provided by
  the IEEE/ScholarOne system.
- **Never upload:** `main_full_internal.pdf` (17-page
  internal record), or any `outputs/`, `data/`, or feature artifacts.

## Portal fields

**Title**
ARSPI-Net: An Event-Driven Reservoir–Graph Substrate for Embodied Affective EEG Perception

**Abstract (plain text)**
We present ARSPI-Net (Affective Reservoir–Spike Processing and Inference
Network), a brain-inspired event-driven reservoir–graph substrate that transforms
noisy affective electroencephalography (EEG) into a set of operationally distinct
neural evidence streams and evaluates those streams as perceptual channels for a
simulated embodied agent. The substrate is anchored in concrete neural
mechanisms: a fixed leaky integrate-and-fire (LIF) spiking reservoir, a binned
spike-count (BSC6) event-driven code, an electrode-level temporal phase-locking
(tPLV) graph, and a structure–function coupling readout kappa. These produce four
evidence streams: a spike-coded embedding E, dynamical descriptors D,
graph-topological descriptors T, and a coupling block C. On 633 trial-averaged
subject–condition observations from 211 adults under subject-grouped
cross-validation across five seeds, a mechanism ablation with negative controls
separates the streams: balanced accuracy is highest for the embedding-containing
configurations (E+D+T, 0.481) and the band-power baseline (0.485), the standalone
D, T, and C streams are lower but above the three-class chance level, and a
shuffled-label control collapses to chance; an independent permutation–FDR
protocol corroborates the ordering. The streams are not linear duplicates of one
another. We characterise the streams under four perturbation families (temporal
jitter, amplitude noise, channel dropout, and graph edge perturbation) and report
their operating regimes rather than a single endpoint. Finally, an explicitly
defined expected-free-energy controller uses the substrate as a perceptual
estimator in a simulated closed-loop affective-control task with action-determined
transition noise; against passive, random, pragmatic-only, epistemic-only, and
perfect-perception oracle policies, the controller reaches and confirms a target
affective state but shows no advantage over pragmatic control, an operating-regime
result. Clinical labels are used only as exploratory contextual validation and are
bounded by a false-discovery-rate result; no diagnostic claim is made. All results
are restricted to the measured study ERP regime and are not a universal claim
about EEG graph neural networks.

**Index terms / keywords**
EEG; neuromorphic computing; spiking neural networks; liquid state machines;
reservoir computing; neural coding; predictive coding; sequential evidence
accumulation; embodied AI; perception–action loops; structure–function coupling;
layer ablation.

## Author-side fields — TO BE COMPLETED BY THE AUTHOR (entered in the portal, not in the blind PDF)
The blind PDF intentionally omits all of these; they are entered in ScholarOne's
author metadata, which is hidden from double-blind reviewers.

| Field | Value (author to confirm) |
|---|---|
| Author list / order | Andrew A. Lane; K. Wendy Tang; Brady D. Nelson |
| Affiliations | *(author to provide)* |
| Corresponding author + email | *(author to provide)* |
| ORCID iDs | *(author to provide)* |
| Funding / grant numbers | *(none stated in the manuscript; author to confirm)* |
| Conflict-of-interest statement | *(author to provide)* |
| Suggested / opposed reviewers | *(optional; author to provide)* |
| Data-availability confirmation | restricted-data only — see statement below |

## Restricted-data statement (as it appears in the manuscript — keep as-is)
"Raw EEG, clinical metadata, and subject-level feature artifacts are maintained in
a restricted research environment … restricted subject-level data are not publicly
distributed and require approved access." No public-repository or public-data
claim is made (verified). For the **camera-ready (non-blind)** version, restore
the laboratory/institution name and the repository link in
`manuscript/main.tex`.

## Pre-upload author checklist (director's items 6–7 — author responsibility)
- [ ] Final human read: title, abstract, contributions, Results, Limitations,
      Data Availability, all figure captions, all tables.
- [ ] Confirm the special issue does not require the system name masked.
- [ ] Confirm author metadata above.
- [ ] Confirm the uploaded review PDF is `main_blind.pdf` (10 pages), not the
      13-page or 17-page internal records.
