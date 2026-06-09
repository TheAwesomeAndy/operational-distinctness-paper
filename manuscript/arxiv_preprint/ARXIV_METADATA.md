# arXiv metadata — ARSPI-Net preprint

**Title**
ARSPI-Net: An Event-Driven Reservoir--Graph Substrate for Embodied Affective EEG Perception

**Authors** (non-blind; confirm final list/order with the author before submission)
Andrew A. Lane, K. Wendy Tang, Brady D. Nelson

> Author fields NOT in the manuscript and NOT to be fabricated — the author must
> supply these on the arXiv submission form / camera-ready:
> - affiliations
> - corresponding author + email
> - ORCID iDs
> - funding statement / grant numbers
> - conflict-of-interest statement (if required)

**Abstract (plain text)**
We present ARSPI-Net (Affective Reservoir-Spike Processing and Inference Network),
a brain-inspired event-driven reservoir-graph substrate that transforms noisy
affective electroencephalography (EEG) into a set of operationally distinct neural
evidence streams and evaluates those streams as perceptual channels for a
simulated embodied agent. The substrate is anchored in concrete neural mechanisms:
a fixed leaky integrate-and-fire (LIF) spiking reservoir, a binned spike-count
(BSC6) event-driven code, an electrode-level temporal phase-locking (tPLV) graph,
and a structure-function coupling readout kappa. These produce four evidence
streams: a spike-coded embedding E, dynamical descriptors D, graph-topological
descriptors T, and a coupling block C. On 633 trial-averaged subject-condition
observations from 211 adults under subject-grouped cross-validation across five
seeds, a mechanism ablation with negative controls separates the streams: balanced
accuracy is highest for the embedding-containing configurations (E+D+T, 0.481) and
the band-power baseline (0.485), the standalone D, T, and C streams are lower but
above the three-class chance level, and a shuffled-label control collapses to
chance; an independent permutation-FDR protocol corroborates the ordering. The
streams are not linear duplicates of one another. We characterise the streams
under four perturbation families (temporal jitter, amplitude noise, channel
dropout, and graph edge perturbation) and report their operating regimes rather
than a single endpoint. Finally, an explicitly defined expected-free-energy
controller uses the substrate as a perceptual estimator in a simulated closed-loop
affective-control task with action-determined transition noise; against passive,
random, pragmatic-only, epistemic-only, and perfect-perception oracle policies, the
controller reaches and confirms a target affective state but shows no advantage
over pragmatic control, an operating-regime result. Clinical labels are used only
as exploratory contextual validation and are bounded by a false-discovery-rate
result; no diagnostic claim is made. All results are restricted to the measured
SHAPE ERP regime and are not a universal claim about EEG graph neural networks.

**Suggested primary category**
eess.SP (Signal Processing)

**Suggested secondary categories** (author selects final on submission)
cs.NE (Neural and Evolutionary Computing); cs.LG (Machine Learning);
q-bio.NC (Neurons and Cognition)

**Comments field**
"10 pages, 5 figures, 6 tables, with supplemental technical appendix. Preprint version."
(Base form without appendix: "10 pages, 5 figures, 6 tables. Preprint version.")
(If a supplemental technical appendix is attached after the risk-reduction pass:
"10 pages, 5 figures, 6 tables, with supplemental technical appendix.")

**License**
Recommend arXiv's default non-exclusive distribution license. Do NOT select a
Creative Commons license automatically — leave to the author.

**Data availability statement (restricted-data; as in the manuscript)**
The raw EEG and clinical metadata used in this study are restricted human-subject
research data and are not publicly released. The manuscript reports aggregate,
deidentified outputs and methodological details sufficient for reproduction under
approved data-access conditions.

**Code/reproducibility statement**
The analysis code and aggregate reproducibility artifacts are maintained in an
access-controlled research repository. Public release is limited by data-governance
and human-subject restrictions. (Do not claim "code is publicly available".)
