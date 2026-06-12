Dear Editor-in-Chief and Editorial Board of IEEE Transactions on Affective Computing,

We submit our manuscript, "ARSPI-Net: A Neuromorphic Reservoir-Graph Substrate for Affective ERP
Decoding and Closed-Loop Neural Evidence Accumulation," for consideration as a Regular Paper.

Affective-computing contribution. ARSPI-Net is a physiological affective-computing method: it
decodes the affective event-related potential (ERP) — a noisy, partially observed
electrophysiological signature of emotional processing — into structured, interpretable evidence
rather than a single recognition score. Treating the EEG as the output of a biological dynamical
system, a fixed leaky integrate-and-fire reservoir, a binned spike-count code, a temporal
phase-locking graph, and a structure-function coupling readout transform each affective
observation into four named neural evidence streams: a spike-coded embedding E, dynamical
descriptors D, graph-topological descriptors T, and a structure-function coupling block C.

Operational distinctness of the evidence streams. The paper's central question is whether E, D, T,
and C behave as different affective measurements or as redundant relabelings of one response.
Under subject-grouped cross-validation with negative controls and false-discovery-rate control, we
test their predictive sufficiency, incremental utility, perturbation response, and representational
redundancy under centered-kernel alignment. The streams are operationally distinct: each exposes
affective structure the others omit, which a single recognition number cannot reveal.

Perturbation robustness. We characterize every stream under temporal jitter, amplitude noise,
channel dropout, and graph edge perturbation, reporting per-stream operating regimes rather than a
single endpoint — which affective descriptors are robust to signal-domain corruption and which are
sensitive to graph-domain corruption. For affective computing this is a practical contribution: it
indicates which physiological evidence to trust as signal quality changes.

Scope and claim discipline. We do not claim recognition-accuracy superiority: a conventional
ERP-amplitude baseline remains stronger for static three-class classification, and we retain it as
a reference, positioning ARSPI-Net as a measurement substrate. As a secondary utility, we show the
substrate can serve as the evidence estimator in a closed-loop neural evidence-accumulation
simulation over recorded ERP observations, with an explicitly defined expected-free-energy
controller read as a precursor to active-inference-style control; this is a simulation over
recorded observations, not an online or real-time system. Clinical labels enter only as exploratory
contextual variables bounded by a false-discovery-rate result, not as diagnostic validation. The
underlying EEG and metadata come from an access-controlled affective ERP cohort; the manuscript
reports aggregate, deidentified outputs.

We confirm that this manuscript is original, is not under consideration elsewhere, and that all
authors have approved the submission. We believe ARSPI-Net offers the affective-computing
readership a rigorous, reproducible methodology for turning affective neural signals into
structured, perturbation-characterized evidence.

Sincerely,
The Authors
