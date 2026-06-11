# CFP Alignment — ARSPI-Net → TCDS "Cognitive Neuroscience Meets BCIs" Special Issue

Special issue: **Cognitive Neuroscience Meets Brain-Computer Interfaces: From Neural Decoding to
Function Augmentation** (IEEE Transactions on Cognitive and Developmental Systems).

This document maps the manuscript to the call's topics. ARSPI-Net is positioned on the
**neural-decoding and evidence-accumulation** side of the call; it does not claim function
augmentation, online deployment, or clinical validation.

## Topic mapping

- **Algorithms for decoding cognitive processes via BCIs.** ARSPI-Net decodes affective ERP
  observations into spike-coded (E), dynamical (D), graph-topological (T), and structure-function
  coupling (C) evidence streams via a fixed LIF reservoir, BSC6 event-driven coding, tPLV
  connectivity, and a coupling readout.

- **Closed-loop and adaptive BCI systems.** A closed-loop adaptive BCI simulation evaluates
  sequential belief updating with an expected-free-energy controller (risk + ambiguity terms) over
  recorded ERP observations under action-determined transition noise.

- **Real-time interpretation of complex neural signals.** The substrate compresses high-dimensional
  ERP observations into compact evidence streams. Real-time operation is **not** claimed or
  demonstrated; all evaluation is offline over recorded observations.

- **System robustness.** Temporal jitter, amplitude noise, channel dropout, and graph edge
  perturbation analyses characterize stream-specific operating regimes rather than a single endpoint.

- **Individualized brain models.** Subject-grouped validation and subject-condition observations;
  structure-function coupling is estimated per observation / subject-condition.

- **Clinical translation relevance.** Exploratory clinical-label sensitivity bounded by a
  false-discovery-rate result; explicitly **not** diagnostic validation or biomarker discovery.
