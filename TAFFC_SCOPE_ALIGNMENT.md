# TAFFC Scope Alignment — ARSPI-Net

IEEE Transactions on Affective Computing publishes research on systems that recognize, interpret,
and model human emotion and related affective phenomena, including affective physiological-signal
processing and methodology. ARSPI-Net aligns on the **measurement-and-methodology** side of that
scope: it is a physiological affective-computing substrate, not an end-to-end emotion classifier.

| TAFFC scope area | How ARSPI-Net addresses it |
|---|---|
| Affect decoding from physiological signals (EEG / ERP) | Decodes the affective ERP into four named neural evidence streams (E, D, T, C) under subject-grouped cross-validation. |
| Representation and feature analysis for affect | Operational-distinctness analysis — predictive sufficiency, incremental utility, and representational redundancy (centered-kernel alignment) of the streams, with negative controls and FDR control. |
| Robustness and reliability of affective sensing | Per-stream perturbation characterization (temporal jitter, amplitude noise, channel dropout, graph edge perturbation), reporting operating regimes rather than a single endpoint. |
| Methodology and evaluation rigor | Subject-grouped validation, permutation-FDR, negative controls; a conventional ERP-amplitude baseline retained as a reference (no recognition-superiority claim). |
| Adaptive / closed-loop affective systems (secondary) | The substrate serves as the evidence estimator in a closed-loop neural evidence-accumulation **simulation** over recorded ERP observations (expected-free-energy controller; precursor to active-inference-style control). |

**Lead message:** affective ERP decoding and the operational distinctness of physiological evidence
streams. **Secondary:** closed-loop neural evidence accumulation.

**Explicitly not claimed (claim boundaries):** online or real-time affective sensing, neurofeedback,
cognitive augmentation, diagnostic or biomarker validation, hardware-energy efficiency, and full
active inference.
