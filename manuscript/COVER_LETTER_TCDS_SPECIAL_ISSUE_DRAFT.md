# Cover letter draft — IEEE TCDS Special Issue (author to finalize)

> Draft only. The author must add the date, salutation/editor name, author
> signatures, and any portal-required statements before submission. Keep the tone
> factual and non-promotional.

---

Dear Editors,

We submit our manuscript, *"ARSPI-Net: An Event-Driven Reservoir–Graph Substrate
for Embodied Affective EEG Perception,"* for consideration in the Special Issue on
Brain-Inspired Computing for Embodied AI.

The manuscript fits the special issue because it studies a concrete translation of
neural mechanisms into an implementable signal-processing substrate and evaluates
that substrate in a simulated embodied perception–action setting. Specifically, it
develops and characterizes:

- event-driven spiking-reservoir dynamics (a fixed leaky integrate-and-fire
  reservoir with a binned spike-count code);
- graph-structured neural evidence (electrode-level temporal phase-locking
  topology and a structure–function coupling readout);
- the perturbation-dependent operating regimes of the resulting evidence streams;
- a simulated embodied affective-control loop driven by an explicitly defined
  expected-free-energy controller.

For scientific precision, we state the scope of the submission. The manuscript
evaluates ARSPI-Net as an event-driven reservoir–graph signal-processing substrate
that transforms noisy affective EEG into structured, perturbation-characterized
evidence streams. The affective EEG task and the associated clinical labels define
the validation setting in which the substrate is characterized. The resource
analysis reports event-rate and operation-count accounting for the event-driven
substrate. The embodied experiment is a simulated active, perceptual-control
analysis over recorded neurophysiological observations. Accordingly, the study is
positioned as a brain-inspired signal-processing and embodied-perception
contribution rather than as diagnostic validation, physical-robot deployment, or
measured hardware-energy characterization.

A supplemental technical appendix accompanies the manuscript and provides the full
robustness tables, an adaptive evidence-routing analysis (which we report as a
**bounded** operating-regime result: the streams are separable, with large oracle
headroom, but label-free routing does not yet outperform the best fixed stream
under the measured regime), resource/event-rate accounting, and a reproducibility
map.

All quantitative results are restricted to the measured study ERP regime. The
underlying EEG and clinical metadata are restricted human-subject research data;
the manuscript reports aggregate, deidentified outputs with methodological detail
sufficient for reproduction under approved data-access conditions.

We confirm that this manuscript is original, is not under consideration elsewhere,
and that all authors have approved the submission.

Thank you for your consideration.

Sincerely,
[Author names and signatures]
[Corresponding author and contact details]

---

## Suggested handling-editor / reviewer expertise profile (author may include)
- neuromorphic signal processing;
- graph signal processing;
- embodied-AI perception–action modeling;
- computational neuroscience;
- EEG/ERP signal processing.

(No specific reviewers are suggested unless the author supplies names.)
