# Risk-reduction changelog

A targeted pass to reduce the main reviewer risk for the TCDS submission without
restructuring the accepted 10-page manuscript. Primary risk addressed: *"why is a
reservoir--graph substrate useful if a conventional band-power baseline is more
robust under amplitude noise and channel dropout?"* Answer: the streams fail
differently, so the architecture exposes separable evidence channels whose
operating regimes are observable and bounded.

## Experiments added (`experiments/tcds_risk_reduction/`)
- `00_preflight_risk_reduction.py` — repo/input/alignment preflight.
- `01_adaptive_evidence_routing.py` — adaptive evidence routing under perturbation.
- `02_resource_event_accounting.py` — resource and event-rate accounting.
- `03_build_arxiv_package.py` — non-blind arXiv source-package builder.
- `99_verify_risk_reduction_package.py` — final quality gates.

## Results obtained
- **Resource/event accounting (succeeded).** Spike sparsity = 0.144 (14.4% of
  reservoir cells active per timestep); the event-driven recurrent layer performs
  ~0.144x the dense-equivalent multiply--accumulate count; reservoir = 256
  neurons/channel x 34 channels; E dim = 2176; PCA projection ~3.34M weights.
  Reported as computational accounting only — **no measured hardware energy.**
- **Adaptive evidence routing (honest null).** Under subject-grouped CV (seeds
  42--46) with leakage-free routers:
  - best fixed stream (E) = 0.468 balanced accuracy (pooled over all regimes);
  - perturbation-label router = 0.461; signal-quality router = 0.440;
    entropy-gated fusion = 0.424;
  - oracle upper bound = 0.942 (non-deployable).
  The deployable routers do **not** improve over the best fixed stream under the
  measured perturbation regime. The large oracle headroom confirms the streams are
  operationally distinct and separable, but label-free selection does not capture
  it here.

## Results that did NOT support the desired interpretation
- Adaptive routing did not beat the best fixed stream. This is reported honestly:
  the analysis **bounds** the practical utility of stream selection under the
  measured SHAPE ERP perturbation regime. No claim that routing recovers lost
  performance is made.

## Manuscript changes (main remains 10 pages)
- Results (robustness pillar): one sentence noting the supplemental routing
  analysis bounds the result (oracle headroom vs. no deployable improvement),
  reframing the contribution as making operating regimes observable.
- Discussion: one sentence on supplemental resource/event accounting, explicitly
  not a measured-energy result.
- Same two sentences synced to `main_blind.tex`. No figure added; four-pillar
  structure, title, and page count unchanged.

## Supplement (`manuscript/supplemental_risk_reduction/supplement.pdf`, 6 pages)
Dataset/privacy; feature-block definitions; perturbation protocol; full A0--A9
ablation; robustness tables; adaptive evidence routing (routers, nested
validation, non-deployable oracle, null result); resource/event accounting (no
energy claim); reproducibility map.

## arXiv package (`manuscript/arxiv_preprint/`)
Non-blind preprint package (main_arxiv.tex/pdf, source zip, metadata, README,
checklist, manifest). 10 pages, privacy-clean. Not submitted; posting timing and
double-anonymous-review interaction flagged for the author.

## Privacy checks
- No raw EEG, feature pickles, `clinical_profile.csv`, or PHI committed.
- Regenerated `E` embedding and `ch6_ch7` pickle stay under gitignored `data/`
  (shape pickle kept `assume-unchanged`); subject IDs hashed.
- Blind manuscript and arXiv package: no identifiers/private paths (gates pass).

## Known remaining reviewer risks
- Absolute affective accuracy is modest (chance 0.333); defended by permutation
  controls and the substrate (not classifier) framing.
- Closed-loop and routing are simulations over recorded EEG, not live deployment.
- Single restricted dataset; no public benchmark.
- Routing null means the "actionable routing" story is bounded, not demonstrated;
  the defensible contribution is observability of stream-specific operating regimes.

## Exact claim changes
- Added: streams are separable (oracle headroom) but label-free routing does not
  uniformly improve over the best fixed stream under the measured regime (bounded).
- Added: event-driven resource accounting (sparsity, event rate, runtime), with an
  explicit non-energy disclaimer.
- No new superiority, universal-robustness, diagnostic, hardware-energy, or
  physical-deployment claims.
