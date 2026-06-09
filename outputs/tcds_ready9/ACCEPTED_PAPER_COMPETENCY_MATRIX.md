# Accepted-paper competency matrix (INTERNAL planning artifact)

> Internal only. This comparison against named accepted papers is a planning aid
> and must NOT appear in the manuscript. The manuscript uses the neutral
> `table_evaluation_coverage.tex` (rows = ARSPI-Net components; columns =
> system-level evaluation requirements). This file is not a reviewer-facing
> rhetorical device and makes no superiority claim — it identifies where the
> ARSPI-Net evaluation is stronger, weaker, or simply different.

Legend: Y = addressed/reported, P = partial, N = not applicable / not reported.

| System / method class | Signal source | Provenance detail | Preproc. visible | Math observable | Neural mechanism | Feature mechanism | Model comparison | Ablation | Robustness/sensitivity | Confusion matrices | Application demo | Reproducibility artifacts | Limitations stated | Dataset status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Multimodal miner fatigue recognition | multimodal physio | P | P | P | N | Y | Y | P | P | Y | Y | P | Y | mixed |
| Touch-gesture pHRI recognition | tactile array | P | P | P | N | Y | Y | P | P | Y | Y | P | Y | mixed |
| Biomimetic SNN auditory-brainstem | audio/spikes | P | P | Y | Y | Y | P | P | P | P | Y | P | Y | public-ish |
| Companion-robot multimodal emotion | multimodal | P | P | P | N | Y | Y | P | P | Y | Y | P | Y | mixed |
| Graph-Laplacian localization backend | sensor graph | P | P | Y | N | Y | Y | N | P | N | Y | P | Y | mixed |
| **ARSPI-Net (this paper)** | clinical EEG/ERP | Y | Y | Y | Y | Y | Y | Y | Y (A0–A9 + controls) | Y (4 families + raw subset) | Y | Y (manifests, fingerprints, map) | Y | restricted/private |

## Where ARSPI-Net is *stronger*
- Explicit mechanism-to-implementation mapping with mathematical observables.
- Provenance + QC + subject-level splits documented as verifiable facts.
- Mechanism ablation with negative controls and uncertainty.
- Perturbation robustness across four families; defined closed-loop controller.
- Reproducibility manifests with input fingerprints.

## Where ARSPI-Net is *weaker / different*
- Dataset is private/restricted (no public benchmark redistribution), unlike some
  comparison systems on public corpora.
- No physical-robot deployment and no measured hardware energy (deliberately not
  claimed).
- Single affective-EEG regime (SHAPE ERP); not a universal EEG-GNN claim.
- Closed-loop is a simulation over recorded observations, not live embodiment.
