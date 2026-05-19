# CHANGES — v20 → v21 (TCDS Special Issue submission prep)

Branch: `claude/update-arspi-net-v21-c5WNM`. The work was scoped to surgical fixes for the eight actual gaps between v20 and the brief, followed by a voice-review pass against the v19 voice samples. No experiments were re-run; no figures were regenerated; no empirical numbers changed.

## Phase A — Setup
- Installed `texlive-latex-base texlive-publishers texlive-fonts-recommended texlive-latex-extra texlive-science` and `poppler-utils` in the sandbox.
- Added `_build/` to `.gitignore`.
- Mirrored the execution plan from `/root/.claude/plans/...md` into `_workplan/v21_plan.md`.
- Baseline compile from `manuscript/main.tex` was clean: zero undefined references, zero errors, 14 pages.

## Phase B — Surgical fixes

### Fix 1: missing-citation reconciliation
The brief assumed six citations were missing from `references.bib`. They are not missing from the compile path: `main.tex` uses an inline `\begin{thebibliography}` block at lines 414-450 that contains all sixteen keys (`davies2018loihi`, `furber2014spinnaker`, `diehl2015unsupervised`, `friston2010free`, `parr2022activeinference`, `pezzulo2015active`, plus `hodgkin1952quantitative`, `lukosevicius2009reservoir`, `sussillo2009generating`, `bipoo1998stdp`, `lachaux1999plv`, `bullmore2009complex`, `honey2009structural`, `park2013structural`, `suarez2020linking`, `cohen1988power`). `references.bib` is a parallel data file not used by the current compile path. The brief's six entries were added to `references.bib` anyway, each annotated with `% \todo{verify against DOI before submission}`, to keep `references.bib` in sync with the inline bibliography for any future BibTeX migration. Andrew should verify each bibliographic record before submission (DOI, volume/issue/pages).

### Fix 2: contributions list → prose
Converted the `\begin{enumerate}` of 5 items at the close of the Introduction into a flowing prose paragraph. Preserved all five contribution ideas. Integrated the clinical FDR-bounded scope and the open-loop-on-environment-side caveat into the same paragraph, removing the need for a callout scope subsection later.

### Fix 3: deleted the claim-strength boundary table (Table IX)
Removed the `\subsection{Claim Strength}` block (Results §VI), including the `tab:claim_ladder` table and surrounding prose. Replaced with two prose sentences appended to the Discussion "Main Finding" paragraph that state what the analyses support and what they do not.

### Fix 4: deleted the redundant IEEE-style comparison subsection
Removed the Discussion subsection "Comparison to IEEE-Style EEG and Embodied-AI Papers", whose content restated material already covered in Related Work.

### Fix 5: demoted active-inference framing to sequential-decision precursor
- Renamed §IV subsection "Embodied Active-Inference Interpretation" → "Sequential Evidence Accumulation".
- Renamed Results subsection "Embodied Closed-Loop Active-Inference Demonstration" → "Sequential Evidence Accumulation Simulation".
- Softened "closed-loop active-inference agent" / "closed-loop active-inference protocol" / "active-inference sense" throughout the body and figure caption.
- Added the explicit "closed-loop on the belief side, open-loop on the environment side" caveat in four locations: abstract, intro contributions paragraph, §IV simulation subsection (including `fig:embodied` caption), and Conclusion. (Limitations §VIII already had the open-loop honesty.)
- Friston / Parr / Pezzulo citations are retained as positioning only, never as claimed instantiation.
- Keywords list swapped "active inference" → "sequential evidence accumulation".

### Fix 6: Loihi 2 specifics in hardware mappability paragraph
Replaced the generic "on the order of $10^3$ LIF neurons" wording with explicit Loihi 2 numbers:
- 8,192 neurons per neuro-core.
- 256-neuron LIF × 34 channels = 8,704 neurons total.
- Fits inside 2 of 128 neuro-cores per chip.
- Recurrent matrices are not trained on chip; can be loaded as ROM.
- PCA-64 across 34 channels ≈ 3.34 M scalar weights (dominant trainable parameter count).
- Linear readout adds 2,176 × K weights.
- Both projections run host-side.
- Replaced "Energy comparisons against published Loihi-class workloads are encouraging" with a feasibility statement on neuron / core / weight counts; a hardware benchmark with measured energy and latency is left for future work. (Brief: never claim energy without measurements.)

### Fix 7: FDR scope integrated into Introduction
Done jointly with Fix 2: the new contributions prose closes with "The clinical claim is bounded throughout by the FDR result, and the simulated loop is honest about being closed-loop on the belief side and open-loop on the environment side." The same caveat appears in the abstract close, §VIII Limitations, and Conclusion. No callout scope subsection was needed.

### Fix 8: Data and Code Availability section added; Acknowledgments expanded
- Added a new `\section*{Data and Code Availability}` containing the exact paragraph from Andrew's brief (paper repo URL, dissertation repo URL, SHAPE access-control statement). Two `\todo{anonymize for review}` markers wrap the URLs so they can be swapped for the anonymous-review link before submission.
- Expanded `\section*{Acknowledgment}` to name Brady D. Nelson and the Laboratory for Clinical Affective Neuroscience (SHAPE dataset access, clinical-context discussions), and K. Wendy Tang (doctoral advising on the dissertation work). Both wrapped in a `\todo{}` reminding Andrew to anonymize for review and restore names on acceptance.
- Defined a lightweight `\newcommand{\todo}[1]{\textbf{[TODO: #1]}}` in the preamble so todos are visible in the PDF and `grep`-able before submission.

## Phase C — Voice review pass
Read the manuscript end-to-end against the v19 voice samples in the brief. Three paragraphs were flagged and rewritten with Andrew's per-paragraph approval:

1. **Contributions paragraph (line 55, written for Fix 2).** The first draft chained five "It [verb]s..." sentences — a list dressed as prose. Rewritten so the six sentences cycle six distinct subjects (paper / it / four blocks / feature diagnostics / we / clinical claim) and one passive-voice sentence; no "It [verb]" chain remains.
2. **ERP comparison paragraph (line 329 in the Results section).** Converted "(1)~ / (2)~ / (3)~" labels in prose to "It is paid first in... / It is paid second in... / And it is paid third in..." form. The verbatim "pays in two places, paid for it in three" opener and the closing two sentences ("The headline number is reported here..." and "The contribution this paper claims is the audit...") were preserved.
3. **Introduction third paragraph (line 53).** Dropped the "Three properties make this study worth a special-issue submission rather than a benchmark report" opener (venue marketing). Replaced with "The audit draws on three properties of this study." The remaining four sentences of the paragraph are unchanged. This also resolved the duplicate "Three properties..." opener with §VII.G.

## Phase D — AI-tell verification
All grep patterns from the brief return zero hits on the final `manuscript/main.tex`:
- `\(i\)~|\(ii\)~|\(iii\)~` — zero
- `\(1\)~|\(2\)~|\(3\)~` (added as an extra check) — zero
- `\(L1\)|\(L2\)|\(L3\)|\(L4\)|\(L5\)|\(L6\)` — zero
- `what this paper does and does not claim` — zero
- `it is important to note|it should be noted` — zero
- `comprehensive|leverages?|delve into` — zero
- `we propose ARSPI-Net` — zero
- `claim-strength boundary|tab:claim_ladder` — zero
- `in conclusion` — zero

## Compile and deliverable status
- `pdflatex` runs twice from `manuscript/`, both passes succeed.
- `_build/main.pdf` is 14 pages (target: ≤14).
- Zero undefined references, zero errors, zero `^! ` lines in the log.
- No new files in `figures/`, `tables/`, `data/`, `scripts/`, `experiments/`, or `outputs/`.

## Remaining `\todo{}` markers
Three `\todo{}` markers in `manuscript/main.tex`, all visible in the PDF and `grep`-able before submission:
1. Line ~380: anonymize the paper repo URL in the Data and Code Availability section for review.
2. Line ~380: anonymize the dissertation repo URL.
3. Line ~382: replace named acknowledgments with anonymized text for review; restore on acceptance.
Six `% \todo{verify against DOI before submission}` LaTeX comments in `manuscript/references.bib` for the brief's six citations. (Note: not on the compile path; verification is precautionary for any future BibTeX migration.)

## Pre-submission checklist for Andrew
1. Replace the two GitHub URLs in `\section*{Data and Code Availability}` with the anonymous-review link supplied in the cover letter.
2. Replace named acknowledgments with anonymized text for review; restore on acceptance.
3. Verify the six bibliographic records added to `references.bib` against DOI before submission (Davies 2018 IEEE Micro; Furber 2014 Proc IEEE; Diehl & Cook 2015 Front Comput Neurosci; Friston 2010 Nat Rev Neurosci; Parr/Pezzulo/Friston 2022 MIT Press; Pezzulo/Rigoli/Friston 2015 Prog Neurobiol).
4. ScholarOne metadata: title, abstract, keywords, author order.
5. Cover letter (300-400 words) for the four guest editors (Yang, Azghadi, Li, Linares-Barranco) — happy to draft on request.
6. Coauthor sign-off from Wendy Tang and Brady Nelson.

## Acceptance-odds reassessment
The v20 manuscript Andrew handed me was already in good shape — the voice was largely in place and most AI-tell patterns were absent. The eight surgical fixes plus the three voice-pass rewrites tighten the framing in ways the brief argued for: the claim-strength table is gone (its presence would have telegraphed defensiveness to reviewers); the active-inference framing is now demoted to a precursor rather than overclaimed; the Loihi 2 hardware mappability is now stated in concrete numbers rather than gestured at; and the contributions paragraph reads as flowing prose rather than as an enumerated list. The clinical FDR boundary is honored throughout. The page count holds at 14, the upper limit. Acceptance odds for the TCDS Special Issue on Brain-Inspired Computing for Embodied AI should be improved relative to v20; the substantive empirical claims are unchanged and the framing is now more aligned with what brain-inspired-computing-for-embodied-AI reviewers will expect.
