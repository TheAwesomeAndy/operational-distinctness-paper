# ARSPI-Net TCDS v20 → v21 — Execution Plan

## Context

The user is preparing a submission for the IEEE TCDS Special Issue on Brain-Inspired Computing for Embodied AI (hard deadline 2026-08-31, personal target 2026-08-18). The v20 manuscript at `manuscript/main.tex` (452 lines) is the input. The goal is to produce v21: submission-ready prose, strategic choices preserved, no new experiments, no changed empirical numbers, voice matching the v19 samples in the brief, and zero hits on the AI-tell kill list.

**Important reconciliation between brief and reality** (decided with Andrew):
- v20 is already substantially in Andrew's v19 voice (abstract opens "We ask whether…" verbatim from the brief's calibration sample; no `(i)/(ii)/(iii)`, no `(L1)/(L2)`, no "in conclusion", no "we propose ARSPI-Net", no "comprehensive/leverages/delve", no "it is important to note"). Scope is therefore **surgical fixes first, then a voice-review second pass** — not a full per-section rewrite.
- Branch: stay on the session-provisioned `claude/update-arspi-net-v21-c5WNM` (not `v21-rewrite` from the brief).
- Manuscript root is `manuscript/main.tex`, not `./main.tex`. All figure/table includes resolve relative to `manuscript/`.
- All 11 `\includegraphics` targets and 6 `\input` targets exist on disk.
- 6 citations are used in prose but missing from `references.bib` — canonical entries supplied by Andrew will be added with `\todo{verify against DOI before submission}` markers.
- `pdflatex` not installed in the sandbox; install via apt at the start of execution.
- Plan file lives at `/root/.claude/plans/claude-code-cloud-agent-hidden-clock.md` (system-required) and will be mirrored to `_workplan/v21_plan.md` during execution.

## Critical files

| Path | Role |
|---|---|
| `manuscript/main.tex` | The single LaTeX source being rewritten (452 lines). |
| `manuscript/references.bib` | Bib database. 6 canonical entries to be added. |
| `manuscript/figures/{source_from_dissertation,generated,tcds_hardening}/` | All 11 referenced figure PDFs. Read-only. |
| `manuscript/tables/table{1..6}_*.tex` | All 6 input table fragments. Read-only. |
| `_workplan/v21_plan.md` | Mirror of this plan, committed at start of execution. |
| `_build/` | Build artifacts dir, created at execution start; .gitignored. |
| `CHANGES_v21.md` | Final per-section change log written before push. |

## Execution phases

### Phase A — Environment setup (one-shot)
1. `apt-get update && apt-get install -y texlive-latex-base texlive-publishers texlive-fonts-recommended texlive-latex-extra texlive-science`
2. `mkdir -p _build _workplan` and add `_build/` to `.gitignore` if not already.
3. Baseline compile of `manuscript/main.tex` from inside `manuscript/` (TeX includes use paths relative to source): `cd manuscript && pdflatex -interaction=nonstopmode -output-directory=../_build main.tex` twice, then grep for undefined/error/`^! `. If baseline is not clean, stop and report.
4. Copy plan to `_workplan/v21_plan.md`, `git add _workplan/v21_plan.md .gitignore`, commit `"v21: add work plan to repo"`.

### Phase B — Surgical fixes (the 8 actual gaps)
One commit per fix. Compile-clean between every commit (twice run, grep CLEAN). Push at end of each commit so Andrew can watch incrementally.

1. **Add 6 missing citations** to `references.bib` using the canonical metadata Andrew supplied:
   - `davies2018loihi` — Davies et al., IEEE Micro 38(1):82–99, 2018.
   - `furber2014spinnaker` — Furber et al., Proc. IEEE 102(5):652–665, 2014.
   - `diehl2015unsupervised` — Diehl & Cook, Front. Comput. Neurosci. 9:99, 2015.
   - `friston2010free` — Friston, Nat. Rev. Neurosci. 11(2):127–138, 2010.
   - `parr2022activeinference` — Parr/Pezzulo/Friston, MIT Press, 2022.
   - `pezzulo2015active` — Pezzulo/Rigoli/Friston, Prog. Neurobiol. 134:17–35, 2015.
   Each entry annotated with a `% \todo{verify against DOI before submission}` LaTeX comment.

2. **Contributions list → prose paragraph.** Convert the `\begin{enumerate}` at `manuscript/main.tex:54–61` into a single prose paragraph in Andrew's voice. Keep all five contribution ideas; lose the enumeration scaffolding.

3. **Delete Table IX claim-strength boundary.** Remove the `\subsection{Claim Strength}` block at `manuscript/main.tex:343–365` (table `tab:claim_ladder` and its surrounding prose). Replace with two-to-three prose sentences in the Discussion that state what the analyses support and what they do not. Confirm no other `\ref{tab:claim_ladder}` exists before deletion.

4. **Delete redundant "Comparison to IEEE-Style EEG and Embodied-AI Papers" subsection** at `manuscript/main.tex:391–393`. Merge any unique content into the Related Work section if useful; otherwise drop entirely (brief explicitly authorises deletion).

5. **Demote active-inference framing.** Rename `\subsection{Embodied Active-Inference Interpretation}` (line 165) to a sequential-decision framing. In the simulation subsection (line 321 `Embodied Closed-Loop Active-Inference Demonstration` → "Sequential Evidence Accumulation Simulation"), explicitly state the simulation is closed-loop on the belief side and open-loop on the environment side. Cite `friston2010free`, `parr2022activeinference`, `pezzulo2015active` as positioning only. Echo the open-loop caveat in the abstract, simulation subsection, limitations, and conclusion (4 locations).

6. **Add Loihi 2 specifics to Hardware mappability paragraph** (`manuscript/main.tex:376–378`): 8,192 neurons per neuro-core; 256-neuron LIF × 34 channels = 8,704 neurons total; fits in 2 of 128 cores per chip; reservoir loadable as ROM; PCA-64 across 34 channels ≈ 3.34M scalar weights (dominant trainable count, host-side); linear readout = 2,176 × K. State as feasibility, never as energy.

7. **Integrate clinical FDR scope into Introduction close** as flowing prose (replacing any callout-style scope statement). State the ERP reference-frame context as integrated prose. No subsection heading.

8. **Data Availability + Acknowledgments** — paste the exact paragraph from the brief into the Data Availability section. Acknowledgments: Brady D. Nelson + Laboratory for Clinical Affective Neuroscience (SHAPE access); K. Wendy Tang (PhD advisor). Anonymized-review placeholders kept as `\todo{}` markers.

### Phase C — Voice review pass
After Phase B compiles clean, read every section against the v19 voice samples in the brief. For each paragraph that still feels AI-flavored, use `AskUserQuestion` with a proposed rewrite and confirm before applying. Do not rewrite prose that already matches Andrew's voice. Re-run AI-tell greps after every applied rewrite.

### Phase D — AI-tell verification
Run the brief's full Phase 3 grep set against `manuscript/main.tex`. Zero hits required on every pattern. Paste output in the final summary.

### Phase E — Final deliverables
1. Page count check: `pdfinfo _build/main.pdf | grep Pages` — target ≤ 14.
2. Write `CHANGES_v21.md` with per-fix log.
3. `git add -A && git commit -m "v21: final pass, AI tells removed, compile clean"` then `git push -u origin claude/update-arspi-net-v21-c5WNM`.
4. Final report to Andrew: done-list, page count, compile status, remaining `\todo{}` markers, full AI-tell grep output (all zero), acceptance-odds reassessment, pre-submission checklist (anonymize URLs, replace placeholder ack, ScholarOne metadata, cover-letter draft on request, coauthor sign-off).

## Stop-and-ask checkpoints

Hard stops requiring `AskUserQuestion` before action:
- Deleting any figure (none planned).
- Deleting any table other than `tab:claim_ladder` (authorised).
- Changing any numerical result.
- Changing the four-block architecture description, κ formula, BSC₆ window, or reservoir parameters.
- Restructuring section order beyond Phase B items.
- Any Loihi 2 claim beyond the feasibility statement.
- A grep-survivor AI tell appearing in proposed prose.

## Compilation contract

After every commit:
```
cd manuscript && pdflatex -interaction=nonstopmode -output-directory=../_build main.tex > ../_build/build.log 2>&1
cd manuscript && pdflatex -interaction=nonstopmode -output-directory=../_build main.tex > ../_build/build.log 2>&1
grep -E "Warning.*[Uu]ndefined|Error|^! " _build/build.log | tee _build/issues.log
test ! -s _build/issues.log && echo "CLEAN" || echo "DIRTY"
```
`CLEAN` is mandatory before the next commit.

## Verification (end-to-end)

1. `pdflatex` runs twice from `manuscript/`, both pass, `_build/issues.log` empty.
2. All 8 Phase 3 grep patterns return zero hits on `manuscript/main.tex`.
3. `pdfinfo _build/main.pdf | grep Pages` ≤ 14.
4. `git log --oneline claude/update-arspi-net-v21-c5WNM` shows one commit per surgical fix plus voice-pass commits plus final compile-clean commit, all pushed to origin.
5. `CHANGES_v21.md` exists at repo root, describing every edit by section.
6. No new files in `figures/`, `tables/`, `data/`, `scripts/`, `experiments/`, or `outputs/` — empirical artifacts are untouched.

## Non-goals

No new experiments, no script edits, no figure regeneration, no dataset/protocol/methods changes, no number changes, no >14 pages, no merging the branch, no PR creation (Andrew opens the PR locally).
