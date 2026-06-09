# Anonymization plan — double-blind submission

IEEE TCDS review is double-blind. Two synchronized manuscript versions are
maintained:

- **Non-blind (author) version:** `manuscript/main.tex` → `main.pdf` (10 pp).
- **Blind (review) version:** `manuscript/main_blind.tex` → `main_blind.pdf`
  (10 pp), generated reproducibly from `main.tex` by the substitutions below.

The blind version is what is uploaded to ScholarOne for review; the non-blind
version is restored on acceptance.

## Substitutions applied for the blind version
| # | Location | Non-blind | Blind |
|---|---|---|---|
| 1 | `\author{}` | "Andrew A. Lane, K. Wendy Tang, and Brady D. Nelson" | "Anonymous Author(s) — Submitted for double-blind review" |
| 2 | Acknowledgment | names the lab, B. D. Nelson, K. W. Tang | "Acknowledgments are withheld for double-blind review and will be restored on acceptance." |
| 3 | Data and Code Availability — institution | "Laboratory for Clinical Affective Neuroscience at Stony Brook University" | "the originating clinical research laboratory (identity withheld for double-blind review)" |
| 4 | Data and Code Availability — repository | "link provided under approved-access and review conditions" | "link withheld for double-blind review" |
| 5 | Dataset name | "SHAPE Community dataset" / "SHAPE ERP regime" | "access-controlled study dataset" / "measured ERP regime" |
| 6 | Related Work + source-figure captions | "prior ARSPI-Net work" / "reproduced from prior work" | "earlier work on this architecture" / "reproduced from earlier work" |

The methodological `\thanks` footnote (permutation-test and FDR description)
contains no identifying information and is retained in both versions.

## Verification of the blind version
- Compiles to 10 pages, 0 undefined references, no overflow.
- Grep confirms no residual identifiers: author names, "Stony Brook",
  "Laboratory for Clinical Affective Neuroscience".
- The system name "ARSPI-Net" is retained (it is the paper's subject and appears
  in the title); only self-revealing *prior-work* phrasing is neutralized. If the
  special-issue chairs require the system name itself to be masked, replace
  "ARSPI-Net" with a placeholder in a final pass.

## Restoration on acceptance
Re-run the inverse of the table above, or simply submit the camera-ready from
`main.tex`. Keep `main_blind.tex` only for the review cycle.

## Items still requiring the author before ScholarOne upload
- Confirm the ORCID / author-order / affiliation block for the camera-ready
  (non-blind) version.
- Confirm whether the special issue requires the system name masked (see above).
- Provide the de-anonymized repository link for the camera-ready Data and Code
  Availability statement.
