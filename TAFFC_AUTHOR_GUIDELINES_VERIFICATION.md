# IEEE Transactions on Affective Computing — Author-Guidelines Verification

**Date of verification:** 2026-06-12.

**Status legend:** ✓ verified from an authoritative source · ⚠ best-available default, confirm on
the ScholarOne portal at submission (the journal's JavaScript-rendered author page could not be
machine-read during this verification).

## 1. Manuscript type selected
✓ **Regular Paper**, IEEE Transactions on Affective Computing (full-length research article). A
special-issue track exists when a relevant call is open; absent an open call, this is a Regular
Paper.

## 2. Page limit / overlength policy
✓ **12 free formatted pages** for Transactions regular papers (including references and author
biographies); **$200 per overlength page**; **18 formatted pages** hard maximum. Source: IEEE
Computer Society Mandatory Overlength Page Charges policy, which **explicitly lists IEEE
Transactions on Affective Computing**.
- **Implication for ARSPI-Net:** the manuscript is **10 formatted pages** → within the free limit,
  no overlength charge. Reuse `manuscript/main.pdf`; **no overlay, no manuscript edits**.

## 3. Supplemental material allowed?
✓ **Yes**, and encouraged: "supporting but nonessential information should be submitted as
supplemental material." The 6-page `supplement.pdf` is uploaded as supplemental material.

## 4. Review model (single- vs double-blind)
⚠ **Single-blind** — the IEEE Transactions default and TAFFC's historical model. Not stated on the
machine-readable sources fetched; confirm on the portal. The package is robust either way:
`manuscript/main.pdf` (non-anonymized) for single-blind, or `manuscript/main_blind.pdf`
(anonymized) if double-blind is required. Both are compiled.

## 5. File-format requirements
✓ / ⚠ IEEE two-column journal format (the "formatted pages" the page policy refers to), prepared
with the IEEEtran class. **PDF** for review; **LaTeX source or Word** on acceptance; vector /
high-resolution figures. Confirm any portal-specific upload requirements at submission.

## 6. Open access vs traditional
✓ **Hybrid journal**, two options: **traditional** (no mandatory article-processing charge) or
optional **open access** at the **2026 IEEE hybrid-journal APC of US$2,800**. Source: 2026 IEEE
Publications APC list / IEEE Open. Confirm the current amount and any society/member discount at
submission.

## 7. Editor-in-Chief
✓ **Prof. Jesse Hoey** (David R. Cheriton School of Computer Science, University of Waterloo) is
the current Editor-in-Chief (per his University of Waterloo and Vector Institute biographies; IEEE
Computer Society has announced a search for a 2027 successor, so his term runs through 2026). The
TAFFC cover letter addresses him by name. Re-verify on the journal masthead at submission.

## 8. Portal-specific cover-letter language
✓ IEEE Computer Society submission requires that any **prior or related conference version** of the
work be **disclosed in the cover letter**, with the new contributions described. ARSPI-Net's
manuscript cites two prior IEEE LISAT conference papers (2023, 2024) in the ARSPI-Net development
line, so the TAFFC cover letter now includes a **prior-work disclosure paragraph** stating that
this journal manuscript substantially extends that earlier work. The cover letter (optional but
recommended) also carries the standard originality / not-under-consideration / all-authors-approve
statements.

## 9. Sources used
- IEEE CS Mandatory Overlength Page Charges policy (PDF):
  https://ieeecs-media.computer.org/assets/pdf/MOPC_policy.pdf — lists TAFFC; 12 free pages,
  $200/page, 18-page maximum, supplemental encouraged.
- 2026 IEEE Publications APC list (PDF):
  https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE-Article-Processing-Charges-List.pdf
  and https://open.ieee.org/for-authors/article-processing-charges/ — hybrid OA APC US$2,800 (2026).
- Editor-in-Chief: https://cs.uwaterloo.ca/~jhoey/ and https://vectorinstitute.ai/team/jesse-hoey/
  — Jesse Hoey, EIC of IEEE Transactions on Affective Computing.
- IEEE CS TAFFC journal page: https://www.computer.org/csdl/journal/ta ; author resources:
  https://www.computer.org/publications/author-resources.
- ScholarOne submission portal: https://mc.manuscriptcentral.com/taffc-cs (IEEE CS pattern; the
  site responds and is login-gated — HTTP 403 to an unauthenticated fetch — consistent with the
  correct portal; confirm the journal name on the portal).

## 10. Packaging decision
TAFFC permits 12 free formatted pages. The TAFFC submission uses an **additive manuscript variant**,
`manuscript/taffc/main_taffc.tex` / `.pdf` (with a double-blind twin `main_taffc_blind`), that
foregrounds physiological affective computing and compiles to **11 pages** — within the free limit.
The shared `manuscript/main.tex` / `.pdf` and `main_blind` (the TCDS manuscript) are preserved
byte-for-byte; this TAFFC layer is purely additive.
