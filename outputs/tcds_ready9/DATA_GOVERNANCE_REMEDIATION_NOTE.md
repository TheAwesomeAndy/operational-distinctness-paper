# Data-governance remediation note (internal)

This is an **internal** governance record. It is not a manuscript-facing
artifact and its contents are not used as claims in the paper.

## Current repository state (observed at audit time)

The following private inputs were found to be **already tracked** in repository
history (committed prior to this task, present on `origin/main`):

| Artifact | Tracked files | Class |
|---|---|---|
| `data/batch_data_full/SHAPE_Community_*_IAPS*_BC.txt` | 636 | raw EEG |
| `data/shape_features_211.pkl` | 1 | subject-level feature pickle |
| `data/clinical_profile.csv` | 1 | clinical metadata |
| `data/ch6_ch7_3class_features.pkl` | 0 (regenerated locally, **not** committed) | feature pickle |

A `.gitignore` already lists `data/`, `*.pkl`, and `clinical_profile.csv`, but
those patterns do not untrack files that were committed before the ignore rules
existed.

## Policy applied by this task

- **No new private data artifacts are added** by this package. In particular the
  locally regenerated `ch6_ch7_3class_features.pkl` is written under the
  gitignored `data/` directory and is **never committed**.
- All manuscript-facing outputs produced here use **aggregate or hashed**
  summaries only. Subject identifiers are SHA-256 hashed; raw absolute paths are
  reduced to basenames in fingerprints.
- The manuscript reports aggregate, deidentified results and adopts
  restricted-data reproducibility language. It does **not** describe the
  already-tracked private files as a strength, and it does not discuss
  repository-history issues.

## Out of scope (requires separate authorization)

Repository-history remediation — i.e. purging the already-committed raw EEG,
feature pickle, and clinical CSV from all commits (e.g. `git filter-repo`) and
force-pushing — is **destructive, rewrites shared history, and is outside this
task**. It requires explicit authorization from the repository owner / data
steward and coordination with anyone holding clones or open branches/PRs. This
note records the condition so that decision can be made deliberately.
