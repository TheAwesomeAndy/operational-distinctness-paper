# Base-branch diff audit

**Purpose.** Document the base from which the submission branch was created, per
the requirement to inspect the working branch before branching.

## Finding

At audit time, the working branch was
**byte-identical to `origin/main`**:

```
git rev-parse HEAD            -> f18f2d332267f83edb09c276fb85e4264c4d195b
git rev-parse origin/main     -> f18f2d332267f83edb09c276fb85e4264c4d195b
git log --oneline origin/main..HEAD   -> (empty)
git log --oneline HEAD..origin/main   -> (empty)
```

There are **no unreviewed commits** in either direction. The working branch
already contains the v21 manuscript and the `experiments/tcds_hardening/`
infrastructure that this package reuses.

## Decision

The submission branch `submission/tcds-ready9-substrate` was created from this
state, which is equivalent to `origin/main`. No content from an unreviewed work
branch is introduced. The package will be opened as a pull request into `main`.

## Method

```
git switch -c submission/tcds-ready9-substrate   # from HEAD == origin/main
```
