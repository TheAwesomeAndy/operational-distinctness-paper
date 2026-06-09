"""Ready-9 hardening package for ARSPI-Net (Paper 1, IEEE TCDS).

This sub-package adds the Ready-9 experimental-hardening layer on top of the
existing ``operational_distinctness`` and ``tcds_hardening`` pipelines. It does
not regenerate the dissertation; it consumes the private feature pickles and
emits privacy-preserving aggregate artifacts (CSV/JSON summaries, figures,
LaTeX tables, manifests) suitable for a special-issue submission.

Nothing in this package commits raw EEG, feature pickles, clinical metadata, or
subject identifiers. All committed artifacts are aggregate or hashed.
"""
