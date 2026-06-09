# Missing inputs - risk-reduction preflight

The risk-reduction pass could not start because required inputs are absent. No downstream experiments were run and no outputs were fabricated.

## Missing

- private feature input missing: ch6_ch7_3class_features

Resolve by regenerating the private feature inputs locally (experiments/tcds_ready9/00_prepare_features.py) or pointing the ARSPI_* environment variables at consistent files, then re-run.
