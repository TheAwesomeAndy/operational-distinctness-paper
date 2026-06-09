# Feature alignment error

Phase 2 detected a mismatch between the consumed private inputs. Downstream Ready-9 analyses were not run, and no fabricated outputs were produced.

## Failures

- ch6_ch7_3class_features.pkl is missing and could not be regenerated locally from raw EEG.

Resolve the input mismatch (regenerate the feature pickle from the matching raw EEG, or point the environment variables at consistent files) and re-run.
