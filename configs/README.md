# configs/

This directory holds experiment configuration files (YAML, TOML, or JSON).

Configuration files are tracked by Git. They must never embed secrets, dataset paths, or model checkpoint paths as hard-coded absolute paths.

## Convention

Use environment variables or relative paths for dataset and model locations. Example:

```yaml
# configs/exp001_lfcc_gmm.yaml
experiment_id: EXP-001
model: lfcc_gmm
data_root: ${DATA_ROOT}          # set in shell, not here
model_root: ${MODEL_ROOT}        # set in shell, not here
feature:
  type: lfcc
  n_coeffs: 60
  sample_rate: 16000
seed: 42
```

## Current status

No configuration files have been created yet. They will be added as experiments are designed.
