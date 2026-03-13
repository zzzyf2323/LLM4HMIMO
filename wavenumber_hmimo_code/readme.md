# Wavenumber HMIMO

## 1. Core static front-end
The static operator-form HMIMO wavenumber-domain estimator is implemented in:

- `core/demo_sbl_block_operator_fixed_angles_report.py`

This file should be kept unchanged and is treated as the reference implementation.

## 2. Experiments
- `python -m experiments.demo_compare_beamforming_mechanisms_dft_strictAligned`
- `python -m experiments.build_temporal_dataset`
- `python -m experiments.train_predictor`
- `python -m experiments.run_predictive_warmstart`

## 3. Data
Processed datasets are stored in:
- `data/processed`

Model checkpoints are stored in:
- `data/checkpoints`

## 4. Outputs
Figures:
- `outputs/figures`

Logs:
- `outputs/logs`

CSV:
- `outputs/csv`