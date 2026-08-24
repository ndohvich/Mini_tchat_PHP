# Reproducible solar-radiation forecasting pipeline

A generic, offline scientific pipeline for a **user-supplied** local radiation dataset. It deliberately does not invent units, data columns, frequencies, outcomes, or physical inputs. The supplied configuration and data inspection determine what is executed; unavailable optional analyses are recorded as skips.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data and configuration

1. Put a `.csv`, `.xlsx`, or `.xls` data file under `data/raw/`.
2. Edit `configs/experiment.yaml` and set `data.path`, for example `data/raw/your_file.xlsx`.
3. Set `sheet_name`, `datetime_column`, and `target_column` when automatic detection is ambiguous. Auto-detection never chooses between multiple candidates.
4. Set only meteorological, sunrise/sunset, and site-coordinate fields actually demonstrated by your data or metadata. Do not provide inferred values.

`config.yaml` is a copy of the default configuration for tools expecting that root-level filename; `configs/experiment.yaml` is the executable default.

## Run

```bash
python run_pipeline.py
pytest -q
```

The pipeline conducts chronological train → validation → locked final-test splitting. Causal target lags and rolling features are constructed with `shift(1)`. The test segment is not passed to model selection or early stopping.

## Outputs

- `results/tables/`: quality gaps, statistics, correlations, MI, final metrics, robustness.
- `results/figures/`: publication-resolution PNG/PDF figures when inputs permit them.
- `results/predictions/`: aligned locked-test actual/predicted values.
- `results/reports/`: environment manifest, intake inventory, splits, bootstrap CI, and `scientific_audit.md`.
- `data/processed/model_input.csv`: inspected/flagged feature data.
- `logs/pipeline.log`: run events.

## Optional backends

The core pipeline needs the packages in `requirements.txt`. Install optional packages only if the relevant experiment is justified: `catboost` (central model), `pvlib` (clear-sky model needing configured coordinates), `optuna` (expanded tuning), `xgboost`, `lightgbm`, and modern forecasting libraries `neuralforecast` (N-BEATS/N-HiTS), `pytorch-forecasting` (TFT), `darts` (PatchTST), and `timesfm`. Their availability is audited without blocking the core run. A single local series is reported as a deep-learning application, not falsely as globally trained forecasting.

## Scientific safeguards

The code logs automatic decisions, preserves raw input snapshots, flags rather than silently removes anomalies, blocks ambiguous schema detection, separates daylight metrics only when supplied sunrise/sunset values parse, applies moving-block bootstrap CIs, and constrains physical models to supplied coordinates. Metric CIs are not prediction intervals. MAPE explicitly records excluded zeros; nRMSE denominator is configured and saved.
