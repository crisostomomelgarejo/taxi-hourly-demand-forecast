# Taxi Hourly Demand Forecast

TripleTen Data Scientist course project (2025). The notebook is in Spanish; this README is in English.

**This is a course dataset, not a production deployment.** The forecasts were not used to dispatch real taxis.

## Problem

A taxi company wants a one-hour-ahead forecast of ride orders so that it can staff the fleet in time. The success criterion set by the course is a test RMSE (RECM) **no higher than 48** orders per hour.

## Data source

- Course file: `/datasets/taxi.csv` (26,496 observations at 10-minute intervals, March–August 2018, as stated in the notebook).

The file is part of the TripleTen Data Scientist program and **may not be redistributable**. It is **not** included in this repository.

How to obtain it: download `taxi.csv` from the TripleTen project page and either keep it at `/datasets/taxi.csv` (the path used in the notebook) or edit the path to a local `datasets/taxi.csv`.

## Approach

1. Inspect the series and resample the 10-minute counts to hourly totals.
2. Decompose the series (trend and daily seasonality).
3. Build 29 features: calendar fields (month, day, day of week, hour), 24 hourly lags, and a 24-hour rolling mean, with shifts applied before rolling so that the target hour is not leaked.
4. Time-series split (5 folds) and `GridSearchCV` for linear regression, a decision tree, a random forest, and LightGBM, using a custom tuner class.
5. Refit the selected model and score it on the held-out test window.

## Results (from executed notebook outputs)

Cross-validation RMSE (RECM CV):

| Model | RECM CV | Train time (s) | Predict time (s) |
|---|---:|---:|---:|
| LightGBM | 24.7620 | 2.834 | 0.0053 |
| Random Forest | 25.2409 | 26.549 | 0.0560 |
| Linear regression | 26.9450 | 0.004 | 0.0031 |
| Decision tree | 30.4741 | 0.472 | 0.0018 |

Selected model: **LightGBM** (`learning_rate=0.1`, `max_depth=5`, `n_estimators=100`).

**Test RMSE (RECM): 41.22**, below the required threshold of 48.00 — printed result: `APROBADO`.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook taxi_hourly_demand_forecast.ipynb
```

The notebook was already executed; outputs (including figures) are saved in the file.

## Requirements

See `requirements.txt`.
