# 🩺 ICU Diabetes Mellitus Screening

Machine-learning project for the MathWorks **Predicting Diabetes Mellitus** challenge
(<https://github.com/mathworks/predicting-diabetes-mellitus>), rebuilt in Python: predict whether an
ICU patient has diabetes mellitus from admission data, then explain, monitor and serve the model.

![CI](https://github.com/YOUR_USERNAME/diabetes_icu_screening/actions/workflows/ci.yml/badge.svg)

## Quick start

**Windows (PowerShell / Cursor terminal)** - one command does everything:
```powershell
.\run_app.bat
```
**macOS / Linux:** `bash run_app.sh`

Or step by step:
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
python -m diabetes_icu.synthetic      # demo data (already included)
python -m diabetes_icu.train          # trains + writes models/ and reports/
streamlit run app/dashboard.py        # opens http://localhost:8501
pytest                                # 8 tests
```
In Cursor: *File → Open Folder* on this folder, then `Ctrl+Shift+P → Python: Select Interpreter → .venv`.
Streamlit apps are started with `streamlit run`, not the ▶ Run button.

## What's inside

| Area | What it does | Where |
|---|---|---|
| **Data mastery** | Drops IDs / constant / mostly-missing columns, winsorises outliers, keeps NaNs and lets each model handle them, robust to unseen columns and categories | `data_utils.py` |
| **Model selection** | Logistic-regression baseline vs. gradient boosting, stratified 5-fold CV, best by ROC-AUC | `features.py`, `train.py` |
| **Screening threshold** | F2-optimised on out-of-fold predictions (recall-weighted), adjustable live | `evaluate.py` |
| **Evaluation** | ROC/PR AUC, Brier, calibration, confusion matrix, sensitivity/specificity, **subgroup fairness table** | `evaluate.py` |
| **Explainability** | Permutation importance (global) + occlusion explanations in plain language (local), no SHAP dependency | `explain.py` |
| **Drift monitoring** | PSI per feature incl. missingness; CLI + dashboard | `drift.py` |
| **Risk trajectory** | Score time-ordered ICU records and flag rising risk; what-if sweeps | `risk_trajectory.py` |
| **Dashboard** | Performance, patient risk with explanation, batch scoring + drift, trajectories, model card | `app/dashboard.py` |
| **Engineering** | Tests, CI (GitHub Actions), Dockerfile, model card, deployment guide | `tests/`, `.github/`, `docs/` |

## Data

Out of the box the project trains on **synthetic** data (`data/sample_icu_synthetic.csv`), so every
metric shown with it describes simulated patients. To use the real challenge data (WiDS Datathon 2021,
Kaggle account required) put `TrainingWiDS2021.csv` in `data/` and run `python -m diabetes_icu.train`.
Details in [`data/README.md`](data/README.md). The `exampleTrain` file in the MathWorks repo is random
dummy data (ROC-AUC ≈ 0.50 for any model, as their tutorial notes); the pipeline runs on it unchanged.

Reference run on the bundled synthetic data: ROC-AUC ≈ 0.91, PR-AUC ≈ 0.83, sensitivity ≈ 0.85 (gradient boosting beat the
logistic baseline, CV 0.913 vs 0.875). Your numbers will differ slightly by library version.

## Project layout
```
app/dashboard.py            Streamlit dashboard
src/diabetes_icu/           config, data_utils, features, train, evaluate, explain, drift,
                            risk_trajectory, predict, synthetic
tests/test_pipeline.py      end-to-end tests
docs/                       MODEL_CARD.md, DEPLOYMENT.md
data/  models/  reports/    data, saved model, plots + metrics (generated)
```

## Command line
```bash
python -m diabetes_icu.train --data data/TrainingWiDS2021.csv [--fast] [--sample 40000]
python -m diabetes_icu.predict --input new_patients.csv --output predictions.csv
python -m diabetes_icu.drift   --data  new_patients.csv
```

## Troubleshooting
* `ModuleNotFoundError: No module named 'joblib'` → you are using the wrong Python (e.g. MSYS2).
  Create/activate `.venv` and select it as the interpreter.
* `No module named 'diabetes_icu'` → run `pip install -r requirements.txt` (installs the package in editable mode).
* Dashboard says *No trained model* → click **Train model now** or run `python -m diabetes_icu.train`.
* Training on the full real dataset is slow → add `--fast --sample 40000`.

## Disclaimer & license
Educational decision-support prototype, **not a medical device**. MIT licensed. Dataset and problem
statement © MathWorks / WiDS Datathon organisers; no MathWorks code is included.
