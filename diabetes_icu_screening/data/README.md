# Data

| File | What it is |
|---|---|
| `sample_icu_synthetic.csv` | 12,000 **simulated** ICU patients (no real data). Used by default so the project runs out of the box. Regenerate with `python -m diabetes_icu.synthetic`. |
| `sample_trajectory.csv` | One simulated patient with 8 time-ordered records, used by the *Risk trajectory* tab. |

## Using the real challenge data

The MathWorks problem statement points to the **WiDS Datathon 2021** dataset on Kaggle
(<https://www.kaggle.com/competitions/widsdatathon2021/data>). It requires a free Kaggle account
and accepting the competition rules, so it cannot be bundled here.

1. Download `TrainingWiDS2021.csv` and save it in this folder.
2. Train: `python -m diabetes_icu.train` (auto-detected, target `diabetes_mellitus`).
   Large file? Try `python -m diabetes_icu.train --fast --sample 40000` first.
3. Restart the dashboard (or click **Retrain** in the sidebar).

The pipeline needs no code changes: it recognises the target column, drops identifier columns
(`encounter_id`, `hospital_id`, `icu_id`, ...), handles 100+ numeric and categorical columns, missing
values and outliers automatically.

## The MathWorks example file

`exampleTrain.zip` in the MathWorks repo is a *dummy* dataset (random values, `OutputVar` target).
It works with this pipeline (`--data path/to/exampleTrain.csv`) but, as the MathWorks tutorial notes,
no model can learn from it: expect ROC-AUC ≈ 0.50.
