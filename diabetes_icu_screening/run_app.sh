#!/usr/bin/env bash
# One-click setup + launch for macOS / Linux:  bash run_app.sh
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
[ -f data/sample_icu_synthetic.csv ] || [ -f data/TrainingWiDS2021.csv ] || python -m diabetes_icu.synthetic
[ -f models/model.joblib ] || python -m diabetes_icu.train --fast
streamlit run app/dashboard.py
