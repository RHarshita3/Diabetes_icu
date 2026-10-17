# Deployment

## Local
```bash
pip install -r requirements.txt
python -m diabetes_icu.train          # creates models/model.joblib
streamlit run app/dashboard.py        # http://localhost:8501
```
Windows one-click: `.\run_app.bat`   macOS/Linux: `bash run_app.sh`

## Docker
```bash
docker build -t diabetes-icu .
docker run -p 8501:8501 diabetes-icu
```
The image trains a fast model on the bundled synthetic data at build time. To use real data, mount it and retrain:
```bash
docker run -p 8501:8501 -v $PWD/data:/app/data diabetes-icu \
  sh -c "python -m diabetes_icu.train && streamlit run app/dashboard.py --server.address=0.0.0.0"
```

## Streamlit Community Cloud
1. Push this repo to GitHub.
2. On <https://share.streamlit.io> create an app: repo = yours, main file = `app/dashboard.py`.
3. First launch shows **Train model now**; it trains on the bundled synthetic data in about a minute.
   (For real data, commit a trained `models/model.joblib` and remove `models/*` from `.gitignore`, using the same scikit-learn version.)

## Batch scoring and monitoring (CLI)
```bash
python -m diabetes_icu.predict --input new_patients.csv --output predictions.csv
python -m diabetes_icu.drift   --data  new_patients.csv
```

## Operations checklist
* Pin library versions; models are pickled and tied to the scikit-learn version in `models/model.joblib` (see `reports/metrics.json` → `versions`).
* Run the drift report on each new batch; retrain when many features show PSI > 0.25.
* Review `reports/subgroups.csv` after every retrain.
* CI (`.github/workflows/ci.yml`) runs tests, a smoke training and a dashboard load check.
