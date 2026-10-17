"""Central configuration: paths, column conventions and modelling constants."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
DOCS_DIR = ROOT / "docs"

MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"

# Target column names we recognise (WiDS 2021 uses diabetes_mellitus,
# the MathWorks example dataset uses OutputVar).
TARGET_CANDIDATES = ("diabetes_mellitus", "OutputVar", "target")

# Identifier columns that must never be used as predictors.
ID_COLUMNS = (
    "encounter_id",
    "patient_id",
    "hospital_id",
    "icu_id",
    "Unnamed: 0",
    "Unnamed: 0.1",
)

# Searched in this order when no --data argument is given.
DATA_CANDIDATES = (
    "TrainingWiDS2021.csv",
    "exampleTrain.csv",
    "sample_icu_synthetic.csv",
)
DATA_ENV_VAR = "DIABETES_ICU_DATA"

SYNTHETIC_DATA_PATH = DATA_DIR / "sample_icu_synthetic.csv"
SAMPLE_TRAJECTORY_PATH = DATA_DIR / "sample_trajectory.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_MISSING_FRACTION = 0.60  # columns missing more than this are dropped
CLIP_QUANTILES = (0.005, 0.995)  # winsorisation bounds for continuous features
SCREENING_BETA = 2.0  # F-beta used to choose the threshold (recall-weighted)

PSI_WARN = 0.10
PSI_ALERT = 0.25


def ensure_dirs() -> None:
    for d in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def env_data_path() -> Path | None:
    value = os.environ.get(DATA_ENV_VAR)
    return Path(value) if value else None
