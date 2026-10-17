"""Data loading and cleaning.

``DataCleaner`` is a scikit-learn transformer so the exact same cleaning
(dropped columns, winsorisation bounds, dtype coercion) is applied at training
time and at prediction time inside the saved pipeline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from . import config


def find_data_path(explicit: str | Path | None = None) -> Path:
    """Resolve the training CSV: explicit arg > env var > known file names in data/."""
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        return path
    env = config.env_data_path()
    if env is not None:
        if not env.exists():
            raise FileNotFoundError(f"{config.DATA_ENV_VAR} points to a missing file: {env}")
        return env
    for name in config.DATA_CANDIDATES:
        candidate = config.DATA_DIR / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No dataset found in data/. Run `python -m diabetes_icu.synthetic` to create "
        "the demo data, or copy the WiDS training CSV into data/ (see data/README.md)."
    )


def detect_target(columns) -> str:
    for name in config.TARGET_CANDIDATES:
        if name in columns:
            return name
    raise ValueError(
        f"Could not find a target column. Expected one of {config.TARGET_CANDIDATES}."
    )


def load_dataset(path: str | Path, sample: int | None = None,
                 random_state: int = config.RANDOM_STATE):
    """Return (X, y, target_name). Optionally draw a stratified subsample."""
    df = pd.read_csv(path)
    target = detect_target(df.columns)
    df = df.dropna(subset=[target]).reset_index(drop=True)
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    if sample and sample < len(df):
        X, _, y, _ = train_test_split(
            X, y, train_size=sample, stratify=y, random_state=random_state
        )
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    return X, y, target


class DataCleaner(BaseEstimator, TransformerMixin):
    """Drop IDs / mostly-missing / constant columns, coerce dtypes, winsorise outliers.

    Missing values are intentionally kept (NaN) - imputation is a modelling choice
    made downstream, and missingness itself is often informative in ICU data.
    """

    def __init__(self, id_columns=config.ID_COLUMNS,
                 max_missing=config.MAX_MISSING_FRACTION,
                 clip_quantiles=config.CLIP_QUANTILES):
        self.id_columns = id_columns
        self.max_missing = max_missing
        self.clip_quantiles = clip_quantiles

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        dropped = {}
        for col in X.columns:
            if col in self.id_columns:
                dropped[col] = "identifier"
        X = X.drop(columns=list(dropped))
        miss = X.isna().mean()
        for col in X.columns:
            if miss[col] > self.max_missing:
                dropped[col] = f"{miss[col]:.0%} missing"
            elif X[col].nunique(dropna=True) <= 1:
                dropped[col] = "constant"
        keep = [c for c in X.columns if c not in dropped]
        self.numeric_ = [c for c in keep if pd.api.types.is_numeric_dtype(X[c])]
        self.categorical_ = [c for c in keep if c not in self.numeric_]
        self.columns_ = self.numeric_ + self.categorical_
        self.dropped_ = dropped
        lo_q, hi_q = self.clip_quantiles
        self.bounds_ = {}
        for col in self.numeric_:
            s = pd.to_numeric(X[col], errors="coerce")
            if s.nunique(dropna=True) > 10:  # never clip flags / small-cardinality codes
                self.bounds_[col] = (float(s.quantile(lo_q)), float(s.quantile(hi_q)))
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = {}
        for col in self.numeric_:
            s = (pd.to_numeric(X[col], errors="coerce").astype(float)
                 if col in X.columns else pd.Series(np.nan, index=X.index))
            if col in self.bounds_:
                s = s.clip(*self.bounds_[col])
            out[col] = s
        for col in self.categorical_:
            if col in X.columns:
                s = X[col].astype("object")
                s = s.where(s.isna(), s.astype(str))
            else:
                s = pd.Series(np.nan, index=X.index, dtype="object")
            out[col] = s
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_, dtype=object)
