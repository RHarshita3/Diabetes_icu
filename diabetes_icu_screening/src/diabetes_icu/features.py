"""Model pipelines and feature metadata used by the dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config
from .data_utils import DataCleaner

MODEL_KINDS = ("logistic", "hist_gradient_boosting")
MODEL_LABELS = {
    "logistic": "Logistic regression (baseline)",
    "hist_gradient_boosting": "Histogram gradient boosting",
}


def _selectors():
    return (make_column_selector(dtype_include=np.number),
            make_column_selector(dtype_exclude=np.number))


def build_pipeline(kind: str, fast: bool = False,
                   random_state: int = config.RANDOM_STATE) -> Pipeline:
    num_sel, cat_sel = _selectors()
    if kind == "logistic":
        pre = ColumnTransformer([
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]), num_sel),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01,
                                         sparse_output=False)),
            ]), cat_sel),
        ])
        model = LogisticRegression(max_iter=3000, C=0.5)
    elif kind == "hist_gradient_boosting":
        # Trees handle NaN natively, so no imputation is needed.
        pre = ColumnTransformer([
            ("num", "passthrough", num_sel),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=np.nan,
                                   encoded_missing_value=np.nan), cat_sel),
        ])
        model = HistGradientBoostingClassifier(
            max_iter=150 if fast else 400,
            learning_rate=0.06,
            max_leaf_nodes=24,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    return Pipeline([("clean", DataCleaner()), ("prep", pre), ("model", model)])


def _nice_step(span: float, integer: bool) -> float:
    if integer:
        return 1.0
    if span <= 0:
        return 0.1
    return float(10 ** np.floor(np.log10(span / 100.0)))


def build_feature_specs(cleaned: pd.DataFrame) -> dict:
    """Describe every cleaned feature so the dashboard can build input widgets."""
    specs = {}
    for col in cleaned.columns:
        s = cleaned[col]
        if pd.api.types.is_numeric_dtype(s):
            vals = s.dropna()
            if vals.empty:
                continue
            uniq = sorted(vals.unique().tolist())
            if len(uniq) <= 2:
                specs[col] = {"kind": "binary", "options": [float(u) for u in uniq],
                              "default": float(vals.mode().iloc[0])}
            else:
                integer = bool(np.allclose(vals, np.round(vals)))
                lo, hi = float(vals.min()), float(vals.max())
                specs[col] = {"kind": "numeric", "min": lo, "max": hi,
                              "median": float(vals.median()), "integer": integer,
                              "step": _nice_step(hi - lo, integer)}
        else:
            vals = s.dropna().astype(str)
            if vals.empty:
                continue
            options = vals.value_counts().index.tolist()[:50]
            specs[col] = {"kind": "categorical", "options": sorted(options),
                          "default": options[0]}
    return specs


def build_reference(cleaned: pd.DataFrame) -> dict:
    """Typical value per feature (median / mode) used for explanations."""
    ref = {}
    for col in cleaned.columns:
        s = cleaned[col].dropna()
        if s.empty:
            ref[col] = np.nan
        elif pd.api.types.is_numeric_dtype(cleaned[col]):
            ref[col] = float(s.median())
        else:
            ref[col] = str(s.mode().iloc[0])
    return ref
