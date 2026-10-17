"""Data-drift monitoring with the Population Stability Index (PSI).

The reference distribution is stored in the model artifact at training time.
Numeric features use decile bins from the training data plus a dedicated
"missing" bin (so shifts in missingness are caught too); categorical features use
category frequencies with an "other" bucket for unseen values.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config

_EPS = 1e-4
MISSING = "__missing__"
OTHER = "__other__"


def build_drift_reference(cleaned: pd.DataFrame) -> dict:
    ref = {}
    for col in cleaned.columns:
        s = cleaned[col]
        if pd.api.types.is_numeric_dtype(s):
            vals = s.dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            edges = np.unique(np.quantile(vals, np.linspace(0.1, 0.9, 9)))
            if len(edges) < 2:  # flag-like feature -> treat as categorical
                ref[col] = {"type": "categorical", "freq": _freq(s.astype(object))}
                continue
            ref[col] = {"type": "numeric", "edges": edges.tolist(),
                        "freq": _numeric_freq(s, edges)}
        else:
            ref[col] = {"type": "categorical", "freq": _freq(s)}
    return ref


def _numeric_freq(s: pd.Series, edges) -> list[float]:
    s = pd.to_numeric(s, errors="coerce")
    n = max(len(s), 1)
    miss = float(s.isna().sum()) / n
    idx = np.digitize(s.dropna().to_numpy(dtype=float), edges)
    counts = np.bincount(idx, minlength=len(edges) + 1) / n
    return counts.tolist() + [miss]


def _freq(s: pd.Series) -> dict:
    n = max(len(s), 1)
    out = {MISSING: float(s.isna().sum()) / n}
    for k, v in s.dropna().astype(str).value_counts().items():
        out[k] = float(v) / n
    return out


def psi(expected, actual) -> float:
    e = np.clip(np.asarray(expected, dtype=float), _EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status_for(value: float) -> str:
    if value >= config.PSI_ALERT:
        return "alert"
    if value >= config.PSI_WARN:
        return "warning"
    return "stable"


def compute_drift(reference: dict, new_data: pd.DataFrame) -> pd.DataFrame:
    """PSI per feature of ``new_data`` (already cleaned) against the reference."""
    rows = []
    for col, spec in reference.items():
        if col not in new_data.columns:
            continue
        s = new_data[col]
        if spec["type"] == "numeric":
            actual = _numeric_freq(s, np.asarray(spec["edges"]))
            value = psi(spec["freq"], actual)
        else:
            ref_freq = spec["freq"]
            keys = list(ref_freq.keys()) + [OTHER]
            act = {k: 0.0 for k in keys}
            n = max(len(s), 1)
            act[MISSING] = float(s.isna().sum()) / n
            for k, v in s.dropna().astype(str).value_counts().items():
                act[k if k in ref_freq else OTHER] += float(v) / n
            value = psi([ref_freq.get(k, 0.0) for k in keys], [act[k] for k in keys])
        rows.append({"feature": col, "psi": value, "status": status_for(value)})
    out = pd.DataFrame(rows, columns=["feature", "psi", "status"])
    return out.sort_values("psi", ascending=False).reset_index(drop=True)


def summarize_drift(report: pd.DataFrame) -> dict:
    if report.empty:
        return {"overall": "unknown", "n_alert": 0, "n_warning": 0, "n_features": 0}
    n_alert = int((report["status"] == "alert").sum())
    n_warn = int((report["status"] == "warning").sum())
    overall = "alert" if n_alert else ("warning" if n_warn else "stable")
    return {"overall": overall, "n_alert": n_alert, "n_warning": n_warn,
            "n_features": int(len(report))}


def main(argv=None) -> None:
    from .predict import load_artifact

    parser = argparse.ArgumentParser(description="PSI drift report vs. training data")
    parser.add_argument("--data", required=True, help="CSV with new patient records")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args(argv)
    art = load_artifact()
    raw = pd.read_csv(args.data)
    cleaned = art["pipeline"].named_steps["clean"].transform(raw)
    report = compute_drift(art["drift_reference"], cleaned)
    print(summarize_drift(report))
    print(report.head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
