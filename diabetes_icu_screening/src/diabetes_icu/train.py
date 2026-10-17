"""Train, compare and persist the screening model.

    python -m diabetes_icu.train                 # auto-detect dataset in data/
    python -m diabetes_icu.train --data data/TrainingWiDS2021.csv
    python -m diabetes_icu.train --fast --sample 20000

Steps: load -> stratified train/test split -> 5-fold CV of two candidate models
(logistic baseline vs gradient boosting) -> choose best by CV ROC-AUC -> pick a
recall-oriented threshold from out-of-fold predictions -> evaluate once on the
held-out test set -> permutation importance, subgroup analysis, drift reference ->
save everything into one artifact (models/model.joblib) plus reports/.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from . import config
from .data_utils import find_data_path, load_dataset
from .drift import build_drift_reference
from .evaluate import best_threshold, compute_metrics, make_plots, subgroup_table
from .features import (MODEL_KINDS, MODEL_LABELS, build_feature_specs,
                       build_pipeline, build_reference)


def run_training(data_path=None, models_dir=None, reports_dir=None, fast=False,
                 sample=None, verbose=True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    models_dir = Path(models_dir or config.MODELS_DIR)
    reports_dir = Path(reports_dir or config.REPORTS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    path = find_data_path(data_path)
    X, y, target = load_dataset(path, sample=sample)
    log(f"[1/6] Loaded {path.name}: {len(X):,} rows x {X.shape[1]} columns, "
        f"target='{target}', prevalence={y.mean():.1%}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)
    n_splits = 3 if fast else 5
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)

    log(f"[2/6] {n_splits}-fold cross-validation of candidate models ...")
    candidates = {}
    for kind in MODEL_KINDS:
        oof = cross_val_predict(build_pipeline(kind, fast=fast), X_tr, y_tr, cv=cv,
                                method="predict_proba")[:, 1]
        candidates[kind] = {
            "label": MODEL_LABELS[kind],
            "cv_roc_auc": float(roc_auc_score(y_tr, oof)),
            "cv_pr_auc": float(average_precision_score(y_tr, oof)),
            "_oof": oof,
        }
        log(f"      {MODEL_LABELS[kind]:<34} CV ROC-AUC={candidates[kind]['cv_roc_auc']:.4f}")
    best_kind = max(candidates, key=lambda k: candidates[k]["cv_roc_auc"])
    threshold = best_threshold(y_tr, candidates[best_kind]["_oof"])
    log(f"[3/6] Selected: {MODEL_LABELS[best_kind]} | screening threshold={threshold:.3f}")

    pipe = build_pipeline(best_kind, fast=fast).fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    metrics = compute_metrics(y_te, proba, threshold)
    log(f"[4/6] Held-out test: ROC-AUC={metrics['roc_auc']:.3f}  PR-AUC={metrics['pr_auc']:.3f}  "
        f"sensitivity={metrics['sensitivity']:.2f}  specificity={metrics['specificity']:.2f}")

    log("[5/6] Permutation importance, subgroup analysis, drift reference ...")
    cleaner = pipe.named_steps["clean"]
    kept = list(cleaner.columns_)
    rng = np.random.default_rng(config.RANDOM_STATE)
    n_imp = min(len(X_te), 3000)
    idx = rng.choice(len(X_te), size=n_imp, replace=False)
    X_imp, y_imp = X_te.iloc[idx][kept], y_te.iloc[idx]
    perm = permutation_importance(pipe, X_imp, y_imp, scoring="roc_auc",
                                  n_repeats=3, random_state=config.RANDOM_STATE, n_jobs=1)
    importance = (pd.DataFrame({"feature": kept, "importance": perm.importances_mean,
                                "std": perm.importances_std})
                  .sort_values("importance", ascending=False).reset_index(drop=True))
    cleaned_train = cleaner.transform(X_tr)
    subgroups = subgroup_table(X_te[kept], y_te, proba, threshold)

    plots = make_plots(y_te, proba, threshold, importance, reports_dir)
    importance.to_csv(reports_dir / "feature_importance.csv", index=False)
    subgroups.to_csv(reports_dir / "subgroups.csv", index=False)
    pd.DataFrame({"y_true": y_te.to_numpy(), "risk": proba}).to_csv(
        reports_dir / "test_predictions.csv", index=False)

    positive_importance = importance[importance["importance"] > 0]["feature"].tolist()
    ranked = positive_importance + [f for f in kept if f not in positive_importance]
    artifact = {
        "pipeline": pipe,
        "model_kind": best_kind,
        "model_label": MODEL_LABELS[best_kind],
        "threshold": float(threshold),
        "target": target,
        "features": kept,
        "dropped_columns": cleaner.dropped_,
        "importance_ranked": ranked,
        "top_features": ranked[:12],
        "feature_specs": build_feature_specs(cleaned_train),
        "reference": build_reference(cleaned_train),
        "drift_reference": build_drift_reference(cleaned_train),
        "metrics": metrics,
        "candidates": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                       for k, v in candidates.items()},
        "data_source": path.name,
        "is_synthetic": "synthetic" in path.name.lower(),
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    model_path = models_dir / "model.joblib"
    joblib.dump(artifact, model_path, compress=3)

    summary = {k: v for k, v in artifact.items()
               if k not in ("pipeline", "reference", "drift_reference", "feature_specs",
                            "importance_ranked")}
    summary["plots"] = plots
    (reports_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    (reports_dir / "RESULTS.md").write_text(_results_markdown(artifact, importance))
    log(f"[6/6] Saved model -> {model_path}  |  reports -> {reports_dir}  "
        f"({time.time() - t0:.0f}s total)")
    return artifact


def _results_markdown(art: dict, importance: pd.DataFrame) -> str:
    m = art["metrics"]
    rows = "\n".join(f"| {v['label']} | {v['cv_roc_auc']:.3f} | {v['cv_pr_auc']:.3f} |"
                     for v in art["candidates"].values())
    top = "\n".join(f"{i + 1}. {r.feature} ({r.importance:.4f})"
                    for i, r in enumerate(importance.head(10).itertuples()))
    note = ("\n> **Synthetic demo data** - these numbers describe simulated patients, "
            "not clinical performance.\n" if art["is_synthetic"] else "")
    return f"""# Training results

Data: `{art['data_source']}` | train {art['n_train']:,} / test {art['n_test']:,} | trained {art['trained_at']}
{note}
## Model selection (cross-validation on training split)
| Model | CV ROC-AUC | CV PR-AUC |
|---|---|---|
{rows}

Selected: **{art['model_label']}**, screening threshold **{m['threshold']:.3f}** (maximises F{config.SCREENING_BETA:g}).

## Held-out test performance
| Metric | Value |
|---|---|
| ROC-AUC | {m['roc_auc']:.3f} |
| PR-AUC | {m['pr_auc']:.3f} |
| Brier score | {m['brier']:.3f} |
| Sensitivity (recall) | {m['sensitivity']:.3f} |
| Specificity | {m['specificity']:.3f} |
| Precision | {m['precision']:.3f} |
| Prevalence | {m['prevalence']:.3f} |

## Top predictors (permutation importance)
{top}
"""


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train the ICU diabetes screening model")
    parser.add_argument("--data", default=None, help="CSV path (default: auto-detect in data/)")
    parser.add_argument("--fast", action="store_true", help="3-fold CV, fewer boosting rounds")
    parser.add_argument("--sample", type=int, default=None, help="stratified subsample size")
    args = parser.parse_args(argv)
    config.ensure_dirs()
    run_training(args.data, fast=args.fast, sample=args.sample)


if __name__ == "__main__":
    main()
