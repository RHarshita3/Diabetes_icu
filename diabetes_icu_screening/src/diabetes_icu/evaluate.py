"""Metrics, threshold selection, subgroup analysis and report plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score, brier_score_loss, confusion_matrix,
    precision_recall_curve, roc_auc_score, roc_curve,
)

from . import config  # noqa: E402


def best_threshold(y_true, proba, beta: float = config.SCREENING_BETA) -> float:
    """Threshold maximising F-beta (beta>1 favours recall, right for screening)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]
    b2 = beta ** 2
    denom = b2 * precision + recall
    fbeta = np.divide((1 + b2) * precision * recall, denom,
                      out=np.zeros_like(denom), where=denom > 0)
    return float(thresholds[int(np.argmax(fbeta))])


def compute_metrics(y_true, proba, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    safe = lambda a, b: float(a / b) if b else 0.0  # noqa: E731
    precision, recall = safe(tp, tp + fp), safe(tp, tp + fn)
    f1 = safe(2 * precision * recall, precision + recall)
    b2 = config.SCREENING_BETA ** 2
    f2 = safe((1 + b2) * precision * recall, b2 * precision + recall)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "prevalence": float(y_true.mean()),
        "accuracy": safe(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "sensitivity": recall,
        "specificity": safe(tn, tn + fp),
        "f1": f1,
        "f2": f2,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(y_true)),
    }


def subgroup_table(X: pd.DataFrame, y_true, proba, threshold: float,
                   min_n: int = 30) -> pd.DataFrame:
    """Performance by subgroup (categorical columns, small-cardinality flags, age bands)."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    X = X.reset_index(drop=True)
    groupings = {}
    for col in X.columns:
        s = X[col]
        if col in config.ID_COLUMNS:
            continue
        nun = s.nunique(dropna=True)
        if 2 <= nun <= 10:
            groupings[col] = s.astype(object).where(s.notna(), "missing").astype(str)
    for col in X.columns:
        if col.lower() == "age" and pd.api.types.is_numeric_dtype(X[col]):
            groupings["age_band"] = pd.cut(X[col], [0, 40, 55, 65, 75, 200],
                                           labels=["<40", "40-54", "55-64", "65-74", "75+"]
                                           ).astype(object).fillna("missing").astype(str)
    rows = []
    for name, groups in groupings.items():
        for level in sorted(groups.unique()):
            mask = (groups == level).to_numpy()
            n = int(mask.sum())
            if n < min_n:
                continue
            yt, pr = y_true[mask], proba[mask]
            pred = pr >= threshold
            pos, neg = (yt == 1).sum(), (yt == 0).sum()
            rows.append({
                "attribute": name, "group": level, "n": n,
                "prevalence": float(yt.mean()),
                "roc_auc": float(roc_auc_score(yt, pr)) if pos and neg else np.nan,
                "sensitivity": float((pred & (yt == 1)).sum() / pos) if pos else np.nan,
                "specificity": float((~pred & (yt == 0)).sum() / neg) if neg else np.nan,
            })
    return pd.DataFrame(rows, columns=["attribute", "group", "n", "prevalence",
                                       "roc_auc", "sensitivity", "specificity"])


def make_plots(y_true, proba, threshold: float, importance: pd.DataFrame | None,
               out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true)
    paths = {}

    fpr, tpr, _ = roc_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y_true, proba):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    ax.legend(loc="lower right")
    paths["roc"] = _save(fig, out_dir / "roc.png")

    prec, rec, _ = precision_recall_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(rec, prec, lw=2, label=f"AP = {average_precision_score(y_true, proba):.3f}")
    ax.axhline(y_true.mean(), ls="--", color="grey", lw=1, label="Prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    ax.legend(loc="upper right")
    paths["pr"] = _save(fig, out_dir / "pr.png")

    pred = (np.asarray(proba) >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=12)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No DM", "DM"],
           yticklabels=["No DM", "DM"], xlabel="Predicted", ylabel="Actual",
           title=f"Confusion matrix (thr={threshold:.2f})")
    paths["confusion"] = _save(fig, out_dir / "confusion.png")

    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(mean_pred, frac_pos, "o-", lw=2)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set(xlabel="Mean predicted risk", ylabel="Observed fraction",
           title="Calibration")
    paths["calibration"] = _save(fig, out_dir / "calibration.png")

    if importance is not None and len(importance):
        top = importance.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6.0, 4.6))
        ax.barh(top["feature"], top["importance"], xerr=top["std"], color="#2a6f97")
        ax.set(xlabel="Drop in ROC-AUC when shuffled", title="Permutation importance")
        paths["importance"] = _save(fig, out_dir / "importance.png")
    return {k: str(v) for k, v in paths.items()}


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
