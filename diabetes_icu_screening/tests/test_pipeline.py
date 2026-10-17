"""End-to-end tests on a small synthetic dataset (fast, no network)."""
import numpy as np
import pandas as pd
import pytest

from diabetes_icu import config
from diabetes_icu.data_utils import DataCleaner, detect_target
from diabetes_icu.drift import compute_drift, psi, summarize_drift
from diabetes_icu.evaluate import best_threshold, compute_metrics
from diabetes_icu.explain import local_explanation, narrative
from diabetes_icu.predict import load_artifact, score_frame
from diabetes_icu.risk_trajectory import (summarize_trajectory, tier_for,
                                          trajectory, what_if_curve)
from diabetes_icu.synthetic import generate, make_sample_trajectory
from diabetes_icu.train import run_training


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("proj")
    csv = root / "sample_icu_synthetic.csv"
    generate(3000, seed=1).to_csv(csv, index=False)
    art = run_training(csv, root / "models", root / "reports", fast=True, verbose=False)
    return art, root, pd.read_csv(csv)


def test_detect_target():
    assert detect_target(["a", "OutputVar"]) == "OutputVar"
    assert detect_target(["diabetes_mellitus", "x"]) == "diabetes_mellitus"
    with pytest.raises(ValueError):
        detect_target(["a", "b"])


def test_cleaner_drops_ids_missing_constant_and_clips():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "patient_id": range(200),
        "mostly_nan": [np.nan] * 150 + [1.0] * 50,
        "const": 1,
        "x": np.append(rng.normal(size=199), 1e6),
        "cat": rng.choice(["a", "b"], 200),
    })
    cl = DataCleaner().fit(df)
    assert set(cl.columns_) == {"x", "cat"}
    assert cl.transform(df)["x"].max() < 1e5  # outlier winsorised
    # unseen / missing columns at predict time do not crash
    out = cl.transform(df[["x"]])
    assert list(out.columns) == ["x", "cat"] and out["cat"].isna().all()


def test_threshold_and_metrics():
    y = np.array([0, 0, 0, 1, 1, 1] * 20)
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9] * 20)
    thr = best_threshold(y, p)
    m = compute_metrics(y, p, thr)
    assert m["roc_auc"] == 1.0 and m["sensitivity"] == 1.0 and m["fp"] == 0


def test_training_artifact(trained):
    art, root, _ = trained
    assert (root / "models" / "model.joblib").exists()
    assert (root / "reports" / "roc.png").exists()
    assert art["metrics"]["roc_auc"] > 0.8
    assert 0 < art["threshold"] < 1
    assert art["is_synthetic"] and art["top_features"]


def test_predict_roundtrip(trained):
    art, root, df = trained
    loaded = load_artifact(root / "models" / "model.joblib")
    scored = score_frame(loaded, df.head(50))
    assert len(scored) == 50 and scored["risk"].between(0, 1).all()
    assert set(scored["tier"]) <= {"low", "moderate", "high"}
    # robust to missing columns and unseen categories
    odd = df.head(3).drop(columns=["glucose"]).assign(admission_type="never_seen")
    assert len(score_frame(loaded, odd)) == 3


def test_explanation_points_the_right_way(trained):
    art, _, _ = trained
    row = pd.DataFrame([{**{f: np.nan for f in art["features"]}, "glucose": 420.0,
                         "hba1c": 11.0}])
    expl = local_explanation(art, row, top_n=5)
    top = expl.set_index("feature")["contribution"]
    assert top.get("hba1c", 0) > 0 or top.get("glucose", 0) > 0
    risk = float(art["pipeline"].predict_proba(row)[0, 1])
    assert isinstance(narrative(expl, risk, "high"), str)


def test_drift_detects_shift(trained):
    art, _, df = trained
    clean = art["pipeline"].named_steps["clean"]
    same = compute_drift(art["drift_reference"], clean.transform(df.sample(1500, random_state=2)))
    assert summarize_drift(same)["overall"] == "stable"
    shifted = df.copy()
    shifted["glucose"] = shifted["glucose"] + 80
    rep = compute_drift(art["drift_reference"], clean.transform(shifted))
    assert rep.iloc[0]["feature"] == "glucose" and rep.iloc[0]["status"] == "alert"
    assert psi([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0)


def test_trajectory_and_whatif(trained):
    art, _, _ = trained
    traj = trajectory(art, make_sample_trajectory(), "hour")
    s = summarize_trajectory(traj, art["threshold"])
    assert s["last"] > s["first"] and s["rising"]
    row = pd.DataFrame([{f: np.nan for f in art["features"]}])
    curve = what_if_curve(art, row, "glucose")
    assert curve["risk"].iloc[-1] > curve["risk"].iloc[0]
    assert tier_for(0.5, 0.3) == "high" and tier_for(0.2, 0.3) == "moderate"
    assert tier_for(0.05, 0.3) == "low"
