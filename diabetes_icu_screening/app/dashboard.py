"""Streamlit dashboard: performance, single-patient risk + explanation,
batch screening with drift monitoring, and risk trajectories.

Run:  streamlit run app/dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # works even without `pip install -e .`

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from diabetes_icu import config  # noqa: E402
from diabetes_icu.drift import compute_drift, summarize_drift  # noqa: E402
from diabetes_icu.explain import local_explanation, narrative, pretty  # noqa: E402
from diabetes_icu.predict import load_artifact, score_frame, strip_target  # noqa: E402
from diabetes_icu.risk_trajectory import (  # noqa: E402
    summarize_trajectory, tier_for, trajectory, what_if_curve)

st.set_page_config(page_title="ICU Diabetes Screening", page_icon="🩺", layout="wide")


@st.cache_resource
def get_artifact(_mtime: float):
    return load_artifact()


def train_now():
    from diabetes_icu.train import run_training
    config.ensure_dirs()
    if not config.SYNTHETIC_DATA_PATH.exists() and not any(
            (config.DATA_DIR / n).exists() for n in config.DATA_CANDIDATES):
        from diabetes_icu.synthetic import main as make_data
        make_data([])
    with st.spinner("Training model (about a minute)..."):
        run_training(fast=True, verbose=False)
    get_artifact.clear()
    st.rerun()


st.title("🩺 ICU Diabetes Mellitus Screening")
st.caption("Decision-support prototype: estimates the chance that an ICU patient has "
           "diabetes mellitus from admission data. Not a medical device.")

if not config.MODEL_PATH.exists():
    st.warning("No trained model found yet.")
    st.button("Train model now", type="primary", on_click=train_now)
    st.code("python -m diabetes_icu.train", language="bash")
    st.stop()

art = get_artifact(config.MODEL_PATH.stat().st_mtime)
specs, metrics = art["feature_specs"], art["metrics"]

with st.sidebar:
    st.header("Model")
    st.write(f"**{art['model_label']}**")
    st.caption(f"Data: `{art['data_source']}`  \nTrained: {art['trained_at'][:16]} UTC")
    if art["is_synthetic"]:
        st.info("Trained on **synthetic** demo data.")
    threshold = st.slider("Screening threshold", 0.02, 0.90, float(art["threshold"]), 0.01,
                          help="Patients at or above this risk are flagged. Lower = more "
                               "sensitive, more false alarms.")
    st.caption(f"Tiers: high ≥ {threshold:.2f}, moderate ≥ {threshold / 2:.2f}")
    st.button("Retrain (fast)", on_click=train_now)

tab_perf, tab_patient, tab_batch, tab_traj, tab_about = st.tabs(
    ["📊 Performance", "🧍 Patient risk", "📁 Batch & drift", "📈 Risk trajectory", "ℹ️ About"])

# ------------------------------------------------------------------ Performance
with tab_perf:
    c = st.columns(5)
    c[0].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    c[1].metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
    c[2].metric("Sensitivity", f"{metrics['sensitivity']:.2f}")
    c[3].metric("Specificity", f"{metrics['specificity']:.2f}")
    c[4].metric("Brier", f"{metrics['brier']:.3f}")
    st.caption(f"Held-out test set: {metrics['n']:,} patients, prevalence "
               f"{metrics['prevalence']:.1%}, at the trained threshold {metrics['threshold']:.2f}.")
    st.subheader("Why this model? (cross-validation)")
    st.dataframe(pd.DataFrame(art["candidates"]).T[["label", "cv_roc_auc", "cv_pr_auc"]]
                 .rename(columns={"label": "Model", "cv_roc_auc": "CV ROC-AUC",
                                  "cv_pr_auc": "CV PR-AUC"}), hide_index=True)
    imgs = [config.REPORTS_DIR / f for f in ("roc.png", "pr.png", "calibration.png", "confusion.png")]
    if all(p.exists() for p in imgs):
        cols = st.columns(4)
        for col, p in zip(cols, imgs):
            col.image(str(p))
    imp_path = config.REPORTS_DIR / "feature_importance.csv"
    if imp_path.exists():
        st.subheader("Top predictors")
        imp = pd.read_csv(imp_path).head(15)
        st.bar_chart(imp.set_index("feature")["importance"])
    sub_path = config.REPORTS_DIR / "subgroups.csv"
    if sub_path.exists():
        sub = pd.read_csv(sub_path)
        if len(sub):
            st.subheader("Subgroup performance (fairness check)")
            attr = st.selectbox("Attribute", sorted(sub["attribute"].unique()))
            st.dataframe(sub[sub["attribute"] == attr].drop(columns="attribute")
                         .style.format({"prevalence": "{:.1%}", "roc_auc": "{:.3f}",
                                        "sensitivity": "{:.2f}", "specificity": "{:.2f}"}),
                         hide_index=True)

# ------------------------------------------------------------------ Patient
patient_row = None
with tab_patient:
    st.subheader("Patient snapshot")
    st.caption("Enter what is known. Tick *Not measured* for missing values - the model "
               "handles missing data. Features not shown are treated as missing.")
    values = {f: np.nan for f in art["features"]}
    form_cols = st.columns(3)
    for i, feat in enumerate(art["top_features"]):
        spec, col = specs.get(feat), form_cols[i % 3]
        if spec is None:
            continue
        with col:
            label = pretty(feat)
            if spec["kind"] == "numeric":
                na = st.checkbox("Not measured", key=f"na_{feat}", value=False)
                v = st.number_input(
                    label, min_value=spec["min"], max_value=spec["max"],
                    value=float(np.clip(spec["median"], spec["min"], spec["max"])),
                    step=spec["step"], key=f"v_{feat}", disabled=na)
                values[feat] = np.nan if na else (round(v) if spec["integer"] else v)
            elif spec["kind"] == "binary":
                values[feat] = st.selectbox(label, spec["options"],
                                            index=spec["options"].index(spec["default"]),
                                            key=f"v_{feat}")
            else:
                values[feat] = st.selectbox(label, spec["options"],
                                            index=spec["options"].index(spec["default"]),
                                            key=f"v_{feat}")
    patient_row = pd.DataFrame([values])
    risk = float(art["pipeline"].predict_proba(patient_row)[0, 1])
    tier = tier_for(risk, threshold)
    st.divider()
    left, right = st.columns([1, 2])
    with left:
        st.metric("Estimated diabetes risk", f"{risk:.1%}",
                  help="Baseline prevalence in training data: "
                       f"{metrics['prevalence']:.1%}")
        st.progress(min(risk, 1.0))
        {"high": st.error, "moderate": st.warning, "low": st.success}[tier](
            f"**{tier.capitalize()} risk** - "
            + ("flag for glycaemic review / HbA1c" if tier != "low" else "no flag"))
    with right:
        expl = local_explanation(art, patient_row, top_n=8)
        st.markdown("**What is driving this estimate**")
        st.write(narrative(expl, risk, tier))
        if len(expl):
            chart = expl.assign(label=expl["feature"].map(pretty)).set_index("label")["contribution"]
            st.bar_chart(chart)
            with st.expander("Details"):
                st.dataframe(expl, hide_index=True)

# ------------------------------------------------------------------ Batch & drift
with tab_batch:
    st.subheader("Batch screening")
    src = st.radio("Records", ["Use bundled sample", "Upload CSV"], horizontal=True)
    batch = None
    if src == "Upload CSV":
        up = st.file_uploader("CSV with patient records (target column optional)", type="csv")
        if up is not None:
            batch = pd.read_csv(up)
    elif config.SYNTHETIC_DATA_PATH.exists():
        batch = pd.read_csv(config.SYNTHETIC_DATA_PATH).sample(500, random_state=1)
    if batch is None:
        st.info("Provide a CSV to score.")
    else:
        scored = score_frame(art, batch, threshold)
        c = st.columns(4)
        c[0].metric("Patients", f"{len(scored):,}")
        c[1].metric("Flagged", f"{int(scored['screen_positive'].sum()):,}")
        c[2].metric("High tier", f"{int((scored['tier'] == 'high').sum()):,}")
        c[3].metric("Mean risk", f"{scored['risk'].mean():.1%}")
        counts, edges = np.histogram(scored["risk"], bins=20, range=(0, 1))
        st.bar_chart(pd.DataFrame({"patients": counts},
                                  index=[f"{e:.2f}" for e in edges[:-1]]))
        show = pd.concat([scored.reset_index(drop=True),
                          strip_target(batch).reset_index(drop=True)
                          .drop(columns=[c for c in scored.columns if c in batch.columns],
                                errors="ignore")], axis=1)
        st.dataframe(show.sort_values("risk", ascending=False).head(100), hide_index=True)
        st.download_button("Download predictions (CSV)",
                           scored.to_csv(index=False).encode(), "predictions.csv", "text/csv")

        st.subheader("Data drift vs. training data (PSI)")
        cleaned = art["pipeline"].named_steps["clean"].transform(strip_target(batch))
        report = compute_drift(art["drift_reference"], cleaned)
        summ = summarize_drift(report)
        msg = (f"{summ['n_alert']} feature(s) with major drift, {summ['n_warning']} moderate, "
               f"of {summ['n_features']} monitored.")
        {"alert": st.error, "warning": st.warning, "stable": st.success}.get(
            summ["overall"], st.info)(("Drift alert: " if summ["overall"] == "alert" else "") + msg)
        st.caption("PSI < 0.10 stable · 0.10-0.25 moderate shift · > 0.25 major shift "
                   "(consider re-validating or retraining).")
        st.dataframe(report.head(15), hide_index=True)

# ------------------------------------------------------------------ Trajectory
with tab_traj:
    mode = st.radio("View", ["What-if sweep (current patient)", "Time-ordered records"],
                    horizontal=True)
    if mode.startswith("What-if"):
        sweepable = [f for f in art["top_features"] if f in specs]
        feat = st.selectbox("Feature to vary", sweepable, format_func=pretty)
        curve = what_if_curve(art, patient_row, feat)
        if specs[feat]["kind"] == "numeric":
            plot = curve.set_index(feat)[["risk"]].assign(threshold=threshold)
            st.line_chart(plot)
        else:
            st.bar_chart(curve.set_index(feat)["risk"])
        st.caption("Risk response when only this feature changes and everything else "
                   "stays as entered in the *Patient risk* tab.")
    else:
        up = st.file_uploader("Time-ordered records for ONE patient (CSV)", type="csv",
                              key="traj_up")
        if up is not None:
            recs = pd.read_csv(up)
        elif config.SAMPLE_TRAJECTORY_PATH.exists():
            recs = pd.read_csv(config.SAMPLE_TRAJECTORY_PATH)
            st.caption("Showing the bundled example: a simulated patient over 24 hours.")
        else:
            recs = None
        if recs is not None:
            time_opts = [None] + list(recs.columns)
            default = time_opts.index("hour") if "hour" in recs.columns else 0
            tcol = st.selectbox("Time column", time_opts, index=default)
            traj = trajectory(art, recs, tcol)
            summ = summarize_trajectory(traj, threshold)
            c = st.columns(4)
            c[0].metric("First", f"{summ['first']:.1%}")
            c[1].metric("Latest", f"{summ['last']:.1%}", f"{summ['change']:+.1%}")
            c[2].metric("Peak", f"{summ['peak']:.1%}")
            c[3].metric("Status", summ["status"].split(" - ")[0].capitalize())
            alert = {"high": st.error, "rising": st.warning}.get(
                summ["status"].split(" - ")[0], st.success)
            alert(summ["status"].capitalize())
            st.line_chart(traj.set_index("time")[["risk"]].assign(threshold=threshold))
            st.dataframe(traj, hide_index=True)

# ------------------------------------------------------------------ About
with tab_about:
    card = config.DOCS_DIR / "MODEL_CARD.md"
    st.markdown(card.read_text(encoding="utf-8") if card.exists() else "Model card missing.")
    res = config.REPORTS_DIR / "RESULTS.md"
    if res.exists():
        with st.expander("Latest training results"):
            st.markdown(res.read_text(encoding="utf-8"))
