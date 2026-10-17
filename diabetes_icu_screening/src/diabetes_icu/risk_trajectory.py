"""Risk trajectories: how predicted risk evolves.

* ``trajectory``: score time-ordered records of one patient (e.g. repeated
  labs / vitals through the ICU stay) and flag rising risk.
* ``what_if_curve``: sweep one feature across its range to see the risk response.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RISE_ALERT = 0.10  # absolute rise in risk that triggers a "rising" alert


def tier_for(risk: float, threshold: float) -> str:
    if risk >= threshold:
        return "high"
    if risk >= 0.5 * threshold:
        return "moderate"
    return "low"


def what_if_curve(artifact: dict, row: pd.DataFrame, feature: str,
                  n_points: int = 30) -> pd.DataFrame:
    spec = artifact["feature_specs"][feature]
    if spec["kind"] == "numeric":
        values = np.linspace(spec["min"], spec["max"], n_points)
        if spec["integer"]:
            values = np.unique(np.round(values))
    else:
        values = spec["options"]
    base = row.iloc[0].to_dict()
    frame = pd.DataFrame([{**base, feature: v} for v in values])
    risk = artifact["pipeline"].predict_proba(frame)[:, 1]
    return pd.DataFrame({feature: values, "risk": risk})


def trajectory(artifact: dict, records: pd.DataFrame,
               time_col: str | None = None) -> pd.DataFrame:
    df = records.copy()
    if time_col and time_col in df.columns:
        df = df.sort_values(time_col)
        times = df[time_col].to_numpy()
    else:
        times = np.arange(len(df))
    risk = artifact["pipeline"].predict_proba(df.drop(columns=[time_col] if time_col in df.columns else []))[:, 1]
    thr = artifact["threshold"]
    out = pd.DataFrame({"time": times, "risk": risk})
    out["tier"] = [tier_for(r, thr) for r in risk]
    out["change_from_first"] = out["risk"] - out["risk"].iloc[0]
    return out.reset_index(drop=True)


def summarize_trajectory(traj: pd.DataFrame, threshold: float) -> dict:
    first, last = float(traj["risk"].iloc[0]), float(traj["risk"].iloc[-1])
    slope = (float(np.polyfit(np.arange(len(traj)), traj["risk"], 1)[0])
             if len(traj) > 1 else 0.0)
    rising = (last - first) >= RISE_ALERT
    if last >= threshold:
        status = "high - above screening threshold"
    elif rising:
        status = "rising - risk climbing, review"
    else:
        status = "stable"
    return {"first": first, "last": last, "peak": float(traj["risk"].max()),
            "change": last - first, "slope_per_step": slope,
            "rising": bool(rising), "status": status}
