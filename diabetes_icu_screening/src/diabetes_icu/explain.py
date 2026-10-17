"""Model-agnostic explanations without extra dependencies.

* Global: permutation importance (computed at training time).
* Local: *occlusion* - replace one feature at a time with its typical value
  (training median / mode) and measure how much the predicted risk moves.
  Positive contribution = the patient's value pushes risk up versus a typical patient.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pretty(name: str) -> str:
    return name.replace("_", " ").strip().capitalize()


def local_explanation(artifact: dict, row: pd.DataFrame, top_n: int = 10,
                      max_candidates: int = 40) -> pd.DataFrame:
    pipe, ref = artifact["pipeline"], artifact["reference"]
    base_row = row.iloc[0].to_dict()
    ranked = [f for f in artifact["importance_ranked"] if f in ref][:max_candidates]
    variants = []
    for feat in ranked:
        variant = dict(base_row)
        variant[feat] = ref[feat]
        variants.append(variant)
    base = float(pipe.predict_proba(row)[0, 1])
    if not variants:
        return pd.DataFrame(columns=["feature", "value", "typical", "contribution"])
    probs = pipe.predict_proba(pd.DataFrame(variants))[:, 1]
    out = pd.DataFrame({
        "feature": ranked,
        "value": [base_row.get(f, np.nan) for f in ranked],
        "typical": [ref[f] for f in ranked],
        "contribution": base - probs,
    })
    out["abs"] = out["contribution"].abs()
    out = out.sort_values("abs", ascending=False).drop(columns="abs")
    return out.head(top_n).reset_index(drop=True)


def narrative(expl: pd.DataFrame, risk: float, tier: str) -> str:
    if expl.empty:
        return f"Estimated risk {risk:.0%} ({tier})."
    up = expl[expl["contribution"] > 0.005].head(3)
    down = expl[expl["contribution"] < -0.005].head(2)
    parts = [f"Estimated risk is {risk:.0%} ({tier} tier)."]
    if len(up):
        parts.append("Raising risk: " + ", ".join(
            f"{pretty(r.feature)} ({_fmt(r.value)}, {r.contribution:+.1%})"
            for r in up.itertuples()) + ".")
    if len(down):
        parts.append("Lowering risk: " + ", ".join(
            f"{pretty(r.feature)} ({_fmt(r.value)}, {r.contribution:+.1%})"
            for r in down.itertuples()) + ".")
    return " ".join(parts)


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "not measured"
    if isinstance(v, (int, float, np.floating)):
        return f"{v:.3g}"
    return str(v)
