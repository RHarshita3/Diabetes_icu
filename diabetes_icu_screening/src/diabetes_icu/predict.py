"""Load the trained artifact and score patient records."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from . import config
from .data_utils import detect_target
from .risk_trajectory import tier_for


def load_artifact(path: str | Path | None = None) -> dict:
    path = Path(path) if path else config.MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python -m diabetes_icu.train` first."
        )
    return joblib.load(path)


def strip_target(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return df.drop(columns=[detect_target(df.columns)])
    except ValueError:
        return df


def score_frame(artifact: dict, df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    thr = artifact["threshold"] if threshold is None else threshold
    X = strip_target(df)
    risk = artifact["pipeline"].predict_proba(X)[:, 1]
    out = pd.DataFrame({"risk": risk})
    out["risk_pct"] = (out["risk"] * 100).round(1)
    out["tier"] = [tier_for(r, thr) for r in risk]
    out["screen_positive"] = (risk >= thr).astype(int)
    for col in config.ID_COLUMNS:
        if col in df.columns:
            out.insert(0, col, df[col].to_numpy())
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Score patient records")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args(argv)
    art = load_artifact()
    result = score_frame(art, pd.read_csv(args.input), args.threshold)
    result.to_csv(args.output, index=False)
    print(f"Scored {len(result):,} patients -> {args.output}")
    print(result["tier"].value_counts().to_string())


if __name__ == "__main__":
    main()
