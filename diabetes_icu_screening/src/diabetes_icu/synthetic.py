"""Generate a realistic *synthetic* ICU dataset so the project runs out of the box.

The data are simulated (no real patients). Columns mirror the clinical features used in
the dashboard; diabetes risk depends on age, BMI, family history, steroids, glucose and
HbA1c, and HbA1c is more often measured when diabetes is suspected (informative
missingness, as in real ICU data). Replace with the real WiDS 2021 data for real work.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def generate(n: int = 12000, seed: int = 7, prevalence: float = 0.215) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(62, 16, n), 18, 95).round()
    sex = rng.choice(["F", "M"], n, p=[0.45, 0.55])
    bmi = np.clip(rng.lognormal(np.log(28), 0.22, n), 15, 60).round(1)
    family = rng.binomial(1, 0.18, n)
    steroids = rng.binomial(1, 0.12, n)
    diuretics = rng.binomial(1, 0.25, n)
    admission = rng.choice(["emergency", "elective", "transfer"], n, p=[0.62, 0.18, 0.20])

    z = (0.035 * (age - 62) + 0.07 * (bmi - 28) + 0.9 * family
         + 0.25 * (sex == "M") + 0.3 * steroids + rng.normal(0, 0.6, n))
    lo, hi = -12.0, 6.0
    for _ in range(60):  # bisection for the intercept that yields the target prevalence
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid)))).mean() > prevalence:
            hi = mid
        else:
            lo = mid
    y = rng.binomial(1, 1 / (1 + np.exp(-(z + (lo + hi) / 2))))

    glucose = np.where(y == 1, rng.normal(172, 65, n), rng.normal(122, 32, n)) + 25 * steroids
    hba1c = np.where(y == 1, rng.normal(7.1, 1.5, n), rng.normal(5.6, 0.65, n))
    heart_rate = rng.normal(88, 16, n)
    systolic = rng.normal(128, 20, n) + 0.2 * (age - 62)
    temperature = rng.normal(37.0, 0.6, n)
    resp = rng.normal(19, 5, n)
    creatinine = rng.lognormal(np.log(1.0), 0.4, n) + 0.25 * y
    potassium = rng.normal(4.1, 0.45, n)
    wbc = rng.lognormal(np.log(10), 0.35, n)

    df = pd.DataFrame({
        "patient_id": np.arange(1, n + 1),
        "age": age, "sex": sex, "bmi": bmi, "admission_type": admission,
        "glucose": np.clip(glucose, 40, 500).round(0),
        "hba1c": np.clip(hba1c, 4, 15).round(1),
        "heart_rate": np.clip(heart_rate, 40, 180).round(0),
        "systolic_bp": np.clip(systolic, 60, 220).round(0),
        "temperature": np.clip(temperature, 34, 41).round(1),
        "respiratory_rate": np.clip(resp, 8, 45).round(0),
        "creatinine": np.clip(creatinine, 0.2, 10).round(2),
        "potassium": np.clip(potassium, 2.2, 6.8).round(1),
        "wbc": np.clip(wbc, 0.5, 40).round(1),
        "prior_steroids": steroids, "prior_diuretics": diuretics,
        "family_history": family,
    })
    # Missingness: HbA1c ordered mostly when diabetes is suspected (informative).
    df.loc[rng.random(n) < np.where(y == 1, 0.35, 0.62), "hba1c"] = np.nan
    for col, rate in [("glucose", 0.06), ("bmi", 0.05), ("creatinine", 0.04),
                      ("potassium", 0.05), ("wbc", 0.04), ("temperature", 0.08),
                      ("respiratory_rate", 0.05)]:
        df.loc[rng.random(n) < rate, col] = np.nan
    df["diabetes_mellitus"] = y
    return df


def make_sample_trajectory() -> pd.DataFrame:
    """One simulated patient with deteriorating glycaemic control over 24 hours."""
    hours = [0, 3, 6, 9, 12, 15, 18, 21]
    base = dict(patient_id=999001, age=67, sex="M", bmi=32.0, admission_type="emergency",
                heart_rate=84, systolic_bp=132, temperature=37.0, respiratory_rate=18,
                creatinine=1.1, potassium=4.0, wbc=9.5, prior_steroids=0,
                prior_diuretics=1, family_history=1)
    glucose = [128, 142, 165, 190, 214, 236, 252, 268]
    rows = []
    for h, g in zip(hours, glucose):
        rows.append({"hour": h, **base, "glucose": g,
                     "hba1c": np.nan if h < 12 else 7.4,
                     "heart_rate": base["heart_rate"] + h // 3 * 2})
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Create synthetic ICU demo data")
    parser.add_argument("--rows", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=str(config.SYNTHETIC_DATA_PATH))
    args = parser.parse_args(argv)
    config.ensure_dirs()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.rows, args.seed)
    df.to_csv(out, index=False)
    make_sample_trajectory().to_csv(config.SAMPLE_TRAJECTORY_PATH, index=False)
    print(f"Wrote {len(df):,} synthetic patients -> {out} "
          f"(prevalence {df['diabetes_mellitus'].mean():.1%})")


if __name__ == "__main__":
    main()
