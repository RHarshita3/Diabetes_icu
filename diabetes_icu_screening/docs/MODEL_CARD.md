# Model card - ICU diabetes mellitus screening

## Intended use
Rank newly admitted ICU patients by the estimated probability of having diabetes mellitus so that
clinicians can prioritise glucose / HbA1c review. It is a **decision-support prototype built for a
hackathon-style challenge, not a medical device**, and must not be used to make or replace
clinical decisions.

## Data
* Challenge: MathWorks *Predicting Diabetes Mellitus* (data from the WiDS Datathon 2021, ICU
  patient records with a binary `diabetes_mellitus` label).
* The repository ships **synthetic** data so it runs immediately; metrics shown with it
  describe simulated patients only. Train on the real data (see `data/README.md`) for meaningful numbers.

## Method
1. **Cleaning** (`DataCleaner`): drop identifiers, constant columns and columns >60% missing;
   winsorise continuous features at the 0.5/99.5 percentiles; keep NaNs (missingness can be informative).
2. **Models compared** with stratified 5-fold CV: logistic regression (interpretable baseline,
   median imputation + missing indicators, scaling, one-hot) vs. histogram gradient boosting
   (native NaN handling, captures non-linearities and interactions). Best CV ROC-AUC wins.
3. **Threshold**: chosen on out-of-fold predictions to maximise F2 (recall-weighted, since a missed
   diabetic patient costs more than an extra glucose check). Adjustable in the dashboard.
4. **Evaluation**: one pass on a 20% stratified hold-out set: ROC-AUC, PR-AUC, Brier score,
   sensitivity/specificity, calibration curve, subgroup performance.

## Explainability and monitoring
* Global: permutation importance. Local: occlusion (replace a feature by its typical value and
  measure the change in risk).
* Drift: Population Stability Index per feature, including missingness, against the training data.
* Risk trajectory: re-scoring time-ordered records and a what-if sweep per feature.

## Limitations
* Observational, single-source data; performance may not transfer to other hospitals or countries.
* Labels come from recorded diagnoses; undiagnosed diabetes is labelled negative.
* Occlusion explanations are associations learned by the model, not causal effects.
* Check the **Subgroup performance** table before any use; small groups give unstable estimates.
* Re-validate after any drift alert.

## Ethical considerations
Do not use for denying or limiting care. Keep a clinician in the loop and audit subgroup
sensitivity regularly. No personal data is stored by the dashboard.
