"""5-fold stratified CV + hyperparameter tuning for the symptom model.

Run from the project root:  python src/cv_tune_symptom_model.py
Does NOT overwrite your existing models. Saves:
  models/symptom_classifier_tuned.joblib
  models/symptom_cv_report.txt
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_validate)

DATA = "data/processed/symptom_dataset.csv"
ENC = "models/label_encoder.joblib"
FEATS = "models/feature_columns.joblib"

for p in (DATA, ENC, FEATS):
    if not os.path.exists(p):
        sys.exit(f"Missing {p}. Are you in the project root? Run `pwd` and `ls`.")

df = pd.read_csv(DATA)
le = joblib.load(ENC)
feature_cols = list(joblib.load(FEATS))
classes = set(le.classes_)

# Find the label column: the one whose values are all known class names.
target = next((c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
               and set(df[c].dropna().unique()) <= classes), None)
if target is None:
    sys.exit("Could not find the label column in the CSV.")

# Build features the same way whether or not they were one-hot encoded.
if all(c in df.columns for c in feature_cols):
    X = df[feature_cols].copy()
else:
    X = pd.get_dummies(df.drop(columns=[target])).reindex(columns=feature_cols, fill_value=0)
X = X.astype(float)
y = le.transform(df[target])
mi = int(le.transform(["Measles"])[0])
print(f"Label column: {target} | rows: {len(X)} | features: {X.shape[1]}")

scoring = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "measles_recall": make_scorer(recall_score, labels=[mi], average="macro", zero_division=0),
    "measles_precision": make_scorer(precision_score, labels=[mi], average="macro", zero_division=0),
}
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def summarize(name, res, out):
    out.append(f"\n{name}")
    for k in scoring:
        v = res[f"test_{k}"]
        out.append(f"  {k:18s} {v.mean():.3f} +/- {v.std():.3f}  (folds: {np.round(v, 3).tolist()})")


report = []
baseline = RandomForestClassifier(n_estimators=200, max_depth=10,
                                  class_weight="balanced", random_state=42, n_jobs=-1)
print("Running baseline 5-fold CV...")
summarize("BASELINE (current settings)", cross_validate(baseline, X, y, cv=outer, scoring=scoring), report)

grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [6, 8, 10, 14, None],
    "min_samples_leaf": [1, 2, 4, 8],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2", 0.5],
    "class_weight": ["balanced", "balanced_subsample"],
}


def make_search():
    return RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1), grid, n_iter=20,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=7),
        scoring="f1_macro", random_state=42, n_jobs=1)


# Nested CV: tuning happens inside each training fold, so this estimate is not
# inflated by choosing the best settings on the same data used to score them.
print("Running nested CV for the tuned model (this can take a few minutes)...")
summarize("TUNED (nested CV, honest estimate)",
          cross_validate(make_search(), X, y, cv=outer, scoring=scoring), report)

print("Fitting final tuned model on all data...")
final = make_search().fit(X, y)
report.append(f"\nBest params: {final.best_params_}")
imp = pd.Series(final.best_estimator_.feature_importances_, index=X.columns)
report.append("\nTop 10 features by importance:")
report.extend(f"  {n:30s} {v:.3f}" for n, v in imp.sort_values(ascending=False).head(10).items())
report.append("\nNOTE: data is synthetic, so these scores measure agreement with the generator's "
              "rules, not performance on real patients. Real validation needs real clinical data.")

joblib.dump(final.best_estimator_, "models/symptom_classifier_tuned.joblib")
text = "\n".join(report)
with open("models/symptom_cv_report.txt", "w") as f:
    f.write(text)
print(text)
print("\nSaved: models/symptom_classifier_tuned.joblib and models/symptom_cv_report.txt")
