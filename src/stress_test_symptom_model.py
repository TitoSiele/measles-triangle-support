"""Robustness stress test for the symptom model.

Run from the project root:  python src/stress_test_symptom_model.py

Trains the CURRENT model settings on clean synthetic data (5-fold stratified),
then measures how performance degrades when real-world messiness is added:
  missing            fraction of values not recorded (symptoms -> 0, numbers -> median)
  symptom_flips      fraction of yes/no symptoms recorded wrongly
  numeric_noise      measurement noise on age/temp/days (noise SD = 2 x level x feature SD)
  hallmark_masked    measles cases presenting WITHOUT Koplik spots / cephalocaudal spread
  train_label_noise  fraction of TRAINING labels wrong (tested on clean data)
  combined           a mix of the first four, applied together

Saves: data/processed/robustness_results.csv and data/processed/robustness_plot.png
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

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
target = next((c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
               and set(df[c].dropna().unique()) <= classes), None)
if target is None:
    sys.exit("Could not find the label column in the CSV.")
if all(c in df.columns for c in feature_cols):
    X = df[feature_cols].copy()
else:
    X = pd.get_dummies(df.drop(columns=[target])).reindex(columns=feature_cols, fill_value=0)
X = X.astype(float)
y = le.transform(df[target])
mi = int(le.transform(["Measles"])[0])
n_classes = len(le.classes_)

binary_cols = [c for c in X.columns if set(np.unique(X[c])) <= {0.0, 1.0}]
numeric_cols = [c for c in X.columns if c not in binary_cols]
HALLMARK = [c for c in ("koplik_spots", "cephalocaudal_spread") if c in X.columns]
LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]
KINDS = ["missing", "symptom_flips", "numeric_noise", "hallmark_masked", "combined"]
if not HALLMARK:
    print("Note: no hallmark columns found, skipping hallmark_masked.")
    KINDS = [k for k in KINDS if k != "hallmark_masked"]
rng = np.random.default_rng(0)


def new_rf():
    return RandomForestClassifier(n_estimators=200, max_depth=10,
                                  class_weight="balanced", random_state=42, n_jobs=-1)


def corrupt(Xt, yt, s, kind, med, sd):
    Xc, n = Xt.copy(), len(Xt)
    if kind in ("missing", "combined"):
        for c in Xc.columns:
            m = rng.random(n) < s
            Xc.loc[m, c] = 0.0 if c in binary_cols else med[c]
    if kind in ("symptom_flips", "combined"):
        p = s / 2 if kind == "combined" else s
        for c in binary_cols:
            m = rng.random(n) < p
            Xc.loc[m, c] = 1.0 - Xc.loc[m, c]
    if kind in ("numeric_noise", "combined"):
        for c in numeric_cols:
            Xc[c] = (Xc[c] + rng.normal(0, 2 * s * sd[c], n)).clip(lower=0)
    if kind in ("hallmark_masked", "combined"):
        for c in HALLMARK:
            m = (yt == mi) & (rng.random(n) < s)
            Xc.loc[m, c] = 0.0
    return Xc


def metrics(yt, pred):
    return (accuracy_score(yt, pred),
            recall_score(yt, pred, labels=[mi], average="macro", zero_division=0),
            precision_score(yt, pred, labels=[mi], average="macro", zero_division=0))


rows = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr, te) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}/5...")
    Xtr, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
    ytr, yte = y[tr], y[te]
    med, sd = Xtr.median(), Xtr.std()
    clean = new_rf().fit(Xtr, ytr)
    for kind in KINDS:
        for s in LEVELS:
            rows.append((kind, s, fold, *metrics(yte, clean.predict(corrupt(Xte, yte, s, kind, med, sd)))))
    for s in LEVELS:
        yn = ytr.copy()
        flip = rng.random(len(yn)) < s
        yn[flip] = [rng.choice([k for k in range(n_classes) if k != v]) for v in yn[flip]]
        rows.append(("train_label_noise", s, fold, *metrics(yte, new_rf().fit(Xtr, yn).predict(Xte))))

res = pd.DataFrame(rows, columns=["scenario", "level", "fold", "accuracy", "measles_recall", "measles_precision"])
avg = res.groupby(["scenario", "level"])[["accuracy", "measles_recall", "measles_precision"]].mean().round(3)
print("\nRESULTS (mean over 5 folds; level 0.00 = clean data)")
print(avg.to_string())

print("\nFIRST LEVEL WHERE MEASLES RECALL DROPS BELOW 0.90:")
for kind in avg.index.get_level_values(0).unique():
    sub = avg.loc[kind]
    bad = sub[sub["measles_recall"] < 0.90]
    print(f"  {kind:18s} " + (f"{bad.index[0]:.2f}" if len(bad) else "not reached within tested range"))

res.to_csv("data/processed/robustness_results.csv", index=False)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col in zip(axes, ["measles_recall", "measles_precision"]):
        for kind in avg.index.get_level_values(0).unique():
            sub = avg.loc[kind]
            ax.plot(sub.index, sub[col], marker="o", label=kind)
        ax.set_xlabel("Corruption level")
        ax.set_ylabel(col.replace("_", " ").title())
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Symptom model robustness (synthetic data, 5-fold mean)")
    fig.tight_layout()
    fig.savefig("data/processed/robustness_plot.png", dpi=150)
    print("\nSaved plot: data/processed/robustness_plot.png")
except Exception as e:
    print(f"\n(Plot skipped: {e})")
print("Saved results: data/processed/robustness_results.csv")
