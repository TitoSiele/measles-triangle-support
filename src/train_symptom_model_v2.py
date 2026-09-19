"""Train an "unknown-aware" symptom model (v2) and compare it with the current one.

Run from the project root:  python src/train_symptom_model_v2.py

What v2 changes:
  1. Every feature gets a companion "<name>_missing" flag (1 = unknown/not recorded).
  2. Training data is augmented with copies that have random values marked unknown,
     so the model learns to cope with gaps instead of treating them as "no"/median.
  3. A "refer" rule: if 2+ of the KEY fields are unknown, the tool should not
     give a score and should say "insufficient information, needs clinical assessment".

Does NOT overwrite your existing models. Saves:
  models/symptom_bundle_v2.joblib   (model + everything the API needs)
  data/processed/robustness_v2_results.csv
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
X = X.astype(float).reset_index(drop=True)
y = le.transform(df[target])
mi = int(le.transform(["Measles"])[0])

binary_cols = [c for c in X.columns if set(np.unique(X[c])) <= {0.0, 1.0}]
numeric_cols = [c for c in X.columns if c not in binary_cols]
HALLMARK = [c for c in ("koplik_spots", "cephalocaudal_spread") if c in X.columns]
LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]
KINDS = ["missing", "combined"]
ABSTAIN_IF_KEY_MISSING = 2
COPIES, MAX_AUG_RATE = 3, 0.4
rng = np.random.default_rng(0)


def new_rf():
    return RandomForestClassifier(n_estimators=200, max_depth=10,
                                  class_weight="balanced", random_state=42, n_jobs=-1)


# Key fields = the 4 most informative original features (from a clean fit).
imp = pd.Series(new_rf().fit(X, y).feature_importances_, index=X.columns)
KEY_FIELDS = list(imp.sort_values(ascending=False).head(4).index)
print("Key fields:", KEY_FIELDS)


def fill_missing(Xt, mask, med):
    Xf = Xt.copy()
    for c in Xf.columns:
        Xf.loc[mask[c].values, c] = 0.0 if c in binary_cols else med[c]
    return Xf


def with_indicators(Xf, mask):
    return pd.concat([Xf, mask.astype(float).add_suffix("_missing")], axis=1)


def augment(Xtr, ytr, med):
    n = len(Xtr)
    none = pd.DataFrame(False, index=Xtr.index, columns=Xtr.columns)
    parts, ys = [with_indicators(Xtr, none)], [ytr]
    for _ in range(COPIES):
        rate = rng.uniform(0, MAX_AUG_RATE, size=(n, 1))
        mask = pd.DataFrame(rng.random(Xtr.shape) < rate, columns=Xtr.columns, index=Xtr.index)
        parts.append(with_indicators(fill_missing(Xtr, mask, med), mask))
        ys.append(ytr)
    return pd.concat(parts, ignore_index=True), np.concatenate(ys)


def corrupt(Xt, yt, s, kind, sd, med):
    """Add real-world mess, then mark values unknown (missing is applied last)."""
    Xc, n = Xt.copy(), len(Xt)
    if kind == "combined":
        for c in binary_cols:
            m = rng.random(n) < s / 2
            Xc.loc[m, c] = 1.0 - Xc.loc[m, c]
        for c in numeric_cols:
            Xc[c] = (Xc[c] + rng.normal(0, 2 * s * sd[c], n)).clip(lower=0)
        for c in HALLMARK:
            m = (yt == mi) & (rng.random(n) < s)
            Xc.loc[m, c] = 0.0
    mask = pd.DataFrame(rng.random(Xc.shape) < s, columns=Xc.columns, index=Xc.index)
    return fill_missing(Xc, mask, med), mask


def score(yt, pred, covered):
    yc, pc = yt[covered], pred[covered]
    n_meas = max((yt == mi).sum(), 1)
    tp = ((pc == mi) & (yc == mi)).sum()
    referred = ((yt == mi) & (~covered)).sum()
    acc = (yc == pc).mean() if len(yc) else np.nan
    return (acc, tp / max((yc == mi).sum(), 1), tp / max((pc == mi).sum(), 1),
            covered.mean(), (tp + referred) / n_meas)


rows = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr, te) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}/5...")
    Xtr, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
    ytr, yte = y[tr], y[te]
    med, sd = Xtr.median(), Xtr.std()
    old = new_rf().fit(Xtr, ytr)
    Xa, ya = augment(Xtr, ytr, med)
    v2 = new_rf().fit(Xa, ya)
    everyone = np.ones(len(yte), dtype=bool)
    for kind in KINDS:
        for s in LEVELS:
            Xc, mask = corrupt(Xte, yte, s, kind, sd, med)
            p_old = old.predict(Xc)
            p_v2 = v2.predict(with_indicators(Xc, mask))
            covered = mask[KEY_FIELDS].sum(axis=1).values < ABSTAIN_IF_KEY_MISSING
            rows.append(("1 old (unknown looks like no)", kind, s, fold, *score(yte, p_old, everyone)))
            rows.append(("2 v2 (unknown-aware)", kind, s, fold, *score(yte, p_v2, everyone)))
            rows.append(("3 v2 + refer if key fields unknown", kind, s, fold, *score(yte, p_v2, covered)))

res = pd.DataFrame(rows, columns=["design", "scenario", "level", "fold", "accuracy",
                                  "measles_recall", "measles_precision", "coverage",
                                  "caught_or_referred"])
avg = (res.groupby(["scenario", "level", "design"])
          [["accuracy", "measles_recall", "measles_precision", "coverage", "caught_or_referred"]]
          .mean().round(3))
print("\nBEFORE vs AFTER (mean over 5 folds; level 0.00 = clean data)")
print("coverage = share of cases the tool scores; caught_or_referred = share of measles cases")
print("that were either flagged as measles or referred to a clinician (the triage-safe number).\n")
print(avg.to_string())
res.to_csv("data/processed/robustness_v2_results.csv", index=False)

# ---- Final model on all data, saved as a self-contained bundle for the API ----
med_all = X.median()
Xa, ya = augment(X, y, med_all)
final = new_rf().fit(Xa, ya)
bundle = {
    "model": final,
    "feature_names": list(Xa.columns),
    "original_cols": list(X.columns),
    "binary_cols": binary_cols,
    "numeric_cols": numeric_cols,
    "medians": {c: float(med_all[c]) for c in X.columns},
    "key_fields": KEY_FIELDS,
    "abstain_if_key_missing": ABSTAIN_IF_KEY_MISSING,
    "class_names": list(le.inverse_transform(final.classes_)),
}
joblib.dump(bundle, "models/symptom_bundle_v2.joblib")


def predict_record(b, record):
    """record: {feature: value or None}. None / missing key = unknown."""
    vals, flags = {}, {}
    for c in b["original_cols"]:
        v = record.get(c)
        unknown = v is None or (isinstance(v, float) and np.isnan(v))
        flags[c + "_missing"] = 1.0 if unknown else 0.0
        vals[c] = (b["medians"][c] if c in b["numeric_cols"] else 0.0) if unknown else float(v)
    row = pd.DataFrame([{**vals, **flags}])[b["feature_names"]]
    proba = dict(zip(b["class_names"], b["model"].predict_proba(row)[0].round(3)))
    n_key = sum(flags[c + "_missing"] for c in b["key_fields"])
    return proba, n_key >= b["abstain_if_key_missing"]


print("\nSMOKE TEST (loading the saved bundle and predicting one measles case)")
b = joblib.load("models/symptom_bundle_v2.joblib")
rec = X[y == mi].iloc[0].to_dict()
for label, r in [("all fields known", rec),
                 ("2 key fields unknown", {**rec, KEY_FIELDS[0]: None, KEY_FIELDS[1]: None}),
                 ("1 key field unknown", {**rec, KEY_FIELDS[0]: None})]:
    proba, refer = predict_record(b, r)
    print(f"  {label:22s} Measles={proba['Measles']}  refer_to_clinician={refer}")
print("\nSaved: models/symptom_bundle_v2.joblib and data/processed/robustness_v2_results.csv")
