"""
Trains a Random Forest classifier on the symptom dataset to distinguish
Measles from lookalike illnesses. Saves the trained model and prints
evaluation metrics with special attention to Measles recall.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/processed/symptom_dataset.csv")

X = df.drop(columns=["disease"])
y = df["disease"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=le.classes_))

report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
measles_recall = report_dict["Measles"]["recall"]
measles_precision = report_dict["Measles"]["precision"]
print(f"\n>>> MEASLES RECALL: {measles_recall:.3f} (of actual measles cases, % correctly caught)")
print(f">>> MEASLES PRECISION: {measles_precision:.3f} (of predicted measles cases, % correct)")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Symptom-Based Classifier")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
os.makedirs("data/processed/eda_plots", exist_ok=True)
plt.savefig("data/processed/eda_plots/confusion_matrix.png", dpi=120)
plt.close()

importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)
print(importances)

plt.figure(figsize=(8, 6))
importances.plot(kind="barh", color="darkgreen")
plt.gca().invert_yaxis()
plt.title("Feature Importance — What Drives the Prediction")
plt.tight_layout()
plt.savefig("data/processed/eda_plots/feature_importance.png", dpi=120)
plt.close()

joblib.dump(clf, "models/symptom_classifier.joblib")
joblib.dump(le, "models/label_encoder.joblib")
joblib.dump(list(X.columns), "models/feature_columns.joblib")

print("\nSaved model to models/symptom_classifier.joblib")
print("Saved confusion matrix and feature importance plots to data/processed/eda_plots/")
