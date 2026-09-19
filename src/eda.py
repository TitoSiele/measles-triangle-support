"""
Exploratory data analysis for the measles symptom dataset.

This script restores the original EDA workflow used in the project:
- load the processed symptom dataset
- inspect class balance and feature distributions
- save summary plots under data/processed/eda_plots/
- exit cleanly for use in notebooks or terminal workflows
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "symptom_dataset.csv"
PLOTS_DIR = ROOT / "data" / "processed" / "eda_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def save_summary(df: pd.DataFrame) -> None:
    """Create a compact EDA report for the symptom dataset."""
    print("\nDataset shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nClass distribution:\n", df["disease"].value_counts().sort_index())
    print("\nSummary:\n", df.describe(include="all").T)

    class_counts = df["disease"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    class_counts.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Disease Class Counts")
    ax.set_xlabel("Disease")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "class_counts.png", dpi=150)
    plt.close(fig)

    numeric_cols = [
        "fever_temp_c",
        "fever_duration_days",
        "rash_onset_day",
        "age_years",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.flat, numeric_cols):
        sns.boxplot(data=df, x="disease", y=col, ax=ax)
        ax.tick_params(axis="x", rotation=30)
        ax.set_title(f"{col} by disease")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_boxplots.png", dpi=150)
    plt.close(fig)

    binary_cols = [
        "cough",
        "coryza",
        "conjunctivitis",
        "koplik_spots",
        "rash_present",
        "cephalocaudal_spread",
        "sore_throat",
        "itchy_rash",
        "lymphadenopathy",
        "vaccinated",
    ]
    binary_summary = df.groupby("disease")[binary_cols].mean().T
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(binary_summary, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    ax.set_title("Proportion of binary symptoms by disease")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "binary_feature_heatmap.png", dpi=150)
    plt.close(fig)

    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=ax)
    ax.set_title("Numeric Feature Correlation")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "numeric_correlation.png", dpi=150)
    plt.close(fig)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Run the data generation script first."
        )

    df = pd.read_csv(DATA_PATH)
    save_summary(df)
    print(f"\nEDA plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
