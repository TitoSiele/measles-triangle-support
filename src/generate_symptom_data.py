"""
Generates a synthetic clinical symptom dataset for measles vs. lookalike
febrile-rash illnesses, based on documented WHO/CDC diagnostic criteria.

Classes:
- Measles
- Chickenpox
- Rubella
- Scarlet Fever
- Roseola
- Nonspecific Viral Fever (no rash / negative class)
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_PER_CLASS = 800

def sample_range(low, high, n, skew=None):
    if skew == "high":
        return np.clip(np.random.beta(4, 2, n) * (high - low) + low, low, high)
    if skew == "low":
        return np.clip(np.random.beta(2, 4, n) * (high - low) + low, low, high)
    return np.random.uniform(low, high, n)

def bernoulli(p, n):
    return np.random.binomial(1, p, n)

def build_class(name, n, params):
    """params: dict of feature -> generator callable(n)"""
    df = pd.DataFrame({k: gen(n) for k, gen in params.items()})
    df["disease"] = name
    return df

# --- Measles ---
measles = build_class("Measles", N_PER_CLASS, {
    "fever_temp_c": lambda n: sample_range(38.3, 40.5, n, skew="high"),
    "fever_duration_days": lambda n: np.random.randint(3, 8, n),
    "cough": lambda n: bernoulli(0.9, n),
    "coryza": lambda n: bernoulli(0.85, n),
    "conjunctivitis": lambda n: bernoulli(0.8, n),
    "koplik_spots": lambda n: bernoulli(0.7, n),
    "rash_present": lambda n: bernoulli(0.95, n),
    "rash_onset_day": lambda n: np.random.randint(3, 6, n),
    "cephalocaudal_spread": lambda n: bernoulli(0.85, n),
    "sore_throat": lambda n: bernoulli(0.3, n),
    "itchy_rash": lambda n: bernoulli(0.15, n),
    "lymphadenopathy": lambda n: bernoulli(0.3, n),
    "age_years": lambda n: sample_range(0.5, 15, n, skew="low"),
    "vaccinated": lambda n: bernoulli(0.15, n),
})

# --- Chickenpox ---
chickenpox = build_class("Chickenpox", N_PER_CLASS, {
    "fever_temp_c": lambda n: sample_range(37.5, 39.0, n),
    "fever_duration_days": lambda n: np.random.randint(1, 4, n),
    "cough": lambda n: bernoulli(0.2, n),
    "coryza": lambda n: bernoulli(0.15, n),
    "conjunctivitis": lambda n: bernoulli(0.05, n),
    "koplik_spots": lambda n: bernoulli(0.01, n),
    "rash_present": lambda n: bernoulli(0.98, n),
    "rash_onset_day": lambda n: np.random.randint(0, 2, n),
    "cephalocaudal_spread": lambda n: bernoulli(0.1, n),
    "sore_throat": lambda n: bernoulli(0.2, n),
    "itchy_rash": lambda n: bernoulli(0.9, n),
    "lymphadenopathy": lambda n: bernoulli(0.2, n),
    "age_years": lambda n: sample_range(0.5, 12, n, skew="low"),
    "vaccinated": lambda n: bernoulli(0.3, n),
})

# --- Rubella ---
rubella = build_class("Rubella", N_PER_CLASS, {
    "fever_temp_c": lambda n: sample_range(37.2, 38.5, n),
    "fever_duration_days": lambda n: np.random.randint(1, 3, n),
    "cough": lambda n: bernoulli(0.3, n),
    "coryza": lambda n: bernoulli(0.3, n),
    "conjunctivitis": lambda n: bernoulli(0.2, n),
    "koplik_spots": lambda n: bernoulli(0.01, n),
    "rash_present": lambda n: bernoulli(0.9, n),
    "rash_onset_day": lambda n: np.random.randint(0, 2, n),
    "cephalocaudal_spread": lambda n: bernoulli(0.5, n),
    "sore_throat": lambda n: bernoulli(0.25, n),
    "itchy_rash": lambda n: bernoulli(0.2, n),
    "lymphadenopathy": lambda n: bernoulli(0.8, n),
    "age_years": lambda n: sample_range(1, 20, n),
    "vaccinated": lambda n: bernoulli(0.2, n),
})

# --- Scarlet Fever ---
scarlet = build_class("Scarlet Fever", N_PER_CLASS, {
    "fever_temp_c": lambda n: sample_range(38.0, 39.5, n, skew="high"),
    "fever_duration_days": lambda n: np.random.randint(1, 5, n),
    "cough": lambda n: bernoulli(0.15, n),
    "coryza": lambda n: bernoulli(0.1, n),
    "conjunctivitis": lambda n: bernoulli(0.05, n),
    "koplik_spots": lambda n: bernoulli(0.01, n),
    "rash_present": lambda n: bernoulli(0.95, n),
    "rash_onset_day": lambda n: np.random.randint(0, 2, n),
    "cephalocaudal_spread": lambda n: bernoulli(0.2, n),
    "sore_throat": lambda n: bernoulli(0.9, n),
    "itchy_rash": lambda n: bernoulli(0.3, n),
    "lymphadenopathy": lambda n: bernoulli(0.4, n),
    "age_years": lambda n: sample_range(3, 12, n),
    "vaccinated": lambda n: bernoulli(0.5, n),
})

# --- Roseola ---
roseola = build_class("Roseola", N_PER_CLASS, {
    "fever_temp_c": lambda n: sample_range(38.5, 40.0, n, skew="high"),
    "fever_duration_days": lambda n: np.random.randint(3, 5, n),
    "cough": lambda n: bernoulli(0.1, n),
    "coryza": lambda n: bernoulli(0.1, n),
    "conjunctivitis": lambda n: bernoulli(0.05, n),
    "koplik_spots": lambda n: bernoulli(0.01, n),
    "rash_present": lambda n: bernoulli(0.9, n),
    "rash_onset_day": lambda n: np.random.randint(4, 6, n),
    "cephalocaudal_spread": lambda n: bernoulli(0.15, n),
    "sore_throat": lambda n: bernoulli(0.1, n),
    "itchy_rash": lambda n: bernoulli(0.1, n),
    "lymphadenopathy": lambda n: bernoulli(0.2, n),
    "age_years": lambda n: sample_range(0.3, 3, n, skew="low"),
    "vaccinated": lambda n: bernoulli(0.4, n),
})

# --- Nonspecific Viral Fever (negative class, no rash) ---
viral = build_class("Nonspecific Viral Fever", N_PER_CLASS, {
    "fever_temp_c": lambda n: sample_range(37.5, 39.0, n),
    "fever_duration_days": lambda n: np.random.randint(1, 4, n),
    "cough": lambda n: bernoulli(0.4, n),
    "coryza": lambda n: bernoulli(0.4, n),
    "conjunctivitis": lambda n: bernoulli(0.1, n),
    "koplik_spots": lambda n: bernoulli(0.0, n),
    "rash_present": lambda n: bernoulli(0.05, n),
    "rash_onset_day": lambda n: np.random.randint(0, 1, n),
    "cephalocaudal_spread": lambda n: bernoulli(0.0, n),
    "sore_throat": lambda n: bernoulli(0.3, n),
    "itchy_rash": lambda n: bernoulli(0.05, n),
    "lymphadenopathy": lambda n: bernoulli(0.15, n),
    "age_years": lambda n: sample_range(0.5, 60, n),
    "vaccinated": lambda n: bernoulli(0.6, n),
})

df = pd.concat([measles, chickenpox, rubella, scarlet, roseola, viral], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df["fever_temp_c"] = df["fever_temp_c"].round(1)
df["age_years"] = df["age_years"].round(1)

df.to_csv("data/processed/symptom_dataset.csv", index=False)
print(f"Generated {len(df)} rows across {df['disease'].nunique()} classes")
print(df["disease"].value_counts())
