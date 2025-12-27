"""
07_models_employment.py

Purpose:
    Empirically assess the predictive power of skills and gender for employment status among skilled U.S. workers (2018–2025).

Inputs:
    - cps_skilled.parquet: Cleaned microdata (IPUMS CPS ASEC), restricted to skilled workers.

Outputs:
    - Console output: Area Under the ROC Curve (AUC) for two logistic regression models:
        1. Skills-only (age, education, occupation, industry, year)
        2. Skills + gender (adds sex)
    - Incremental AUC lift from including gender.

Modeling Approach:
    - Logistic regression with one-hot encoding for categorical features.
    - Two feature sets compared to isolate the incremental predictive value of gender.
    - Model performance evaluated using AUC on a held-out test set.

Interpretation of Results:
    - AUC quantifies the model's ability to distinguish employed from non-employed individuals.
    - The difference in AUC between the two models reflects the additional predictive power of gender, controlling for skills and occupation.
    - A negligible incremental lift suggests that, conditional on skills, gender does not meaningfully improve employment prediction.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------
# 1) Load data
# --------------------------------------------------
df = pd.read_parquet("cps_skilled.parquet").copy()

# Keep ONLY skilled workers
df = df[df["SKILLED"] == 1].copy()

print("Rows (skilled only):", len(df))

# --------------------------------------------------
# 2) Target variable
# --------------------------------------------------
y = df["EMPLOYED"].astype(int)

# --------------------------------------------------
# 3) Feature sets
# --------------------------------------------------
# Skills-only features
base_features = ["AGE", "EDUC", "OCC", "IND", "YEAR"]

# Skills + gender
gender_feature = ["SEX"]

# --------------------------------------------------
# 4) Training function
# --------------------------------------------------
def train_and_score(feature_cols):
    X = df[feature_cols].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    categorical = [c for c in feature_cols if c in ["EDUC", "OCC", "IND", "YEAR", "SEX"]]
    numeric = [c for c in feature_cols if c == "AGE"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )

    model = LogisticRegression(max_iter=1000)

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)

    probs = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    return auc

# --------------------------------------------------
# 5) Run models
# --------------------------------------------------
auc_skills = train_and_score(base_features)
auc_skills_gender = train_and_score(base_features + gender_feature)

print("\n✅ EMPLOYMENT MODEL RESULTS (Skilled workers)")
print("Model A — Skills only AUC:", round(auc_skills, 4))
print("Model B — Skills + Gender AUC:", round(auc_skills_gender, 4))
print("Incremental lift from adding gender:",
      round(auc_skills_gender - auc_skills, 6))
