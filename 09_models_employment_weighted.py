"""
09_models_employment_weighted.py

Purpose:
    Estimate the predictive value of skills and gender for employment status among skilled U.S. workers (2018–2025), using survey weights for population-representative inference.

Inputs:
    - cps_skilled.parquet: Cleaned microdata (IPUMS CPS ASEC), restricted to skilled workers, with survey weights (ASECWT).

Outputs:
    - Console output: Weighted Area Under the ROC Curve (AUC) for two logistic regression models:
        1. Skills-only (age, education, occupation, industry, year)
        2. Skills + gender (adds sex)
    - Incremental AUC lift from including gender.

Modeling Approach:
    - Logistic regression with one-hot encoding for categorical features.
    - Survey weights (ASECWT) applied to both model fitting and evaluation for population-representative results.
    - Two feature sets compared to isolate the incremental predictive value of gender.
    - Model performance evaluated using weighted AUC on a held-out test set.

Interpretation of Results:
    - Weighted AUC quantifies the model's ability to distinguish employed from non-employed individuals, accounting for survey design.
    - The difference in AUC between the two models reflects the additional predictive power of gender, controlling for skills and occupation.
    - A negligible incremental lift suggests that, conditional on skills, gender does not meaningfully improve employment prediction in the weighted (population) sample.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

df = pd.read_parquet("cps_skilled.parquet").copy()
df = df[df["SKILLED"] == 1].copy()

y = df["EMPLOYED"].astype(int)
weights = df["ASECWT"]

base_features = ["AGE", "EDUC", "OCC", "IND", "YEAR"]
gender_feature = ["SEX"]

def train_auc(features):
    X = df[features]

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.25, random_state=42, stratify=y
    )

    categorical = [c for c in features if c in ["EDUC", "OCC", "IND", "YEAR", "SEX"]]
    numeric = ["AGE"]

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )

    model = LogisticRegression(max_iter=1000)

    pipe = Pipeline([("pre", pre), ("model", model)])

    pipe.fit(X_train, y_train, model__sample_weight=w_train)

    probs = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs, sample_weight=w_test)

    return auc

auc_a = train_auc(base_features)
auc_b = train_auc(base_features + gender_feature)

print("\n✅ WEIGHTED EMPLOYMENT MODEL")
print("Skills only AUC:", round(auc_a, 4))
print("Skills + gender AUC:", round(auc_b, 4))
print("Incremental lift:", round(auc_b - auc_a, 6))
