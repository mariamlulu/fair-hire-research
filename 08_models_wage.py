"""
08_models_wage.py

Purpose:
    Quantitatively assess the predictive value of skills and gender for wage outcomes among skilled and employed U.S. workers (2018–2025).

Inputs:
    - cps_skilled.parquet: Cleaned microdata (IPUMS CPS ASEC), filtered for skilled and employed individuals with valid wage data.

Outputs:
    - Console output: R² (coefficient of determination) and MAE (mean absolute error) for two linear regression models:
        1. Skills-only (age, education, occupation, industry, year)
        2. Skills + gender (adds sex)
    - Incremental R² from including gender.

Modeling Approach:
    - Linear regression with one-hot encoding for categorical features.
    - Two feature sets compared to isolate the incremental predictive value of gender.
    - Model performance evaluated using R² and MAE on a held-out test set.

Interpretation of Results:
    - R² measures the proportion of variance in log wages explained by the model.
    - The difference in R² between the two models reflects the additional predictive power of gender, controlling for skills and occupation.
    - A negligible incremental R² suggests that, conditional on skills, gender does not meaningfully improve wage prediction.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# --------------------------------------------------
# 1) Load data
# --------------------------------------------------
df = pd.read_parquet("cps_skilled.parquet").copy()

# Keep only skilled + employed + valid wage
df = df[
    (df["SKILLED"] == 1) &
    (df["EMPLOYED"] == 1) &
    (df["LOG_WAGE"].notna())
].copy()

print("Rows (skilled + employed):", len(df))

# --------------------------------------------------
# 2) Target
# --------------------------------------------------
y = df["LOG_WAGE"]

# --------------------------------------------------
# 3) Feature sets
# --------------------------------------------------
base_features = ["AGE", "EDUC", "OCC", "IND", "YEAR"]
gender_feature = ["SEX"]

# --------------------------------------------------
# 4) Train + evaluate
# --------------------------------------------------
def train_and_score(feature_cols):
    X = df[feature_cols].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    categorical = [c for c in feature_cols if c in ["EDUC", "OCC", "IND", "YEAR", "SEX"]]
    numeric = [c for c in feature_cols if c == "AGE"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )

    model = LinearRegression()

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    return r2, mae

# --------------------------------------------------
# 5) Run models
# --------------------------------------------------
r2_a, mae_a = train_and_score(base_features)
r2_b, mae_b = train_and_score(base_features + gender_feature)

print("\n✅ WAGE MODEL RESULTS (Skilled & employed)")
print("Model A — Skills only:")
print("  R²:", round(r2_a, 4), "| MAE:", round(mae_a, 4))

print("Model B — Skills + Gender:")
print("  R²:", round(r2_b, 4), "| MAE:", round(mae_b, 4))

print("\nIncremental R² from adding gender:", round(r2_b - r2_a, 6))
