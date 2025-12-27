"""
10_plot_coefficients.py

Purpose:
    Visualize the most influential features in a logistic regression model predicting employment status among skilled U.S. workers (2018–2025).

Inputs:
    - cps_skilled.parquet: Cleaned microdata (IPUMS CPS ASEC), restricted to skilled workers.

Outputs:
    - employment_coefficients.png: Bar plot of the top 20 model coefficients by absolute value.
    - Console output: Confirmation of plot file creation.

Modeling Approach:
    - Logistic regression with one-hot encoding for categorical features.
    - All relevant demographic and occupational variables included as predictors.
    - Model fitted to the full skilled worker sample.

Interpretation of Results:
    - The plot displays the features with the largest positive or negative association with employment probability, as estimated by the model.
    - Coefficient magnitude reflects the relative importance of each feature, conditional on other variables.
    - Interpretation should consider the direction and size of coefficients, but not causal effects.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

df = pd.read_parquet("cps_skilled.parquet")
df = df[df["SKILLED"] == 1]

X = df[["AGE", "EDUC", "OCC", "IND", "YEAR", "SEX"]]
y = df["EMPLOYED"]

cat = ["EDUC", "OCC", "IND", "YEAR", "SEX"]
num = ["AGE"]

pre = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", "passthrough", num),
    ]
)

model = LogisticRegression(max_iter=1000)

pipe = Pipeline([("pre", pre), ("model", model)])
pipe.fit(X, y)

feature_names = pipe.named_steps["pre"].get_feature_names_out()
coefs = pipe.named_steps["model"].coef_[0]

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coef": coefs
}).sort_values("coef", key=abs, ascending=False)

top = coef_df.head(20)

plt.figure(figsize=(8, 6))
plt.barh(top["feature"], top["coef"])
plt.title("Top Coefficients — Employment Model")
plt.xlabel("Coefficient magnitude")
plt.tight_layout()

# Save the figure
plt.savefig("employment_coefficients.png", dpi=150)
print("✅ Plot saved as employment_coefficients.png")

plt.show()

