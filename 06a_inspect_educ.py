import pandas as pd

df = pd.read_parquet("cps_clean.parquet")

print("Unique EDUC values (sorted, first 50):")
print(sorted(df["EDUC"].dropna().unique())[:50])

print("\nTop EDUC value counts:")
print(df["EDUC"].value_counts().head(20))

