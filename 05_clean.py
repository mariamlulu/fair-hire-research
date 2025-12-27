import numpy as np
import pandas as pd

df = pd.read_parquet("cps_raw.parquet").copy()

print("Start rows:", len(df))

# Keep adults in stable working ages
df = df[(df["AGE"] >= 25) & (df["AGE"] <= 60)].copy()
print("After age filter (25–60):", len(df))

# EMPLOYED target:
# IPUMS CPS: EMPSTAT 10 = At work, 12 = Has job not at work (common CPS codes)
df["EMPLOYED"] = df["EMPSTAT"].isin([10, 12]).astype(int)

# Clean wage income:
# 0 and special high values (like 99999999) should be treated as missing
df["INCWAGE"] = pd.to_numeric(df["INCWAGE"], errors="coerce")
df.loc[df["INCWAGE"] <= 0, "INCWAGE"] = np.nan
df.loc[df["INCWAGE"] >= 9.9e7, "INCWAGE"] = np.nan  # catches 99999999 topcode/missing

# Log transform (only for rows with wage)
df["LOG_WAGE"] = np.log1p(df["INCWAGE"])

# Make simple categorical versions (keeps models clean)
df["OCC"] = df["OCC"].astype("int32")
df["IND"] = df["IND"].astype("int32")
df["EDUC"] = df["EDUC"].astype("int32")
df["SEX"] = df["SEX"].astype("int32")
df["YEAR"] = df["YEAR"].astype("int32")

df.to_parquet("cps_clean.parquet", index=False)
print("✅ Saved cps_clean.parquet")
print("Columns:", df.columns.tolist())
print(df[["YEAR","AGE","SEX","EDUC","EMPSTAT","EMPLOYED","INCWAGE","LOG_WAGE"]].head(10))
