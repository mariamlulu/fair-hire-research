import pandas as pd

df = pd.read_parquet("cps_clean.parquet").copy()

# EDUC codes in your extract:
# 111, 123, 124, 125 represent Bachelor's+ (and above)
df["SKILLED"] = (df["EDUC"] >= 111).astype(int)

# Save
df.to_parquet("cps_skilled.parquet", index=False)

print("✅ Saved cps_skilled.parquet")
print("Total rows:", len(df))
print("Skilled rows:", int(df["SKILLED"].sum()))
print("Skilled share:", round(df["SKILLED"].mean(), 4))

print("\nEDUC counts among SKILLED (top):")
print(df[df["SKILLED"] == 1]["EDUC"].value_counts())
