import pandas as pd
from pathlib import Path
from ipumspy import readers

# Load data
df = pd.read_parquet("cps_clean.parquet").copy()

# Load codebook (gives us labels for EDUC codes)
ddi = readers.read_ipums_ddi(Path("downloads/cps_00001.xml"))

# Get EDUC labels from the codebook
educ_var = ddi.variables["EDUC"]
educ_map = {int(k): v for k, v in educ_var.codes.items()}

# Add readable label column
df["EDUC_LABEL"] = df["EDUC"].map(educ_map)

# Define skilled: Bachelor's or higher
# We'll do it by checking label text (robust across coding)
def is_ba_plus(label: str) -> int:
    if label is None:
        return 0
    label_low = str(label).lower()
    keywords = [
        "bachelor", "ba", "bs", "b.s", "b.a",
        "master", "ma", "ms", "m.s", "m.a",
        "professional", "doctor", "phd", "jd", "md"
    ]
    return int(any(k in label_low for k in keywords))

df["SKILLED"] = df["EDUC_LABEL"].apply(is_ba_plus).astype(int)

# Save
df.to_parquet("cps_skilled.parquet", index=False)

print("✅ Saved cps_skilled.parquet")
print("Skilled rate:", df["SKILLED"].mean())

print("\nTop EDUC labels (to sanity check):")
print(df["EDUC_LABEL"].value_counts().head(15))

print("\nSkilled EDUC labels (sample):")
print(df.loc[df["SKILLED"] == 1, "EDUC_LABEL"].value_counts().head(15))
