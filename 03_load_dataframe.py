from pathlib import Path
from ipumspy import readers

xml_path = Path("downloads/cps_00001.xml")
data_path = Path("downloads/cps_00001.dat.gz")

print("1) Reading DDI (XML instructions)...")
ddi = readers.read_ipums_ddi(xml_path)
print("✅ DDI loaded")

print("2) Reading microdata (.dat.gz)... (this is the heavy step)")
df = readers.read_microdata(ddi, data_path)
print("✅ Data loaded successfully!")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Column names:", list(df.columns))
print("\nFirst 5 rows:")
print(df.head())
