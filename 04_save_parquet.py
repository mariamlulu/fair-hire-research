from pathlib import Path
from ipumspy import readers

xml_path = Path("downloads/cps_00001.xml")
data_path = Path("downloads/cps_00001.dat.gz")

print("Reading DDI...")
ddi = readers.read_ipums_ddi(xml_path)

print("Reading microdata...")
df = readers.read_microdata(ddi, data_path)

out = Path("cps_raw.parquet")
df.to_parquet(out, index=False)

print("✅ Saved:", out.resolve())
print("Rows:", df.shape[0], "Cols:", df.shape[1])
