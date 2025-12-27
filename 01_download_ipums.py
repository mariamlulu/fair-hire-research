import os
from pathlib import Path
import dotenv


from ipumspy import IpumsApiClient, MicrodataExtract

# 1) Load key
dotenv.load_dotenv()
API_KEY = os.getenv("IPUMS_API_KEY")
if not API_KEY:
    raise RuntimeError("No IPUMS_API_KEY found. Check your .env file.")

# 2) Connect to IPUMS
ipums = IpumsApiClient(API_KEY)

# 3) Choose samples (ASEC = March supplement = _03s)
# These sample IDs are listed by IPUMS CPS. :contentReference[oaicite:2]{index=2}
samples = [
    "cps2018_03s",
    "cps2019_03s",
    "cps2020_03s",
    "cps2021_03s",
    "cps2022_03s",
    "cps2023_03s",
    "cps2024_03s",
    "cps2025_03s",
]

# 4) Choose variables (simple starter set)
# You can add more later.
variables = [
    "AGE",
    "SEX",
    "EDUC",
    "OCC",
    "IND",
    "EMPSTAT",
    "INCWAGE",
]

# 5) Define the extract
extract = MicrodataExtract(
    collection="cps",
    samples=samples,
    variables=variables,
    description="FairHire: skills vs gender (ASEC 2018–2025)"
)

# 6) Submit extract
ipums.submit_extract(extract)
print(f"✅ Submitted extract. Extract ID = {extract.extract_id}")

# 7) Wait until IPUMS finishes it
ipums.wait_for_extract(extract)
print("✅ Extract is ready!")

# 8) Download it
download_dir = Path("downloads")
download_dir.mkdir(exist_ok=True)
ipums.download_extract(extract, download_dir=download_dir)
print(f"✅ Downloaded to: {download_dir.resolve()}")

