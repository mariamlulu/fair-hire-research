from pathlib import Path
import zipfile

downloads = Path("downloads")
zips = sorted(downloads.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)

if not zips:
    raise RuntimeError("No .zip file found in downloads/. Run 01_download_ipums.py again.")

zip_path = zips[0]
print("✅ Using zip:", zip_path.name)

# Extract it
out_dir = downloads / zip_path.stem
out_dir.mkdir(exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(out_dir)

print("✅ Extracted to:", out_dir)

# List files inside (so we know how to read it)
all_files = sorted([p.relative_to(out_dir) for p in out_dir.rglob("*") if p.is_file()])
print("\nFILES INSIDE EXTRACT:")
for f in all_files:
    print(" -", f)
