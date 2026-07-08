import pandas as pd
from pathlib import Path

# Change this to your actual metadata CSV path
csv_path = Path(r"C:\Users\admin\Downloads\FYP\Codes\prep+models\EarthFormerProject\data\train.csv")

# Backup
backup_path = csv_path.with_name(csv_path.stem + "_backup_before_nan_day_removal.csv")

df = pd.read_csv(csv_path)
df.to_csv(backup_path, index=False)

print("Original rows:", len(df))
print("Backup saved to:", backup_path)

# Normalize dates
df["input_day"] = pd.to_datetime(df["input_day"]).dt.date.astype(str)
df["target_day"] = pd.to_datetime(df["target_day"]).dt.date.astype(str)

bad_input_day = "2019-10-12"
bad_target_day = "2019-10-13"

mask_bad = (df["input_day"] == bad_input_day) | (df["target_day"] == bad_target_day)

print("Rows to remove:", mask_bad.sum())
print(df.loc[mask_bad, ["sample_id", "location", "input_day", "target_day", "input_zarr"]])

df_clean = df.loc[~mask_bad].copy()
df_clean.to_csv(csv_path, index=False)

print("Clean rows:", len(df_clean))
print("Clean CSV overwritten:", csv_path)