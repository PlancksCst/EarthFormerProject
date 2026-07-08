from pathlib import Path

root = Path(r"C:\Users\admin\Downloads\FYP\Codes\verification_dataset")  # adjust if needed
for p in root.rglob("*.csv"):
    print(p)