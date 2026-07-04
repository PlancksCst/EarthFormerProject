"""CSV logging utilities."""

from __future__ import annotations

import csv
from pathlib import Path


class CSVLogger:
    """Append epoch metrics to a CSV file."""

    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "CSI_MAE",
        "CSI_RMSE",
        "CSI_nRMSE",
        "CSI_R2",
        "GHI_MAE",
        "GHI_RMSE",
        "GHI_nRMSE",
        "GHI_R2",
        "learning_rate",
        "epoch_time",
    ]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, **row: float | int | str) -> None:
        """Append one epoch row."""
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({field: row.get(field, "") for field in self.fieldnames})
