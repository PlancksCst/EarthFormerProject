"""CSV logging utilities."""

from __future__ import annotations

import csv
from pathlib import Path


class CSVLogger:
    """Append epoch metrics to a CSV file."""

    fieldnames = [
        "epoch",
        "train_loss",
        "train_csi_loss",
        "train_ghi_loss",
        "train_valid_fraction",
        "val_loss",
        "val_csi_loss",
        "val_ghi_loss",
        "valid_fraction",
        "CSI_MAE",
        "CSI_RMSE",
        "CSI_nRMSE",
        "CSI_R2",
        "CSI_MBE",
        "GHI_MAE",
        "GHI_RMSE",
        "GHI_nRMSE",
        "GHI_R2",
        "GHI_MBE",
        "learning_rate",
        "lr_backbone",
        "lr_head",
        "best_val_loss",
        "patience_counter",
        "epoch_time",
    ]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
        else:
            self._upgrade_header_if_needed()

    def _upgrade_header_if_needed(self) -> None:
        """Rewrite an older log file with any newly added columns."""
        with self.path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames or [])
            if existing_fields == self.fieldnames:
                return
            rows = list(reader)
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in self.fieldnames})

    def log(self, **row: float | int | str) -> None:
        """Append one epoch row."""
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({field: row.get(field, "") for field in self.fieldnames})
