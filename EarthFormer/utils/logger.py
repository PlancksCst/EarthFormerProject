"""CSV logging utilities."""

from __future__ import annotations

import csv
from pathlib import Path


class CSVLogger:
    """Append epoch metrics to a CSV file."""

    fieldnames = ["epoch", "train_loss", "validation_loss", "learning_rate", "epoch_time"]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(
        self,
        epoch: int,
        train_loss: float,
        validation_loss: float,
        learning_rate: float,
        epoch_time: float,
    ) -> None:
        """Append one epoch row."""
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "learning_rate": learning_rate,
                    "epoch_time": epoch_time,
                }
            )
