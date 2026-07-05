"""Artifact persistence helpers for local/Colab training runs."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def discover_drive_root() -> Path | None:
    """Return the Google Drive EarthFormer root when it is mounted."""
    env_value = os.environ.get("EARTHFORMER_DRIVE_ROOT")
    candidate = Path(env_value) if env_value else Path("/content/drive/MyDrive/EarthFormer")
    return candidate if candidate.exists() else None


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


class ArtifactMirror:
    """Mirror generated files from the primary run directories to Google Drive."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        output_dir: str | Path,
        enabled: bool = True,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.drive_root = discover_drive_root() if enabled else None
        self.drive_checkpoint_dir: Path | None = None
        self.drive_output_dir: Path | None = None
        if self.drive_root is not None:
            checkpoint_candidate = self.drive_root / "checkpoints"
            output_candidate = self.drive_root / "outputs"
            if not _is_relative_to(self.checkpoint_dir, self.drive_root):
                self.drive_checkpoint_dir = checkpoint_candidate
            if not _is_relative_to(self.output_dir, self.drive_root):
                self.drive_output_dir = output_candidate

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.drive_checkpoint_dir is not None:
            self.drive_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.drive_output_dir is not None:
            self.drive_output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def has_drive(self) -> bool:
        """Return whether a Drive mirror is active."""
        return self.drive_checkpoint_dir is not None or self.drive_output_dir is not None

    def _destination(self, path: Path, primary_root: Path, mirror_root: Path) -> Path:
        try:
            relative = path.resolve().relative_to(primary_root.resolve())
        except ValueError:
            relative = Path(path.name)
        return mirror_root / relative

    def mirror_checkpoint_file(self, path: str | Path) -> Path | None:
        """Copy one checkpoint file to Drive when a mirror is available."""
        if self.drive_checkpoint_dir is None:
            return None
        return self._copy_file(Path(path), self.checkpoint_dir, self.drive_checkpoint_dir)

    def mirror_output_file(self, path: str | Path) -> Path | None:
        """Copy one output file to Drive when a mirror is available."""
        if self.drive_output_dir is None:
            return None
        return self._copy_file(Path(path), self.output_dir, self.drive_output_dir)

    def mirror_output_tree(self, path: str | Path) -> None:
        """Mirror every file under an output directory to Drive."""
        if self.drive_output_dir is None:
            return
        source = Path(path)
        if not source.exists():
            return
        if source.is_file():
            self.mirror_output_file(source)
            return
        for child in source.rglob("*"):
            if child.is_file():
                try:
                    child.resolve().relative_to(self.output_dir.resolve())
                    self._copy_file(child, self.output_dir, self.drive_output_dir)
                except ValueError:
                    destination = self.drive_output_dir / source.name / child.relative_to(source)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(child, destination)

    def _copy_file(self, source: Path, primary_root: Path, mirror_root: Path) -> Path | None:
        if not source.exists() or not source.is_file():
            return None
        destination = self._destination(source, primary_root, mirror_root)
        if source.resolve() == destination.resolve():
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination
