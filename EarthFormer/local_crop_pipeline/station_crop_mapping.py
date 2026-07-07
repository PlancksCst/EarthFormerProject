"""Map station latitude/longitude to pixels in the 200x200 Lebanon crop."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from configs.config import discover_locations_csv
except Exception:
    discover_locations_csv = None  # type: ignore


STATION_NAMES = (
    "LBRAS",
    "LBAMA",
    "BEIRUT",
    "TRIPOLI",
    "TYRE",
    "AKKAR_HIGHLANDS",
    "MOUNT_LEBANON",
    "HERMEL",
    "NABATIEH",
    "SOUTH_HIGHLANDS",
)

# Conservative station-coordinate fallback used only when no locations CSV is present.
DEFAULT_STATION_COORDS: dict[str, tuple[float, float]] = {
    "LBRAS": (33.887, 35.513),
    "LBAMA": (33.563, 35.368),
    "BEIRUT": (33.893, 35.502),
    "TRIPOLI": (34.436, 35.849),
    "TYRE": (33.270, 35.203),
    "AKKAR_HIGHLANDS": (34.530, 36.080),
    "MOUNT_LEBANON": (33.880, 35.720),
    "HERMEL": (34.394, 36.384),
    "NABATIEH": (33.378, 35.483),
    "SOUTH_HIGHLANDS": (33.120, 35.420),
}


@dataclass(frozen=True)
class CropBounds:
    """Approximate geographic bounds for the 200x200 shared crop."""

    lat_min: float = 33.0
    lat_max: float = 34.7
    lon_min: float = 35.0
    lon_max: float = 36.7
    height: int = 200
    width: int = 200


def _normalise_location(value: Any) -> str:
    return str(value).strip().upper()


def _column_lookup(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def load_station_locations(locations_csv: Path | None = None) -> dict[str, tuple[float, float]]:
    """Load station latitude/longitude from project metadata or fallback defaults."""
    path = locations_csv
    if path is None and discover_locations_csv is not None:
        path = Path(discover_locations_csv())
    stations: dict[str, tuple[float, float]] = {}

    if path is not None and path.exists():
        frame = pd.read_csv(path)
        columns = list(frame.columns)
        location_column = _column_lookup(
            columns,
            ("location", "station", "name", "site", "id"),
        )
        lat_column = _column_lookup(columns, ("lat", "latitude", "Latitude"))
        lon_column = _column_lookup(columns, ("lon", "longitude", "lng", "Longitude"))
        if lat_column is None or lon_column is None:
            raise ValueError(
                f"Could not identify latitude/longitude columns in {path}. "
                f"Columns: {columns}"
            )
        if location_column is None:
            location_column = columns[0]
        for _, row in frame.iterrows():
            location = _normalise_location(row[location_column])
            lat = float(row[lat_column])
            lon = float(row[lon_column])
            if math.isfinite(lat) and math.isfinite(lon):
                stations[location] = (lat, lon)

    for name, coords in DEFAULT_STATION_COORDS.items():
        stations.setdefault(name, coords)

    missing = [name for name in STATION_NAMES if name not in stations]
    if missing:
        raise RuntimeError(f"Missing station coordinates for: {', '.join(missing)}")
    return {name: stations[name] for name in STATION_NAMES}


def latlon_to_pixel(lat: float, lon: float, bounds: CropBounds) -> tuple[int, int, bool]:
    """Linearly map lat/lon to integer pixel coordinates."""
    if bounds.lat_max <= bounds.lat_min or bounds.lon_max <= bounds.lon_min:
        raise ValueError("Invalid crop geographic bounds.")
    y_float = (bounds.lat_max - lat) / (bounds.lat_max - bounds.lat_min) * (bounds.height - 1)
    x_float = (lon - bounds.lon_min) / (bounds.lon_max - bounds.lon_min) * (bounds.width - 1)
    y = int(round(y_float))
    x = int(round(x_float))
    inside = 0 <= y < bounds.height and 0 <= x < bounds.width
    return y, x, inside


def crop_window(
    center_y: int,
    center_x: int,
    crop_size: int,
    height: int = 200,
    width: int = 200,
) -> dict[str, int | bool]:
    """Return clipped crop bounds and whether padding is needed."""
    if crop_size <= 0:
        raise ValueError("crop_size must be positive")
    half = crop_size // 2
    y0_raw = center_y - half
    x0_raw = center_x - half
    y1_raw = y0_raw + crop_size
    x1_raw = x0_raw + crop_size
    y0 = max(0, y0_raw)
    x0 = max(0, x0_raw)
    y1 = min(height, y1_raw)
    x1 = min(width, x1_raw)
    padding_needed = y0 != y0_raw or y1 != y1_raw or x0 != x0_raw or x1 != x1_raw
    return {
        "crop_y0": y0,
        "crop_y1": y1,
        "crop_x0": x0,
        "crop_x1": x1,
        "padding_needed": padding_needed,
    }


def build_station_mapping(
    locations_csv: Path | None = None,
    bounds: CropBounds | None = None,
    local_crop_size: int = 64,
) -> list[dict[str, object]]:
    """Build mapping rows for all configured stations."""
    bounds = bounds or CropBounds()
    stations = load_station_locations(locations_csv)
    rows: list[dict[str, object]] = []
    for location, (lat, lon) in stations.items():
        pixel_y, pixel_x, inside = latlon_to_pixel(lat, lon, bounds)
        window = crop_window(
            pixel_y,
            pixel_x,
            crop_size=local_crop_size,
            height=bounds.height,
            width=bounds.width,
        )
        rows.append(
            {
                "location": location,
                "lat": lat,
                "lon": lon,
                "pixel_y": pixel_y,
                "pixel_x": pixel_x,
                "inside_bounds": bool(inside),
                **window,
            }
        )
    return rows


def write_station_mapping_csv(rows: list[dict[str, object]], output_path: Path) -> Path:
    """Write station mapping rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "location",
        "lat",
        "lon",
        "pixel_y",
        "pixel_x",
        "inside_bounds",
        "crop_y0",
        "crop_y1",
        "crop_x0",
        "crop_x1",
        "padding_needed",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def mapping_by_location(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Return mapping rows keyed by normalized station name."""
    mapping = {_normalise_location(row["location"]): row for row in rows}
    for location, row in mapping.items():
        if not bool(row["inside_bounds"]):
            raise RuntimeError(f"Station {location} maps outside the configured crop bounds.")
    return mapping


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build station pixel mapping for local crops.")
    parser.add_argument("--locations-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/local_crop_pipeline"))
    parser.add_argument("--local-crop-size", type=int, default=64)
    parser.add_argument("--crop-lat-min", type=float, default=CropBounds.lat_min)
    parser.add_argument("--crop-lat-max", type=float, default=CropBounds.lat_max)
    parser.add_argument("--crop-lon-min", type=float, default=CropBounds.lon_min)
    parser.add_argument("--crop-lon-max", type=float, default=CropBounds.lon_max)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    bounds = CropBounds(
        lat_min=args.crop_lat_min,
        lat_max=args.crop_lat_max,
        lon_min=args.crop_lon_min,
        lon_max=args.crop_lon_max,
    )
    rows = build_station_mapping(
        locations_csv=args.locations_csv,
        bounds=bounds,
        local_crop_size=args.local_crop_size,
    )
    path = write_station_mapping_csv(rows, args.output_dir / "station_pixel_mapping.csv")
    print(f"Wrote station pixel mapping: {path}")


if __name__ == "__main__":
    main()
