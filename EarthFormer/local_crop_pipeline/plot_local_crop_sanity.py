"""Visual sanity checks for station-centered local crops."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_crop_pipeline.local_crop_dataset import LocalCropDataset  # noqa: E402
from local_crop_pipeline.station_crop_mapping import (  # noqa: E402
    CropBounds,
    build_station_mapping,
    write_station_mapping_csv,
)


def _as_list(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().float().tolist()
    return list(value)


def plot_sample(
    base_item: dict,
    crop_item: dict,
    output_path: Path,
    channel_index: int,
) -> None:
    """Save a figure for one base/crop sample pair."""
    full = base_item["satellite"].detach().cpu()
    crop = crop_item["satellite"].detach().cpu()
    target = _as_list(crop_item.get("target", []))
    clear = _as_list(crop_item.get("clear_sky_ghi", []))
    target_ghi = _as_list(crop_item.get("target_ghi", []))
    center_y = int(crop_item["local_crop_center_y"])
    center_x = int(crop_item["local_crop_center_x"])
    y0 = int(crop_item["local_crop_y0"])
    y1 = int(crop_item["local_crop_y1"])
    x0 = int(crop_item["local_crop_x0"])
    x1 = int(crop_item["local_crop_x1"])

    fig = plt.figure(figsize=(16, 13), constrained_layout=True)
    grid = fig.add_gridspec(5, 5)
    ax_full = fig.add_subplot(grid[0:2, 0:2])
    ax_full.imshow(full[0, channel_index], cmap="gray")
    rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", linewidth=2)
    ax_full.add_patch(rect)
    ax_full.scatter([center_x], [center_y], c="cyan", s=20)
    ax_full.set_title(f"{crop_item['location']} full 200x200")
    ax_full.axis("off")

    ax_crop = fig.add_subplot(grid[0:2, 2:4])
    ax_crop.imshow(crop[0, channel_index], cmap="gray")
    ax_crop.set_title("local 64x64 crop")
    ax_crop.axis("off")

    for t in range(min(13, crop.shape[0])):
        ax = fig.add_subplot(grid[2 + (t // 5), t % 5])
        ax.imshow(crop[t, channel_index], cmap="gray")
        ax.set_title(f"t={t}")
        ax.axis("off")

    ax_curve = fig.add_subplot(grid[:, 4])
    if target:
        ax_curve.plot(target, label="CSI target")
    if target_ghi:
        ax_curve.plot(target_ghi, label="GHI target")
    if clear:
        ax_curve.plot(clear, label="clear-sky GHI")
    ax_curve.legend(fontsize=8)
    ax_curve.set_title("target curves")
    ax_curve.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot local crop sanity figures.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--channel-index", type=int, default=0)
    parser.add_argument("--local-crop-size", type=int, default=64)
    parser.add_argument("--crop-padding-mode", choices=("edge", "reflect"), default="edge")
    parser.add_argument("--locations-csv", type=Path, default=None)
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
    dataset = LocalCropDataset(
        dataset_root=args.dataset_root,
        split=args.split,
        local_crop_size=args.local_crop_size,
        crop_padding_mode=args.crop_padding_mode,
        crop_bounds=bounds,
        locations_csv=args.locations_csv,
        include_target=True,
    )
    mapping_rows = build_station_mapping(args.locations_csv, bounds, args.local_crop_size)
    write_station_mapping_csv(mapping_rows, args.output_dir.parent / "station_pixel_mapping.csv")

    rows = []
    for index in range(min(args.num_samples, len(dataset))):
        base_item = dataset.base_dataset[index]
        crop_item = dataset[index]
        location = str(crop_item["location"])
        output_path = args.output_dir / f"sample_{index}_{location}_local_crop.png"
        plot_sample(base_item, crop_item, output_path, args.channel_index)
        rows.append(
            {
                "sample_index": index,
                "sample_id": crop_item.get("sample_id"),
                "location": location,
                "local_crop_center_y": int(crop_item["local_crop_center_y"]),
                "local_crop_center_x": int(crop_item["local_crop_center_x"]),
                "local_crop_y0": int(crop_item["local_crop_y0"]),
                "local_crop_y1": int(crop_item["local_crop_y1"]),
                "local_crop_x0": int(crop_item["local_crop_x0"]),
                "local_crop_x1": int(crop_item["local_crop_x1"]),
                "figure": str(output_path),
            }
        )

    summary_path = args.output_dir / "local_crop_sanity_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["sample_index"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote local crop sanity outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
