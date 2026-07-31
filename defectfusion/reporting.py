from __future__ import annotations

import csv
from pathlib import Path


def experiment_output_dir(output: str) -> Path:
    path = Path(output)
    return path.parent / path.stem if path.suffix else path


def write_metrics_csv(path: Path, category_metrics: list[dict], macro: dict) -> None:
    fields = [
        "category", "images", "image_auroc", "image_aupr", "image_f1_max",
        "pixel_auroc", "pixel_aupr", "pixel_aupro", "pixel_f1_max", "defect_type_accuracy",
        "defect_type_macro_f1", "total_seconds", "memory_patch_count", "memory_bytes",
    ]
    rows = []
    for metrics in category_metrics:
        row = {field: metrics.get(field, "") for field in fields}
        row["total_seconds"] = metrics.get("timing_seconds", {}).get("total", "")
        rows.append(row)
    macro_row = {field: "" for field in fields}
    macro_row.update({"category": "macro_average", **macro})
    rows.append(macro_row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
