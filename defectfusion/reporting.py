from __future__ import annotations

import csv
import json
from pathlib import Path


def experiment_output_dir(output: str) -> Path:
    path = Path(output)
    return path.parent / path.stem if path.suffix else path


def completed_category_metrics(path: Path, category: str) -> dict | None:
    """Return metrics only when a category output is complete and reusable."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    metrics = payload.get("metrics")
    predictions = payload.get("predictions")
    if not isinstance(metrics, dict) or not isinstance(predictions, list):
        return None
    if metrics.get("category") != category:
        return None
    return metrics


def write_metrics_csv(path: Path, category_metrics: list[dict], macro: dict) -> None:
    fields = [
        "category", "images", "good_images", "good_decision_images", "good_predicted_normal",
        "good_predicted_anomaly", "good_accuracy", "defect_images", "defect_decision_images",
        "defect_predicted_anomaly", "defect_predicted_normal", "defect_recall",
        "balanced_accuracy", "decision_accuracy", "pixel_metric_images", "good_decision_threshold",
        "good_decision_threshold_source", "good_decision_quantile", "good_decision_quantile_method",
        "good_decision_reference_images", "normal_decision_calibration",
        "normal_decision_augment_count", "normal_decision_view_quantile",
        "normal_decision_fit_augment_count",
        "normal_decision_folds", "normal_decision_seed",
        "image_auroc", "image_aupr", "image_f1_max",
        "pixel_auroc", "pixel_aupr", "pixel_aupro", "pixel_f1_max", "defect_type_accuracy",
        "defect_type_macro_precision", "defect_type_macro_recall", "defect_type_macro_f1",
        "defect_type_weighted_f1", "defect_type_calibration", "defect_type_calibration_samples",
        "defect_type_calibration_macro_f1", "defect_type_unknown_threshold",
        "total_seconds", "memory_patch_count", "memory_bytes",
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
