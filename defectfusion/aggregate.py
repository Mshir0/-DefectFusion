from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


METRIC_FIELDS = (
    "image_auroc", "image_aupr", "image_f1_max",
    "pixel_auroc", "pixel_aupr", "pixel_aupro", "pixel_f1_max",
    "defect_type_accuracy", "defect_type_macro_precision",
    "defect_type_macro_recall", "defect_type_macro_f1",
    "defect_type_weighted_f1",
)

MACRO_FIELDS = (
    "experiment", "dataset", "normal_shots", "defect_shots", "seed",
    "category_count", "category_names", "images", *METRIC_FIELDS,
    "total_seconds", "peak_memory_patch_count", "peak_memory_bytes",
    "normal_fit_max_patches", "source",
)

CATEGORY_LEADING_FIELDS = (
    "experiment", "dataset", "normal_shots", "defect_shots", "seed",
    "category", "images", *METRIC_FIELDS, "total_seconds",
    "timing_prediction_seconds", "timing_pixel_preparation_seconds",
    "timing_json_output_seconds", "timing_metrics_seconds",
    "memory_patch_count", "memory_bytes",
)


def _common_value(categories: list[dict], key: str):
    values = {item.get(key) for item in categories if item.get(key) is not None}
    return values.pop() if len(values) == 1 else ""


def _dataset_name(categories: list[dict], experiment: str) -> str:
    dataset = _common_value(categories, "dataset")
    if dataset:
        return str(dataset)
    lowered = experiment.lower()
    if "mvtec" in lowered:
        return "mvtec"
    if "visa" in lowered:
        return "visa"
    return "unknown"


def _timing_total(metrics: dict) -> float:
    timing = metrics.get("timing_seconds", {})
    return float(timing.get("total", 0.0)) if isinstance(timing, dict) else 0.0


def _category_row(experiment: str, metrics: dict) -> dict:
    row = {"experiment": experiment}
    for key, value in metrics.items():
        if key == "timing_seconds" and isinstance(value, dict):
            row["total_seconds"] = value.get("total", "")
            for timing_name, timing_value in value.items():
                if timing_name != "total":
                    row[f"timing_{timing_name}_seconds"] = timing_value
        elif value is None or isinstance(value, (str, int, float, bool)):
            row[key] = "" if value is None else value
    return row


def collect_results(input_root: Path) -> tuple[list[dict], list[dict], list[str]]:
    input_root = input_root.resolve()
    macro_rows: list[dict] = []
    category_rows: list[dict] = []
    warnings: list[str] = []

    for result_path in sorted(input_root.rglob("results.json")):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"skipped {result_path}: {exc}")
            continue
        categories = payload.get("categories")
        macro = payload.get("macro_average")
        if not isinstance(categories, list) or not categories or not isinstance(macro, dict):
            warnings.append(f"skipped {result_path}: missing categories or macro_average")
            continue

        experiment_path = result_path.parent.relative_to(input_root)
        experiment = experiment_path.as_posix() if experiment_path.parts else result_path.parent.name
        category_names = [str(item.get("category", "")) for item in categories]
        row = {
            "experiment": experiment,
            "dataset": _dataset_name(categories, experiment),
            "normal_shots": _common_value(categories, "normal_shots"),
            "defect_shots": _common_value(categories, "defect_shots"),
            "seed": _common_value(categories, "seed"),
            "category_count": len(categories),
            "category_names": ";".join(category_names),
            "images": sum(int(item.get("images", 0)) for item in categories),
            "total_seconds": sum(_timing_total(item) for item in categories),
            "peak_memory_patch_count": max(int(item.get("memory_patch_count", 0)) for item in categories),
            "peak_memory_bytes": max(int(item.get("memory_bytes", 0)) for item in categories),
            "normal_fit_max_patches": _common_value(categories, "normal_fit_max_patches"),
            "source": str(result_path),
        }
        row.update({field: macro.get(field, "") for field in METRIC_FIELDS})
        row.update({
            key: value
            for key, value in macro.items()
            if key not in row and (value is None or isinstance(value, (str, int, float, bool)))
        })
        macro_rows.append(row)
        category_rows.extend(_category_row(experiment, metrics) for metrics in categories)

    return macro_rows, category_rows, warnings


def _write_csv(path: Path, rows: list[dict], leading_fields: tuple[str, ...]) -> None:
    all_fields = {key for row in rows for key in row}
    fields = [field for field in leading_fields if field in all_fields]
    fields.extend(sorted(all_fields - set(fields)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_statistics(input_root: Path, output_dir: Path) -> tuple[Path, Path, int, list[str]]:
    macro_rows, category_rows, warnings = collect_results(input_root)
    if not macro_rows:
        raise ValueError(f"No complete results.json files found under {input_root}")
    macro_path = output_dir / "experiment_metrics.csv"
    category_path = output_dir / "category_metrics.csv"
    _write_csv(macro_path, macro_rows, MACRO_FIELDS)
    _write_csv(category_path, category_rows, CATEGORY_LEADING_FIELDS)
    return macro_path, category_path, len(macro_rows), warnings


def _format_metric(value) -> str:
    if value in (None, ""):
        return "-"
    return f"{100 * float(value):.2f}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate all completed DefectFusion experiments")
    parser.add_argument("--input", default="outputs", help="root containing experiment results.json files")
    parser.add_argument("--output", default="outputs/all-results-summary", help="aggregate output directory")
    args = parser.parse_args(argv)

    try:
        macro_path, category_path, count, warnings = write_statistics(Path(args.input), Path(args.output))
    except ValueError as exc:
        parser.error(str(exc))
    for warning in warnings:
        print(f"[summary] warning: {warning}", file=sys.stderr)

    macro_rows, _, _ = collect_results(Path(args.input))
    print("experiment\tnormal\tdefect\tI-AUROC\tP-AUROC\tPRO\tP-F1\tType-F1")
    for row in macro_rows:
        print(
            f"{row['experiment']}\t{row['normal_shots']}\t{row['defect_shots']}\t"
            f"{_format_metric(row['image_auroc'])}\t{_format_metric(row['pixel_auroc'])}\t"
            f"{_format_metric(row['pixel_aupro'])}\t{_format_metric(row['pixel_f1_max'])}\t"
            f"{_format_metric(row['defect_type_macro_f1'])}"
        )
    print(f"[summary] {count} experiments -> {macro_path}")
    print(f"[summary] category details -> {category_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
