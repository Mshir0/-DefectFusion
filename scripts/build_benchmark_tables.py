#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


METRICS = (
    "image_auroc",
    "image_aupr",
    "image_f1_max",
    "pixel_auroc",
    "pixel_aupr",
    "pixel_aupro",
    "pixel_f1_max",
    "good_accuracy",
    "defect_recall",
    "balanced_accuracy",
    "decision_accuracy",
)

METRIC_LABELS = {
    "image_auroc": "Image AUROC",
    "image_aupr": "Image AUPR",
    "image_f1_max": "Image F1-max",
    "pixel_auroc": "Pixel AUROC",
    "pixel_aupr": "Pixel AUPR",
    "pixel_aupro": "Pixel AUPRO",
    "pixel_f1_max": "Pixel F1-max",
    "good_accuracy": "Good accuracy",
    "defect_recall": "Defect recall",
    "balanced_accuracy": "Balanced accuracy",
    "decision_accuracy": "Decision accuracy",
}

EXPERIMENT_FIELDS = (
    "method",
    "method_label",
    "dataset",
    "normal_shots",
    "seed",
    "category_count",
    *METRICS,
    "best_balanced_accuracy",
    "normal_decision_calibration",
    "good_decision_quantile",
    "good_decision_quantile_method",
    "normal_decision_view_quantile",
    "map_postprocess",
    "total_seconds",
    "experiment",
    "result_file",
)

CATEGORY_FIELDS = (
    "method",
    "method_label",
    "dataset",
    "normal_shots",
    "seed",
    "experiment",
    "category",
    "images",
    "good_images",
    "good_predicted_normal",
    "good_predicted_anomaly",
    "defect_images",
    "defect_predicted_anomaly",
    "defect_predicted_normal",
    *METRICS,
    "normal_decision_calibration",
    "good_decision_threshold",
    "good_decision_quantile",
    "good_decision_quantile_method",
    "normal_decision_view_quantile",
    "normal_decision_folds",
    "map_postprocess",
    "total_seconds",
    "lora_adapter",
    "result_file",
)

BEST_FIELDS = (
    "dataset",
    "metric",
    "metric_label",
    "best_value",
    "method",
    "method_label",
    "normal_shots",
    "experiment",
    "result_file",
)


def _common(categories: list[dict], key: str):
    values = {item.get(key) for item in categories if item.get(key) is not None}
    return values.pop() if len(values) == 1 else ""


def _number(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _dataset(categories: list[dict], relative: Path) -> str:
    value = _common(categories, "dataset")
    if value:
        return str(value).lower()
    lowered = relative.as_posix().lower()
    if "mvtec" in lowered:
        return "mvtec"
    if "visa" in lowered:
        return "visa"
    return "unknown"


def _method(relative: Path) -> tuple[str, str]:
    lowered = relative.as_posix().lower()
    if "distill" in lowered:
        return "distilled_lora", "Distilled ViT-S+ LoRA"
    return "main_pca", "Main DINOv3 PCA"


def _normal_shots(categories: list[dict], relative: Path):
    value = _common(categories, "normal_shots")
    if value != "":
        return value
    match = re.search(r"(?:^|/)(1|2|4|8)shot(?:/|$)", relative.as_posix())
    return int(match.group(1)) if match else ""


def _timing_total(category: dict) -> float:
    timing = category.get("timing_seconds")
    if isinstance(timing, dict):
        return float(timing.get("total", 0.0))
    return float(category.get("total_seconds", 0.0) or 0.0)


def _sort_key(row: dict):
    shots = row.get("normal_shots", "")
    try:
        shot_order = int(shots)
    except (TypeError, ValueError):
        shot_order = 10_000
    method_order = 0 if row.get("method") == "main_pca" else 1
    return str(row.get("dataset", "")), shot_order, method_order, str(row.get("experiment", ""))


def collect_benchmark_results(input_root: Path) -> tuple[list[dict], list[dict], list[str]]:
    input_root = input_root.resolve()
    experiments: list[dict] = []
    category_rows: list[dict] = []
    warnings: list[str] = []

    for result_path in sorted(input_root.rglob("results.json")):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"Skipped {result_path}: {exc}")
            continue
        categories = payload.get("categories")
        macro = payload.get("macro_average")
        if not isinstance(categories, list) or not categories or not isinstance(macro, dict):
            warnings.append(f"Skipped {result_path}: missing categories or macro_average")
            continue

        relative = result_path.parent.relative_to(input_root)
        experiment = relative.as_posix()
        dataset = _dataset(categories, relative)
        method, method_label = _method(relative)
        normal_shots = _normal_shots(categories, relative)
        experiment_row = {
            "method": method,
            "method_label": method_label,
            "dataset": dataset,
            "normal_shots": normal_shots,
            "seed": _common(categories, "seed"),
            "category_count": len(categories),
            "normal_decision_calibration": _common(categories, "normal_decision_calibration"),
            "good_decision_quantile": _common(categories, "good_decision_quantile"),
            "good_decision_quantile_method": _common(categories, "good_decision_quantile_method"),
            "normal_decision_view_quantile": _common(categories, "normal_decision_view_quantile"),
            "map_postprocess": _common(categories, "map_postprocess"),
            "total_seconds": sum(_timing_total(item) for item in categories),
            "experiment": experiment,
            "result_file": str(result_path),
            "best_balanced_accuracy": False,
        }
        experiment_row.update({metric: macro.get(metric, "") for metric in METRICS})
        experiments.append(experiment_row)

        for category in categories:
            category_row = {
                "method": method,
                "method_label": method_label,
                "dataset": dataset,
                "normal_shots": category.get("normal_shots", normal_shots),
                "seed": category.get("seed", ""),
                "experiment": experiment,
                "category": category.get("category", ""),
                "total_seconds": _timing_total(category),
                "result_file": str(result_path),
            }
            for field in CATEGORY_FIELDS:
                if field not in category_row:
                    category_row[field] = category.get(field, "")
            category_rows.append(category_row)

    experiments.sort(key=_sort_key)
    category_rows.sort(key=lambda row: (*_sort_key(row), str(row.get("category", ""))))
    for dataset in {str(row["dataset"]) for row in experiments}:
        candidates = [
            row for row in experiments
            if row["dataset"] == dataset and _number(row.get("balanced_accuracy")) is not None
        ]
        if not candidates:
            continue
        best_value = max(float(row["balanced_accuracy"]) for row in candidates)
        for row in candidates:
            row["best_balanced_accuracy"] = math.isclose(
                float(row["balanced_accuracy"]), best_value, rel_tol=1e-12, abs_tol=1e-12
            )
    return experiments, category_rows, warnings


def best_metric_rows(experiments: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for dataset in sorted({str(row["dataset"]) for row in experiments}):
        dataset_rows = [row for row in experiments if row["dataset"] == dataset]
        for metric in METRICS:
            candidates = [(row, _number(row.get(metric))) for row in dataset_rows]
            candidates = [(row, value) for row, value in candidates if value is not None]
            if not candidates:
                continue
            best_value = max(value for _, value in candidates)
            for row, value in candidates:
                if not math.isclose(value, best_value, rel_tol=1e-12, abs_tol=1e-12):
                    continue
                rows.append({
                    "dataset": dataset,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "best_value": best_value,
                    "method": row["method"],
                    "method_label": row["method_label"],
                    "normal_shots": row["normal_shots"],
                    "experiment": row["experiment"],
                    "result_file": row["result_file"],
                })
    return rows


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _percent(value, *, bold: bool = False) -> str:
    number = _number(value)
    if number is None:
        return "-"
    rendered = f"{100.0 * number:.2f}"
    return f"**{rendered}**" if bold else rendered


def _markdown(experiments: list[dict], best_rows: list[dict]) -> str:
    lines = [
        "# Shot and Distillation Benchmark",
        "",
        "All metric values are percentages. Bold values are the best completed result within each dataset.",
        "Balanced accuracy is the primary metric for selecting a good/anomaly decision threshold.",
        "",
    ]
    best_lookup = {
        (row["dataset"], row["metric"], row["experiment"])
        for row in best_rows
    }
    shown_metrics = (
        "image_auroc", "image_aupr", "pixel_auroc", "pixel_aupr",
        "pixel_aupro", "pixel_f1_max", "good_accuracy", "defect_recall",
        "balanced_accuracy",
    )
    headers = (
        "Method", "Shots", "I-AUROC", "I-AUPR", "P-AUROC", "P-AUPR",
        "P-AUPRO", "P-F1", "Good Acc", "Defect Recall", "Balanced Acc",
    )
    for dataset in sorted({str(row["dataset"]) for row in experiments}):
        lines.extend([
            f"## {dataset.upper()}",
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---", "---:", *("---:" for _ in shown_metrics)]) + " |",
        ])
        for row in (item for item in experiments if item["dataset"] == dataset):
            values = [
                _percent(
                    row.get(metric),
                    bold=(dataset, metric, row["experiment"]) in best_lookup,
                )
                for metric in shown_metrics
            ]
            lines.append(
                "| " + " | ".join([
                    str(row["method_label"]), str(row["normal_shots"]), *values,
                ]) + " |"
            )
        lines.append("")

    lines.extend([
        "## Best Threshold Result",
        "",
        "| Dataset | Method | Shots | Balanced Acc | Good Acc | Defect Recall | Result |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in (item for item in experiments if item.get("best_balanced_accuracy")):
        lines.append(
            "| " + " | ".join([
                str(row["dataset"]).upper(),
                str(row["method_label"]),
                str(row["normal_shots"]),
                _percent(row.get("balanced_accuracy")),
                _percent(row.get("good_accuracy")),
                _percent(row.get("defect_recall")),
                f"`{row['experiment']}`",
            ]) + " |"
        )
    lines.append("")
    return "\n".join(lines)


def build_tables(input_root: Path, output_dir: Path) -> dict[str, Path]:
    experiments, categories, warnings = collect_benchmark_results(input_root)
    if not experiments:
        raise ValueError(f"No complete benchmark results.json files found under {input_root}")
    best_rows = best_metric_rows(experiments)
    best_balanced = [row for row in experiments if row.get("best_balanced_accuracy")]

    paths = {
        "experiments": output_dir / "experiment_results.csv",
        "categories": output_dir / "category_results.csv",
        "best": output_dir / "best_results.csv",
        "best_balanced": output_dir / "best_balanced_results.csv",
        "markdown": output_dir / "results.md",
    }
    _write_csv(paths["experiments"], EXPERIMENT_FIELDS, experiments)
    _write_csv(paths["categories"], CATEGORY_FIELDS, categories)
    _write_csv(paths["best"], BEST_FIELDS, best_rows)
    _write_csv(paths["best_balanced"], EXPERIMENT_FIELDS, best_balanced)
    paths["markdown"].write_text(_markdown(experiments, best_rows), encoding="utf-8")
    if warnings:
        (output_dir / "warnings.txt").write_text("\n".join(warnings) + "\n", encoding="utf-8")
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build shot/distillation result tables")
    parser.add_argument("--input-root", required=True, help="benchmark output root")
    parser.add_argument("--output-dir", help="table directory; defaults to INPUT_ROOT/tables")
    args = parser.parse_args(argv)
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir) if args.output_dir else input_root / "tables"
    try:
        paths = build_tables(input_root, output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"[benchmark-table] experiments: {paths['experiments']}")
    print(f"[benchmark-table] category details: {paths['categories']}")
    print(f"[benchmark-table] per-metric best: {paths['best']}")
    print(f"[benchmark-table] threshold best: {paths['best_balanced']}")
    print(f"[benchmark-table] markdown: {paths['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
