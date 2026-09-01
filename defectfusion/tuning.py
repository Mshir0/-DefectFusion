from __future__ import annotations

import argparse
import json
from pathlib import Path


def underperforming_categories(results_path: Path, metric: str, threshold: float) -> list[str]:
    payload = json.loads(Path(results_path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"results file has no category metrics: {results_path}")
    categories = payload.get("categories")
    if not isinstance(categories, list):
        raise ValueError(f"results file has no category metrics: {results_path}")
    selected = []
    for item in categories:
        if not isinstance(item, dict) or not item.get("category"):
            continue
        value = item.get(metric)
        if isinstance(value, (int, float)) and value < threshold:
            selected.append(str(item["category"]))
    return sorted(set(selected))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Select low-performing categories from DefectFusion results")
    parser.add_argument("--results", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--below", required=True, type=float)
    args = parser.parse_args(argv)
    try:
        categories = underperforming_categories(Path(args.results), args.metric, args.below)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    for category in categories:
        print(category)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
