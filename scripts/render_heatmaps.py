#!/usr/bin/env python3
"""Render one real DefectFusion anomaly map as one heatmap PNG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


COLORMAPS = {
    "turbo": (
        (0.00, (48, 18, 59)),
        (0.18, (42, 85, 174)),
        (0.38, (30, 165, 188)),
        (0.58, (77, 202, 94)),
        (0.76, (245, 220, 55)),
        (0.90, (244, 111, 32)),
        (1.00, (151, 17, 24)),
    ),
    "magma": (
        (0.00, (0, 0, 4)),
        (0.22, (57, 15, 110)),
        (0.45, (137, 34, 106)),
        (0.68, (213, 69, 68)),
        (0.86, (251, 140, 60)),
        (1.00, (252, 253, 191)),
    ),
    "jet": (
        (0.00, (0, 0, 128)),
        (0.20, (0, 80, 255)),
        (0.40, (0, 220, 255)),
        (0.60, (160, 255, 95)),
        (0.80, (255, 180, 0)),
        (1.00, (180, 0, 0)),
    ),
}


def parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not layers:
        raise argparse.ArgumentTypeError("feature layers cannot be empty")
    return layers


def load_prediction_rows(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict) and "predictions" in payload:
        rows = payload["predictions"]
    elif isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and "anomaly_map" in payload:
        rows = [payload]
    else:
        raise ValueError("prediction JSON must contain a prediction, a list, or a predictions field")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("predictions must be a list of JSON objects")
    missing = [index for index, row in enumerate(rows) if "image" not in row or "anomaly_map" not in row]
    if missing:
        raise ValueError(f"predictions missing image/anomaly_map at rows: {missing[:5]}")
    if not rows:
        raise ValueError("prediction JSON contains no predictions")
    return rows


def select_prediction(rows: list[dict], image: str | None, selection: str) -> dict:
    if image:
        requested = Path(image)
        matches = [
            row for row in rows
            if Path(row["image"]) == requested
            or Path(row["image"]).name == requested.name
            or Path(row["image"]).stem == requested.stem
        ]
        if not matches:
            raise ValueError(f"prediction image not found: {image}")
        if len(matches) > 1:
            exact = [row for row in matches if Path(row["image"]) == requested]
            if len(exact) == 1:
                return exact[0]
            raise ValueError(f"prediction image filter is ambiguous: {image}")
        return matches[0]
    if selection == "first":
        return rows[0]
    return max(rows, key=lambda row: float(row.get("anomaly_score", float("-inf"))))


def run_inference(args) -> dict:
    if not args.image:
        raise ValueError("--model-state mode requires --image")
    if not args.model:
        raise ValueError("--model-state mode requires --model")
    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"inference image not found: {image_path}")

    from defectfusion.features import DinoFeatureExtractor
    from defectfusion.pipeline import DefectFusion

    state = json.loads(Path(args.model_state).read_text(encoding="utf-8"))
    image_size = args.image_size or state.get("pixel_image_size") or 672
    extractor = DinoFeatureExtractor(
        args.model,
        image_size=image_size,
        resize_mode=args.resize_mode,
        device=args.device,
        debias=args.debias,
        svd_components=args.svd_components,
        feature_layers=args.feature_layers,
        layer_aggregation=args.layer_aggregation,
        layer_normalization=args.layer_normalization,
    )
    return DefectFusion.load(args.model_state, extractor).predict(str(image_path))


def percentile_window(values: np.ndarray, lower: float, upper: float) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 0.0, 1.0
    low, high = np.percentile(finite, [lower, upper]).astype(float)
    if high <= low:
        high = low + max(abs(low) * 1e-6, 1e-6)
    return low, high


def normalize_map(anomaly_map: np.ndarray, low: float, high: float) -> np.ndarray:
    values = np.asarray(anomaly_map, dtype=np.float32)
    values = np.nan_to_num(values, nan=low, posinf=high, neginf=low)
    return np.clip((values - low) / max(high - low, 1e-12), 0.0, 1.0)


def colorize(normalized: np.ndarray, colormap: str) -> np.ndarray:
    stops = COLORMAPS[colormap]
    positions = np.asarray([item[0] for item in stops], dtype=np.float32)
    colors = np.asarray([item[1] for item in stops], dtype=np.float32)
    flat = np.asarray(normalized, dtype=np.float32).reshape(-1)
    channels = [np.interp(flat, positions, colors[:, channel]) for channel in range(3)]
    return np.stack(channels, axis=1).reshape(normalized.shape + (3,)).round().astype(np.uint8)


def resize_score_map(anomaly_map: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    source = Image.fromarray(np.asarray(anomaly_map, dtype=np.float32), mode="F")
    return np.asarray(source.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)


def render_heatmap(
    prediction: dict,
    output: str | Path,
    *,
    lower_percentile: float,
    upper_percentile: float,
    vmin: float | None,
    vmax: float | None,
    colormap: str,
) -> tuple[Path, float, float]:
    image_path = Path(prediction["image"])
    if not image_path.exists():
        raise FileNotFoundError(f"source image not found: {image_path}")
    anomaly_map = np.asarray(prediction["anomaly_map"], dtype=np.float32)
    if anomaly_map.ndim != 2 or not anomaly_map.size:
        raise ValueError("anomaly_map must be a non-empty 2D array")
    with Image.open(image_path) as image:
        size = image.size
    resized = resize_score_map(anomaly_map, size)
    if (vmin is None) != (vmax is None):
        raise ValueError("--vmin and --vmax must be provided together")
    if vmin is not None:
        if vmax <= vmin:
            raise ValueError("--vmax must be greater than --vmin")
        low, high = float(vmin), float(vmax)
    else:
        low, high = percentile_window(resized, lower_percentile, upper_percentile)
    heatmap = Image.fromarray(colorize(normalize_map(resized, low, high), colormap), mode="RGB")
    output_path = Path(output)
    if output_path.suffix.lower() != ".png":
        raise ValueError("--output must be a .png file")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap.save(output_path)
    return output_path, low, high


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render one real DefectFusion heatmap PNG")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", help="category JSON produced by evaluate-mvtec/evaluate-visa")
    source.add_argument("--model-state", help="saved DefectFusion JSON state for direct inference")
    parser.add_argument("--image", help="prediction image selector, or direct-inference image")
    parser.add_argument("--select", choices=["highest", "first"], default="highest",
                        help="prediction selected when --image is omitted")
    parser.add_argument("--output", required=True, help="single output heatmap PNG")
    parser.add_argument("--lower-percentile", type=float, default=1.0)
    parser.add_argument("--upper-percentile", type=float, default=99.0)
    parser.add_argument("--vmin", type=float, help="fixed color minimum for cross-image comparison")
    parser.add_argument("--vmax", type=float, help="fixed color maximum for cross-image comparison")
    parser.add_argument("--colormap", choices=sorted(COLORMAPS), default="turbo")

    parser.add_argument("--model", help="DINOv3 model name/path for direct inference")
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--resize-mode", choices=["direct", "longest_pad"], default="direct")
    parser.add_argument("--feature-layers", type=parse_layers, default=(1, 17, 21, 23))
    parser.add_argument("--layer-aggregation", choices=["mean", "concat"], default="mean")
    parser.add_argument("--layer-normalization", choices=["none", "l2"], default="none")
    parser.add_argument("--debias", action="store_true")
    parser.add_argument("--svd-components", type=int, default=20)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 <= args.lower_percentile < args.upper_percentile <= 100:
        parser.error("percentiles must satisfy 0 <= lower < upper <= 100")
    try:
        if args.predictions:
            prediction = select_prediction(load_prediction_rows(args.predictions), args.image, args.select)
        else:
            prediction = run_inference(args)
        output, low, high = render_heatmap(
            prediction,
            args.output,
            lower_percentile=args.lower_percentile,
            upper_percentile=args.upper_percentile,
            vmin=args.vmin,
            vmax=args.vmax,
            colormap=args.colormap,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(f"image={prediction['image']}")
    print(f"anomaly_score={prediction.get('anomaly_score', 'n/a')}")
    print(f"color_range=[{low:.8g}, {high:.8g}]")
    print(f"heatmap={output}")


if __name__ == "__main__":
    main()
