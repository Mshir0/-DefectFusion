from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np


def _region_masks(mask, connectivity):
    try:
        from scipy.ndimage import generate_binary_structure, label as connected_components
        structure = generate_binary_structure(2, 2 if connectivity == 8 else 1)
        labeled, count = connected_components(mask, structure=structure)
        return [labeled == region_id for region_id in range(1, count + 1)]
    except ImportError:
        mask = np.asarray(mask, dtype=bool)
        visited = np.zeros_like(mask, dtype=bool)
        regions = []
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if connectivity == 8:
            offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        height, width = mask.shape
        for row, column in np.argwhere(mask):
            if visited[row, column]:
                continue
            region = np.zeros_like(mask, dtype=bool)
            stack = [(int(row), int(column))]
            visited[row, column] = True
            while stack:
                current_row, current_column = stack.pop()
                region[current_row, current_column] = True
                for row_offset, column_offset in offsets:
                    next_row = current_row + row_offset
                    next_column = current_column + column_offset
                    if 0 <= next_row < height and 0 <= next_column < width and mask[next_row, next_column] and not visited[next_row, next_column]:
                        visited[next_row, next_column] = True
                        stack.append((next_row, next_column))
            regions.append(region)
        return regions


def compute_aupro(anomaly_maps, gt_masks, fpr_limit=0.3, num_thresholds=300, connectivity=8):
    """Compute MVTec AUPRO, normalized over the false-positive range [0, fpr_limit]."""
    predictions = np.stack([np.asarray(item, dtype=np.float32) for item in anomaly_maps])
    masks = np.stack([np.asarray(item, dtype=np.uint8) for item in gt_masks])
    if predictions.shape != masks.shape:
        raise ValueError(f"AUPRO shape mismatch: {predictions.shape} vs {masks.shape}")
    if not np.isfinite(predictions).all():
        return float("nan")

    regions = []
    for index, mask in enumerate(masks):
        for region_mask in _region_masks(mask, connectivity):
            regions.append(np.sort(predictions[index][region_mask]))
    negative_scores = np.sort(predictions[masks == 0])
    if not regions or negative_scores.size == 0:
        return float("nan")

    target_fprs = np.linspace(0.0, fpr_limit, num_thresholds + 1)[1:]
    quantile_indices = np.clip(
        np.floor((1.0 - target_fprs) * (len(negative_scores) - 1)).astype(np.int64),
        0,
        len(negative_scores) - 1,
    )
    thresholds = negative_scores[quantile_indices]
    false_positives = len(negative_scores) - np.searchsorted(negative_scores, thresholds, side="left")
    fprs = false_positives.astype(np.float64) / len(negative_scores)

    pros = np.zeros(num_thresholds, dtype=np.float64)
    for region_scores in regions:
        detected = len(region_scores) - np.searchsorted(region_scores, thresholds, side="left")
        pros += detected / len(region_scores)
    pros /= len(regions)

    order = np.argsort(fprs, kind="stable")
    fprs = np.concatenate([[0.0], fprs[order]])
    pros = np.concatenate([[0.0], pros[order]])
    if fprs[-1] > fpr_limit:
        cut = np.searchsorted(fprs, fpr_limit, side="right")
        f0, f1 = fprs[cut - 1], fprs[cut]
        p0, p1 = pros[cut - 1], pros[cut]
        boundary = p0 + (p1 - p0) * (fpr_limit - f0) / (f1 - f0) if f1 > f0 else p0
        fprs = np.concatenate([fprs[:cut], [fpr_limit]])
        pros = np.concatenate([pros[:cut], [boundary]])
    elif fprs[-1] < fpr_limit:
        fprs = np.concatenate([fprs, [fpr_limit]])
        pros = np.concatenate([pros, [pros[-1]]])
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(pros, fprs) / fpr_limit)


def compute_binary_metrics(labels, scores):
    """Compute exact AUROC, average precision, and maximum F1 from one sort."""
    labels = np.asarray(labels, dtype=np.uint8).ravel()
    scores = np.asarray(scores).ravel()
    if labels.shape != scores.shape:
        raise ValueError(f"Binary metric shape mismatch: {labels.shape} vs {scores.shape}")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan"), float("nan"), float("nan")
    if not np.isfinite(scores).all():
        return float("nan"), float("nan"), float("nan")

    order = np.argsort(scores, kind="stable")[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    threshold_ends = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    cumulative_true = np.cumsum(sorted_labels, dtype=np.int64)[threshold_ends]
    cumulative_count = np.arange(1, len(labels) + 1, dtype=np.int64)[threshold_ends]
    cumulative_false = cumulative_count - cumulative_true

    true_positive_rate = np.r_[0.0, cumulative_true / positives]
    false_positive_rate = np.r_[0.0, cumulative_false / negatives]
    integrate = getattr(np, "trapezoid", np.trapz)
    auroc = float(integrate(true_positive_rate, false_positive_rate))

    precision = cumulative_true / cumulative_count
    recall = cumulative_true / positives
    aupr = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    false_negatives = positives - cumulative_true
    denominator = 2 * cumulative_true + cumulative_false + false_negatives
    f1_max = float(np.max(np.divide(
        2 * cumulative_true,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator > 0,
    )))
    return auroc, aupr, f1_max


def compute_binary_auroc_aupr(labels, scores):
    """Backward-compatible AUROC/AP pair computed by ``compute_binary_metrics``."""
    auroc, aupr, _ = compute_binary_metrics(labels, scores)
    return auroc, aupr


def _images(path):
    return sorted(p for p in Path(path).glob("*.*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def evaluate_samples(fusion, category, samples, output, *, progress=True, excluded_images=None):
    """Evaluate image/mask records shared by MVTec AD and VisA."""
    excluded_images = {str(Path(p).resolve()) for p in (excluded_images or [])}
    rows, image_y, image_s, pixel_masks, pixel_maps = [], [], [], [], []
    started = time.perf_counter()
    prediction_seconds = 0.0
    pixel_preparation_seconds = 0.0
    for image, defect, truth, mask_path in samples:
        image = Path(image)
        if str(image.resolve()) in excluded_images:
            continue
        prediction_started = time.perf_counter()
        result = fusion.predict(str(image))
        prediction_seconds += time.perf_counter() - prediction_started
        result.update({"category": category, "ground_truth_type": defect, "ground_truth_anomaly": bool(truth)})
        mask_path = None if mask_path is None else Path(mask_path)
        if (mask_path is not None and mask_path.exists()) or not truth:
            pixel_started = time.perf_counter()
            from PIL import Image as PILImage
            if mask_path is not None and mask_path.exists():
                mask = np.asarray(PILImage.open(mask_path).convert("L")) > 0
            else:
                with PILImage.open(image) as source:
                    mask = np.zeros((source.height, source.width), dtype=bool)
            score = np.asarray(result["anomaly_map"], dtype=np.float32)
            # Resize the coarse patch map to the mask resolution for pixel metrics.
            score = np.asarray(PILImage.fromarray(score.astype("float32"), mode="F").resize(mask.shape[::-1], PILImage.Resampling.BILINEAR))
            pixel_masks.append(mask); pixel_maps.append(score)
            pixel_preparation_seconds += time.perf_counter() - pixel_started
        image_y.append(int(truth)); image_s.append(float(result["anomaly_score"])); rows.append(result)
        if progress:
            print(f"[{len(rows):04d}] {image.name} anomaly={result['anomaly_score']:.5f} type={result['defect_type']}", flush=True)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output_started = time.perf_counter()
    output.write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    output_seconds = time.perf_counter() - output_started
    metrics = {"category": category, "images": len(rows), "results": str(output)}
    metrics_started = time.perf_counter()
    if len(set(image_y)) > 1:
        metrics["image_auroc"], metrics["image_aupr"], metrics["image_f1_max"] = compute_binary_metrics(image_y, image_s)
    if pixel_masks:
        pixel_y = np.concatenate([item.ravel() for item in pixel_masks]).astype(np.uint8)
        pixel_s = np.concatenate([item.ravel() for item in pixel_maps]).astype(np.float32, copy=False)
        if pixel_y.min() == 0 and pixel_y.max() == 1:
            metrics["pixel_auroc"], metrics["pixel_aupr"], metrics["pixel_f1_max"] = compute_binary_metrics(pixel_y, pixel_s)
            del pixel_y, pixel_s
            metrics["pixel_aupro"] = compute_aupro(pixel_maps, pixel_masks)
    try:
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
        defect_rows = [r for r in rows if r["ground_truth_anomaly"]]
        true_types = [r["ground_truth_type"] for r in defect_rows]
        pred_types = [r["defect_type"] for r in defect_rows]
        if any(p != "unknown" for p in pred_types):
            labels = sorted(set(true_types) | set(pred_types))
            metrics["defect_type_accuracy"] = float(accuracy_score(true_types, pred_types))
            metrics["defect_type_macro_f1"] = float(f1_score(true_types, pred_types, labels=labels, average="macro", zero_division=0))
            metrics["defect_type_labels"] = labels
            metrics["defect_type_confusion_matrix"] = confusion_matrix(true_types, pred_types, labels=labels).tolist()
        else:
            metrics["defect_type_note"] = "No prototypes supplied; type metrics are unavailable"
    except ImportError:
        metrics["type_metrics_note"] = "Install scikit-learn to compute defect-type metrics"
    metrics_seconds = time.perf_counter() - metrics_started
    metrics["timing_seconds"] = {
        "prediction": prediction_seconds,
        "pixel_preparation": pixel_preparation_seconds,
        "json_output": output_seconds,
        "metrics": metrics_seconds,
        "total": time.perf_counter() - started,
    }
    return metrics


def evaluate_mvtec(fusion, category_dir, output, *, progress=True, excluded_images=None):
    """Evaluate a fitted model on one MVTec category and write JSON predictions."""
    root = Path(category_dir)
    samples = []
    for defect_dir in sorted(p for p in (root / "test").iterdir() if p.is_dir()):
        defect = defect_dir.name
        for image in _images(defect_dir):
            truth = defect != "good"
            mask = root / "ground_truth" / defect / f"{image.stem}_mask.png"
            samples.append((image, defect, truth, mask if truth else None))
    return evaluate_samples(fusion, root.name, samples, output, progress=progress, excluded_images=excluded_images)
