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


def compute_type_metrics(true_types, pred_types):
    """Compute multiclass defect-type metrics without an optional dependency."""
    if len(true_types) != len(pred_types):
        raise ValueError(f"Defect-type metric shape mismatch: {len(true_types)} vs {len(pred_types)}")
    labels = sorted(set(true_types) | set(pred_types))
    if not labels:
        return {}

    label_indices = {label: index for index, label in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for truth, prediction in zip(true_types, pred_types):
        confusion[label_indices[truth], label_indices[prediction]] += 1

    true_positive = np.diag(confusion).astype(np.float64)
    predicted_count = confusion.sum(axis=0).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros_like(true_positive),
        where=predicted_count > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support > 0,
    )
    f1_denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        f1_denominator,
        out=np.zeros_like(precision),
        where=f1_denominator > 0,
    )
    total = float(support.sum())
    return {
        "defect_type_accuracy": float(true_positive.sum() / total) if total else float("nan"),
        "defect_type_macro_precision": float(precision.mean()),
        "defect_type_macro_recall": float(recall.mean()),
        "defect_type_macro_f1": float(f1.mean()),
        "defect_type_weighted_f1": float(np.sum(f1 * support) / total) if total else float("nan"),
        "defect_type_labels": labels,
        "defect_type_confusion_matrix": confusion.tolist(),
    }


def _images(path):
    return sorted(p for p in Path(path).glob("*.*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def _normal_reference_threshold(fusion, normal_reference_images, source, quantile=1.0):
    """Return a decision threshold from normal reference-image scores.

    Image-level AUROC and related ranking metrics do not need a threshold, but
    a per-image ``good``/``anomaly`` result does.  Calibrating from the normal
    images used to fit the detector avoids using test labels for that decision.
    """

    if not 0 < quantile <= 1:
        raise ValueError("decision threshold quantile must be in (0, 1]")
    references = list(normal_reference_images or [])
    if not references:
        return None, "unavailable", 0, 0.0

    started = time.perf_counter()
    scores = []
    for image in references:
        result = fusion.predict(image)
        score = float(result["anomaly_score"])
        if not np.isfinite(score):
            raise ValueError(f"Normal reference produced a non-finite anomaly score: {image}")
        scores.append(score)
    return float(np.quantile(scores, quantile)), source, len(scores), time.perf_counter() - started


def evaluate_samples(
    fusion,
    category,
    samples,
    output,
    *,
    progress=True,
    excluded_images=None,
    excluded_type_images=None,
    normal_reference_images=None,
    decision_threshold_source="normal_reference_max",
    decision_threshold_quantile=1.0,
):
    """Evaluate image/mask records shared by MVTec AD and VisA."""
    excluded_images = {str(Path(p).resolve()) for p in (excluded_images or [])}
    excluded_type_images = {str(Path(p).resolve()) for p in (excluded_type_images or [])}
    rows, image_y, image_s, pixel_masks, pixel_maps = [], [], [], [], []
    started = time.perf_counter()
    decision_threshold, resolved_threshold_source, reference_count, threshold_seconds = _normal_reference_threshold(
        fusion,
        normal_reference_images,
        decision_threshold_source,
        decision_threshold_quantile,
    )
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
        if decision_threshold is None:
            result.update({
                "predicted_anomaly": None,
                "predicted_label": "unavailable",
                "prediction_correct": None,
                "decision_threshold": None,
                "decision_threshold_source": resolved_threshold_source,
            })
        else:
            predicted_anomaly = float(result["anomaly_score"]) > decision_threshold
            result.update({
                "predicted_anomaly": predicted_anomaly,
                "predicted_label": "anomaly" if predicted_anomaly else "good",
                "prediction_correct": predicted_anomaly == bool(truth),
                "decision_threshold": decision_threshold,
                "decision_threshold_source": resolved_threshold_source,
            })
        # Good images only receive an image-level normal/anomaly decision.
        # Pixel metrics are based solely on defective samples with masks.
        if truth and mask_path is not None and mask_path.exists():
            pixel_started = time.perf_counter()
            from PIL import Image as PILImage
            mask = np.asarray(PILImage.open(mask_path).convert("L")) > 0
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
    good_rows = [row for row in rows if not row["ground_truth_anomaly"]]
    good_decision_rows = [row for row in good_rows if row["predicted_anomaly"] is not None]
    good_predicted_normal = sum(row["predicted_label"] == "good" for row in good_decision_rows)
    defect_rows = [row for row in rows if row["ground_truth_anomaly"]]
    metrics = {
        "category": category,
        "images": len(rows),
        "good_images": len(good_rows),
        "good_decision_images": len(good_decision_rows),
        "good_predicted_normal": good_predicted_normal,
        "good_predicted_anomaly": sum(row["predicted_label"] == "anomaly" for row in good_decision_rows),
        "defect_images": len(defect_rows),
        "pixel_metric_images": len(pixel_masks),
        "good_decision_threshold": decision_threshold,
        "good_decision_threshold_source": resolved_threshold_source,
        "good_decision_quantile": decision_threshold_quantile,
        "good_decision_reference_images": reference_count,
        "results": str(output),
    }
    if good_decision_rows:
        metrics["good_accuracy"] = good_predicted_normal / len(good_decision_rows)
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
    defect_type_rows = [
        r for r in rows
        if r["ground_truth_anomaly"]
        and str(Path(r["image"]).resolve()) not in excluded_type_images
    ]
    true_types = [r["ground_truth_type"] for r in defect_type_rows]
    pred_types = [r["defect_type"] for r in defect_type_rows]
    type_metrics_enabled = bool(excluded_type_images) or any(p != "unknown" for p in pred_types)
    if true_types and type_metrics_enabled:
        metrics.update(compute_type_metrics(true_types, pred_types))
    else:
        metrics["defect_type_note"] = "No prototypes supplied; type metrics are unavailable"
    metrics_seconds = time.perf_counter() - metrics_started
    metrics["timing_seconds"] = {
        "threshold_calibration": threshold_seconds,
        "prediction": prediction_seconds,
        "pixel_preparation": pixel_preparation_seconds,
        "json_output": output_seconds,
        "metrics": metrics_seconds,
        "total": time.perf_counter() - started,
    }
    return metrics


def evaluate_mvtec(
    fusion,
    category_dir,
    output,
    *,
    progress=True,
    excluded_images=None,
    excluded_type_images=None,
    normal_reference_images=None,
    decision_threshold_source="normal_reference_max",
    decision_threshold_quantile=1.0,
):
    """Evaluate a fitted model on one MVTec category and write JSON predictions."""
    root = Path(category_dir)
    samples = []
    for defect_dir in sorted(p for p in (root / "test").iterdir() if p.is_dir()):
        defect = defect_dir.name
        for image in _images(defect_dir):
            truth = defect != "good"
            mask = root / "ground_truth" / defect / f"{image.stem}_mask.png"
            samples.append((image, defect, truth, mask if truth else None))
    return evaluate_samples(
        fusion,
        root.name,
        samples,
        output,
        progress=progress,
        excluded_images=excluded_images,
        excluded_type_images=excluded_type_images,
        normal_reference_images=normal_reference_images,
        decision_threshold_source=decision_threshold_source,
        decision_threshold_quantile=decision_threshold_quantile,
    )
