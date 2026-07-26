from __future__ import annotations

import json
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


def _images(path):
    return sorted(p for p in Path(path).glob("*.*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def evaluate_mvtec(fusion, category_dir, output, *, progress=True):
    """Evaluate a fitted model on one MVTec category and write JSONL results."""
    root = Path(category_dir)
    rows, image_y, image_s, pixel_masks, pixel_maps = [], [], [], [], []
    test = root / "test"
    for defect_dir in sorted(p for p in test.iterdir() if p.is_dir()):
        defect = defect_dir.name
        for image in _images(defect_dir):
            result = fusion.predict(str(image))
            truth = defect != "good"
            result.update({"category": root.name, "ground_truth_type": defect, "ground_truth_anomaly": truth})
            mask_path = root / "ground_truth" / defect / f"{image.stem}_mask.png"
            if mask_path.exists() or not truth:
                from PIL import Image as PILImage
                if mask_path.exists():
                    mask = np.asarray(PILImage.open(mask_path).convert("L")) > 0
                else:
                    with PILImage.open(image) as source:
                        mask = np.zeros((source.height, source.width), dtype=bool)
                score = np.asarray(result["anomaly_map"], dtype=float)
                # Resize the coarse patch map to the mask resolution for pixel metrics.
                score = np.asarray(PILImage.fromarray(score.astype("float32"), mode="F").resize(mask.shape[::-1], PILImage.Resampling.BILINEAR))
                pixel_masks.append(mask); pixel_maps.append(score)
            image_y.append(int(truth)); image_s.append(float(result["anomaly_score"])); rows.append(result)
            if progress:
                print(f"[{len(rows):04d}] {image.name} anomaly={result['anomaly_score']:.5f}", flush=True)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    metrics = {"category": root.name, "images": len(rows), "results": str(output)}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        if len(set(image_y)) > 1:
            metrics["image_auroc"] = float(roc_auc_score(image_y, image_s))
            metrics["image_aupr"] = float(average_precision_score(image_y, image_s))
        if pixel_masks:
            pixel_y = np.concatenate([item.ravel() for item in pixel_masks]).astype(np.uint8)
            pixel_s = np.concatenate([item.ravel() for item in pixel_maps])
            if len(np.unique(pixel_y)) > 1:
                metrics["pixel_auroc"] = float(roc_auc_score(pixel_y, pixel_s))
                metrics["pixel_aupr"] = float(average_precision_score(pixel_y, pixel_s))
                metrics["pixel_aupro"] = compute_aupro(pixel_maps, pixel_masks)
    except ImportError:
        metrics["metrics_note"] = "Install scikit-learn to compute evaluation metrics"
    return metrics
