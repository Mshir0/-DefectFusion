from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def _images(path):
    return sorted(p for p in Path(path).glob("*.*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def evaluate_mvtec(fusion, category_dir, output, *, progress=True):
    """Evaluate a fitted model on one MVTec category and write JSONL results."""
    root = Path(category_dir)
    rows, image_y, image_s, pixel_y, pixel_s = [], [], [], [], []
    test = root / "test"
    for defect_dir in sorted(p for p in test.iterdir() if p.is_dir()):
        defect = defect_dir.name
        for image in _images(defect_dir):
            result = fusion.predict(str(image))
            truth = defect != "good"
            result.update({"category": root.name, "ground_truth_type": defect, "ground_truth_anomaly": truth})
            mask_path = root / "ground_truth" / defect / f"{image.stem}_mask.png"
            if mask_path.exists():
                from PIL import Image
                mask = np.asarray(Image.open(mask_path).convert("L")) > 0
                score = np.asarray(result["anomaly_map"], dtype=float)
                # Resize the coarse patch map to the mask resolution for pixel metrics.
                from PIL import Image as PILImage
                score = np.asarray(PILImage.fromarray(score.astype("float32"), mode="F").resize(mask.shape[::-1], PILImage.Resampling.BILINEAR))
                pixel_y.extend(mask.ravel().astype(int)); pixel_s.extend(score.ravel())
            image_y.append(int(truth)); image_s.append(float(result["anomaly_score"])); rows.append(result)
            if progress:
                print(f"[{len(rows):04d}] {image.name} anomaly={result['anomaly_score']:.5f} type={result['defect_type']}", flush=True)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    metrics = {"category": root.name, "images": len(rows), "results": str(output)}
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(image_y)) > 1: metrics["image_auroc"] = float(roc_auc_score(image_y, image_s))
        if pixel_y and len(set(pixel_y)) > 1: metrics["pixel_auroc"] = float(roc_auc_score(pixel_y, pixel_s))
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
        true_types = [r["ground_truth_type"] for r in rows]
        pred_types = [r["defect_type"] for r in rows]
        if any(p != "unknown" for p in pred_types):
            labels = sorted(set(true_types) | set(pred_types))
            metrics["defect_type_accuracy"] = float(accuracy_score(true_types, pred_types))
            metrics["defect_type_macro_f1"] = float(f1_score(true_types, pred_types, labels=labels, average="macro", zero_division=0))
            metrics["defect_type_labels"] = labels
            metrics["defect_type_confusion_matrix"] = confusion_matrix(true_types, pred_types, labels=labels).tolist()
        else:
            metrics["defect_type_note"] = "No prototypes supplied; type metrics are unavailable"
    except ImportError:
        metrics["metrics_note"] = "Install scikit-learn to compute AUROC"
    return metrics
