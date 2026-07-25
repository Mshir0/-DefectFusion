from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .model import NormalSubspace, PrototypeBank


class DefectFusion:
    def __init__(self, extractor, *, alpha: float = 0.5, unknown_threshold: float = 0.35, top_k_ratio: float = 0.05, image_score: str = "mtop1p", type_matching: str = "bidirectional_patch"):
        self.extractor = extractor
        self.alpha = alpha
        self.subspace = NormalSubspace()
        self.prototype_bank = PrototypeBank()
        self.prototype_bank.unknown_threshold = unknown_threshold
        if not 0 < top_k_ratio <= 1:
            raise ValueError("top_k_ratio must be in (0, 1]")
        self.top_k_ratio = top_k_ratio
        if image_score not in {"mtop1p", "mean", "max", "p99"}:
            raise ValueError("image_score must be one of: mtop1p, mean, max, p99")
        self.image_score = image_score
        if type_matching not in {"prototype_mean", "bidirectional_patch"}:
            raise ValueError("type_matching must be prototype_mean or bidirectional_patch")
        self.type_matching = type_matching
        self.reference_grid = None
        self.reference_shape = None

    def fit_normal(self, image_paths):
        patch_batches = []
        for path in image_paths:
            image = Image.open(path)
            views = self._extract_views(image)
            patch_batches.extend(patches for patches, _, _ in views)
            self.reference_shape = views[0][1]
        if not patch_batches:
            raise ValueError("No normal images were provided")
        features = np.concatenate(patch_batches, axis=0)
        self.subspace.fit(features)
        self.reference_grid = features.shape[1]
        return self

    def add_prototype(self, label: str, image_path):
        image = Image.open(image_path)
        patches = np.concatenate([item[0] for item in self._extract_views(image)], axis=0)
        self.prototype_bank.add(label, self._anomaly_patches(patches))
        return self

    def _extract_views(self, image):
        if hasattr(self.extractor, "extract_views"):
            return self.extractor.extract_views(image)
        patches, grid = self.extractor.extract(image)
        return [(patches, grid, (0, 0, image.width, image.height))]

    def _anomaly_patches(self, patches):
        scores = self.subspace.score(patches)
        keep = max(1, int(np.ceil(len(scores) * self.top_k_ratio)))
        indices = np.argpartition(scores, -keep)[-keep:]
        return patches[indices]

    def _aggregate_image_score(self, scores):
        scores = np.asarray(scores, dtype=np.float64)
        if self.image_score == "mean":
            return float(scores.mean())
        if self.image_score == "max":
            return float(scores.max())
        if self.image_score == "p99":
            return float(np.percentile(scores, 99))
        keep = max(1, int(np.ceil(scores.size * 0.01)))
        return float(np.partition(scores, -keep)[-keep:].mean())

    def predict(self, image_path):
        image = Image.open(image_path)
        views = self._extract_views(image)
        patches = np.concatenate([item[0] for item in views], axis=0)
        grid = views[0][1]
        if self.reference_grid is None:
            self.reference_grid = patches.shape[1]
        canvas = np.zeros((image.height, image.width), dtype=np.float64)
        counts = np.zeros_like(canvas)
        for view_patches, view_grid, (x1, y1, x2, y2) in views:
            scores = self.subspace.score(view_patches).reshape(view_grid).astype("float32")
            resized = np.asarray(Image.fromarray(scores, mode="F").resize((x2 - x1, y2 - y1), Image.Resampling.BILINEAR))
            canvas[y1:y2, x1:x2] += resized
            counts[y1:y2, x1:x2] += 1
        fused_map = np.divide(canvas, counts, out=np.zeros_like(canvas), where=counts > 0)
        coarse_map = np.asarray(Image.fromarray(fused_map.astype("float32"), mode="F").resize(grid[::-1], Image.Resampling.BILINEAR))
        anomaly_map = coarse_map.tolist()
        fused_score = self._aggregate_image_score(coarse_map.ravel())
        typing_patches = self._anomaly_patches(patches)
        typing_features = typing_patches if self.type_matching == "bidirectional_patch" else typing_patches.mean(axis=0)
        label, label_score = self.prototype_bank.predict(typing_features)
        return {
            "image": str(image_path),
            "grid": list(grid),
            "views": len(views),
            "anomaly_score": fused_score,
            "anomaly_map": anomaly_map,
            "defect_type": label,
            "defect_type_score": float(label_score),
            "fused_score": fused_score * self.alpha + float(label_score) * (1.0 - self.alpha),
        }

    def save(self, path):
        state = {
            "alpha": self.alpha,
            "subspace": self.subspace.to_dict(),
            "prototype_bank": self.prototype_bank.to_dict(),
            "unknown_threshold": self.prototype_bank.unknown_threshold,
            "top_k_ratio": self.top_k_ratio,
            "image_score": self.image_score,
            "type_matching": self.type_matching,
            "reference_grid": self.reference_grid,
            "reference_shape": self.reference_shape,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path, extractor):
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(extractor, alpha=state.get("alpha", 0.5), unknown_threshold=state.get("unknown_threshold", 0.35), top_k_ratio=state.get("top_k_ratio", 0.05), image_score=state.get("image_score", "mean"), type_matching=state.get("type_matching", "prototype_mean"))
        obj.subspace = NormalSubspace.from_dict(state["subspace"])
        obj.prototype_bank = PrototypeBank.from_dict(state.get("prototype_bank", {}))
        obj.prototype_bank.unknown_threshold = state.get("unknown_threshold", 0.35)
        obj.reference_grid = state.get("reference_grid")
        obj.reference_shape = state.get("reference_shape")
        return obj
