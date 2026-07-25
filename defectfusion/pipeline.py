from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .model import NormalSubspace, PrototypeBank


class DefectFusion:
    def __init__(self, extractor, *, alpha: float = 0.5, unknown_threshold: float = 0.35):
        self.extractor = extractor
        self.alpha = alpha
        self.subspace = NormalSubspace()
        self.prototype_bank = PrototypeBank()
        self.prototype_bank.unknown_threshold = unknown_threshold
        self.reference_grid = None
        self.reference_shape = None

    def fit_normal(self, image_paths):
        patch_batches = []
        for path in image_paths:
            image = Image.open(path)
            patches, grid = self.extractor.extract(image)
            patch_batches.append(patches)
            self.reference_shape = grid
        if not patch_batches:
            raise ValueError("No normal images were provided")
        features = np.concatenate(patch_batches, axis=0)
        self.subspace.fit(features)
        self.reference_grid = features.shape[1]
        return self

    def add_prototype(self, label: str, image_path):
        image = Image.open(image_path)
        patches, _ = self.extractor.extract(image)
        self.prototype_bank.add(label, patches.mean(axis=0))
        return self

    def predict(self, image_path):
        image = Image.open(image_path)
        patches, grid = self.extractor.extract(image)
        if self.reference_grid is None:
            self.reference_grid = patches.shape[1]
        anomaly_scores = self.subspace.score(patches)
        anomaly_map = anomaly_scores.reshape(grid).tolist()
        fused_score = float(np.mean(anomaly_scores))
        label, label_score = self.prototype_bank.predict(patches.mean(axis=0))
        return {
            "image": str(image_path),
            "grid": list(grid),
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
        obj = cls(extractor, alpha=state.get("alpha", 0.5), unknown_threshold=state.get("unknown_threshold", 0.35))
        obj.subspace = NormalSubspace.from_dict(state["subspace"])
        obj.prototype_bank = PrototypeBank.from_dict(state.get("prototype_bank", {}))
        obj.reference_grid = state.get("reference_grid")
        obj.reference_shape = state.get("reference_shape")
        return obj
