from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .model import NormalSubspace, PrototypeBank


class DefectFusion:
    def __init__(self, extractor, *, alpha: float = 0.5, unknown_threshold: float = 0.35, top_k_ratio: float = 0.05, image_score: str = "mtop1p", type_matching: str = "bidirectional_patch", map_postprocess: str = "none", gaussian_sigma: float = 1.0):
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
        if map_postprocess not in {"none", "gaussian", "crf"}:
            raise ValueError("map_postprocess must be none, gaussian, or crf")
        self.map_postprocess = map_postprocess
        self.gaussian_sigma = gaussian_sigma
        self.reference_grid = None
        self.reference_shape = None

    def fit_normal(self, image_paths):
        patch_batches = []
        for path in image_paths:
            image = path.copy() if isinstance(path, Image.Image) else Image.open(path)
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
        self.prototype_bank.add(label, self._anomaly_patches(patches))
        return self

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

    def _postprocess_map(self, anomaly_map, image):
        if self.map_postprocess == "none":
            return anomaly_map
        if self.map_postprocess == "gaussian":
            import torch
            import torch.nn.functional as F
            sigma = max(float(self.gaussian_sigma), 1e-6)
            radius = max(1, int(np.ceil(3 * sigma)))
            axis = torch.arange(-radius, radius + 1, dtype=torch.float32)
            kernel = torch.exp(-(axis ** 2) / (2 * sigma ** 2)); kernel /= kernel.sum()
            x = torch.as_tensor(anomaly_map, dtype=torch.float32)[None, None]
            x = F.pad(x, (radius, radius, radius, radius), mode="reflect")
            x = F.conv2d(x, kernel[None, None, None, :])
            x = F.conv2d(x, kernel[None, None, :, None])
            return x[0, 0].numpy()
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax
        except ImportError as exc:
            raise RuntimeError("CRF requires: pip install 'defectfusion[crf]'") from exc
        full = np.asarray(Image.fromarray(anomaly_map.astype("float32"), mode="F").resize(image.size, Image.Resampling.BILINEAR))
        prob = (full - full.min()) / max(float(full.max() - full.min()), 1e-12)
        unary = unary_from_softmax(np.stack([1 - prob, prob]).astype("float32"))
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        crf = dcrf.DenseCRF2D(image.width, image.height, 2); crf.setUnaryEnergy(unary)
        crf.addPairwiseGaussian(sxy=3, compat=3); crf.addPairwiseBilateral(sxy=50, srgb=10, rgbim=rgb, compat=5)
        refined = np.asarray(crf.inference(5), dtype=np.float32)[1].reshape(image.height, image.width)
        return np.asarray(Image.fromarray(refined, mode="F").resize(anomaly_map.shape[::-1], Image.Resampling.BILINEAR))

    def predict(self, image_path):
        image = Image.open(image_path)
        patches, grid = self.extractor.extract(image)
        if self.reference_grid is None:
            self.reference_grid = patches.shape[1]
        anomaly_scores = self.subspace.score(patches)
        anomaly_map = self._postprocess_map(anomaly_scores.reshape(grid), image).tolist()
        fused_score = self._aggregate_image_score(anomaly_scores)
        typing_patches = self._anomaly_patches(patches)
        typing_features = typing_patches if self.type_matching == "bidirectional_patch" else typing_patches.mean(axis=0)
        label, label_score = self.prototype_bank.predict(typing_features)
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
            "top_k_ratio": self.top_k_ratio,
            "image_score": self.image_score,
            "type_matching": self.type_matching,
            "map_postprocess": self.map_postprocess,
            "gaussian_sigma": self.gaussian_sigma,
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
        obj = cls(extractor, alpha=state.get("alpha", 0.5), unknown_threshold=state.get("unknown_threshold", 0.35), top_k_ratio=state.get("top_k_ratio", 0.05), image_score=state.get("image_score", "mean"), type_matching=state.get("type_matching", "prototype_mean"), map_postprocess=state.get("map_postprocess", "none"), gaussian_sigma=state.get("gaussian_sigma", 1.0))
        obj.subspace = NormalSubspace.from_dict(state["subspace"])
        obj.prototype_bank = PrototypeBank.from_dict(state.get("prototype_bank", {}))
        obj.prototype_bank.unknown_threshold = state.get("unknown_threshold", 0.35)
        obj.reference_grid = state.get("reference_grid")
        obj.reference_shape = state.get("reference_shape")
        return obj
