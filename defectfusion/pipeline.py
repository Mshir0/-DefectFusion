from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .model import NormalPatchMemory, NormalSubspace, PrototypeBank


class DefectFusion:
    def __init__(self, extractor, *, alpha: float = 0.5, unknown_threshold: float = 0.35, top_k_ratio: float = 0.05, image_score: str = "mtop1p", image_top_ratio: float = 0.01, image_fusion_stage: str = "patch", image_spatial_weight: float = 0.0, type_matching: str = "bidirectional_patch", map_postprocess: str = "none", gaussian_sigma: float = 1.0, anomaly_method: str = "pca", pca_residual_metric: str = "squared_l2", knn_weight: float = 0.5, memory_max_patches: int = 50000, knn_chunk_size: int = 256, knn_backend: str = "auto", knn_dtype: str = "float32", knn_spatial_radius: float = -1.0, dual_branch: bool = False, fusion_mode: str = "fixed", gate_temperature: float = 1.0):
        self.extractor = extractor
        knn_device = getattr(extractor, "device", None)
        self.dual_branch = bool(dual_branch)
        self.image_subspace = NormalSubspace(residual_metric=pca_residual_metric) if self.dual_branch else None
        self.image_memory = NormalPatchMemory(memory_max_patches, knn_chunk_size, backend=knn_backend, device=knn_device if self.dual_branch else None, dtype=knn_dtype, spatial_radius=knn_spatial_radius) if self.dual_branch else None
        self.alpha = alpha
        self.subspace = NormalSubspace(residual_metric=pca_residual_metric)
        self.normal_memory = NormalPatchMemory(memory_max_patches, knn_chunk_size, backend=knn_backend, device=knn_device, dtype=knn_dtype, spatial_radius=knn_spatial_radius)
        self.prototype_bank = PrototypeBank()
        self.prototype_bank.unknown_threshold = unknown_threshold
        if not 0 < top_k_ratio <= 1:
            raise ValueError("top_k_ratio must be in (0, 1]")
        self.top_k_ratio = top_k_ratio
        if image_score not in {"mtop1p", "mean", "max", "p99"}:
            raise ValueError("image_score must be one of: mtop1p, mean, max, p99")
        self.image_score = image_score
        if not 0 < image_top_ratio <= 1:
            raise ValueError("image_top_ratio must be in (0, 1]")
        self.image_top_ratio = float(image_top_ratio)
        if image_fusion_stage not in {"patch", "score"}:
            raise ValueError("image_fusion_stage must be patch or score")
        self.image_fusion_stage = image_fusion_stage
        if image_spatial_weight < 0:
            raise ValueError("image_spatial_weight must be non-negative")
        self.image_spatial_weight = float(image_spatial_weight)
        if type_matching not in {"prototype_mean", "bidirectional_patch", "rbf_svm"}:
            raise ValueError("type_matching must be prototype_mean, bidirectional_patch, or rbf_svm")
        self.type_matching = type_matching
        if map_postprocess not in {"none", "gaussian", "crf"}:
            raise ValueError("map_postprocess must be none, gaussian, or crf")
        self.map_postprocess = map_postprocess
        self.gaussian_sigma = gaussian_sigma
        if anomaly_method not in {"pca", "knn", "pca_knn"}:
            raise ValueError("anomaly_method must be pca, knn, or pca_knn")
        if not 0 <= knn_weight <= 1:
            raise ValueError("knn_weight must be in [0, 1]")
        self.anomaly_method = anomaly_method
        self.pca_residual_metric = pca_residual_metric
        self.knn_weight = float(knn_weight)
        if fusion_mode not in {"fixed", "gated"}:
            raise ValueError("fusion_mode must be fixed or gated")
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive")
        self.fusion_mode = fusion_mode
        self.gate_temperature = float(gate_temperature)
        self.reference_grid = None
        self.reference_shape = None

    def fit_normal(self, image_paths):
        patch_batches, image_patch_batches, position_batches = [], [], []
        for path in image_paths:
            image = path.copy() if isinstance(path, Image.Image) else Image.open(path)
            if self.dual_branch:
                patches, image_patches, grid = self.extractor.extract_dual(image)
                image_patch_batches.append(image_patches)
            else:
                patches, grid = self.extractor.extract(image)
            patch_batches.append(patches)
            position_batches.append(self._patch_positions(grid))
            self.reference_shape = grid
        if not patch_batches:
            raise ValueError("No normal images were provided")
        features = np.concatenate(patch_batches, axis=0)
        self.subspace.fit(features)
        if self.anomaly_method in {"knn", "pca_knn"}:
            self.normal_memory.fit(features, np.concatenate(position_batches, axis=0))
        if self.dual_branch:
            image_features = np.concatenate(image_patch_batches, axis=0)
            self.image_subspace.fit(image_features)
            if self.anomaly_method in {"knn", "pca_knn"}:
                self.image_memory.fit(image_features, np.concatenate(position_batches, axis=0))
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

    @staticmethod
    def _patch_positions(grid):
        rows, columns = np.indices(grid, dtype=np.float32)
        return np.stack([(rows.ravel() + 0.5) / grid[0], (columns.ravel() + 0.5) / grid[1]], axis=1)

    def _aggregate_image_score(self, scores):
        scores = np.asarray(scores, dtype=np.float64)
        if self.image_score == "mean":
            return float(scores.mean())
        if self.image_score == "max":
            return float(scores.max())
        if self.image_score == "p99":
            return float(np.percentile(scores, 99))
        keep = max(1, int(np.ceil(scores.size * self.image_top_ratio)))
        return float(np.partition(scores, -keep)[-keep:].mean())

    def _anomaly_scores(self, patches, positions=None):
        pca_scores = self.subspace.score(patches)
        if self.anomaly_method == "pca":
            return pca_scores, pca_scores, None, None
        knn_scores = self.normal_memory.score(patches, positions=positions)
        if self.anomaly_method == "knn":
            return knn_scores, pca_scores, knn_scores, None
        if self.fusion_mode == "gated":
            pca_evidence = self.subspace.tail_evidence(pca_scores)
            knn_evidence = self.normal_memory.tail_evidence(knn_scores)
            logits = np.clip((knn_evidence - pca_evidence) / self.gate_temperature, -60.0, 60.0)
            knn_gate = 1.0 / (1.0 + np.exp(-logits))
            fused = (1.0 - knn_gate) * pca_evidence + knn_gate * knn_evidence
            return fused, pca_scores, knn_scores, knn_gate
        pca_calibrated = self.subspace.calibrated(pca_scores)
        knn_calibrated = self.normal_memory.calibrated(knn_scores)
        fused = (1.0 - self.knn_weight) * pca_calibrated + self.knn_weight * knn_calibrated
        return fused, pca_scores, knn_scores, None

    def _image_scores(self, patches, positions=None):
        if not self.dual_branch:
            return self._anomaly_scores(patches, positions)[0]
        pca_scores = self.image_subspace.score(patches)
        if self.anomaly_method == "pca":
            return pca_scores
        knn_scores = self.image_memory.score(patches, positions=positions)
        if self.anomaly_method == "knn":
            return knn_scores
        if self.fusion_mode == "gated":
            pca_evidence = self.image_subspace.tail_evidence(pca_scores)
            knn_evidence = self.image_memory.tail_evidence(knn_scores)
            logits = np.clip((knn_evidence - pca_evidence) / self.gate_temperature, -60.0, 60.0)
            gate = 1.0 / (1.0 + np.exp(-logits))
            return (1.0 - gate) * pca_evidence + gate * knn_evidence
        return (1.0 - self.knn_weight) * self.image_subspace.calibrated(pca_scores) + self.knn_weight * self.image_memory.calibrated(knn_scores)

    def _spatial_consistency(self, scores, grid):
        scores = np.asarray(scores, dtype=np.float64)
        if scores.size != grid[0] * grid[1]:
            raise ValueError(f"Score count {scores.size} does not match grid {grid}")
        keep = max(1, int(np.ceil(scores.size * self.image_top_ratio)))
        selected = np.argpartition(scores, -keep)[-keep:]
        mask = np.zeros(scores.size, dtype=bool)
        mask[selected] = True
        mask = mask.reshape(grid)
        visited = np.zeros_like(mask, dtype=bool)
        largest = 0
        for row, column in np.argwhere(mask):
            if visited[row, column]:
                continue
            size = 0
            stack = [(int(row), int(column))]
            visited[row, column] = True
            while stack:
                current_row, current_column = stack.pop()
                size += 1
                for row_offset in (-1, 0, 1):
                    for column_offset in (-1, 0, 1):
                        if row_offset == column_offset == 0:
                            continue
                        next_row, next_column = current_row + row_offset, current_column + column_offset
                        if 0 <= next_row < grid[0] and 0 <= next_column < grid[1] and mask[next_row, next_column] and not visited[next_row, next_column]:
                            visited[next_row, next_column] = True
                            stack.append((next_row, next_column))
            largest = max(largest, size)
        return largest / keep

    def _image_anomaly_score(self, patches, positions=None, grid=None):
        if self.image_fusion_stage == "patch" or self.anomaly_method != "pca_knn" or self.fusion_mode != "fixed":
            patch_scores = self._image_scores(patches, positions)
            base_score = self._aggregate_image_score(patch_scores)
        else:
            subspace = self.image_subspace if self.dual_branch else self.subspace
            memory = self.image_memory if self.dual_branch else self.normal_memory
            pca = subspace.calibrated(subspace.score(patches))
            knn = memory.calibrated(memory.score(patches, positions=positions))
            patch_scores = (1.0 - self.knn_weight) * pca + self.knn_weight * knn
            base_score = (1.0 - self.knn_weight) * self._aggregate_image_score(pca) + self.knn_weight * self._aggregate_image_score(knn)
        consistency = self._spatial_consistency(patch_scores, grid) if grid is not None else 0.0
        score = base_score + self.image_spatial_weight * abs(base_score) * consistency
        return score, consistency

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
        if self.dual_branch:
            patches, image_patches, grid = self.extractor.extract_dual(image)
        else:
            patches, grid = self.extractor.extract(image)
        if self.reference_grid is None:
            self.reference_grid = patches.shape[1]
        positions = self._patch_positions(grid)
        anomaly_scores, pca_scores, knn_scores, knn_gate = self._anomaly_scores(patches, positions)
        anomaly_map = self._postprocess_map(anomaly_scores.reshape(grid), image).tolist()
        fused_score, spatial_consistency = self._image_anomaly_score(image_patches if self.dual_branch else patches, positions, grid)
        typing_patches = self._anomaly_patches(patches)
        if self.type_matching == "rbf_svm":
            label, label_score = self.prototype_bank.predict_rbf_svm(typing_patches)
        else:
            typing_features = typing_patches if self.type_matching == "bidirectional_patch" else typing_patches.mean(axis=0)
            label, label_score = self.prototype_bank.predict(typing_features)
        result = {
            "image": str(image_path),
            "grid": list(grid),
            "anomaly_score": fused_score,
            "anomaly_map": anomaly_map,
            "defect_type": label,
            "defect_type_score": float(label_score),
            "fused_score": fused_score * self.alpha + float(label_score) * (1.0 - self.alpha),
            "anomaly_method": self.anomaly_method,
            "pca_residual_metric": self.pca_residual_metric,
            "fusion_mode": self.fusion_mode,
            "pca_anomaly_score": self._aggregate_image_score(pca_scores),
            "image_spatial_consistency": float(spatial_consistency),
        }
        if knn_scores is not None:
            result["knn_anomaly_score"] = self._aggregate_image_score(knn_scores)
        if knn_gate is not None:
            result["knn_gate_mean"] = float(np.mean(knn_gate))
            result["knn_gate_top1p_mean"] = self._aggregate_image_score(knn_gate)
        return result

    def save(self, path):
        state = {
            "alpha": self.alpha,
            "subspace": self.subspace.to_dict(),
            "prototype_bank": self.prototype_bank.to_dict(),
            "unknown_threshold": self.prototype_bank.unknown_threshold,
            "top_k_ratio": self.top_k_ratio,
            "image_score": self.image_score,
            "image_top_ratio": self.image_top_ratio,
            "image_fusion_stage": self.image_fusion_stage,
            "image_spatial_weight": self.image_spatial_weight,
            "type_matching": self.type_matching,
            "map_postprocess": self.map_postprocess,
            "gaussian_sigma": self.gaussian_sigma,
            "anomaly_method": self.anomaly_method,
            "pca_residual_metric": self.pca_residual_metric,
            "knn_weight": self.knn_weight,
            "memory_max_patches": self.normal_memory.max_patches,
            "knn_chunk_size": self.normal_memory.query_chunk_size,
            "knn_backend": self.normal_memory.backend,
            "knn_dtype": self.normal_memory.dtype,
            "knn_spatial_radius": self.normal_memory.spatial_radius,
            "dual_branch": self.dual_branch,
            "fusion_mode": self.fusion_mode,
            "gate_temperature": self.gate_temperature,
            "knn_center": self.normal_memory.center,
            "knn_scale": self.normal_memory.scale,
            "knn_calibration_scores": self.normal_memory.calibration_scores.tolist() if self.normal_memory.calibration_scores is not None else None,
            "reference_grid": self.reference_grid,
            "reference_shape": self.reference_shape,
        }
        if self.dual_branch:
            state["image_subspace"] = self.image_subspace.to_dict()
            state["image_knn_center"] = self.image_memory.center
            state["image_knn_scale"] = self.image_memory.scale
            state["image_knn_calibration_scores"] = self.image_memory.calibration_scores.tolist() if self.image_memory.calibration_scores is not None else None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.normal_memory.features is not None:
            memory_path = path.with_suffix(path.suffix + ".normal-memory.npz")
            memory_data = {"features": self.normal_memory.features.astype(np.float16)}
            if self.normal_memory.positions is not None:
                memory_data["positions"] = self.normal_memory.positions
            if self.dual_branch and self.image_memory.features is not None:
                memory_data["image_features"] = self.image_memory.features.astype(np.float16)
                if self.image_memory.positions is not None:
                    memory_data["image_positions"] = self.image_memory.positions
            np.savez_compressed(memory_path, **memory_data)
            state["normal_memory_file"] = memory_path.name
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path, extractor):
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(extractor, alpha=state.get("alpha", 0.5), unknown_threshold=state.get("unknown_threshold", 0.35), top_k_ratio=state.get("top_k_ratio", 0.05), image_score=state.get("image_score", "mean"), image_top_ratio=state.get("image_top_ratio", 0.01), image_fusion_stage=state.get("image_fusion_stage", "patch"), image_spatial_weight=state.get("image_spatial_weight", 0.0), type_matching=state.get("type_matching", "prototype_mean"), map_postprocess=state.get("map_postprocess", "none"), gaussian_sigma=state.get("gaussian_sigma", 1.0), anomaly_method=state.get("anomaly_method", "pca"), pca_residual_metric=state.get("pca_residual_metric", "squared_l2"), knn_weight=state.get("knn_weight", 0.5), memory_max_patches=state.get("memory_max_patches", 50000), knn_chunk_size=state.get("knn_chunk_size", 256), knn_backend=state.get("knn_backend", "auto"), knn_dtype=state.get("knn_dtype", "float32"), knn_spatial_radius=state.get("knn_spatial_radius", -1.0), dual_branch=state.get("dual_branch", False), fusion_mode=state.get("fusion_mode", "fixed"), gate_temperature=state.get("gate_temperature", 1.0))
        obj.subspace = NormalSubspace.from_dict(state["subspace"])
        if obj.dual_branch and "image_subspace" in state:
            obj.image_subspace = NormalSubspace.from_dict(state["image_subspace"])
        obj.prototype_bank = PrototypeBank.from_dict(state.get("prototype_bank", {}))
        obj.prototype_bank.unknown_threshold = state.get("unknown_threshold", 0.35)
        memory_file = state.get("normal_memory_file")
        if memory_file:
            memory_path = Path(path).parent / memory_file
            memory = np.load(memory_path)
            obj.normal_memory.features = memory["features"].astype(np.float32)
            obj.normal_memory.positions = memory["positions"].astype(np.float32) if "positions" in memory else None
            obj.normal_memory.center = float(state.get("knn_center", 0.0))
            obj.normal_memory.scale = float(state.get("knn_scale", 1.0))
            calibration = state.get("knn_calibration_scores")
            obj.normal_memory.calibration_scores = None if calibration is None else np.asarray(calibration, dtype=np.float64)
            if obj.dual_branch and "image_features" in memory:
                obj.image_memory.features = memory["image_features"].astype(np.float32)
                obj.image_memory.positions = memory["image_positions"].astype(np.float32) if "image_positions" in memory else None
                obj.image_memory.center = float(state.get("image_knn_center", 0.0))
                obj.image_memory.scale = float(state.get("image_knn_scale", 1.0))
                image_calibration = state.get("image_knn_calibration_scores")
                obj.image_memory.calibration_scores = None if image_calibration is None else np.asarray(image_calibration, dtype=np.float64)
        obj.reference_grid = state.get("reference_grid")
        obj.reference_shape = state.get("reference_shape")
        return obj
