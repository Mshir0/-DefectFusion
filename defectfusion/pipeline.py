from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .model import NormalPatchMemory, NormalSubspace, NormalTextureModel, PrototypeBank


@dataclass(frozen=True)
class NormalTrainingView:
    """A normal training image with an augmented-to-canonical position map."""
    image: Image.Image
    inverse_position_matrix: np.ndarray | None = None


class DefectFusion:
    def __init__(self, extractor, *, alpha: float = 0.5, unknown_threshold: float = 0.35, top_k_ratio: float = 0.05, image_score: str = "mtop1p", image_top_ratio: float = 0.01, image_fusion_stage: str = "patch", image_spatial_weight: float = 0.0, type_matching: str = "bidirectional_patch", map_postprocess: str = "none", gaussian_sigma: float = 1.0, anomaly_method: str = "pca", pca_residual_metric: str = "squared_l2", knn_weight: float = 0.5, anoco_neighbors: int = 16, anoco_query_weight: float = 1.0, anoco_temperature: float = 0.07, anoco_weight: float = 0.5, memory_max_patches: int = 50000, knn_chunk_size: int = 256, knn_backend: str = "auto", knn_dtype: str = "float32", knn_spatial_radius: float = -1.0, align_training_positions: bool = False, dual_branch: bool = False, fusion_mode: str = "fixed", gate_temperature: float = 1.0, test_augmentations=(), texture_evidence: bool = False, texture_weight: float = 0.25, texture_candidate_ratio: float = 0.1):
        self.extractor = extractor
        knn_device = getattr(extractor, "device", None)
        self.dual_branch = bool(dual_branch)
        self.image_subspace = NormalSubspace(residual_metric=pca_residual_metric) if self.dual_branch else None
        self.image_memory = NormalPatchMemory(memory_max_patches, knn_chunk_size, backend=knn_backend, device=knn_device if self.dual_branch else None, dtype=knn_dtype, spatial_radius=knn_spatial_radius) if self.dual_branch else None
        self.alpha = alpha
        self.subspace = NormalSubspace(residual_metric=pca_residual_metric)
        self.normal_memory = NormalPatchMemory(memory_max_patches, knn_chunk_size, backend=knn_backend, device=knn_device, dtype=knn_dtype, spatial_radius=knn_spatial_radius)
        self.align_training_positions = bool(align_training_positions)
        if self.align_training_positions and knn_spatial_radius < 0:
            raise ValueError("align_training_positions requires a non-negative knn_spatial_radius")
        if self.align_training_positions and getattr(extractor, "resize_mode", "direct") != "direct":
            raise ValueError("align_training_positions currently requires resize_mode=direct")
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
        if anomaly_method not in {"pca", "knn", "pca_knn", "anoco", "pca_anoco"}:
            raise ValueError("anomaly_method must be pca, knn, pca_knn, anoco, or pca_anoco")
        if not 0 <= knn_weight <= 1:
            raise ValueError("knn_weight must be in [0, 1]")
        self.anomaly_method = anomaly_method
        self.pca_residual_metric = pca_residual_metric
        self.knn_weight = float(knn_weight)
        if anoco_neighbors <= 0 or anoco_query_weight <= 0 or anoco_temperature <= 0:
            raise ValueError("ANoCo neighbors, query weight, and temperature must be positive")
        if not 0 <= anoco_weight <= 1:
            raise ValueError("anoco_weight must be in [0, 1]")
        self.anoco_neighbors = int(anoco_neighbors)
        self.anoco_query_weight = float(anoco_query_weight)
        self.anoco_temperature = float(anoco_temperature)
        self.anoco_weight = float(anoco_weight)
        if fusion_mode not in {"fixed", "gated"}:
            raise ValueError("fusion_mode must be fixed or gated")
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive")
        self.fusion_mode = fusion_mode
        self.gate_temperature = float(gate_temperature)
        invalid_augmentations = set(test_augmentations) - {"hflip", "vflip"}
        if invalid_augmentations:
            raise ValueError(f"Unknown test augmentations: {sorted(invalid_augmentations)}")
        self.test_augmentations = tuple(dict.fromkeys(test_augmentations))
        self.texture_evidence = bool(texture_evidence)
        if texture_weight < 0:
            raise ValueError("texture_weight must be non-negative")
        if not 0 < texture_candidate_ratio <= 1:
            raise ValueError("texture_candidate_ratio must be in (0, 1]")
        self.texture_weight = float(texture_weight)
        self.texture_candidate_ratio = float(texture_candidate_ratio)
        self.texture_model = NormalTextureModel() if self.texture_evidence else None
        self.reference_grid = None
        self.reference_shape = None

    def fit_normal(self, image_paths):
        patch_batches, image_patch_batches, position_batches, texture_batches = [], [], [], []
        memory_patch_batches, image_memory_patch_batches, memory_position_batches = [], [], []
        for path in image_paths:
            if isinstance(path, NormalTrainingView):
                image = path.image.copy()
                inverse_position_matrix = path.inverse_position_matrix
            else:
                image = path.copy() if isinstance(path, Image.Image) else Image.open(path)
                inverse_position_matrix = None
            if self.dual_branch:
                patches, image_patches, grid = self.extractor.extract_dual(image)
                image_patch_batches.append(image_patches)
            else:
                patches, grid = self.extractor.extract(image)
            patch_batches.append(patches)
            if self.texture_evidence:
                texture_batches.append(self._texture_descriptors(image, grid))
            positions, valid = self._training_positions(grid, inverse_position_matrix)
            position_batches.append(positions)
            memory_patch_batches.append(patches[valid])
            if self.dual_branch:
                image_memory_patch_batches.append(image_patches[valid])
            memory_position_batches.append(positions)
            self.reference_shape = grid
        if not patch_batches:
            raise ValueError("No normal images were provided")
        features = np.concatenate(patch_batches, axis=0)
        self.subspace.fit(features)
        if self.texture_evidence:
            self.texture_model.fit(np.concatenate(texture_batches, axis=0))
        if self.anomaly_method != "pca":
            memory_features = np.concatenate(memory_patch_batches, axis=0) if self.align_training_positions else features
            memory_positions = np.concatenate(memory_position_batches if self.align_training_positions else position_batches, axis=0)
            self.normal_memory.fit(memory_features, memory_positions)
            if self.anomaly_method in {"anoco", "pca_anoco"}:
                self.normal_memory.fit_anoco_calibration(self.anoco_neighbors, self.anoco_query_weight, self.anoco_temperature)
        if self.dual_branch:
            image_features = np.concatenate(image_patch_batches, axis=0)
            self.image_subspace.fit(image_features)
            if self.anomaly_method != "pca":
                image_memory_features = np.concatenate(image_memory_patch_batches, axis=0) if self.align_training_positions else image_features
                self.image_memory.fit(image_memory_features, memory_positions)
                if self.anomaly_method in {"anoco", "pca_anoco"}:
                    self.image_memory.fit_anoco_calibration(self.anoco_neighbors, self.anoco_query_weight, self.anoco_temperature)
        self.reference_grid = features.shape[1]
        return self

    def add_prototype(self, label: str, image_path):
        image = Image.open(image_path)
        patches, _ = self.extractor.extract(image)
        self.prototype_bank.add(label, self._anomaly_patches(patches))
        return self

    def _anomaly_patches(self, patches, scores=None):
        scores = self.subspace.score(patches) if scores is None else np.asarray(scores)
        if len(scores) != len(patches):
            raise ValueError("Anomaly score count must match patch count")
        keep = max(1, int(np.ceil(len(scores) * self.top_k_ratio)))
        indices = np.argpartition(scores, -keep)[-keep:]
        return patches[indices]

    @staticmethod
    def _patch_positions(grid):
        rows, columns = np.indices(grid, dtype=np.float32)
        return np.stack([(rows.ravel() + 0.5) / grid[0], (columns.ravel() + 0.5) / grid[1]], axis=1)

    def _training_positions(self, grid, inverse_position_matrix):
        positions = self._patch_positions(grid)
        if not self.align_training_positions or inverse_position_matrix is None:
            return positions, np.ones(len(positions), dtype=bool)
        matrix = np.asarray(inverse_position_matrix, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError("inverse_position_matrix must have shape (3, 3)")
        homogeneous = np.concatenate([positions, np.ones((len(positions), 1))], axis=1)
        canonical = homogeneous @ matrix.T
        canonical = canonical[:, :2] / np.maximum(canonical[:, 2:], 1e-12)
        valid = np.all((canonical >= 0.0) & (canonical <= 1.0), axis=1)
        if not np.any(valid):
            raise ValueError("Training augmentation has no valid canonical patch positions")
        return canonical[valid].astype(np.float32), valid

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

    @staticmethod
    def _texture_descriptors(image, grid):
        height, width = grid
        resized = image.convert("L").resize((width * 16, height * 16), Image.Resampling.BILINEAR)
        gray = np.asarray(resized, dtype=np.float32) / 255.0
        gradient_y, gradient_x = np.gradient(gray)
        gradient = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        padded = np.pad(gray, 1, mode="reflect")
        laplacian = np.abs(
            padded[1:-1, :-2] + padded[1:-1, 2:] + padded[:-2, 1:-1] + padded[2:, 1:-1]
            - 4.0 * gray
        )
        gray_blocks = gray.reshape(height, 16, width, 16)
        gradient_blocks = gradient.reshape(height, 16, width, 16)
        laplacian_blocks = laplacian.reshape(height, 16, width, 16)
        return np.stack([
            gray_blocks.std(axis=(1, 3)),
            gradient_blocks.mean(axis=(1, 3)),
            laplacian_blocks.mean(axis=(1, 3)),
        ], axis=-1).reshape(-1, 3).astype(np.float32)

    @staticmethod
    def _connected_regions(mask):
        mask = np.asarray(mask, dtype=bool)
        visited = np.zeros_like(mask, dtype=bool)
        regions = []
        for row, column in np.argwhere(mask):
            if visited[row, column]:
                continue
            region = []
            stack = [(int(row), int(column))]
            visited[row, column] = True
            while stack:
                current_row, current_column = stack.pop()
                region.append((current_row, current_column))
                for row_offset in (-1, 0, 1):
                    for column_offset in (-1, 0, 1):
                        if row_offset == column_offset == 0:
                            continue
                        next_row, next_column = current_row + row_offset, current_column + column_offset
                        if 0 <= next_row < mask.shape[0] and 0 <= next_column < mask.shape[1] and mask[next_row, next_column] and not visited[next_row, next_column]:
                            visited[next_row, next_column] = True
                            stack.append((next_row, next_column))
            regions.append(region)
        return regions

    def _reason_reject(self, scores, texture_scores, grid):
        scores = np.asarray(scores, dtype=np.float64)
        texture_scores = np.asarray(texture_scores, dtype=np.float64)
        if scores.size != grid[0] * grid[1] or texture_scores.size != scores.size:
            raise ValueError("Reason-and-reject scores must match the patch grid")
        keep = max(1, int(np.ceil(scores.size * self.texture_candidate_ratio)))
        selected = np.argpartition(scores, -keep)[-keep:]
        candidate_mask = np.zeros(scores.size, dtype=bool)
        candidate_mask[selected] = True
        candidate_mask = candidate_mask.reshape(grid)
        refined = scores.reshape(grid).copy()
        evidences = []
        texture_map = texture_scores.reshape(grid)
        for region in self._connected_regions(candidate_mask):
            rows, columns = zip(*region)
            evidence = float(np.clip(np.mean(texture_map[rows, columns]), -5.0, 5.0))
            refined[rows, columns] += self.texture_weight * evidence
            evidences.append(evidence)
        return refined.ravel(), evidences

    def _branch_scores(self, patches, subspace, memory, positions=None):
        pca_scores = subspace.score(patches)
        if self.anomaly_method == "pca":
            return pca_scores, pca_scores, None, None
        if self.anomaly_method in {"anoco", "pca_anoco"}:
            normal_scores = memory.score_anoco(patches, self.anoco_neighbors, self.anoco_query_weight, self.anoco_temperature, positions=positions)
            if self.anomaly_method == "anoco":
                return normal_scores, pca_scores, normal_scores, None
            if self.fusion_mode == "gated":
                pca_evidence = subspace.tail_evidence(pca_scores)
                normal_evidence = memory.tail_evidence_anoco(normal_scores)
                logits = np.clip((normal_evidence - pca_evidence) / self.gate_temperature, -60.0, 60.0)
                gate = 1.0 / (1.0 + np.exp(-logits))
                return (1.0 - gate) * pca_evidence + gate * normal_evidence, pca_scores, normal_scores, gate
            fused = ((1.0 - self.anoco_weight) * subspace.calibrated(pca_scores)
                     + self.anoco_weight * memory.calibrated_anoco(normal_scores))
            return fused, pca_scores, normal_scores, None
        knn_scores = memory.score(patches, positions=positions)
        if self.anomaly_method == "knn":
            return knn_scores, pca_scores, knn_scores, None
        if self.fusion_mode == "gated":
            pca_evidence = subspace.tail_evidence(pca_scores)
            knn_evidence = memory.tail_evidence(knn_scores)
            logits = np.clip((knn_evidence - pca_evidence) / self.gate_temperature, -60.0, 60.0)
            knn_gate = 1.0 / (1.0 + np.exp(-logits))
            fused = (1.0 - knn_gate) * pca_evidence + knn_gate * knn_evidence
            return fused, pca_scores, knn_scores, knn_gate
        pca_calibrated = subspace.calibrated(pca_scores)
        knn_calibrated = memory.calibrated(knn_scores)
        fused = (1.0 - self.knn_weight) * pca_calibrated + self.knn_weight * knn_calibrated
        return fused, pca_scores, knn_scores, None

    def _anomaly_scores(self, patches, positions=None):
        return self._branch_scores(patches, self.subspace, self.normal_memory, positions)

    def _image_score_bundle(self, patches, positions=None):
        subspace = self.image_subspace if self.dual_branch else self.subspace
        memory = self.image_memory if self.dual_branch else self.normal_memory
        return self._branch_scores(patches, subspace, memory, positions)

    def _image_scores(self, patches, positions=None):
        return self._image_score_bundle(patches, positions)[0]

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

    def _image_anomaly_score(self, patches, positions=None, grid=None, texture_scores=None, score_bundle=None):
        if score_bundle is None:
            score_bundle = self._image_score_bundle(patches, positions)
        fused_scores, pca_scores, knn_scores, _ = score_bundle
        if self.image_fusion_stage == "patch" or self.anomaly_method not in {"pca_knn", "pca_anoco"} or self.fusion_mode != "fixed":
            patch_scores = fused_scores
            base_score = self._aggregate_image_score(patch_scores)
        else:
            subspace = self.image_subspace if self.dual_branch else self.subspace
            memory = self.image_memory if self.dual_branch else self.normal_memory
            pca = subspace.calibrated(pca_scores)
            if self.anomaly_method == "pca_anoco":
                normal = memory.calibrated_anoco(knn_scores)
                weight = self.anoco_weight
            else:
                normal = memory.calibrated(knn_scores)
                weight = self.knn_weight
            patch_scores = (1.0 - weight) * pca + weight * normal
            base_score = (1.0 - weight) * self._aggregate_image_score(pca) + weight * self._aggregate_image_score(normal)
        if self.texture_evidence and texture_scores is not None:
            patch_scores, _ = self._reason_reject(patch_scores, texture_scores, grid)
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

    def _predict_single(self, image, image_name):
        if self.dual_branch:
            patches, image_patches, grid = self.extractor.extract_dual(image)
        else:
            patches, grid = self.extractor.extract(image)
        if self.reference_grid is None:
            self.reference_grid = patches.shape[1]
        positions = self._patch_positions(grid)
        anomaly_scores, pca_scores, knn_scores, knn_gate = self._anomaly_scores(patches, positions)
        if self.dual_branch:
            image_score_bundle = self._image_score_bundle(image_patches, positions)
        else:
            image_score_bundle = (anomaly_scores, pca_scores, knn_scores, knn_gate)
        texture_scores = None
        texture_region_evidence = []
        if self.texture_evidence:
            descriptors = self._texture_descriptors(image, grid)
            texture_scores = self.texture_model.score(descriptors)
            anomaly_scores, texture_region_evidence = self._reason_reject(anomaly_scores, texture_scores, grid)
        anomaly_map = self._postprocess_map(anomaly_scores.reshape(grid), image).tolist()
        fused_score, spatial_consistency = self._image_anomaly_score(
            image_patches if self.dual_branch else patches,
            positions,
            grid,
            texture_scores,
            image_score_bundle,
        )
        if not self.prototype_bank.prototypes:
            label, label_score = "unknown", 0.0
        else:
            typing_patches = self._anomaly_patches(patches, pca_scores)
            if self.type_matching == "rbf_svm":
                label, label_score = self.prototype_bank.predict_rbf_svm(typing_patches)
            else:
                typing_features = typing_patches if self.type_matching == "bidirectional_patch" else typing_patches.mean(axis=0)
                label, label_score = self.prototype_bank.predict(typing_features)
        result = {
            "image": str(image_name),
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
            "texture_region_evidence": texture_region_evidence,
        }
        if knn_scores is not None:
            score_name = "anoco_anomaly_score" if self.anomaly_method in {"anoco", "pca_anoco"} else "knn_anomaly_score"
            result[score_name] = self._aggregate_image_score(knn_scores)
        if knn_gate is not None:
            result["knn_gate_mean"] = float(np.mean(knn_gate))
            result["knn_gate_top1p_mean"] = self._aggregate_image_score(knn_gate)
        return result

    @staticmethod
    def _test_view(image, augmentation):
        if augmentation == "hflip":
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if augmentation == "vflip":
            return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        return image

    @staticmethod
    def _restore_test_map(anomaly_map, augmentation):
        anomaly_map = np.asarray(anomaly_map, dtype=np.float64)
        if augmentation == "hflip":
            return np.fliplr(anomaly_map)
        if augmentation == "vflip":
            return np.flipud(anomaly_map)
        return anomaly_map

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        base = self._predict_single(image, image_path)
        if not self.test_augmentations:
            base["test_augmentations"] = []
            base["tta_view_scores"] = [float(base["anomaly_score"])]
            return base

        views = [("identity", base)]
        for augmentation in self.test_augmentations:
            transformed = self._test_view(image, augmentation)
            views.append((augmentation, self._predict_single(transformed, image_path)))
        view_scores = [float(result["anomaly_score"]) for _, result in views]
        maps = [self._restore_test_map(result["anomaly_map"], name) for name, result in views]
        base["anomaly_map"] = np.mean(maps, axis=0).tolist()
        averaged_fields = ["anomaly_score", "pca_anomaly_score", "image_spatial_consistency"]
        for field in ("knn_anomaly_score", "anoco_anomaly_score"):
            if all(field in result for _, result in views):
                averaged_fields.append(field)
        if all("knn_gate_mean" in result for _, result in views):
            averaged_fields.extend(["knn_gate_mean", "knn_gate_top1p_mean"])
        for field in averaged_fields:
            base[field] = float(np.mean([result[field] for _, result in views]))
        base["fused_score"] = base["anomaly_score"] * self.alpha + float(base["defect_type_score"]) * (1.0 - self.alpha)
        base["test_augmentations"] = list(self.test_augmentations)
        base["tta_view_scores"] = view_scores
        return base

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
            "anoco_neighbors": self.anoco_neighbors,
            "anoco_query_weight": self.anoco_query_weight,
            "anoco_temperature": self.anoco_temperature,
            "anoco_weight": self.anoco_weight,
            "memory_max_patches": self.normal_memory.max_patches,
            "knn_chunk_size": self.normal_memory.query_chunk_size,
            "knn_backend": self.normal_memory.backend,
            "knn_dtype": self.normal_memory.dtype,
            "knn_spatial_radius": self.normal_memory.spatial_radius,
            "align_training_positions": self.align_training_positions,
            "dual_branch": self.dual_branch,
            "fusion_mode": self.fusion_mode,
            "gate_temperature": self.gate_temperature,
            "test_augmentations": list(self.test_augmentations),
            "texture_evidence": self.texture_evidence,
            "texture_weight": self.texture_weight,
            "texture_candidate_ratio": self.texture_candidate_ratio,
            "texture_model": self.texture_model.to_dict() if self.texture_model is not None else None,
            "knn_center": self.normal_memory.center,
            "knn_scale": self.normal_memory.scale,
            "knn_calibration_scores": self.normal_memory.calibration_scores.tolist() if self.normal_memory.calibration_scores is not None else None,
            "anoco_center": self.normal_memory.anoco_center,
            "anoco_scale": self.normal_memory.anoco_scale,
            "anoco_calibration_scores": self.normal_memory.anoco_calibration_scores.tolist() if self.normal_memory.anoco_calibration_scores is not None else None,
            "reference_grid": self.reference_grid,
            "reference_shape": self.reference_shape,
        }
        if self.dual_branch:
            state["image_subspace"] = self.image_subspace.to_dict()
            state["image_knn_center"] = self.image_memory.center
            state["image_knn_scale"] = self.image_memory.scale
            state["image_knn_calibration_scores"] = self.image_memory.calibration_scores.tolist() if self.image_memory.calibration_scores is not None else None
            state["image_anoco_center"] = self.image_memory.anoco_center
            state["image_anoco_scale"] = self.image_memory.anoco_scale
            state["image_anoco_calibration_scores"] = self.image_memory.anoco_calibration_scores.tolist() if self.image_memory.anoco_calibration_scores is not None else None
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
        obj = cls(extractor, alpha=state.get("alpha", 0.5), unknown_threshold=state.get("unknown_threshold", 0.35), top_k_ratio=state.get("top_k_ratio", 0.05), image_score=state.get("image_score", "mean"), image_top_ratio=state.get("image_top_ratio", 0.01), image_fusion_stage=state.get("image_fusion_stage", "patch"), image_spatial_weight=state.get("image_spatial_weight", 0.0), type_matching=state.get("type_matching", "prototype_mean"), map_postprocess=state.get("map_postprocess", "none"), gaussian_sigma=state.get("gaussian_sigma", 1.0), anomaly_method=state.get("anomaly_method", "pca"), pca_residual_metric=state.get("pca_residual_metric", "squared_l2"), knn_weight=state.get("knn_weight", 0.5), anoco_neighbors=state.get("anoco_neighbors", 16), anoco_query_weight=state.get("anoco_query_weight", 1.0), anoco_temperature=state.get("anoco_temperature", 0.07), anoco_weight=state.get("anoco_weight", 0.5), memory_max_patches=state.get("memory_max_patches", 50000), knn_chunk_size=state.get("knn_chunk_size", 256), knn_backend=state.get("knn_backend", "auto"), knn_dtype=state.get("knn_dtype", "float32"), knn_spatial_radius=state.get("knn_spatial_radius", -1.0), align_training_positions=state.get("align_training_positions", False), dual_branch=state.get("dual_branch", False), fusion_mode=state.get("fusion_mode", "fixed"), gate_temperature=state.get("gate_temperature", 1.0), test_augmentations=state.get("test_augmentations", ()), texture_evidence=state.get("texture_evidence", False), texture_weight=state.get("texture_weight", 0.25), texture_candidate_ratio=state.get("texture_candidate_ratio", 0.1))
        obj.subspace = NormalSubspace.from_dict(state["subspace"])
        if obj.texture_evidence and state.get("texture_model") is not None:
            obj.texture_model = NormalTextureModel.from_dict(state["texture_model"])
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
            obj.normal_memory.anoco_center = float(state.get("anoco_center", 0.0))
            obj.normal_memory.anoco_scale = float(state.get("anoco_scale", 1.0))
            anoco_calibration = state.get("anoco_calibration_scores")
            obj.normal_memory.anoco_calibration_scores = None if anoco_calibration is None else np.asarray(anoco_calibration, dtype=np.float64)
            if obj.dual_branch and "image_features" in memory:
                obj.image_memory.features = memory["image_features"].astype(np.float32)
                obj.image_memory.positions = memory["image_positions"].astype(np.float32) if "image_positions" in memory else None
                obj.image_memory.center = float(state.get("image_knn_center", 0.0))
                obj.image_memory.scale = float(state.get("image_knn_scale", 1.0))
                image_calibration = state.get("image_knn_calibration_scores")
                obj.image_memory.calibration_scores = None if image_calibration is None else np.asarray(image_calibration, dtype=np.float64)
                obj.image_memory.anoco_center = float(state.get("image_anoco_center", 0.0))
                obj.image_memory.anoco_scale = float(state.get("image_anoco_scale", 1.0))
                image_anoco_calibration = state.get("image_anoco_calibration_scores")
                obj.image_memory.anoco_calibration_scores = None if image_anoco_calibration is None else np.asarray(image_anoco_calibration, dtype=np.float64)
        obj.reference_grid = state.get("reference_grid")
        obj.reference_shape = state.get("reference_shape")
        return obj
