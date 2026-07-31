from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .model import NormalPatchMemory, NormalSubspace, PrototypeBank


@dataclass(frozen=True)
class NormalTrainingView:
    """A normal training image with an augmented-to-canonical position map."""
    image: Image.Image
    inverse_position_matrix: np.ndarray | None = None


class DefectFusion:
    def __init__(self, extractor, *, alpha: float = 0.5, unknown_threshold: float = 0.35, top_k_ratio: float = 0.05, image_score: str = "mtop1p", image_top_ratio: float = 0.01, image_fusion_stage: str = "patch", image_spatial_weight: float = 0.0, image_min_component_size: int = 1, type_matching: str = "bidirectional_patch", map_postprocess: str = "none", gaussian_sigma: float = 1.0, anomaly_method: str = "pca", pca_residual_metric: str = "squared_l2", knn_weight: float = 0.5, anoco_neighbors: int = 16, anoco_query_weight: float = 1.0, anoco_temperature: float = 0.07, anoco_affinity: str = "softmax", anoco_anchor_ranking: str = "mean", anoco_norm_compatibility: bool = False, anoco_weight: float = 0.5, anoco_layer_consensus: bool = False, memory_max_patches: int = 50000, knn_chunk_size: int = 256, knn_backend: str = "auto", knn_dtype: str = "float32", knn_spatial_radius: float = -1.0, align_training_positions: bool = False, dual_branch: bool = False, fusion_mode: str = "fixed", gate_temperature: float = 1.0, test_augmentations=(), pixel_image_size: int | None = None, image_head_image_size: int | None = None, secondary_pixel_image_size: int | None = None, pixel_multiscale_weight: float = 0.5):
        self.extractor = extractor
        knn_device = getattr(extractor, "device", None)
        self.dual_branch = bool(dual_branch)
        default_image_size = getattr(extractor, "image_size", None)
        self.pixel_image_size = default_image_size if pixel_image_size is None else int(pixel_image_size)
        self.image_head_image_size = default_image_size if image_head_image_size is None else int(image_head_image_size)
        self.secondary_pixel_image_size = None if secondary_pixel_image_size is None else int(secondary_pixel_image_size)
        if not 0 <= pixel_multiscale_weight <= 1:
            raise ValueError("pixel_multiscale_weight must be in [0, 1]")
        self.pixel_multiscale_weight = float(pixel_multiscale_weight)
        self._positional_basis_by_size = {}
        if default_image_size is not None and getattr(extractor, "positional_basis", None) is not None:
            self._positional_basis_by_size[default_image_size] = extractor.positional_basis
        if self.pixel_image_size is not None and self.pixel_image_size <= 0:
            raise ValueError("pixel_image_size must be positive")
        if self.image_head_image_size is not None and self.image_head_image_size <= 0:
            raise ValueError("image_head_image_size must be positive")
        if self.secondary_pixel_image_size is not None and self.secondary_pixel_image_size <= 0:
            raise ValueError("secondary_pixel_image_size must be positive")
        if self.secondary_pixel_image_size is not None and self.secondary_pixel_image_size == self.pixel_image_size:
            raise ValueError("secondary_pixel_image_size must differ from pixel_image_size")
        if not self.dual_branch and self.pixel_image_size != self.image_head_image_size:
            raise ValueError("different pixel and image-head sizes require dual_branch=true")
        self.image_subspace = NormalSubspace(residual_metric=pca_residual_metric) if self.dual_branch else None
        self.image_memory = NormalPatchMemory(memory_max_patches, knn_chunk_size, backend=knn_backend, device=knn_device if self.dual_branch else None, dtype=knn_dtype, spatial_radius=knn_spatial_radius) if self.dual_branch else None
        self.alpha = alpha
        self.subspace = NormalSubspace(residual_metric=pca_residual_metric)
        self.normal_memory = NormalPatchMemory(memory_max_patches, knn_chunk_size, backend=knn_backend, device=knn_device, dtype=knn_dtype, spatial_radius=knn_spatial_radius)
        self.secondary_subspace = NormalSubspace(residual_metric=pca_residual_metric) if self.secondary_pixel_image_size is not None else None
        self.secondary_normal_memory = NormalPatchMemory(memory_max_patches, knn_chunk_size, backend=knn_backend, device=knn_device, dtype=knn_dtype, spatial_radius=knn_spatial_radius) if self.secondary_pixel_image_size is not None else None
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
        if image_min_component_size <= 0:
            raise ValueError("image_min_component_size must be positive")
        self.image_min_component_size = int(image_min_component_size)
        if type_matching not in {"prototype_mean", "bidirectional_patch", "rbf_svm"}:
            raise ValueError("type_matching must be prototype_mean, bidirectional_patch, or rbf_svm")
        self.type_matching = type_matching
        if map_postprocess not in {"none", "gaussian", "crf"}:
            raise ValueError("map_postprocess must be none, gaussian, or crf")
        self.map_postprocess = map_postprocess
        self.gaussian_sigma = gaussian_sigma
        if anomaly_method not in {"pca", "knn", "pca_knn", "anoco", "pca_anoco", "pca_knn_anoco"}:
            raise ValueError("anomaly_method must be pca, knn, pca_knn, anoco, pca_anoco, or pca_knn_anoco")
        if anomaly_method == "pca_knn_anoco" and not self.dual_branch:
            raise ValueError("pca_knn_anoco requires dual_branch=true")
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
        if anoco_affinity not in {"softmax", "cosine"}:
            raise ValueError("anoco_affinity must be softmax or cosine")
        self.anoco_affinity = anoco_affinity
        if anoco_anchor_ranking not in {"mean", "minimum"}:
            raise ValueError("anoco_anchor_ranking must be mean or minimum")
        self.anoco_anchor_ranking = anoco_anchor_ranking
        self.anoco_norm_compatibility = bool(anoco_norm_compatibility)
        self.anoco_weight = float(anoco_weight)
        self.anoco_layer_consensus = bool(anoco_layer_consensus)
        if self.anoco_layer_consensus and (not self.dual_branch or anomaly_method not in {"pca_anoco", "pca_knn_anoco"} or image_fusion_stage != "patch"):
            raise ValueError("anoco_layer_consensus requires dual_branch, a PCA+ANoCo image head, and patch fusion")
        self.image_layer_memories = []
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
        self.reference_grid = None
        self.reference_shape = None

    def _set_extractor_image_size(self, image_size):
        if image_size is None or not hasattr(self.extractor, "image_size"):
            return
        if self.extractor.image_size != image_size:
            current_basis = getattr(self.extractor, "positional_basis", None)
            if current_basis is not None:
                self._positional_basis_by_size[self.extractor.image_size] = current_basis
            self.extractor.image_size = image_size
            if hasattr(self.extractor, "positional_basis"):
                self.extractor.positional_basis = self._positional_basis_by_size.get(image_size)

    def memory_stats(self):
        memories = [self.normal_memory]
        if self.dual_branch and self.image_memory is not None:
            memories.append(self.image_memory)
            memories.extend(self.image_layer_memories)
        if self.secondary_normal_memory is not None:
            memories.append(self.secondary_normal_memory)
        return {
            "patch_count": int(sum(len(memory.features) for memory in memories if memory.features is not None)),
            "bytes": int(sum(memory.memory_bytes for memory in memories)),
        }

    def _extract_branches(self, image):
        if not self.dual_branch:
            self._set_extractor_image_size(self.pixel_image_size)
            patches, grid = self.extractor.extract(image)
            return patches, None, None, grid, None
        if self.pixel_image_size == self.image_head_image_size:
            self._set_extractor_image_size(self.pixel_image_size)
            if self.anoco_layer_consensus:
                patches, image_patches, image_layers, grid = self.extractor.extract_dual_layers(image)
            else:
                patches, image_patches, grid = self.extractor.extract_dual(image)
                image_layers = None
            return patches, image_patches, image_layers, grid, grid
        self._set_extractor_image_size(self.pixel_image_size)
        patches, pixel_grid = self.extractor.extract(image)
        self._set_extractor_image_size(self.image_head_image_size)
        patches_unused, image_patches, image_layers, image_grid = self.extractor.extract_dual_layers(image)
        return patches, image_patches, image_layers if self.anoco_layer_consensus else None, pixel_grid, image_grid

    def fit_normal(self, image_paths):
        patch_batches, image_patch_batches, image_layer_batches, position_batches, image_position_batches = [], [], [], [], []
        memory_patch_batches, image_memory_patch_batches, image_layer_memory_batches = [], [], []
        memory_position_batches, image_memory_position_batches = [], []
        secondary_patch_batches, secondary_memory_patch_batches, secondary_position_batches = [], [], []
        for path in image_paths:
            if isinstance(path, NormalTrainingView):
                image = path.image.copy()
                inverse_position_matrix = path.inverse_position_matrix
            else:
                image = path.copy() if isinstance(path, Image.Image) else Image.open(path)
                inverse_position_matrix = None
            patches, image_patches, image_layers, grid, image_grid = self._extract_branches(image)
            if self.dual_branch:
                image_patch_batches.append(image_patches)
                if self.anoco_layer_consensus:
                    image_layer_batches.append(image_layers)
            patch_batches.append(patches)
            positions, valid = self._training_positions(grid, inverse_position_matrix)
            position_batches.append(positions)
            memory_patch_batches.append(patches[valid])
            if self.dual_branch:
                image_positions, image_valid = self._training_positions(image_grid, inverse_position_matrix)
                image_position_batches.append(image_positions)
                image_memory_patch_batches.append(image_patches[image_valid])
                image_memory_position_batches.append(image_positions)
                if self.anoco_layer_consensus:
                    image_layer_memory_batches.append(image_layers[:, image_valid])
            memory_position_batches.append(positions)
            if self.secondary_pixel_image_size is not None:
                self._set_extractor_image_size(self.secondary_pixel_image_size)
                secondary_patches, secondary_grid = self.extractor.extract(image)
                secondary_positions, secondary_valid = self._training_positions(secondary_grid, inverse_position_matrix)
                secondary_patch_batches.append(secondary_patches)
                secondary_memory_patch_batches.append(secondary_patches[secondary_valid])
                secondary_position_batches.append(secondary_positions)
            self.reference_shape = grid
        if not patch_batches:
            raise ValueError("No normal images were provided")
        features = np.concatenate(patch_batches, axis=0)
        self.subspace.fit(features)
        if self.anomaly_method != "pca":
            memory_features = np.concatenate(memory_patch_batches, axis=0) if self.align_training_positions else features
            memory_positions = np.concatenate(memory_position_batches if self.align_training_positions else position_batches, axis=0)
            self.normal_memory.fit(memory_features, memory_positions)
            if self.anomaly_method in {"anoco", "pca_anoco"}:
                self.normal_memory.fit_anoco_calibration(self.anoco_neighbors, self.anoco_query_weight, self.anoco_temperature, affinity=self.anoco_affinity, anchor_ranking=self.anoco_anchor_ranking, norm_compatibility=self.anoco_norm_compatibility)
        if self.secondary_pixel_image_size is not None:
            secondary_features = np.concatenate(secondary_patch_batches, axis=0)
            self.secondary_subspace.fit(secondary_features)
            if self.anomaly_method != "pca":
                secondary_memory_features = np.concatenate(secondary_memory_patch_batches, axis=0) if self.align_training_positions else secondary_features
                secondary_positions = np.concatenate(secondary_position_batches, axis=0)
                self.secondary_normal_memory.fit(secondary_memory_features, secondary_positions)
                if self.anomaly_method in {"anoco", "pca_anoco"}:
                    self.secondary_normal_memory.fit_anoco_calibration(self.anoco_neighbors, self.anoco_query_weight, self.anoco_temperature, affinity=self.anoco_affinity, anchor_ranking=self.anoco_anchor_ranking, norm_compatibility=self.anoco_norm_compatibility)
        if self.dual_branch:
            image_features = np.concatenate(image_patch_batches, axis=0)
            image_memory_positions = np.concatenate(image_memory_position_batches if self.align_training_positions else image_position_batches, axis=0)
            self.image_subspace.fit(image_features)
            if self.anomaly_method != "pca":
                image_memory_features = np.concatenate(image_memory_patch_batches, axis=0) if self.align_training_positions else image_features
                self.image_memory.fit(image_memory_features, image_memory_positions)
                if self.anomaly_method in {"anoco", "pca_anoco", "pca_knn_anoco"}:
                    self.image_memory.fit_anoco_calibration(self.anoco_neighbors, self.anoco_query_weight, self.anoco_temperature, affinity=self.anoco_affinity, anchor_ranking=self.anoco_anchor_ranking, norm_compatibility=self.anoco_norm_compatibility)
                if self.anoco_layer_consensus:
                    layer_count = image_layer_batches[0].shape[0]
                    self.image_layer_memories = []
                    for layer_index in range(layer_count):
                        source = image_layer_memory_batches if self.align_training_positions else image_layer_batches
                        layer_features = np.concatenate([batch[layer_index] for batch in source], axis=0)
                        layer_memory = NormalPatchMemory(
                            self.normal_memory.max_patches, self.normal_memory.query_chunk_size,
                            backend=self.normal_memory.backend, device=self.normal_memory.device,
                            dtype=self.normal_memory.dtype, spatial_radius=self.normal_memory.spatial_radius,
                        )
                        layer_memory.fit(layer_features, image_memory_positions)
                        layer_memory.fit_anoco_calibration(
                            self.anoco_neighbors, self.anoco_query_weight, self.anoco_temperature,
                            affinity=self.anoco_affinity,
                            anchor_ranking=self.anoco_anchor_ranking,
                            norm_compatibility=self.anoco_norm_compatibility,
                        )
                        self.image_layer_memories.append(layer_memory)
        self.reference_grid = features.shape[1]
        return self

    def add_prototype(self, label: str, image_path, mask_path=None):
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        self._set_extractor_image_size(self.pixel_image_size)
        patches, grid = self.extractor.extract(image)
        if mask_path is None:
            prototype_patches = self._anomaly_patches(patches)
        else:
            mask_path = Path(mask_path)
            if not mask_path.is_file():
                raise FileNotFoundError(f"Prototype mask not found: {mask_path}")
            with Image.open(mask_path) as source:
                prototype_patches = self._masked_anomaly_patches(patches, grid, source.convert("L"))
        self.prototype_bank.add(label, prototype_patches)
        return self

    def _masked_anomaly_patches(self, patches, grid, mask):
        patches = np.asarray(patches)
        if len(patches) != grid[0] * grid[1]:
            raise ValueError("Prototype patch count must match its spatial grid")
        binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)
        if not binary.any():
            raise ValueError("Prototype mask is empty")
        coverage = np.asarray(
            Image.fromarray(binary, mode="F").resize(grid[::-1], Image.Resampling.BOX),
            dtype=np.float32,
        ).ravel()
        candidates = np.flatnonzero(coverage > 0)
        if not len(candidates):
            candidates = np.asarray([int(np.argmax(coverage))])
        scores = np.asarray(self.subspace.score(patches))
        keep = min(len(candidates), max(1, int(np.ceil(len(patches) * self.top_k_ratio))))
        if len(candidates) > keep:
            selected = np.argpartition(scores[candidates], -keep)[-keep:]
            candidates = candidates[selected]
        return patches[candidates]

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

    def _branch_scores(self, patches, subspace, memory, positions=None, method=None):
        method = self.anomaly_method if method is None else method
        pca_scores = subspace.score(patches)
        if method == "pca":
            return pca_scores, pca_scores, None, None
        if method in {"anoco", "pca_anoco"}:
            normal_scores = memory.score_anoco(patches, self.anoco_neighbors, self.anoco_query_weight, self.anoco_temperature, positions=positions, affinity=self.anoco_affinity, anchor_ranking=self.anoco_anchor_ranking, norm_compatibility=self.anoco_norm_compatibility)
            if method == "anoco":
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
        if method == "knn":
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
        method = "pca_knn" if self.anomaly_method == "pca_knn_anoco" else self.anomaly_method
        return self._branch_scores(patches, self.subspace, self.normal_memory, positions, method)

    def _secondary_anomaly_scores(self, patches, positions=None):
        method = "pca_knn" if self.anomaly_method == "pca_knn_anoco" else self.anomaly_method
        return self._branch_scores(patches, self.secondary_subspace, self.secondary_normal_memory, positions, method)

    @staticmethod
    def _resize_patch_scores(scores, source_grid, target_grid):
        if source_grid == target_grid:
            return np.asarray(scores, dtype=np.float64)
        image = Image.fromarray(np.asarray(scores, dtype=np.float32).reshape(source_grid), mode="F")
        resized = image.resize((target_grid[1], target_grid[0]), Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.float64).reshape(-1)

    def _image_score_bundle(self, patches, positions=None, layer_patches=None):
        subspace = self.image_subspace if self.dual_branch else self.subspace
        memory = self.image_memory if self.dual_branch else self.normal_memory
        method = "pca_anoco" if self.anomaly_method == "pca_knn_anoco" else self.anomaly_method
        if self.anoco_layer_consensus:
            if layer_patches is None or len(layer_patches) != len(self.image_layer_memories):
                raise ValueError("Per-layer image features do not match fitted ANoCo memories")
            pca_scores = subspace.score(patches)
            layer_evidence = []
            for features, layer_memory in zip(layer_patches, self.image_layer_memories):
                scores = layer_memory.score_anoco(
                    features, self.anoco_neighbors, self.anoco_query_weight,
                    self.anoco_temperature, positions=positions, affinity=self.anoco_affinity,
                    anchor_ranking=self.anoco_anchor_ranking,
                    norm_compatibility=self.anoco_norm_compatibility,
                )
                layer_evidence.append(layer_memory.calibrated_anoco(scores))
            consensus = np.median(np.stack(layer_evidence, axis=0), axis=0)
            pca_calibrated = subspace.calibrated(pca_scores)
            fused = (1.0 - self.anoco_weight) * pca_calibrated + self.anoco_weight * consensus
            return fused, pca_scores, consensus, None
        return self._branch_scores(patches, subspace, memory, positions, method)

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

    def _region_rejected_score(self, scores, grid):
        scores = np.asarray(scores, dtype=np.float64)
        if self.image_min_component_size <= 1 or self.image_score != "mtop1p":
            return self._aggregate_image_score(scores)
        keep = max(1, int(np.ceil(scores.size * self.image_top_ratio)))
        selected = np.argpartition(scores, -keep)[-keep:]
        candidate = np.zeros(scores.size, dtype=bool)
        candidate[selected] = True
        candidate = candidate.reshape(grid)
        accepted = np.zeros_like(candidate)
        visited = np.zeros_like(candidate)
        for row, column in np.argwhere(candidate):
            if visited[row, column]:
                continue
            component = []
            stack = [(int(row), int(column))]
            visited[row, column] = True
            while stack:
                current = stack.pop()
                component.append(current)
                for row_offset in (-1, 0, 1):
                    for column_offset in (-1, 0, 1):
                        if row_offset == column_offset == 0:
                            continue
                        next_row = current[0] + row_offset
                        next_column = current[1] + column_offset
                        if 0 <= next_row < grid[0] and 0 <= next_column < grid[1] and candidate[next_row, next_column] and not visited[next_row, next_column]:
                            visited[next_row, next_column] = True
                            stack.append((next_row, next_column))
            if len(component) >= self.image_min_component_size:
                for component_row, component_column in component:
                    accepted[component_row, component_column] = True
        return float(scores.reshape(grid)[accepted].mean()) if accepted.any() else float(scores.mean())

    def _image_anomaly_score(self, patches, positions=None, grid=None, score_bundle=None):
        if score_bundle is None:
            score_bundle = self._image_score_bundle(patches, positions)
        fused_scores, pca_scores, knn_scores, _ = score_bundle
        image_method = "pca_anoco" if self.anomaly_method == "pca_knn_anoco" else self.anomaly_method
        if self.image_fusion_stage == "patch" or image_method not in {"pca_knn", "pca_anoco"} or self.fusion_mode != "fixed":
            patch_scores = fused_scores
            base_score = self._region_rejected_score(patch_scores, grid) if grid is not None else self._aggregate_image_score(patch_scores)
        else:
            subspace = self.image_subspace if self.dual_branch else self.subspace
            memory = self.image_memory if self.dual_branch else self.normal_memory
            pca = subspace.calibrated(pca_scores)
            if image_method == "pca_anoco":
                normal = memory.calibrated_anoco(knn_scores)
                weight = self.anoco_weight
            else:
                normal = memory.calibrated(knn_scores)
                weight = self.knn_weight
            patch_scores = (1.0 - weight) * pca + weight * normal
            base_score = (1.0 - weight) * self._aggregate_image_score(pca) + weight * self._aggregate_image_score(normal)
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
        patches, image_patches, image_layers, grid, image_grid = self._extract_branches(image)
        if self.reference_grid is None:
            self.reference_grid = patches.shape[1]
        positions = self._patch_positions(grid)
        anomaly_scores, pca_scores, knn_scores, knn_gate = self._anomaly_scores(patches, positions)
        if self.secondary_pixel_image_size is not None:
            self._set_extractor_image_size(self.secondary_pixel_image_size)
            secondary_patches, secondary_grid = self.extractor.extract(image)
            secondary_positions = self._patch_positions(secondary_grid)
            secondary_scores, _, _, _ = self._secondary_anomaly_scores(secondary_patches, secondary_positions)
            secondary_scores = self._resize_patch_scores(secondary_scores, secondary_grid, grid)
            anomaly_scores = ((1.0 - self.pixel_multiscale_weight) * anomaly_scores
                              + self.pixel_multiscale_weight * secondary_scores)
        if self.dual_branch:
            image_positions = self._patch_positions(image_grid)
            image_score_bundle = self._image_score_bundle(image_patches, image_positions, image_layers)
        else:
            image_positions, image_grid = positions, grid
            image_score_bundle = (anomaly_scores, pca_scores, knn_scores, knn_gate)
        anomaly_map = self._postprocess_map(anomaly_scores.reshape(grid), image).tolist()
        fused_score, spatial_consistency = self._image_anomaly_score(
            image_patches if self.dual_branch else patches,
            image_positions,
            image_grid,
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
            "pixel_image_size": self.pixel_image_size,
            "image_head_image_size": self.image_head_image_size,
            "secondary_pixel_image_size": self.secondary_pixel_image_size,
            "pixel_multiscale_weight": self.pixel_multiscale_weight,
            "pca_anomaly_score": self._aggregate_image_score(pca_scores),
            "image_spatial_consistency": float(spatial_consistency),
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
            "image_min_component_size": self.image_min_component_size,
            "type_matching": self.type_matching,
            "map_postprocess": self.map_postprocess,
            "gaussian_sigma": self.gaussian_sigma,
            "anomaly_method": self.anomaly_method,
            "pca_residual_metric": self.pca_residual_metric,
            "knn_weight": self.knn_weight,
            "anoco_neighbors": self.anoco_neighbors,
            "anoco_query_weight": self.anoco_query_weight,
            "anoco_temperature": self.anoco_temperature,
            "anoco_affinity": self.anoco_affinity,
            "anoco_anchor_ranking": self.anoco_anchor_ranking,
            "anoco_norm_compatibility": self.anoco_norm_compatibility,
            "anoco_weight": self.anoco_weight,
            "anoco_layer_consensus": self.anoco_layer_consensus,
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
            "pixel_image_size": self.pixel_image_size,
            "image_head_image_size": self.image_head_image_size,
            "secondary_pixel_image_size": self.secondary_pixel_image_size,
            "pixel_multiscale_weight": self.pixel_multiscale_weight,
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
            if self.anoco_layer_consensus:
                state["image_layer_anoco"] = [
                    {
                        "center": memory.anoco_center,
                        "scale": memory.anoco_scale,
                        "calibration_scores": memory.anoco_calibration_scores.tolist(),
                    }
                    for memory in self.image_layer_memories
                ]
        if self.secondary_pixel_image_size is not None:
            state["secondary_subspace"] = self.secondary_subspace.to_dict()
            state["secondary_knn_center"] = self.secondary_normal_memory.center
            state["secondary_knn_scale"] = self.secondary_normal_memory.scale
            state["secondary_knn_calibration_scores"] = self.secondary_normal_memory.calibration_scores.tolist() if self.secondary_normal_memory.calibration_scores is not None else None
            state["secondary_anoco_center"] = self.secondary_normal_memory.anoco_center
            state["secondary_anoco_scale"] = self.secondary_normal_memory.anoco_scale
            state["secondary_anoco_calibration_scores"] = self.secondary_normal_memory.anoco_calibration_scores.tolist() if self.secondary_normal_memory.anoco_calibration_scores is not None else None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.normal_memory.features is not None:
            memory_path = path.with_suffix(path.suffix + ".normal-memory.npz")
            memory_data = {"features": self.normal_memory.features.astype(np.float16), "norms": self.normal_memory.norms.astype(np.float32)}
            if self.normal_memory.positions is not None:
                memory_data["positions"] = self.normal_memory.positions
            if self.dual_branch and self.image_memory.features is not None:
                memory_data["image_features"] = self.image_memory.features.astype(np.float16)
                memory_data["image_norms"] = self.image_memory.norms.astype(np.float32)
                if self.image_memory.positions is not None:
                    memory_data["image_positions"] = self.image_memory.positions
                for index, memory in enumerate(self.image_layer_memories):
                    memory_data[f"image_layer_{index}_features"] = memory.features.astype(np.float16)
                    memory_data[f"image_layer_{index}_norms"] = memory.norms.astype(np.float32)
                    if memory.positions is not None:
                        memory_data[f"image_layer_{index}_positions"] = memory.positions
            if self.secondary_normal_memory is not None and self.secondary_normal_memory.features is not None:
                memory_data["secondary_features"] = self.secondary_normal_memory.features.astype(np.float16)
                memory_data["secondary_norms"] = self.secondary_normal_memory.norms.astype(np.float32)
                if self.secondary_normal_memory.positions is not None:
                    memory_data["secondary_positions"] = self.secondary_normal_memory.positions
            np.savez_compressed(memory_path, **memory_data)
            state["normal_memory_file"] = memory_path.name
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path, extractor):
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(extractor, alpha=state.get("alpha", 0.5), unknown_threshold=state.get("unknown_threshold", 0.35), top_k_ratio=state.get("top_k_ratio", 0.05), image_score=state.get("image_score", "mean"), image_top_ratio=state.get("image_top_ratio", 0.01), image_fusion_stage=state.get("image_fusion_stage", "patch"), image_spatial_weight=state.get("image_spatial_weight", 0.0), image_min_component_size=state.get("image_min_component_size", 1), type_matching=state.get("type_matching", "prototype_mean"), map_postprocess=state.get("map_postprocess", "none"), gaussian_sigma=state.get("gaussian_sigma", 1.0), anomaly_method=state.get("anomaly_method", "pca"), pca_residual_metric=state.get("pca_residual_metric", "squared_l2"), knn_weight=state.get("knn_weight", 0.5), anoco_neighbors=state.get("anoco_neighbors", 16), anoco_query_weight=state.get("anoco_query_weight", 1.0), anoco_temperature=state.get("anoco_temperature", 0.07), anoco_affinity=state.get("anoco_affinity", "softmax"), anoco_anchor_ranking=state.get("anoco_anchor_ranking", "mean"), anoco_norm_compatibility=state.get("anoco_norm_compatibility", False), anoco_weight=state.get("anoco_weight", 0.5), anoco_layer_consensus=state.get("anoco_layer_consensus", False), memory_max_patches=state.get("memory_max_patches", 50000), knn_chunk_size=state.get("knn_chunk_size", 256), knn_backend=state.get("knn_backend", "auto"), knn_dtype=state.get("knn_dtype", "float32"), knn_spatial_radius=state.get("knn_spatial_radius", -1.0), align_training_positions=state.get("align_training_positions", False), dual_branch=state.get("dual_branch", False), fusion_mode=state.get("fusion_mode", "fixed"), gate_temperature=state.get("gate_temperature", 1.0), test_augmentations=state.get("test_augmentations", ()), pixel_image_size=state.get("pixel_image_size"), image_head_image_size=state.get("image_head_image_size"), secondary_pixel_image_size=state.get("secondary_pixel_image_size"), pixel_multiscale_weight=state.get("pixel_multiscale_weight", 0.5))
        obj.subspace = NormalSubspace.from_dict(state["subspace"])
        if obj.dual_branch and "image_subspace" in state:
            obj.image_subspace = NormalSubspace.from_dict(state["image_subspace"])
        if obj.secondary_pixel_image_size is not None and "secondary_subspace" in state:
            obj.secondary_subspace = NormalSubspace.from_dict(state["secondary_subspace"])
        obj.prototype_bank = PrototypeBank.from_dict(state.get("prototype_bank", {}))
        obj.prototype_bank.unknown_threshold = state.get("unknown_threshold", 0.35)
        memory_file = state.get("normal_memory_file")
        if memory_file:
            memory_path = Path(path).parent / memory_file
            memory = np.load(memory_path)
            obj.normal_memory.features = memory["features"].astype(np.float32)
            obj.normal_memory.norms = memory["norms"].astype(np.float32) if "norms" in memory else np.ones(len(obj.normal_memory.features), dtype=np.float32)
            obj.normal_memory.positions = memory["positions"].astype(np.float32) if "positions" in memory else None
            obj.normal_memory.center = float(state.get("knn_center", 0.0))
            obj.normal_memory.scale = float(state.get("knn_scale", 1.0))
            calibration = state.get("knn_calibration_scores")
            obj.normal_memory.calibration_scores = None if calibration is None else np.asarray(calibration, dtype=np.float64)
            obj.normal_memory.anoco_center = float(state.get("anoco_center", 0.0))
            obj.normal_memory.anoco_scale = float(state.get("anoco_scale", 1.0))
            anoco_calibration = state.get("anoco_calibration_scores")
            obj.normal_memory.anoco_calibration_scores = None if anoco_calibration is None else np.asarray(anoco_calibration, dtype=np.float64)
            if obj.secondary_normal_memory is not None and "secondary_features" in memory:
                obj.secondary_normal_memory.features = memory["secondary_features"].astype(np.float32)
                obj.secondary_normal_memory.norms = memory["secondary_norms"].astype(np.float32) if "secondary_norms" in memory else np.ones(len(obj.secondary_normal_memory.features), dtype=np.float32)
                obj.secondary_normal_memory.positions = memory["secondary_positions"].astype(np.float32) if "secondary_positions" in memory else None
                obj.secondary_normal_memory.center = float(state.get("secondary_knn_center", 0.0))
                obj.secondary_normal_memory.scale = float(state.get("secondary_knn_scale", 1.0))
                secondary_calibration = state.get("secondary_knn_calibration_scores")
                obj.secondary_normal_memory.calibration_scores = None if secondary_calibration is None else np.asarray(secondary_calibration, dtype=np.float64)
                obj.secondary_normal_memory.anoco_center = float(state.get("secondary_anoco_center", 0.0))
                obj.secondary_normal_memory.anoco_scale = float(state.get("secondary_anoco_scale", 1.0))
                secondary_anoco_calibration = state.get("secondary_anoco_calibration_scores")
                obj.secondary_normal_memory.anoco_calibration_scores = None if secondary_anoco_calibration is None else np.asarray(secondary_anoco_calibration, dtype=np.float64)
            if obj.dual_branch and "image_features" in memory:
                obj.image_memory.features = memory["image_features"].astype(np.float32)
                obj.image_memory.norms = memory["image_norms"].astype(np.float32) if "image_norms" in memory else np.ones(len(obj.image_memory.features), dtype=np.float32)
                obj.image_memory.positions = memory["image_positions"].astype(np.float32) if "image_positions" in memory else None
                obj.image_memory.center = float(state.get("image_knn_center", 0.0))
                obj.image_memory.scale = float(state.get("image_knn_scale", 1.0))
                image_calibration = state.get("image_knn_calibration_scores")
                obj.image_memory.calibration_scores = None if image_calibration is None else np.asarray(image_calibration, dtype=np.float64)
                obj.image_memory.anoco_center = float(state.get("image_anoco_center", 0.0))
                obj.image_memory.anoco_scale = float(state.get("image_anoco_scale", 1.0))
                image_anoco_calibration = state.get("image_anoco_calibration_scores")
                obj.image_memory.anoco_calibration_scores = None if image_anoco_calibration is None else np.asarray(image_anoco_calibration, dtype=np.float64)
                obj.image_layer_memories = []
                for index, layer_state in enumerate(state.get("image_layer_anoco", [])):
                    layer_memory = NormalPatchMemory(
                        obj.normal_memory.max_patches, obj.normal_memory.query_chunk_size,
                        backend=obj.normal_memory.backend, device=obj.normal_memory.device,
                        dtype=obj.normal_memory.dtype, spatial_radius=obj.normal_memory.spatial_radius,
                    )
                    layer_memory.features = memory[f"image_layer_{index}_features"].astype(np.float32)
                    norm_key = f"image_layer_{index}_norms"
                    layer_memory.norms = memory[norm_key].astype(np.float32) if norm_key in memory else np.ones(len(layer_memory.features), dtype=np.float32)
                    position_key = f"image_layer_{index}_positions"
                    layer_memory.positions = memory[position_key].astype(np.float32) if position_key in memory else None
                    layer_memory.anoco_center = float(layer_state["center"])
                    layer_memory.anoco_scale = float(layer_state["scale"])
                    layer_memory.anoco_calibration_scores = np.asarray(layer_state["calibration_scores"], dtype=np.float64)
                    obj.image_layer_memories.append(layer_memory)
        obj.reference_grid = state.get("reference_grid")
        obj.reference_shape = state.get("reference_shape")
        return obj
