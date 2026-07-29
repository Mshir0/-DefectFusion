from __future__ import annotations
import numpy as np


class NormalPatchMemory:
    def __init__(self, max_patches=50000, query_chunk_size=256, calibration_patches=4096, backend="auto", device=None, dtype="float32", spatial_radius=-1.0):
        self.max_patches = int(max_patches)
        self.query_chunk_size = int(query_chunk_size)
        self.calibration_patches = int(calibration_patches)
        if backend not in {"auto", "numpy", "torch"}:
            raise ValueError("kNN backend must be auto, numpy, or torch")
        if dtype not in {"float32", "float16"}:
            raise ValueError("kNN dtype must be float32 or float16")
        if spatial_radius != -1 and not 0 <= spatial_radius <= 1:
            raise ValueError("spatial_radius must be -1 (global) or in [0, 1]")
        self.backend = backend
        self.device = None if device is None else str(device)
        self.dtype = dtype
        self.spatial_radius = float(spatial_radius)
        self.features = None
        self.positions = None
        self.view_ids = None
        self._view_groups = ()
        self._torch_bank = None
        self._torch_positions = None
        self.center = 0.0
        self.scale = 1.0
        self.calibration_scores = None
        self.anoco_center = 0.0
        self.anoco_scale = 1.0
        self.anoco_calibration_scores = None

    @staticmethod
    def _normalize(features):
        x = np.asarray(features, dtype=np.float32)
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    @staticmethod
    def _robust_stats(scores):
        scores = np.asarray(scores, dtype=np.float64)
        center = float(np.median(scores))
        mad_scale = float(1.4826 * np.median(np.abs(scores - center)))
        std_floor = float(scores.std() * 0.1)
        return center, max(mad_scale, std_floor, 1e-12)

    def set_view_ids(self, view_ids):
        self.view_ids = None if view_ids is None else np.ascontiguousarray(view_ids, dtype=np.int32)
        self._view_groups = () if self.view_ids is None else tuple(
            np.flatnonzero(self.view_ids == view_id) for view_id in np.unique(self.view_ids)
        )

    def fit(self, features, positions=None, view_ids=None):
        bank = self._normalize(features)
        if len(bank) < 2:
            raise ValueError("Normal patch kNN requires at least two reference patches")
        if positions is not None:
            positions = np.asarray(positions, dtype=np.float32)
            if positions.shape != (len(bank), 2):
                raise ValueError("Normal patch positions must have shape (patches, 2)")
        if view_ids is not None:
            view_ids = np.asarray(view_ids, dtype=np.int32)
            if view_ids.shape != (len(bank),):
                raise ValueError("Normal patch view_ids must have shape (patches,)")
        if self.max_patches > 0 and len(bank) > self.max_patches:
            indices = np.linspace(0, len(bank) - 1, self.max_patches, dtype=np.int64)
            bank = bank[indices]
            if positions is not None:
                positions = positions[indices]
            if view_ids is not None:
                view_ids = view_ids[indices]
        self.features = np.ascontiguousarray(bank)
        self.positions = None if positions is None else np.ascontiguousarray(positions)
        self.set_view_ids(view_ids)
        self._torch_bank = None
        self._torch_positions = None
        sample_count = min(len(bank), max(2, self.calibration_patches))
        sample_indices = np.linspace(0, len(bank) - 1, sample_count, dtype=np.int64)
        calibration_positions = None if self.positions is None else self.positions[sample_indices]
        calibration = self.score(bank[sample_indices], exclude_indices=sample_indices, positions=calibration_positions)
        self.center, self.scale = self._robust_stats(calibration)
        self.calibration_scores = np.sort(np.asarray(calibration, dtype=np.float64))
        return self

    def _resolved_backend(self):
        if self.backend != "auto":
            return self.backend
        if self.device and self.device.startswith("cuda"):
            try:
                import torch
                if torch.cuda.is_available():
                    return "torch"
            except ImportError:
                pass
        return "numpy"

    @property
    def resolved_backend(self):
        return self._resolved_backend()

    def _score_numpy(self, query, exclude, positions):
        scores = np.empty(len(query), dtype=np.float32)
        for start in range(0, len(query), self.query_chunk_size):
            end = min(start + self.query_chunk_size, len(query))
            similarity = query[start:end] @ self.features.T
            if self.spatial_radius >= 0 and positions is not None and self.positions is not None:
                distance = np.max(np.abs(positions[start:end, None, :] - self.positions[None, :, :]), axis=2)
                allowed = distance <= self.spatial_radius
            else:
                allowed = np.ones_like(similarity, dtype=bool)
            if exclude is not None:
                rows = np.arange(end - start)
                allowed[rows, exclude[start:end]] = False
            empty = ~allowed.any(axis=1)
            if np.any(empty):
                allowed[empty] = True
                if exclude is not None:
                    allowed[np.where(empty)[0], exclude[start:end][empty]] = False
            similarity[~allowed] = -np.inf
            scores[start:end] = 1.0 - similarity.max(axis=1)
        return scores

    def _score_torch(self, query, exclude, positions):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Torch kNN backend requires PyTorch") from exc
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Torch kNN backend requested CUDA, but CUDA is unavailable")
        dtype = torch.float16 if self.dtype == "float16" else torch.float32
        if self._torch_bank is None or self._torch_bank.device != torch.device(device) or self._torch_bank.dtype != dtype:
            self._torch_bank = torch.as_tensor(self.features, device=device, dtype=dtype).contiguous()
        if self.positions is not None and (self._torch_positions is None or self._torch_positions.device != torch.device(device)):
            self._torch_positions = torch.as_tensor(self.positions, device=device, dtype=torch.float32).contiguous()
        scores = np.empty(len(query), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(query), self.query_chunk_size):
                end = min(start + self.query_chunk_size, len(query))
                query_chunk = torch.as_tensor(query[start:end], device=device, dtype=dtype)
                similarity = query_chunk @ self._torch_bank.T
                if self.spatial_radius >= 0 and positions is not None and self.positions is not None:
                    query_positions = torch.as_tensor(positions[start:end], device=device, dtype=torch.float32)
                    distance = torch.amax(torch.abs(query_positions[:, None, :] - self._torch_positions[None, :, :]), dim=2)
                    allowed = distance <= self.spatial_radius
                else:
                    allowed = torch.ones_like(similarity, dtype=torch.bool)
                if exclude is not None:
                    rows = torch.arange(end - start, device=device)
                    columns = torch.as_tensor(exclude[start:end], device=device)
                    allowed[rows, columns] = False
                empty = ~allowed.any(dim=1)
                if empty.any():
                    allowed[empty] = True
                    if exclude is not None:
                        allowed[rows[empty], columns[empty]] = False
                similarity[~allowed] = -torch.inf
                scores[start:end] = (1.0 - similarity.max(dim=1).values).float().cpu().numpy()
        return scores

    def score(self, features, exclude_indices=None, positions=None):
        if self.features is None or len(self.features) == 0:
            raise ValueError("Normal patch memory has not been fitted")
        query = self._normalize(features)
        exclude = None if exclude_indices is None else np.asarray(exclude_indices)
        if positions is not None:
            positions = np.asarray(positions, dtype=np.float32)
            if positions.shape != (len(query), 2):
                raise ValueError("Query patch positions must have shape (patches, 2)")
        if self.spatial_radius >= 0 and (positions is None or self.positions is None):
            raise ValueError("Spatial kNN requires patch positions for normal and query features")
        if self._resolved_backend() == "torch":
            return self._score_torch(query, exclude, positions)
        return self._score_numpy(query, exclude, positions)

    def _anoco_score_numpy(self, query, exclude, positions, neighbor_count, query_weight, temperature, view_balance=False):
        scores = np.empty(len(query), dtype=np.float32)
        for start in range(0, len(query), self.query_chunk_size):
            end = min(start + self.query_chunk_size, len(query))
            query_chunk = query[start:end]
            similarity = query_chunk @ self.features.T
            if self.spatial_radius >= 0 and positions is not None and self.positions is not None:
                distance = np.max(np.abs(positions[start:end, None, :] - self.positions[None, :, :]), axis=2)
                allowed = distance <= self.spatial_radius
            else:
                allowed = np.ones_like(similarity, dtype=bool)
            if exclude is not None:
                rows = np.arange(end - start)
                allowed[rows, exclude[start:end]] = False
            empty = ~allowed.any(axis=1)
            if np.any(empty):
                allowed[empty] = True
                if exclude is not None:
                    allowed[np.where(empty)[0], exclude[start:end][empty]] = False
            masked_similarity = np.where(allowed, similarity, -np.inf)
            anchors = np.argmax(masked_similarity, axis=1)
            anchor_similarity = self.features[anchors] @ self.features.T
            joint = np.where(allowed, 0.5 * (similarity + anchor_similarity), -np.inf)
            if view_balance and len(self._view_groups) > 1:
                candidates = []
                for columns in self._view_groups:
                    local = np.argmax(joint[:, columns], axis=1)
                    candidates.append(columns[local])
                candidate_indices = np.stack(candidates, axis=1)
                candidate_scores = np.take_along_axis(joint, candidate_indices, axis=1)
                keep = min(neighbor_count, candidate_scores.shape[1])
                selected = np.argpartition(candidate_scores, -keep, axis=1)[:, -keep:]
                neighbor_indices = np.take_along_axis(candidate_indices, selected, axis=1)
            else:
                keep = min(neighbor_count, joint.shape[1])
                neighbor_indices = np.argpartition(joint, -keep, axis=1)[:, -keep:]
            neighbor_similarity = np.take_along_axis(similarity, neighbor_indices, axis=1)
            neighbor_allowed = np.take_along_axis(allowed, neighbor_indices, axis=1)
            logits = neighbor_similarity / temperature
            logits = np.where(neighbor_allowed, logits, -np.inf)
            logits -= np.max(logits, axis=1, keepdims=True)
            weights = np.where(neighbor_allowed, np.exp(logits), 0.0)
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
            normal_target = np.sum(self.features[neighbor_indices] * weights[:, :, None], axis=1)
            updated = (query_weight * query_chunk + normal_target) / (query_weight + 1.0)
            displacement = np.sum((updated - query_chunk) ** 2, axis=1)
            updated_normalized = updated / np.maximum(np.linalg.norm(updated, axis=1, keepdims=True), 1e-12)
            angular = 1.0 - np.sum(updated_normalized * query_chunk, axis=1)
            scores[start:end] = displacement * np.maximum(angular, 0.0)
        return scores

    def _anoco_score_torch(self, query, exclude, positions, neighbor_count, query_weight, temperature, view_balance=False):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Torch ANoCo backend requires PyTorch") from exc
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Torch ANoCo backend requested CUDA, but CUDA is unavailable")
        dtype = torch.float16 if self.dtype == "float16" else torch.float32
        if self._torch_bank is None or self._torch_bank.device != torch.device(device) or self._torch_bank.dtype != dtype:
            self._torch_bank = torch.as_tensor(self.features, device=device, dtype=dtype).contiguous()
        if self.positions is not None and (self._torch_positions is None or self._torch_positions.device != torch.device(device)):
            self._torch_positions = torch.as_tensor(self.positions, device=device, dtype=torch.float32).contiguous()
        scores = np.empty(len(query), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(query), self.query_chunk_size):
                end = min(start + self.query_chunk_size, len(query))
                query_chunk = torch.as_tensor(query[start:end], device=device, dtype=dtype)
                similarity = query_chunk @ self._torch_bank.T
                if self.spatial_radius >= 0 and positions is not None and self.positions is not None:
                    query_positions = torch.as_tensor(positions[start:end], device=device, dtype=torch.float32)
                    distance = torch.amax(torch.abs(query_positions[:, None, :] - self._torch_positions[None, :, :]), dim=2)
                    allowed = distance <= self.spatial_radius
                else:
                    allowed = torch.ones_like(similarity, dtype=torch.bool)
                if exclude is not None:
                    rows = torch.arange(end - start, device=device)
                    columns = torch.as_tensor(exclude[start:end], device=device)
                    allowed[rows, columns] = False
                empty = ~allowed.any(dim=1)
                if empty.any():
                    allowed[empty] = True
                    if exclude is not None:
                        allowed[rows[empty], columns[empty]] = False
                masked_similarity = similarity.masked_fill(~allowed, -torch.inf)
                anchors = masked_similarity.argmax(dim=1)
                anchor_similarity = self._torch_bank[anchors] @ self._torch_bank.T
                joint = (0.5 * (similarity + anchor_similarity)).masked_fill(~allowed, -torch.inf)
                if view_balance and len(self._view_groups) > 1:
                    candidates = []
                    for group in self._view_groups:
                        columns = torch.as_tensor(group, device=device)
                        local = torch.argmax(joint[:, columns], dim=1)
                        candidates.append(columns[local])
                    candidate_indices = torch.stack(candidates, dim=1)
                    candidate_scores = torch.gather(joint, 1, candidate_indices)
                    keep = min(neighbor_count, candidate_scores.shape[1])
                    selected = torch.topk(candidate_scores, keep, dim=1).indices
                    neighbor_indices = torch.gather(candidate_indices, 1, selected)
                else:
                    keep = min(neighbor_count, joint.shape[1])
                    neighbor_indices = torch.topk(joint, keep, dim=1).indices
                neighbor_similarity = torch.gather(similarity, 1, neighbor_indices)
                neighbor_allowed = torch.gather(allowed, 1, neighbor_indices)
                logits = (neighbor_similarity.float() / temperature).masked_fill(~neighbor_allowed, -torch.inf)
                weights = torch.softmax(logits, dim=1)
                normal_features = self._torch_bank[neighbor_indices].float()
                normal_target = torch.sum(normal_features * weights[:, :, None], dim=1)
                query_float = query_chunk.float()
                updated = (query_weight * query_float + normal_target) / (query_weight + 1.0)
                displacement = torch.sum((updated - query_float) ** 2, dim=1)
                angular = 1.0 - torch.sum(torch.nn.functional.normalize(updated, p=2, dim=1) * query_float, dim=1)
                scores[start:end] = (displacement * torch.clamp(angular, min=0.0)).cpu().numpy()
        return scores

    def score_anoco(self, features, neighbor_count=16, query_weight=1.0, temperature=0.07, view_balance=False, exclude_indices=None, positions=None):
        if self.features is None or len(self.features) == 0:
            raise ValueError("Normal patch memory has not been fitted")
        if neighbor_count <= 0:
            raise ValueError("ANoCo neighbor_count must be positive")
        if query_weight <= 0:
            raise ValueError("ANoCo query_weight must be positive")
        if temperature <= 0:
            raise ValueError("ANoCo temperature must be positive")
        query = self._normalize(features)
        exclude = None if exclude_indices is None else np.asarray(exclude_indices)
        if positions is not None:
            positions = np.asarray(positions, dtype=np.float32)
            if positions.shape != (len(query), 2):
                raise ValueError("Query patch positions must have shape (patches, 2)")
        if self.spatial_radius >= 0 and (positions is None or self.positions is None):
            raise ValueError("Spatial ANoCo requires patch positions for normal and query features")
        if self._resolved_backend() == "torch":
            return self._anoco_score_torch(query, exclude, positions, neighbor_count, query_weight, temperature, view_balance)
        return self._anoco_score_numpy(query, exclude, positions, neighbor_count, query_weight, temperature, view_balance)

    def fit_anoco_calibration(self, neighbor_count=16, query_weight=1.0, temperature=0.07, view_balance=False, calibration_patches=1024):
        sample_count = min(len(self.features), max(2, int(calibration_patches)))
        sample_indices = np.linspace(0, len(self.features) - 1, sample_count, dtype=np.int64)
        calibration_positions = None if self.positions is None else self.positions[sample_indices]
        calibration = self.score_anoco(
            self.features[sample_indices],
            neighbor_count=neighbor_count,
            query_weight=query_weight,
            temperature=temperature,
            view_balance=view_balance,
            exclude_indices=sample_indices,
            positions=calibration_positions,
        )
        self.anoco_center, self.anoco_scale = self._robust_stats(calibration)
        self.anoco_calibration_scores = np.sort(np.asarray(calibration, dtype=np.float64))
        return self

    def calibrated_anoco(self, scores):
        return (np.asarray(scores, dtype=np.float64) - self.anoco_center) / self.anoco_scale

    def tail_evidence_anoco(self, scores):
        return self._tail_evidence(scores, self.anoco_calibration_scores)

    def calibrated(self, scores):
        return (np.asarray(scores, dtype=np.float64) - self.center) / self.scale

    def tail_evidence(self, scores):
        return self._tail_evidence(scores, self.calibration_scores)

    @staticmethod
    def _tail_evidence(scores, calibration_scores):
        reference = np.asarray(calibration_scores, dtype=np.float64)
        if reference.size < 2:
            raise ValueError("Tail calibration requires at least two normal scores")
        reference = np.sort(reference)
        ranks = np.arange(1, len(reference) + 1, dtype=np.float64) / (len(reference) + 1)
        values = np.asarray(scores, dtype=np.float64)
        cdf = np.interp(values, reference, ranks, left=0.0, right=ranks[-1])
        evidence = -np.log(np.maximum(1.0 - cdf, 1.0 / (len(reference) + 1)))
        above = values > reference[-1]
        if np.any(above):
            tail_index = max(0, len(reference) - min(16, len(reference)))
            tail_scale = max(reference[-1] - reference[tail_index], np.std(reference) * 0.1, 1e-12)
            evidence = np.asarray(evidence)
            evidence[above] += (values[above] - reference[-1]) / tail_scale
        return evidence

class NormalSubspace:
    def __init__(self, explained_variance=0.99, max_components=None, residual_metric="squared_l2", covariance_shrinkage=0.1):
        if residual_metric not in {"squared_l2", "mahalanobis"}:
            raise ValueError("residual_metric must be squared_l2 or mahalanobis")
        if not 0 <= covariance_shrinkage <= 1:
            raise ValueError("covariance_shrinkage must be in [0, 1]")
        self.explained_variance = explained_variance; self.max_components = max_components
        self.residual_metric = residual_metric
        self.covariance_shrinkage = float(covariance_shrinkage)
        self.mean = self.components = None
        self.residual_variance = None
        self.score_center = 0.0; self.score_scale = 1.0
        self.calibration_scores = None
    def fit(self, features):
        x = np.asarray(features, dtype=np.float64); self.mean = x.mean(0)
        _, s, vt = np.linalg.svd(x - self.mean, full_matrices=False); var = s * s
        keep = np.searchsorted(np.cumsum(var) / max(var.sum(), 1e-12), self.explained_variance) + 1
        if self.max_components: keep = min(keep, self.max_components)
        self.components = vt[:max(1, keep)]
        residual = self._residual(x)
        variance = np.mean(residual ** 2, axis=0)
        target = float(np.mean(variance))
        variance = (1.0 - self.covariance_shrinkage) * variance + self.covariance_shrinkage * target
        self.residual_variance = np.maximum(variance, max(target * 1e-6, 1e-12))
        scores = self.score(x)
        self.score_center, self.score_scale = NormalPatchMemory._robust_stats(scores)
        sample_count = min(len(scores), 4096)
        sample_indices = np.linspace(0, len(scores) - 1, sample_count, dtype=np.int64)
        self.calibration_scores = np.sort(np.asarray(scores[sample_indices], dtype=np.float64))
        return self
    def _residual(self, features):
        z = np.asarray(features, dtype=np.float64) - self.mean; recon = (z @ self.components.T) @ self.components
        return z - recon
    def score(self, features):
        residual = self._residual(features)
        if self.residual_metric == "mahalanobis":
            if self.residual_variance is None:
                raise ValueError("Mahalanobis residual requires a fitted residual variance")
            return np.mean((residual ** 2) / self.residual_variance, axis=1)
        return np.sum(residual ** 2, axis=1)
    def calibrated(self, scores): return (np.asarray(scores, dtype=np.float64) - self.score_center) / self.score_scale
    def tail_evidence(self, scores): return NormalPatchMemory._tail_evidence(scores, self.calibration_scores)
    def to_dict(self): return {"mean": self.mean.tolist(), "components": self.components.tolist(), "residual_metric": self.residual_metric, "covariance_shrinkage": self.covariance_shrinkage, "residual_variance": self.residual_variance.tolist() if self.residual_variance is not None else None, "score_center": self.score_center, "score_scale": self.score_scale, "calibration_scores": self.calibration_scores.tolist() if self.calibration_scores is not None else None}
    @classmethod
    def from_dict(cls, d):
        obj = cls(residual_metric=d.get("residual_metric", "squared_l2"), covariance_shrinkage=d.get("covariance_shrinkage", 0.1)); obj.mean = np.asarray(d["mean"]); obj.components = np.asarray(d["components"]); variance = d.get("residual_variance"); obj.residual_variance = None if variance is None else np.asarray(variance, dtype=np.float64); obj.score_center = float(d.get("score_center", 0.0)); obj.score_scale = float(d.get("score_scale", 1.0)); calibration = d.get("calibration_scores"); obj.calibration_scores = None if calibration is None else np.asarray(calibration, dtype=np.float64); return obj

class PrototypeBank:
    def __init__(self, unknown_threshold=0.35):
        self.prototypes = {}; self.patch_banks = {}; self.counts = {}; self.unknown_threshold = unknown_threshold
        self._svm = None
    def add(self, label, features):
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 0:
            raise ValueError("Prototype features must be a vector or a matrix")
        p = x if x.ndim == 1 else x.mean(axis=0)
        p = p.copy(); p /= max(np.linalg.norm(p), 1e-12)
        count = self.counts.get(label, 0)
        if count: p = (self.prototypes[label] * count + p) / (count + 1); p /= max(np.linalg.norm(p), 1e-12)
        self.prototypes[label] = p; self.counts[label] = count + 1
        patches = x[None, :] if x.ndim == 1 else x
        patches = patches / np.maximum(np.linalg.norm(patches, axis=1, keepdims=True), 1e-12)
        self.patch_banks[label] = np.concatenate([self.patch_banks[label], patches], axis=0) if label in self.patch_banks else patches
        self._svm = None
    def predict(self, feature):
        if not self.prototypes: return "unknown", 0.0
        x = np.asarray(feature, dtype=np.float64)
        if x.ndim == 2 and self.patch_banks:
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
            candidates = []
            for label, reference in self.patch_banks.items():
                similarity = x @ reference.T
                forward = similarity.max(axis=1).mean()
                backward = similarity.max(axis=0).mean()
                candidates.append((label, float(0.5 * (forward + backward))))
            label, score = max(candidates, key=lambda kv: kv[1])
        else:
            x = x.mean(axis=0) if x.ndim == 2 else x
            x /= max(np.linalg.norm(x), 1e-12)
            label, score = max(((k, float(x @ v)) for k, v in self.prototypes.items()), key=lambda kv: kv[1])
        return (label, score) if score >= self.unknown_threshold else ("unknown", score)
    def predict_rbf_svm(self, features):
        if not self.patch_banks:
            return "unknown", 0.0
        labels = sorted(self.patch_banks)
        if len(labels) == 1:
            return labels[0], 1.0
        if self._svm is None:
            from sklearn.svm import SVC
            training = np.concatenate([self.patch_banks[label] for label in labels], axis=0)
            targets = np.concatenate([
                np.full(len(self.patch_banks[label]), label, dtype=object)
                for label in labels
            ])
            self._svm = SVC(
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                decision_function_shape="ovr",
            )
            self._svm.fit(training, targets)
        query = np.asarray(features, dtype=np.float64)
        query = query[None, :] if query.ndim == 1 else query
        query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
        margins = np.asarray(self._svm.decision_function(query), dtype=np.float64)
        if margins.ndim == 1:
            margins = np.stack([-margins, margins], axis=1)
        margins -= margins.max(axis=1, keepdims=True)
        probabilities = (np.exp(margins) / np.exp(margins).sum(axis=1, keepdims=True)).mean(axis=0)
        index = int(np.argmax(probabilities))
        label = str(self._svm.classes_[index])
        score = float(probabilities[index])
        return (label, score) if score >= self.unknown_threshold else ("unknown", score)
    def to_dict(self): return {"prototypes": {k: v.tolist() for k, v in self.prototypes.items()}, "patch_banks": {k: v.tolist() for k, v in self.patch_banks.items()}, "counts": self.counts}
    @classmethod
    def from_dict(cls, d):
        b = cls()
        if "prototypes" in d:
            b.prototypes = {k: np.asarray(v) for k, v in d["prototypes"].items()}
            b.patch_banks = {k: np.asarray(v) for k, v in d.get("patch_banks", {}).items()}
            b.counts = {k: int(v) for k, v in d.get("counts", {}).items()}
        else:
            b.prototypes = {k: np.asarray(v) for k, v in d.items()}; b.counts = {k: 1 for k in d}
        return b
