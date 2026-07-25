from __future__ import annotations
import numpy as np


class NormalPatchMemory:
    def __init__(self, max_patches=50000, query_chunk_size=256, calibration_patches=4096, backend="auto", device=None, dtype="float32"):
        self.max_patches = int(max_patches)
        self.query_chunk_size = int(query_chunk_size)
        self.calibration_patches = int(calibration_patches)
        if backend not in {"auto", "numpy", "torch"}:
            raise ValueError("kNN backend must be auto, numpy, or torch")
        if dtype not in {"float32", "float16"}:
            raise ValueError("kNN dtype must be float32 or float16")
        self.backend = backend
        self.device = None if device is None else str(device)
        self.dtype = dtype
        self.features = None
        self._torch_bank = None
        self.center = 0.0
        self.scale = 1.0
        self.calibration_scores = None

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

    def fit(self, features):
        bank = self._normalize(features)
        if len(bank) < 2:
            raise ValueError("Normal patch kNN requires at least two reference patches")
        if self.max_patches > 0 and len(bank) > self.max_patches:
            indices = np.linspace(0, len(bank) - 1, self.max_patches, dtype=np.int64)
            bank = bank[indices]
        self.features = np.ascontiguousarray(bank)
        self._torch_bank = None
        sample_count = min(len(bank), max(2, self.calibration_patches))
        sample_indices = np.linspace(0, len(bank) - 1, sample_count, dtype=np.int64)
        calibration = self.score(bank[sample_indices], exclude_indices=sample_indices)
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

    def _score_numpy(self, query, exclude):
        scores = np.empty(len(query), dtype=np.float32)
        for start in range(0, len(query), self.query_chunk_size):
            end = min(start + self.query_chunk_size, len(query))
            similarity = query[start:end] @ self.features.T
            if exclude is not None:
                rows = np.arange(end - start)
                similarity[rows, exclude[start:end]] = -np.inf
            scores[start:end] = 1.0 - similarity.max(axis=1)
        return scores

    def _score_torch(self, query, exclude):
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
        scores = np.empty(len(query), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(query), self.query_chunk_size):
                end = min(start + self.query_chunk_size, len(query))
                query_chunk = torch.as_tensor(query[start:end], device=device, dtype=dtype)
                similarity = query_chunk @ self._torch_bank.T
                if exclude is not None:
                    rows = torch.arange(end - start, device=device)
                    columns = torch.as_tensor(exclude[start:end], device=device)
                    similarity[rows, columns] = -torch.inf
                scores[start:end] = (1.0 - similarity.max(dim=1).values).float().cpu().numpy()
        return scores

    def score(self, features, exclude_indices=None):
        if self.features is None or len(self.features) == 0:
            raise ValueError("Normal patch memory has not been fitted")
        query = self._normalize(features)
        exclude = None if exclude_indices is None else np.asarray(exclude_indices)
        if self._resolved_backend() == "torch":
            return self._score_torch(query, exclude)
        return self._score_numpy(query, exclude)

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
    def __init__(self, explained_variance=0.99, max_components=None):
        self.explained_variance = explained_variance; self.max_components = max_components
        self.mean = self.components = None
        self.score_center = 0.0; self.score_scale = 1.0
        self.calibration_scores = None
    def fit(self, features):
        x = np.asarray(features, dtype=np.float64); self.mean = x.mean(0)
        _, s, vt = np.linalg.svd(x - self.mean, full_matrices=False); var = s * s
        keep = np.searchsorted(np.cumsum(var) / max(var.sum(), 1e-12), self.explained_variance) + 1
        if self.max_components: keep = min(keep, self.max_components)
        self.components = vt[:max(1, keep)]
        scores = self.score(x)
        self.score_center, self.score_scale = NormalPatchMemory._robust_stats(scores)
        sample_count = min(len(scores), 4096)
        sample_indices = np.linspace(0, len(scores) - 1, sample_count, dtype=np.int64)
        self.calibration_scores = np.sort(np.asarray(scores[sample_indices], dtype=np.float64))
        return self
    def score(self, features):
        z = np.asarray(features, dtype=np.float64) - self.mean; recon = (z @ self.components.T) @ self.components
        return np.sum((z - recon) ** 2, axis=1)
    def calibrated(self, scores): return (np.asarray(scores, dtype=np.float64) - self.score_center) / self.score_scale
    def tail_evidence(self, scores): return NormalPatchMemory._tail_evidence(scores, self.calibration_scores)
    def to_dict(self): return {"mean": self.mean.tolist(), "components": self.components.tolist(), "score_center": self.score_center, "score_scale": self.score_scale, "calibration_scores": self.calibration_scores.tolist() if self.calibration_scores is not None else None}
    @classmethod
    def from_dict(cls, d):
        obj = cls(); obj.mean = np.asarray(d["mean"]); obj.components = np.asarray(d["components"]); obj.score_center = float(d.get("score_center", 0.0)); obj.score_scale = float(d.get("score_scale", 1.0)); calibration = d.get("calibration_scores"); obj.calibration_scores = None if calibration is None else np.asarray(calibration, dtype=np.float64); return obj

class PrototypeBank:
    def __init__(self, unknown_threshold=0.35):
        self.prototypes = {}; self.patch_banks = {}; self.counts = {}; self.unknown_threshold = unknown_threshold
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
