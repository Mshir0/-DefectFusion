from __future__ import annotations
import numpy as np

class NormalSubspace:
    def __init__(self, explained_variance=0.99, max_components=None):
        self.explained_variance = explained_variance; self.max_components = max_components
        self.mean = self.components = None
    def fit(self, features):
        x = np.asarray(features, dtype=np.float64); self.mean = x.mean(0)
        _, s, vt = np.linalg.svd(x - self.mean, full_matrices=False); var = s * s
        keep = np.searchsorted(np.cumsum(var) / max(var.sum(), 1e-12), self.explained_variance) + 1
        if self.max_components: keep = min(keep, self.max_components)
        self.components = vt[:max(1, keep)]; return self
    def score(self, features):
        z = np.asarray(features, dtype=np.float64) - self.mean; recon = (z @ self.components.T) @ self.components
        return np.sum((z - recon) ** 2, axis=1)
    def to_dict(self): return {"mean": self.mean.tolist(), "components": self.components.tolist()}
    @classmethod
    def from_dict(cls, d):
        obj = cls(); obj.mean = np.asarray(d["mean"]); obj.components = np.asarray(d["components"]); return obj

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
