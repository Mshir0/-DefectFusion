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
    def __init__(self, unknown_threshold=0.35): self.prototypes = {}; self.counts = {}; self.unknown_threshold = unknown_threshold
    def add(self, label, features):
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 0:
            raise ValueError("Prototype features must be a vector or a matrix")
        p = x if x.ndim == 1 else x.mean(axis=0)
        p = p.copy(); p /= max(np.linalg.norm(p), 1e-12)
        count = self.counts.get(label, 0)
        if count: p = (self.prototypes[label] * count + p) / (count + 1); p /= max(np.linalg.norm(p), 1e-12)
        self.prototypes[label] = p; self.counts[label] = count + 1
    def predict(self, feature):
        if not self.prototypes: return "unknown", 0.0
        x = np.asarray(feature, dtype=np.float64); x /= max(np.linalg.norm(x), 1e-12)
        label, score = max(((k, float(x @ v)) for k, v in self.prototypes.items()), key=lambda kv: kv[1])
        return (label, score) if score >= self.unknown_threshold else ("unknown", score)
    def to_dict(self): return {k: v.tolist() for k, v in self.prototypes.items()}
    @classmethod
    def from_dict(cls, d):
        b = cls(); b.prototypes = {k: np.asarray(v) for k, v in d.items()}; b.counts = {k: 1 for k in d}; return b
