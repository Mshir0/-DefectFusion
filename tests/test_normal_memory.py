import tempfile
import unittest
from pathlib import Path

import numpy as np

from defectfusion.model import NormalPatchMemory, NormalSubspace
from defectfusion.pipeline import DefectFusion


class NormalPatchMemoryTest(unittest.TestCase):
    def test_independent_layer_scores_are_calibrated_then_averaged(self):
        extractor = type("Extractor", (), {"debias": False, "layer_aggregation": "mean"})()
        fusion = DefectFusion(extractor, layer_fusion="score_mean")
        normal_a = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
        normal_b = np.array([[0, 0], [2, 0], [0, 2], [2, 2]], dtype=np.float32)
        fusion.layer_subspaces = [NormalSubspace(max_components=1).fit(normal_a), NormalSubspace(max_components=1).fit(normal_b)]
        query_a = np.array([[2, 2], [0.5, 0.5]], dtype=np.float32)
        query_b = np.array([[4, 4], [1, 1]], dtype=np.float32)
        scores, calibrated = fusion._pca_patch_scores(
            np.zeros_like(query_a), [query_a, query_b]
        )
        expected = np.stack([
            subspace.calibrated(subspace.score(query))
            for subspace, query in zip(fusion.layer_subspaces, [query_a, query_b])
        ]).mean(axis=0)
        self.assertTrue(calibrated)
        np.testing.assert_allclose(scores, expected)

    def test_numpy_backend_matches_auto_cpu(self):
        normal = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
        query = np.array([[1, -1], [-1, 1]], dtype=np.float32)
        automatic = NormalPatchMemory(backend="auto").fit(normal)
        numpy_memory = NormalPatchMemory(backend="numpy").fit(normal)
        np.testing.assert_allclose(automatic.score(query), numpy_memory.score(query))

    def test_leave_one_out_calibration_is_finite(self):
        features = np.eye(4, dtype=np.float32)
        memory = NormalPatchMemory(max_patches=0, query_chunk_size=2).fit(features)
        self.assertTrue(np.isfinite(memory.center))
        self.assertGreater(memory.scale, 0)
        np.testing.assert_allclose(memory.score(features), 0.0, atol=1e-6)

    def test_fused_weight_endpoints_match_calibrated_components(self):
        normal = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
        query = np.array([[1, -1], [-1, 1]], dtype=np.float32)

        pca_only = DefectFusion(object(), anomaly_method="pca_knn", knn_weight=0.0)
        pca_only.subspace.fit(normal)
        pca_only.normal_memory.fit(normal)
        fused, pca, _ = pca_only._anomaly_scores(query)
        np.testing.assert_allclose(fused, pca_only.subspace.calibrated(pca))

        knn_only = DefectFusion(object(), anomaly_method="pca_knn", knn_weight=1.0)
        knn_only.subspace.fit(normal)
        knn_only.normal_memory.fit(normal)
        fused, _, knn = knn_only._anomaly_scores(query)
        np.testing.assert_allclose(fused, knn_only.normal_memory.calibrated(knn))

    def test_memory_uses_npz_sidecar(self):
        fusion = DefectFusion(object(), anomaly_method="knn", memory_max_patches=10)
        normal = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
        fusion.subspace.fit(normal)
        fusion.normal_memory.fit(normal)
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "model.json"
            fusion.save(state_path)
            self.assertTrue(state_path.with_suffix(".json.normal-memory.npz").exists())
            loaded = DefectFusion.load(state_path, object())
            np.testing.assert_allclose(
                loaded.normal_memory.features,
                fusion.normal_memory.features,
                atol=5e-4,
            )

    def test_layer_subspaces_round_trip_without_aggregate_pca(self):
        extractor = type("Extractor", (), {"debias": False, "layer_aggregation": "mean"})()
        fusion = DefectFusion(extractor, layer_fusion="score_max")
        normal = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
        fusion.layer_subspaces = [NormalSubspace(max_components=1).fit(normal)]
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "model.json"
            fusion.save(state_path)
            loaded = DefectFusion.load(state_path, extractor)
            self.assertEqual(loaded.layer_fusion, "score_max")
            self.assertIsNone(loaded.subspace.components)
            self.assertEqual(len(loaded.layer_subspaces), 1)


if __name__ == "__main__":
    unittest.main()
