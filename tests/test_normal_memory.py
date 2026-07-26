import tempfile
import unittest
from pathlib import Path

import numpy as np

from defectfusion.model import NormalPatchMemory
from defectfusion.pipeline import DefectFusion


class NormalPatchMemoryTest(unittest.TestCase):
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
        fused, pca, _, _ = pca_only._anomaly_scores(query)
        np.testing.assert_allclose(fused, pca_only.subspace.calibrated(pca))

        knn_only = DefectFusion(object(), anomaly_method="pca_knn", knn_weight=1.0)
        knn_only.subspace.fit(normal)
        knn_only.normal_memory.fit(normal)
        fused, _, knn, _ = knn_only._anomaly_scores(query)
        np.testing.assert_allclose(fused, knn_only.normal_memory.calibrated(knn))

    def test_tail_evidence_is_monotonic(self):
        calibration = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        evidence = NormalPatchMemory._tail_evidence([0.05, 0.25, 0.5, 0.6], calibration)
        self.assertTrue(np.all(np.diff(evidence) > 0))

    def test_gated_fusion_produces_patchwise_weights(self):
        normal = np.array([[1, 0], [0, 1], [1, 1], [-1, 1]], dtype=np.float32)
        query = np.array([[1, -1], [-1, -1]], dtype=np.float32)
        fusion = DefectFusion(object(), anomaly_method="pca_knn", fusion_mode="gated")
        fusion.subspace.fit(normal)
        fusion.normal_memory.fit(normal)
        fused, _, _, gate = fusion._anomaly_scores(query)
        self.assertEqual(gate.shape, (2,))
        self.assertTrue(np.all((gate >= 0) & (gate <= 1)))
        self.assertTrue(np.all(np.isfinite(fused)))

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
            np.testing.assert_allclose(
                loaded.normal_memory.calibration_scores,
                fusion.normal_memory.calibration_scores,
            )


if __name__ == "__main__":
    unittest.main()
