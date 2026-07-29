import tempfile
import unittest
import importlib.util
from pathlib import Path

import numpy as np

from defectfusion.model import NormalPatchMemory, NormalSubspace, PrototypeBank
from defectfusion.pipeline import DefectFusion


class NormalPatchMemoryTest(unittest.TestCase):
    def test_mahalanobis_residual_emphasizes_low_variance_dimensions(self):
        training = np.array([
            [-4.0, -0.1, 0.0], [-2.0, 0.1, 0.0], [0.0, -0.1, 0.0],
            [2.0, 0.1, 0.0], [4.0, -0.1, 0.0],
        ])
        subspace = NormalSubspace(explained_variance=0.8, residual_metric="mahalanobis").fit(training)
        high_variance_score = subspace.score([[0.0, 1.0, 0.0]])[0]
        low_variance_score = subspace.score([[0.0, 0.0, 1.0]])[0]
        self.assertGreater(low_variance_score, high_variance_score)

    def test_subspace_serialization_preserves_residual_metric(self):
        training = np.array([[0.0, 0.0], [1.0, 0.1], [2.0, -0.1]])
        fitted = NormalSubspace(explained_variance=0.8, residual_metric="mahalanobis").fit(training)
        loaded = NormalSubspace.from_dict(fitted.to_dict())
        self.assertEqual(loaded.residual_metric, "mahalanobis")
        np.testing.assert_allclose(loaded.score(training), fitted.score(training))

    def test_image_top_ratio_controls_aggregation(self):
        scores = np.arange(1, 101, dtype=np.float64)
        top_one = DefectFusion(object(), image_top_ratio=0.01)._aggregate_image_score(scores)
        top_five = DefectFusion(object(), image_top_ratio=0.05)._aggregate_image_score(scores)
        self.assertEqual(top_one, 100.0)
        self.assertEqual(top_five, 98.0)

    def test_score_first_image_fusion_preserves_separate_peaks(self):
        class Scores:
            def __init__(self, values): self.values = np.asarray(values, dtype=np.float64)
            def score(self, features, positions=None): return self.values
            def calibrated(self, values): return np.asarray(values)

        patches = np.zeros((4, 2), dtype=np.float32)
        patch_first = DefectFusion(object(), anomaly_method="pca_knn", image_top_ratio=0.25, image_fusion_stage="patch")
        patch_first.subspace = Scores([10, 0, 0, 0]); patch_first.normal_memory = Scores([0, 10, 0, 0])
        score_first = DefectFusion(object(), anomaly_method="pca_knn", image_top_ratio=0.25, image_fusion_stage="score")
        score_first.subspace = Scores([10, 0, 0, 0]); score_first.normal_memory = Scores([0, 10, 0, 0])
        self.assertEqual(patch_first._image_anomaly_score(patches)[0], 5.0)
        self.assertEqual(score_first._image_anomaly_score(patches)[0], 10.0)

    def test_spatial_consistency_rewards_connected_top_patches(self):
        fusion = DefectFusion(object(), image_top_ratio=0.25)
        connected = np.array([4, 3, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        scattered = np.array([4, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1])
        self.assertEqual(fusion._spatial_consistency(connected, (4, 4)), 1.0)
        self.assertEqual(fusion._spatial_consistency(scattered, (4, 4)), 0.25)

    def test_spatial_weight_zero_preserves_base_score(self):
        patches = np.zeros((4, 2), dtype=np.float32)
        fusion = DefectFusion(object(), image_top_ratio=0.5, image_spatial_weight=0.0)
        fusion._image_scores = lambda features, positions=None: np.array([4.0, 3.0, 0.0, 0.0])
        score, consistency = fusion._image_anomaly_score(patches, grid=(2, 2))
        self.assertEqual(score, 3.5)
        self.assertEqual(consistency, 1.0)

    def test_spatial_weight_boosts_connected_scores_more(self):
        fusion = DefectFusion(object(), image_top_ratio=0.25, image_spatial_weight=0.5)
        connected = np.array([4, 3, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        scattered = np.array([4, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1])
        patches = np.zeros((16, 2), dtype=np.float32)
        fusion._image_scores = lambda features, positions=None: connected
        connected_score, _ = fusion._image_anomaly_score(patches, grid=(4, 4))
        fusion._image_scores = lambda features, positions=None: scattered
        scattered_score, _ = fusion._image_anomaly_score(patches, grid=(4, 4))
        self.assertGreater(connected_score, scattered_score)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is not installed")
    def test_rbf_svm_classifies_separable_patch_sets(self):
        bank = PrototypeBank(unknown_threshold=0.0)
        bank.add("crack", np.array([[1.0, 0.0], [0.9, 0.1], [1.1, -0.1]]))
        bank.add("color", np.array([[0.0, 1.0], [0.1, 0.9], [-0.1, 1.1]]))
        label, score = bank.predict_rbf_svm(np.array([[0.95, 0.05], [1.05, -0.05]]))
        self.assertEqual(label, "crack")
        self.assertGreater(score, 0.5)

    def test_numpy_backend_matches_auto_cpu(self):
        normal = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
        query = np.array([[1, -1], [-1, 1]], dtype=np.float32)
        automatic = NormalPatchMemory(backend="auto").fit(normal)
        numpy_memory = NormalPatchMemory(backend="numpy").fit(normal)
        np.testing.assert_allclose(automatic.score(query), numpy_memory.score(query))

    def test_spatial_knn_excludes_distant_better_match(self):
        features = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        positions = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float32)
        spatial = NormalPatchMemory(max_patches=0, backend="numpy", spatial_radius=0.1).fit(features, positions)
        global_memory = NormalPatchMemory(max_patches=0, backend="numpy").fit(features)
        query = np.array([[1.0, 0.0]], dtype=np.float32)
        query_position = np.array([[0.5, 0.5]], dtype=np.float32)
        self.assertGreater(spatial.score(query, positions=query_position)[0], global_memory.score(query)[0])

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

    def test_pipeline_save_load_preserves_mahalanobis_residual(self):
        fusion = DefectFusion(object(), pca_residual_metric="mahalanobis")
        normal = np.array([[0.0, 0.0], [1.0, 0.1], [2.0, -0.1]])
        fusion.subspace.fit(normal)
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "model.json"
            fusion.save(state_path)
            loaded = DefectFusion.load(state_path, object())
            self.assertEqual(loaded.pca_residual_metric, "mahalanobis")
            self.assertEqual(loaded.subspace.residual_metric, "mahalanobis")


if __name__ == "__main__":
    unittest.main()
