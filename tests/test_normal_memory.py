import tempfile
import unittest
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

from defectfusion.model import NormalPatchMemory, NormalSubspace, PrototypeBank
from defectfusion.pipeline import DefectFusion, NormalTrainingView


class NormalPatchMemoryTest(unittest.TestCase):
    @staticmethod
    def _resolution_extractor():
        class Extractor:
            image_size = 672
            positional_basis = "cached"
            resize_mode = "direct"
            device = None

            def __init__(self):
                self.calls = []

            def _features(self):
                side = 2 if self.image_size == 672 else 3
                count = side * side
                values = np.arange(count * 2, dtype=np.float32).reshape(count, 2)
                return values, (side, side)

            def extract(self, image):
                self.calls.append(("extract", self.image_size))
                return self._features()

            def extract_dual(self, image):
                self.calls.append(("extract_dual", self.image_size))
                raw, grid = self._features()
                return raw, raw + 0.25, grid

            def extract_dual_layers(self, image):
                self.calls.append(("extract_dual_layers", self.image_size))
                raw, grid = self._features()
                return raw, raw + 0.25, np.stack([raw + 0.5, raw + 0.75]), grid

        return Extractor()

    def test_equal_head_resolutions_reuse_one_dual_forward(self):
        extractor = self._resolution_extractor()
        fusion = DefectFusion(extractor, anomaly_method="pca", dual_branch=True, pixel_image_size=672, image_head_image_size=672)
        fusion.fit_normal([Image.new("RGB", (8, 8))])
        self.assertEqual(extractor.calls, [("extract_dual", 672)])

    def test_different_head_resolutions_use_separate_grids_and_positions(self):
        extractor = self._resolution_extractor()
        fusion = DefectFusion(
            extractor, anomaly_method="knn", dual_branch=True,
            pixel_image_size=672, image_head_image_size=896, memory_max_patches=0,
        ).fit_normal([Image.new("RGB", (8, 8))])
        self.assertEqual(extractor.calls, [("extract", 672), ("extract_dual_layers", 896)])
        self.assertEqual(len(fusion.normal_memory.positions), 4)
        self.assertEqual(len(fusion.image_memory.positions), 9)
        self.assertIsNone(extractor.positional_basis)

        seen = {}
        original = fusion._image_anomaly_score
        def capture(patches, positions=None, grid=None, score_bundle=None):
            seen["grid"] = grid
            seen["positions"] = positions
            return original(patches, positions, grid, score_bundle)
        fusion._image_anomaly_score = capture
        result = fusion._predict_single(Image.new("RGB", (8, 8)), "sample.png")
        self.assertEqual(result["grid"], [2, 2])
        self.assertEqual(np.asarray(result["anomaly_map"]).shape, (2, 2))
        self.assertEqual(seen["grid"], (3, 3))
        self.assertEqual(len(seen["positions"]), 9)

    def test_pipeline_save_load_preserves_head_resolutions(self):
        extractor = self._resolution_extractor()
        fusion = DefectFusion(
            extractor, anomaly_method="pca", dual_branch=True,
            pixel_image_size=672, image_head_image_size=896,
        ).fit_normal([Image.new("RGB", (8, 8))])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            fusion.save(path)
            loaded = DefectFusion.load(path, self._resolution_extractor())
        self.assertEqual(loaded.pixel_image_size, 672)
        self.assertEqual(loaded.image_head_image_size, 896)

    def test_empty_prototype_bank_skips_duplicate_typing_pca(self):
        class Extractor:
            def extract_dual(self, image):
                patches = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]])
                return patches, patches.copy(), (2, 2)

        class CountingSubspace:
            def __init__(self):
                self.calls = 0

            def score(self, features):
                self.calls += 1
                return np.arange(len(features), dtype=np.float64)

        fusion = DefectFusion(Extractor(), anomaly_method="pca", dual_branch=True)
        fusion.subspace = CountingSubspace()
        fusion.image_subspace = CountingSubspace()
        result = fusion._predict_single(Image.new("RGB", (2, 2)), "image.png")
        self.assertEqual(result["defect_type"], "unknown")
        self.assertEqual(fusion.subspace.calls, 1)
        self.assertEqual(fusion.image_subspace.calls, 1)

    def test_single_branch_reuses_patch_scores_for_image_aggregation(self):
        class Extractor:
            def extract(self, image):
                return np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]]), (2, 2)

        class CountingSubspace:
            def __init__(self):
                self.calls = 0

            def score(self, features):
                self.calls += 1
                return np.arange(len(features), dtype=np.float64)

        fusion = DefectFusion(Extractor(), anomaly_method="pca")
        fusion.subspace = CountingSubspace()
        fusion._predict_single(Image.new("RGB", (2, 2)), "image.png")
        self.assertEqual(fusion.subspace.calls, 1)

    def test_score_first_reuses_pca_and_knn_component_scores(self):
        class Scores:
            def __init__(self, values):
                self.values = np.asarray(values, dtype=np.float64)
                self.calls = 0

            def score(self, features, positions=None):
                self.calls += 1
                return self.values

            def calibrated(self, values):
                return np.asarray(values)

        patches = np.zeros((4, 2), dtype=np.float32)
        fusion = DefectFusion(object(), anomaly_method="pca_knn", image_top_ratio=0.25, image_fusion_stage="score")
        fusion.subspace = Scores([10, 0, 0, 0])
        fusion.normal_memory = Scores([0, 10, 0, 0])
        bundle = fusion._image_score_bundle(patches)
        score, _ = fusion._image_anomaly_score(patches, score_bundle=bundle)
        self.assertEqual(score, 10.0)
        self.assertEqual(fusion.subspace.calls, 1)
        self.assertEqual(fusion.normal_memory.calls, 1)

    def test_typing_reuses_existing_pca_scores(self):
        patches = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]])
        supplied_scores = np.array([0.0, 1.0, 2.0, 3.0])
        fusion = DefectFusion(object(), top_k_ratio=0.25)
        fusion.subspace.score = lambda features: self.fail("PCA scores should be reused")
        selected = fusion._anomaly_patches(patches, supplied_scores)
        np.testing.assert_array_equal(selected, patches[[3]])

    def test_aligned_training_positions_restore_rotation_coordinates(self):
        fusion = DefectFusion(object(), knn_spatial_radius=0.1, align_training_positions=True)
        # A 90-degree counter-clockwise augmentation maps its top-center patch back to right-center.
        matrix = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        positions, valid = fusion._training_positions((2, 2), matrix)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(positions[0], [0.25, 0.75])

    def test_aligned_training_positions_drop_rotated_padding(self):
        fusion = DefectFusion(object(), knn_spatial_radius=0.1, align_training_positions=True)
        radians = np.deg2rad(45)
        cosine, sine = np.cos(radians), np.sin(radians)
        matrix = np.array([[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]])
        matrix[:2, 2] = 0.5 - matrix[:2, :2] @ np.array([0.5, 0.5])
        _, valid = fusion._training_positions((4, 4), matrix)
        self.assertLess(valid.sum(), len(valid))

    def test_aligned_positions_are_used_by_spatial_memory(self):
        class Extractor:
            resize_mode = "direct"
            device = None

            def extract(self, image):
                return np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]]), (2, 2)

        rotation = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        image = Image.new("RGB", (2, 2))
        fusion = DefectFusion(Extractor(), anomaly_method="knn", knn_spatial_radius=0.1, align_training_positions=True)
        fusion.fit_normal([NormalTrainingView(image, rotation)])
        np.testing.assert_allclose(fusion.normal_memory.positions[0], [0.25, 0.75])

    def test_tta_map_inverse_restores_flip_coordinates(self):
        anomaly_map = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(
            DefectFusion._restore_test_map(np.fliplr(anomaly_map), "hflip"),
            anomaly_map,
        )
        np.testing.assert_array_equal(
            DefectFusion._restore_test_map(np.flipud(anomaly_map), "vflip"),
            anomaly_map,
        )

    def test_tta_rejects_unknown_transforms(self):
        with self.assertRaises(ValueError):
            DefectFusion(object(), test_augmentations=["rotate45"])

    def test_tta_averages_scores_and_aligned_maps(self):
        fusion = DefectFusion(object(), test_augmentations=["hflip"])
        results = [
            {"image": "image.png", "anomaly_score": 2.0, "anomaly_map": [[1.0, 2.0], [3.0, 4.0]], "defect_type": "unknown", "defect_type_score": 0.0, "fused_score": 1.0, "pca_anomaly_score": 2.0, "image_spatial_consistency": 0.0},
            {"image": "image.png", "anomaly_score": 4.0, "anomaly_map": [[6.0, 5.0], [8.0, 7.0]], "defect_type": "unknown", "defect_type_score": 0.0, "fused_score": 2.0, "pca_anomaly_score": 4.0, "image_spatial_consistency": 0.0},
        ]
        fusion._predict_single = lambda image, image_name: results.pop(0)
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            Image.new("RGB", (2, 2)).save(image_path)
            result = fusion.predict(image_path)
        self.assertEqual(result["anomaly_score"], 3.0)
        np.testing.assert_allclose(result["anomaly_map"], [[3.0, 4.0], [5.0, 6.0]])
        self.assertEqual(result["tta_view_scores"], [2.0, 4.0])

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
        scores = np.array([4.0, 3.0, 0.0, 0.0])
        score, consistency = fusion._image_anomaly_score(patches, grid=(2, 2), score_bundle=(scores, scores, None, None))
        self.assertEqual(score, 3.5)
        self.assertEqual(consistency, 1.0)

    def test_spatial_weight_boosts_connected_scores_more(self):
        fusion = DefectFusion(object(), image_top_ratio=0.25, image_spatial_weight=0.5)
        connected = np.array([4, 3, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        scattered = np.array([4, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1])
        patches = np.zeros((16, 2), dtype=np.float32)
        connected_score, _ = fusion._image_anomaly_score(patches, grid=(4, 4), score_bundle=(connected, connected, None, None))
        scattered_score, _ = fusion._image_anomaly_score(patches, grid=(4, 4), score_bundle=(scattered, scattered, None, None))
        self.assertGreater(connected_score, scattered_score)

    def test_minimum_component_size_rejects_isolated_top_patches(self):
        fusion = DefectFusion(object(), image_top_ratio=0.25, image_min_component_size=2)
        connected = np.array([4, 3, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        isolated = np.array([4, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1], dtype=np.float32)

        connected_score = fusion._region_rejected_score(connected, (4, 4))
        isolated_score = fusion._region_rejected_score(isolated, (4, 4))

        self.assertEqual(connected_score, 2.5)
        self.assertEqual(isolated_score, float(isolated.mean()))
        self.assertGreater(connected_score, isolated_score)

    def test_minimum_component_size_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            DefectFusion(object(), image_min_component_size=0)

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

    def test_anoco_scores_off_manifold_query_higher(self):
        angles = np.linspace(-0.2, 0.2, 9)
        normal = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1).astype(np.float32)
        memory = NormalPatchMemory(max_patches=0, backend="numpy").fit(normal)
        scores = memory.score_anoco(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32), neighbor_count=4)
        self.assertGreater(scores[1], scores[0])

    def test_anoco_calibration_is_finite(self):
        features = np.eye(6, dtype=np.float32)
        memory = NormalPatchMemory(max_patches=0, backend="numpy").fit(features)
        memory.fit_anoco_calibration(neighbor_count=3)
        self.assertTrue(np.isfinite(memory.anoco_center))
        self.assertGreater(memory.anoco_scale, 0)
        self.assertTrue(np.all(np.isfinite(memory.anoco_calibration_scores)))

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

    def test_pca_anoco_weight_endpoints_match_calibrated_components(self):
        normal = np.array([[1, 0], [0, 1], [1, 1], [-1, 1]], dtype=np.float32)
        query = np.array([[1, -1], [-1, -1]], dtype=np.float32)
        for weight in (0.0, 1.0):
            fusion = DefectFusion(object(), anomaly_method="pca_anoco", anoco_weight=weight, anoco_neighbors=2)
            fusion.subspace.fit(normal)
            fusion.normal_memory.fit(normal).fit_anoco_calibration(neighbor_count=2)
            fused, pca, anoco, _ = fusion._anomaly_scores(query)
            expected = fusion.subspace.calibrated(pca) if weight == 0 else fusion.normal_memory.calibrated_anoco(anoco)
            np.testing.assert_allclose(fused, expected)

    def test_hybrid_head_uses_knn_for_pixels_and_anoco_for_images(self):
        normal = np.array([[1, 0], [0, 1], [1, 1], [-1, 1]], dtype=np.float32)
        query = np.array([[1, -1], [-1, -1]], dtype=np.float32)
        fusion = DefectFusion(
            object(), anomaly_method="pca_knn_anoco", dual_branch=True,
            knn_weight=0.5, anoco_weight=0.25, anoco_neighbors=2,
        )
        fusion.subspace.fit(normal)
        fusion.normal_memory.fit(normal)
        fusion.image_subspace.fit(normal)
        fusion.image_memory.fit(normal).fit_anoco_calibration(neighbor_count=2)

        pixel_fused, pixel_pca, pixel_knn, _ = fusion._anomaly_scores(query)
        expected_pixel = 0.5 * fusion.subspace.calibrated(pixel_pca) + 0.5 * fusion.normal_memory.calibrated(pixel_knn)
        np.testing.assert_allclose(pixel_fused, expected_pixel)

        image_fused, image_pca, image_anoco, _ = fusion._image_score_bundle(query)
        expected_image = 0.75 * fusion.image_subspace.calibrated(image_pca) + 0.25 * fusion.image_memory.calibrated_anoco(image_anoco)
        np.testing.assert_allclose(image_fused, expected_image)

    def test_anoco_layer_consensus_uses_median_calibrated_drift(self):
        normal = np.array([
            [1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0],
            [0.1, 0.9, 0.0], [0.0, 0.0, 1.0], [0.1, 0.0, 0.9],
        ], dtype=np.float32)
        query = np.array([[0.8, 0.2, 0.1], [0.1, 0.7, 0.3]], dtype=np.float32)
        layer_queries = np.stack([query, query[:, [1, 2, 0]]], axis=0)
        fusion = DefectFusion(
            object(), anomaly_method="pca_knn_anoco", dual_branch=True,
            anoco_layer_consensus=True, anoco_neighbors=2, anoco_weight=0.25,
        )
        fusion.image_subspace.fit(normal)
        layer_normal = [normal, normal[:, [1, 2, 0]]]
        for features in layer_normal:
            memory = NormalPatchMemory(backend="numpy").fit(features)
            memory.fit_anoco_calibration(neighbor_count=2)
            fusion.image_layer_memories.append(memory)
        fused, pca_scores, consensus, gate = fusion._image_score_bundle(query, layer_patches=layer_queries)
        expected_layers = []
        for features, memory in zip(layer_queries, fusion.image_layer_memories):
            scores = memory.score_anoco(features, neighbor_count=2)
            expected_layers.append(memory.calibrated_anoco(scores))
        expected_consensus = np.median(np.stack(expected_layers), axis=0)
        expected = 0.75 * fusion.image_subspace.calibrated(pca_scores) + 0.25 * expected_consensus
        np.testing.assert_allclose(consensus, expected_consensus)
        np.testing.assert_allclose(fused, expected)
        self.assertIsNone(gate)

    def test_anoco_layer_consensus_requires_matching_layers(self):
        fusion = DefectFusion(
            object(), anomaly_method="pca_knn_anoco", dual_branch=True,
            anoco_layer_consensus=True,
        )
        fusion.image_subspace.fit(np.eye(3, dtype=np.float32))
        fusion.image_layer_memories = [NormalPatchMemory(backend="numpy").fit(np.eye(3, dtype=np.float32))]
        with self.assertRaises(ValueError):
            fusion._image_score_bundle(np.eye(3, dtype=np.float32), layer_patches=np.empty((0, 3, 3)))

    def test_hybrid_head_requires_dual_branch(self):
        with self.assertRaisesRegex(ValueError, "requires dual_branch"):
            DefectFusion(object(), anomaly_method="pca_knn_anoco")

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

    def test_pipeline_save_load_preserves_anoco_state(self):
        fusion = DefectFusion(object(), anomaly_method="anoco", anoco_neighbors=3, anoco_temperature=0.2)
        normal = np.eye(5, dtype=np.float32)
        fusion.subspace.fit(normal)
        fusion.normal_memory.fit(normal).fit_anoco_calibration(neighbor_count=3, temperature=0.2)
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "model.json"
            fusion.save(state_path)
            loaded = DefectFusion.load(state_path, object())
            self.assertEqual(loaded.anoco_neighbors, 3)
            self.assertEqual(loaded.anoco_temperature, 0.2)
            np.testing.assert_allclose(loaded.normal_memory.anoco_calibration_scores, fusion.normal_memory.anoco_calibration_scores)

    def test_pipeline_save_load_preserves_anoco_layer_consensus(self):
        normal = np.eye(5, dtype=np.float32)
        fusion = DefectFusion(
            object(), anomaly_method="pca_knn_anoco", dual_branch=True,
            anoco_layer_consensus=True, anoco_neighbors=2,
        )
        fusion.subspace.fit(normal)
        fusion.image_subspace.fit(normal)
        fusion.normal_memory.fit(normal)
        fusion.image_memory.fit(normal).fit_anoco_calibration(neighbor_count=2)
        for features in (normal, normal[::-1].copy()):
            memory = NormalPatchMemory(backend="numpy").fit(features)
            memory.fit_anoco_calibration(neighbor_count=2)
            fusion.image_layer_memories.append(memory)
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "model.json"
            fusion.save(state_path)
            loaded = DefectFusion.load(state_path, object())
            self.assertTrue(loaded.anoco_layer_consensus)
            self.assertEqual(len(loaded.image_layer_memories), 2)
            for expected, actual in zip(fusion.image_layer_memories, loaded.image_layer_memories):
                np.testing.assert_allclose(actual.features, expected.features, atol=5e-4)
                np.testing.assert_allclose(actual.anoco_calibration_scores, expected.anoco_calibration_scores)

    def test_pipeline_save_load_preserves_mahalanobis_residual(self):
        fusion = DefectFusion(
            object(), pca_residual_metric="mahalanobis",
            image_min_component_size=2, test_augmentations=["hflip"],
        )
        normal = np.array([[0.0, 0.0], [1.0, 0.1], [2.0, -0.1]])
        fusion.subspace.fit(normal)
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "model.json"
            fusion.save(state_path)
            loaded = DefectFusion.load(state_path, object())
            self.assertEqual(loaded.pca_residual_metric, "mahalanobis")
            self.assertEqual(loaded.subspace.residual_metric, "mahalanobis")
            self.assertEqual(loaded.image_min_component_size, 2)
            self.assertEqual(loaded.test_augmentations, ("hflip",))


if __name__ == "__main__":
    unittest.main()
