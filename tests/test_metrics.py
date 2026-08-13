import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from defectfusion.mvtec import compute_aupro, compute_binary_auroc_aupr, compute_binary_metrics, evaluate_mvtec, evaluate_samples
from defectfusion.pipeline import NormalTrainingView


class AuproTest(unittest.TestCase):
    def test_threshold_calibration_uses_score_only_prediction_when_available(self):
        class Fusion:
            def __init__(self):
                self.score_calls = []

            def predict_anomaly_score(self, image):
                self.score_calls.append(Path(image).stem)
                return {"reference-low": 0.25, "reference-high": 0.75}[Path(image).stem]

            def predict(self, image):
                if Path(image).stem.startswith("reference"):
                    raise AssertionError("threshold calibration must not compute anomaly maps")
                return {
                    "image": str(image), "anomaly_score": 0.5,
                    "anomaly_map": [[0.0, 0.0], [0.0, 0.0]],
                    "defect_type": "unknown", "defect_type_score": 0.0,
                }

        fusion = Fusion()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            references = [root / "reference-low.png", root / "reference-high.png"]
            metrics = evaluate_samples(
                fusion,
                "bottle",
                [(root / "good.png", "good", False, None)],
                root / "results.json",
                progress=False,
                normal_reference_images=references,
                decision_threshold_quantile=1.0,
            )

        self.assertEqual(fusion.score_calls, ["reference-low", "reference-high"])
        self.assertEqual(metrics["good_decision_threshold"], 0.75)

    def test_good_samples_get_a_thresholded_decision_without_pixel_metrics(self):
        class Fusion:
            scores = {"normal-reference": 0.25, "good": 0.20, "defect": 0.80}

            def predict(self, image):
                score = self.scores[Path(image).stem]
                anomaly_map = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
                if Path(image).stem == "defect":
                    anomaly_map[0, 0] = 1.0
                return {
                    "image": image,
                    "anomaly_score": score,
                    "anomaly_map": anomaly_map.tolist(),
                    "defect_type": "unknown",
                    "defect_type_score": 0.0,
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normal_reference = root / "normal-reference.png"
            good = root / "good.png"
            defect = root / "defect.png"
            mask = root / "defect_mask.png"
            for image in (normal_reference, good, defect):
                Image.new("RGB", (2, 2)).save(image)
            Image.fromarray(np.array([[255, 0], [0, 0]], dtype=np.uint8)).save(mask)
            output = root / "results.json"
            metrics = evaluate_samples(
                Fusion(),
                "bottle",
                [(good, "good", False, None), (defect, "crack", True, mask)],
                output,
                progress=False,
                normal_reference_images=[normal_reference],
                decision_threshold_source="normal_validation_max",
            )
            rows = {Path(row["image"]).stem: row for row in json.loads(output.read_text(encoding="utf-8"))}

        self.assertEqual(rows["good"]["ground_truth_anomaly"], False)
        self.assertEqual(rows["good"]["predicted_anomaly"], False)
        self.assertEqual(rows["good"]["predicted_label"], "good")
        self.assertEqual(rows["good"]["prediction_correct"], True)
        self.assertEqual(rows["defect"]["predicted_label"], "anomaly")
        self.assertEqual(metrics["good_images"], 1)
        self.assertEqual(metrics["good_decision_images"], 1)
        self.assertEqual(metrics["good_predicted_normal"], 1)
        self.assertEqual(metrics["good_predicted_anomaly"], 0)
        self.assertEqual(metrics["good_accuracy"], 1.0)
        self.assertEqual(metrics["defect_images"], 1)
        self.assertEqual(metrics["defect_decision_images"], 1)
        self.assertEqual(metrics["defect_predicted_anomaly"], 1)
        self.assertEqual(metrics["defect_predicted_normal"], 0)
        self.assertEqual(metrics["defect_recall"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(metrics["decision_accuracy"], 1.0)
        self.assertEqual(metrics["pixel_metric_images"], 1)
        self.assertEqual(metrics["good_decision_threshold"], 0.25)
        self.assertEqual(metrics["good_decision_threshold_source"], "normal_validation_max")
        self.assertEqual(metrics["good_decision_quantile"], 1.0)
        self.assertEqual(metrics["good_decision_reference_images"], 1)
        self.assertEqual(metrics["pixel_auroc"], 1.0)

    def test_reference_scores_use_requested_decision_quantile(self):
        class Fusion:
            def __init__(self, view_scores):
                self.view_scores = view_scores

            def predict(self, image):
                if isinstance(image, NormalTrainingView):
                    score = self.view_scores[id(image)]
                else:
                    score = 0.75
                return {
                    "image": str(image), "anomaly_score": score,
                    "anomaly_map": [[0.0, 0.0], [0.0, 0.0]],
                    "defect_type": "unknown", "defect_type_score": 0.0,
                }

        references = [NormalTrainingView(Image.new("RGB", (2, 2))) for _ in range(4)]
        view_scores = {id(view): score for view, score in zip(references, (0.1, 0.2, 0.4, 0.8))}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / "good.png"
            output = root / "results.json"
            metrics = evaluate_samples(
                Fusion(view_scores),
                "bottle",
                [(good, "good", False, None)],
                output,
                progress=False,
                normal_reference_images=references,
                decision_threshold_source="normal_augmentation_holdout_quantile",
                decision_threshold_quantile=0.995,
            )

        self.assertAlmostEqual(metrics["good_decision_threshold"], np.quantile([0.1, 0.2, 0.4, 0.8], 0.995))
        self.assertEqual(metrics["good_decision_threshold_source"], "normal_augmentation_holdout_quantile")
        self.assertEqual(metrics["good_decision_quantile"], 0.995)
        self.assertEqual(metrics["good_decision_reference_images"], 4)
        self.assertEqual(metrics["good_accuracy"], 1.0)

    def test_precomputed_scores_use_higher_quantile_and_include_calibration_time(self):
        class Fusion:
            def predict(self, image):
                return {
                    "image": str(image), "anomaly_score": 0.35,
                    "anomaly_map": [[0.0, 0.0], [0.0, 0.0]],
                    "defect_type": "unknown", "defect_type_score": 0.0,
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = evaluate_samples(
                Fusion(),
                "bottle",
                [(root / "good.png", "good", False, None)],
                root / "results.json",
                progress=False,
                normal_reference_scores=[0.1, 0.2, 0.4, 0.8],
                normal_reference_seconds=2.5,
                decision_threshold_source="normal_leave_one_out_quantile",
                decision_threshold_quantile=0.5,
                decision_threshold_quantile_method="higher",
            )

        self.assertEqual(metrics["good_decision_threshold"], 0.4)
        self.assertEqual(metrics["good_decision_quantile_method"], "higher")
        self.assertEqual(metrics["good_decision_reference_images"], 4)
        self.assertGreaterEqual(metrics["timing_seconds"]["threshold_calibration"], 2.5)
        self.assertGreaterEqual(metrics["timing_seconds"]["total"], 2.5)

    def test_threshold_metrics_report_false_positive_and_false_negative_counts(self):
        class Fusion:
            scores = {
                "reference": 0.5,
                "good-correct": 0.2,
                "good-wrong": 0.8,
                "defect-correct": 0.9,
                "defect-wrong": 0.3,
            }

            def predict(self, image):
                return {
                    "image": str(image), "anomaly_score": self.scores[Path(image).stem],
                    "anomaly_map": [[0.0, 0.0], [0.0, 0.0]],
                    "defect_type": "unknown", "defect_type_score": 0.0,
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = [
                (root / "good-correct.png", "good", False, None),
                (root / "good-wrong.png", "good", False, None),
                (root / "defect-correct.png", "defect", True, None),
                (root / "defect-wrong.png", "defect", True, None),
            ]
            metrics = evaluate_samples(
                Fusion(), "widget", samples, root / "results.json", progress=False,
                normal_reference_images=[root / "reference.png"],
            )

        self.assertEqual(metrics["good_predicted_normal"], 1)
        self.assertEqual(metrics["good_predicted_anomaly"], 1)
        self.assertEqual(metrics["defect_predicted_anomaly"], 1)
        self.assertEqual(metrics["defect_predicted_normal"], 1)
        self.assertEqual(metrics["good_accuracy"], 0.5)
        self.assertEqual(metrics["defect_recall"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertEqual(metrics["decision_accuracy"], 0.5)

    def test_evaluation_writes_json_array_not_jsonl(self):
        class Fusion:
            def predict(self, image):
                return {
                    "image": image, "anomaly_score": 0.0,
                    "anomaly_map": [[0.0, 0.0], [0.0, 0.0]],
                    "defect_type": "unknown", "defect_type_score": 0.0,
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bottle"
            image_dir = root / "test" / "good"
            image_dir.mkdir(parents=True)
            Image.new("RGB", (4, 4)).save(image_dir / "001.png")
            output = Path(directory) / "bottle.json"
            evaluate_mvtec(Fusion(), root, output, progress=False)
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["category"], "bottle")

    def test_shared_binary_metrics_are_exact_for_perfect_ranking(self):
        auroc, aupr = compute_binary_auroc_aupr([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(auroc, 1.0)
        self.assertEqual(aupr, 1.0)

        _, _, f1_max = compute_binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(f1_max, 1.0)

    def test_shared_binary_metrics_handle_tied_scores(self):
        auroc, aupr = compute_binary_auroc_aupr([0, 1], [0.5, 0.5])
        self.assertEqual(auroc, 0.5)
        self.assertEqual(aupr, 0.5)

        _, _, f1_max = compute_binary_metrics([0, 1], [0.5, 0.5])
        self.assertAlmostEqual(f1_max, 2.0 / 3.0)

    def test_shared_binary_metrics_return_nan_for_one_class(self):
        metrics = compute_binary_metrics([0, 0], [0.1, 0.2])
        self.assertTrue(all(np.isnan(value) for value in metrics))

    def test_shared_binary_metrics_match_known_reversed_ranking(self):
        auroc, aupr = compute_binary_auroc_aupr([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
        self.assertEqual(auroc, 0.0)
        self.assertAlmostEqual(aupr, 5.0 / 12.0)

    def test_perfect_map_has_near_perfect_aupro(self):
        masks = np.zeros((2, 8, 8), dtype=np.uint8)
        masks[1, 2:6, 2:6] = 1
        predictions = np.linspace(0.0, 0.1, masks.size, dtype=np.float32).reshape(masks.shape)
        predictions[masks > 0] = 1.0
        # The finite 8x8 background makes the smallest realizable non-zero FPR coarse.
        self.assertGreater(compute_aupro(predictions, masks), 0.95)

    def test_reversed_map_is_worse(self):
        masks = np.zeros((2, 8, 8), dtype=np.uint8)
        masks[1, 2:6, 2:6] = 1
        perfect = np.linspace(0.0, 0.1, masks.size, dtype=np.float32).reshape(masks.shape)
        perfect[masks > 0] = 1.0
        reversed_map = 1.0 - perfect
        self.assertGreater(compute_aupro(perfect, masks), compute_aupro(reversed_map, masks))


if __name__ == "__main__":
    unittest.main()
