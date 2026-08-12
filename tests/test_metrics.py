import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from defectfusion.mvtec import compute_aupro, compute_binary_auroc_aupr, compute_binary_metrics, evaluate_mvtec, evaluate_samples
from defectfusion.pipeline import NormalTrainingView


class AuproTest(unittest.TestCase):
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
        self.assertEqual(metrics["pixel_metric_images"], 1)
        self.assertEqual(metrics["good_decision_threshold"], 0.25)
        self.assertEqual(metrics["good_decision_threshold_source"], "normal_validation_max")
        self.assertEqual(metrics["good_decision_quantile"], 1.0)
        self.assertEqual(metrics["good_decision_reference_images"], 1)
        self.assertEqual(metrics["pixel_auroc"], 1.0)

    def test_training_augmentation_scores_use_requested_decision_quantile(self):
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
                decision_threshold_source="normal_training_augmented_quantile",
                decision_threshold_quantile=0.995,
            )

        self.assertAlmostEqual(metrics["good_decision_threshold"], np.quantile([0.1, 0.2, 0.4, 0.8], 0.995))
        self.assertEqual(metrics["good_decision_threshold_source"], "normal_training_augmented_quantile")
        self.assertEqual(metrics["good_decision_quantile"], 0.995)
        self.assertEqual(metrics["good_decision_reference_images"], 4)
        self.assertEqual(metrics["good_accuracy"], 1.0)

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
