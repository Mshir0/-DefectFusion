import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from defectfusion.mvtec import compute_aupro, compute_binary_auroc_aupr, compute_binary_metrics, evaluate_mvtec


class AuproTest(unittest.TestCase):
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
