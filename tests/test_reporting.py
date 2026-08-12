import csv
import tempfile
import unittest
from pathlib import Path

from defectfusion.reporting import experiment_output_dir, write_metrics_csv


class ReportingTest(unittest.TestCase):
    def test_filename_output_becomes_same_stem_directory(self):
        self.assertEqual(
            experiment_output_dir("outputs/experiment.jsonl"),
            Path("outputs/experiment"),
        )
        self.assertEqual(experiment_output_dir("outputs/experiment"), Path("outputs/experiment"))

    def test_summary_csv_contains_categories_and_macro_average(self):
        metrics = [{
            "category": "bottle", "images": 10, "image_auroc": 0.9,
            "image_f1_max": 0.85, "pixel_auroc": 0.8, "pixel_f1_max": 0.75,
            "good_images": 4, "good_decision_images": 4,
            "good_predicted_normal": 3, "good_predicted_anomaly": 1, "good_accuracy": 0.75,
            "defect_images": 6, "pixel_metric_images": 6,
            "good_decision_threshold": 0.42,
            "good_decision_threshold_source": "normal_reference_max",
            "good_decision_quantile": 1.0,
            "good_decision_reference_images": 8,
            "defect_type_accuracy": 0.7, "defect_type_macro_precision": 0.6,
            "defect_type_macro_recall": 0.65, "defect_type_macro_f1": 0.62,
            "defect_type_weighted_f1": 0.68,
            "timing_seconds": {"total": 12.5},
            "memory_patch_count": 123, "memory_bytes": 456,
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            write_metrics_csv(path, metrics, {
                "image_auroc": 0.9, "image_f1_max": 0.85,
                "pixel_auroc": 0.8, "pixel_f1_max": 0.75,
            })
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["category"], "bottle")
        self.assertEqual(rows[0]["image_f1_max"], "0.85")
        self.assertEqual(rows[0]["pixel_f1_max"], "0.75")
        self.assertEqual(rows[0]["good_images"], "4")
        self.assertEqual(rows[0]["good_decision_images"], "4")
        self.assertEqual(rows[0]["good_predicted_normal"], "3")
        self.assertEqual(rows[0]["good_predicted_anomaly"], "1")
        self.assertEqual(rows[0]["good_accuracy"], "0.75")
        self.assertEqual(rows[0]["pixel_metric_images"], "6")
        self.assertEqual(rows[0]["good_decision_threshold"], "0.42")
        self.assertEqual(rows[0]["good_decision_threshold_source"], "normal_reference_max")
        self.assertEqual(rows[0]["good_decision_quantile"], "1.0")
        self.assertEqual(rows[0]["total_seconds"], "12.5")
        self.assertEqual(rows[0]["memory_patch_count"], "123")
        self.assertEqual(rows[0]["memory_bytes"], "456")
        self.assertEqual(rows[0]["defect_type_macro_precision"], "0.6")
        self.assertEqual(rows[0]["defect_type_macro_recall"], "0.65")
        self.assertEqual(rows[0]["defect_type_weighted_f1"], "0.68")
        self.assertEqual(rows[1]["category"], "macro_average")
        self.assertEqual(rows[1]["image_auroc"], "0.9")
        self.assertEqual(rows[1]["image_f1_max"], "0.85")
        self.assertEqual(rows[1]["pixel_f1_max"], "0.75")


if __name__ == "__main__":
    unittest.main()
