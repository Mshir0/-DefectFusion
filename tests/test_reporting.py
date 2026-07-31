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
            "timing_seconds": {"total": 12.5},
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
        self.assertEqual(rows[0]["total_seconds"], "12.5")
        self.assertEqual(rows[1]["category"], "macro_average")
        self.assertEqual(rows[1]["image_auroc"], "0.9")
        self.assertEqual(rows[1]["image_f1_max"], "0.85")
        self.assertEqual(rows[1]["pixel_f1_max"], "0.75")


if __name__ == "__main__":
    unittest.main()
