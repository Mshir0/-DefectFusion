import csv
import json
import tempfile
import unittest
from pathlib import Path

from defectfusion.aggregate import collect_results, write_statistics


class AggregateResultsTest(unittest.TestCase):
    @staticmethod
    def _write_result(root: Path):
        experiment = root / "visa-normal-1shot"
        experiment.mkdir()
        categories = [
            {
                "category": "candle", "dataset": "visa", "normal_shots": 1,
                "defect_shots": 0, "seed": 42, "images": 10,
                "image_auroc": 0.9, "pixel_auroc": 0.95,
                "timing_seconds": {"prediction": 2.0, "total": 3.0},
                "memory_patch_count": 100, "memory_bytes": 1000,
            },
            {
                "category": "capsules", "dataset": "visa", "normal_shots": 1,
                "defect_shots": 0, "seed": 42, "images": 20,
                "image_auroc": 1.0, "pixel_auroc": 0.97,
                "timing_seconds": {"prediction": 4.0, "total": 5.0},
                "memory_patch_count": 200, "memory_bytes": 2000,
            },
        ]
        payload = {
            "macro_average": {
                "image_auroc": 0.95,
                "pixel_auroc": 0.96,
                "future_metric": 0.97,
            },
            "categories": categories,
        }
        (experiment / "results.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_collects_macro_and_category_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_result(root)
            macro, categories, warnings = collect_results(root)
        self.assertEqual(warnings, [])
        self.assertEqual(macro[0]["experiment"], "visa-normal-1shot")
        self.assertEqual(macro[0]["images"], 30)
        self.assertEqual(macro[0]["total_seconds"], 8.0)
        self.assertEqual(macro[0]["peak_memory_bytes"], 2000)
        self.assertEqual(macro[0]["future_metric"], 0.97)
        self.assertEqual(categories[0]["timing_prediction_seconds"], 2.0)

    def test_writes_two_csv_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_result(root)
            output = root / "aggregate"
            macro_path, category_path, count, warnings = write_statistics(root, output)
            with macro_path.open(encoding="utf-8-sig", newline="") as stream:
                macro_rows = list(csv.DictReader(stream))
            with category_path.open(encoding="utf-8-sig", newline="") as stream:
                category_rows = list(csv.DictReader(stream))
        self.assertEqual(count, 1)
        self.assertEqual(warnings, [])
        self.assertEqual(macro_rows[0]["image_auroc"], "0.95")
        self.assertEqual(len(category_rows), 2)


if __name__ == "__main__":
    unittest.main()
