import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_benchmark_tables import build_tables, collect_benchmark_results


class BenchmarkTablesTest(unittest.TestCase):
    @staticmethod
    def _write_result(root: Path, relative: str, *, balanced: float, method_shots: int):
        result_dir = root / relative
        result_dir.mkdir(parents=True)
        category = {
            "category": "candle",
            "dataset": "visa",
            "normal_shots": method_shots,
            "seed": 42,
            "images": 200,
            "good_images": 100,
            "defect_images": 100,
            "good_accuracy": balanced + 0.02,
            "defect_recall": balanced - 0.02,
            "balanced_accuracy": balanced,
            "image_auroc": balanced + 0.05,
            "pixel_auroc": balanced + 0.04,
            "normal_decision_calibration": "leave-one-out",
            "good_decision_quantile": 0.995,
            "good_decision_quantile_method": "higher",
            "map_postprocess": "none",
            "timing_seconds": {"total": 10.0},
        }
        payload = {
            "macro_average": {
                "good_accuracy": category["good_accuracy"],
                "defect_recall": category["defect_recall"],
                "balanced_accuracy": balanced,
                "image_auroc": category["image_auroc"],
                "pixel_auroc": category["pixel_auroc"],
            },
            "categories": [category],
        }
        (result_dir / "results.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_collects_main_and_distilled_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_result(root, "main/1shot/visa", balanced=0.70, method_shots=1)
            self._write_result(root, "distillation/visa/evaluation", balanced=0.80, method_shots=8)
            experiments, categories, warnings = collect_benchmark_results(root)

        self.assertEqual(warnings, [])
        self.assertEqual(len(experiments), 2)
        self.assertEqual(len(categories), 2)
        self.assertEqual(experiments[0]["method"], "main_pca")
        self.assertEqual(experiments[1]["method"], "distilled_lora")
        self.assertFalse(experiments[0]["best_balanced_accuracy"])
        self.assertTrue(experiments[1]["best_balanced_accuracy"])

    def test_writes_csv_and_markdown_best_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_result(root, "main/4shot/visa", balanced=0.75, method_shots=4)
            self._write_result(root, "main/8shot/visa", balanced=0.85, method_shots=8)
            paths = build_tables(root, root / "tables")
            with paths["best_balanced"].open(encoding="utf-8-sig", newline="") as stream:
                best_balanced = list(csv.DictReader(stream))
            with paths["best"].open(encoding="utf-8-sig", newline="") as stream:
                per_metric = list(csv.DictReader(stream))
            markdown = paths["markdown"].read_text(encoding="utf-8")

        self.assertEqual(len(best_balanced), 1)
        self.assertEqual(best_balanced[0]["normal_shots"], "8")
        self.assertTrue(any(row["metric"] == "balanced_accuracy" for row in per_metric))
        self.assertIn("## VISA", markdown)
        self.assertIn("**85.00**", markdown)


if __name__ == "__main__":
    unittest.main()
