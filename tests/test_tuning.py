import json
import tempfile
import unittest
from pathlib import Path

from defectfusion.tuning import underperforming_categories


class TuningTest(unittest.TestCase):
    def test_selects_categories_below_metric_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text(json.dumps({"categories": [
                {"category": "candle", "balanced_accuracy": 0.57},
                {"category": "pcb4", "balanced_accuracy": 0.91},
                {"category": "missing"},
            ]}), encoding="utf-8")
            self.assertEqual(
                underperforming_categories(path, "balanced_accuracy", 0.8),
                ["candle"],
            )

    def test_rejects_non_result_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no category metrics"):
                underperforming_categories(path, "defect_type_macro_f1", 0.6)


if __name__ == "__main__":
    unittest.main()
