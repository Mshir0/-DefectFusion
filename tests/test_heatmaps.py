import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.render_heatmaps import load_prediction_rows, render_heatmap, select_prediction


class HeatmapRenderingTest(unittest.TestCase):
    def test_selects_highest_prediction_and_writes_one_heatmap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_image = root / "first.png"
            second_image = root / "second.png"
            Image.new("RGB", (20, 12), (90, 100, 110)).save(first_image)
            Image.new("RGB", (18, 10), (80, 90, 100)).save(second_image)
            rows = [
                {
                    "image": str(first_image),
                    "anomaly_score": 0.25,
                    "anomaly_map": [[0.0, 0.5], [0.25, 1.0]],
                },
                {
                    "image": str(second_image),
                    "anomaly_score": 1.25,
                    "anomaly_map": [[0.0, 0.2], [0.4, 1.0]],
                },
            ]
            predictions = root / "category.json"
            predictions.write_text(
                json.dumps({"metrics": {}, "predictions": rows}), encoding="utf-8",
            )
            selected = select_prediction(load_prediction_rows(predictions), None, "highest")
            self.assertEqual(selected["image"], str(second_image))

            output = root / "heatmap.png"
            path, low, high = render_heatmap(
                selected,
                output,
                lower_percentile=0.0,
                upper_percentile=100.0,
                vmin=None,
                vmax=None,
                colormap="turbo",
            )
            self.assertEqual(path, output)
            self.assertLess(low, high)
            self.assertEqual(set(root.glob("*.png")), {first_image, second_image, output})
            with Image.open(output) as rendered:
                self.assertEqual(rendered.size, (18, 10))
                self.assertGreater(np.asarray(rendered).std(), 0)


if __name__ == "__main__":
    unittest.main()
