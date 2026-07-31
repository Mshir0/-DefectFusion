import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from defectfusion.mvtec import evaluate_samples
from defectfusion.visa import load_visa_categories


class VisaTest(unittest.TestCase):
    def test_loads_raw_release_layout_without_split_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "candle" / "Data"
            normal_dir = data / "Images" / "Normal"
            anomaly_dir = data / "Images" / "Anomaly"
            mask_dir = data / "Masks" / "Anomaly"
            normal_dir.mkdir(parents=True)
            anomaly_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            Image.new("RGB", (2, 2)).save(normal_dir / "normal.jpg")
            Image.new("RGB", (2, 2)).save(anomaly_dir / "broken.jpg")
            Image.new("L", (2, 2)).save(mask_dir / "broken.png")

            categories = load_visa_categories(root)

        self.assertEqual([item.name for item in categories], ["candle"])
        self.assertEqual(len(categories[0].normal_images), 1)
        anomalous = [item for item in categories[0].test_samples if item.anomalous]
        self.assertEqual(anomalous[0].mask.name, "broken.png")

    def test_loads_official_one_class_split_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "split_csv" / "1cls.csv"
            split.parent.mkdir()
            split.write_text(
                "\ufeffobject,split,label,image,mask\n"
                "candle,train,normal,candle/train.png,\n"
                "candle,test,normal,candle/good.png,\n"
                "candle,test,anomaly,candle/bad.png,candle/mask.png\n",
                encoding="utf-8",
            )

            categories = load_visa_categories(root)

        self.assertEqual([item.name for item in categories], ["candle"])
        self.assertEqual(categories[0].normal_images, (root / "candle" / "train.png",))
        samples = {item.image.name: item for item in categories[0].test_samples}
        self.assertFalse(samples["good.png"].anomalous)
        self.assertEqual(samples["bad.png"].mask, root / "candle" / "mask.png")

    def test_visa_samples_use_csv_masks_for_pixel_metrics(self):
        class Fusion:
            def predict(self, image):
                anomalous = Path(image).stem == "bad"
                score = np.zeros((2, 2), dtype=np.float32)
                if anomalous:
                    score[0, 0] = 1.0
                return {
                    "image": image, "anomaly_score": float(anomalous),
                    "anomaly_map": score.tolist(), "defect_type": "unknown",
                    "defect_type_score": 0.0,
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good, bad, mask = root / "good.png", root / "bad.png", root / "mask.png"
            Image.new("RGB", (2, 2)).save(good)
            Image.new("RGB", (2, 2)).save(bad)
            mask_data = np.zeros((2, 2), dtype=np.uint8)
            mask_data[0, 0] = 255
            Image.fromarray(mask_data).save(mask)
            output = root / "results.json"
            samples = [(good, "good", False, None), (bad, "anomaly", True, mask)]

            metrics = evaluate_samples(Fusion(), "candle", samples, output, progress=False)

        self.assertEqual(metrics["image_auroc"], 1.0)
        self.assertEqual(metrics["image_f1_max"], 1.0)
        self.assertEqual(metrics["pixel_auroc"], 1.0)
        self.assertEqual(metrics["pixel_f1_max"], 1.0)

    def test_visa_type_metrics_include_precision_recall_and_f1(self):
        class Fusion:
            def predict(self, image):
                predicted = {"a1": "scratch", "a2": "dent", "b1": "dent"}[Path(image).stem]
                score = np.ones((2, 2), dtype=np.float32)
                return {
                    "image": image, "anomaly_score": 1.0,
                    "anomaly_map": score.tolist(), "defect_type": predicted,
                    "defect_type_score": 1.0,
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = []
            for name, truth in (("a1", "scratch"), ("a2", "scratch"), ("b1", "dent")):
                image = root / f"{name}.png"
                Image.new("RGB", (2, 2)).save(image)
                samples.append((image, truth, True, None))
            metrics = evaluate_samples(Fusion(), "widget", samples, root / "results.json", progress=False)

        self.assertAlmostEqual(metrics["defect_type_accuracy"], 2.0 / 3.0)
        self.assertIn("defect_type_macro_precision", metrics)
        self.assertIn("defect_type_macro_recall", metrics)
        self.assertIn("defect_type_macro_f1", metrics)
        self.assertIn("defect_type_weighted_f1", metrics)

    def test_type_prototypes_are_excluded_only_from_type_metrics(self):
        class Fusion:
            def predict(self, image):
                stem = Path(image).stem
                anomalous = stem != "good"
                predicted = "scratch" if stem in {"prototype", "heldout"} else "dent"
                score = np.zeros((2, 2), dtype=np.float32)
                if anomalous:
                    score[0, 0] = 1.0
                return {
                    "image": image, "anomaly_score": float(anomalous),
                    "anomaly_map": score.tolist(), "defect_type": predicted,
                    "defect_type_score": 1.0,
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = []
            good = root / "good.png"
            Image.new("RGB", (2, 2)).save(good)
            samples.append((good, "good", False, None))
            for name, truth in (("prototype", "scratch"), ("heldout", "scratch"), ("dent", "dent")):
                image = root / f"{name}.png"
                Image.new("RGB", (2, 2)).save(image)
                mask = root / f"{name}_mask.png"
                Image.fromarray(np.array([[255, 0], [0, 0]], dtype=np.uint8)).save(mask)
                samples.append((image, truth, True, mask))
            metrics = evaluate_samples(
                Fusion(), "widget", samples, root / "results.json", progress=False,
                excluded_type_images=[root / "prototype.png"],
            )

        self.assertEqual(metrics["images"], 4)
        self.assertAlmostEqual(metrics["defect_type_accuracy"], 1.0)
        self.assertEqual(metrics["defect_type_labels"], ["dent", "scratch"])
        self.assertEqual(sum(sum(row) for row in metrics["defect_type_confusion_matrix"]), 2)
        self.assertEqual(metrics["pixel_auroc"], 1.0)


if __name__ == "__main__":
    unittest.main()
