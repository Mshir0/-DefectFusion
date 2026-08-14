import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

if "defectfusion.features" not in sys.modules:
    features_stub = types.ModuleType("defectfusion.features")
    features_stub.DinoFeatureExtractor = object
    sys.modules["defectfusion.features"] = features_stub

from defectfusion.cli import _leave_one_out_normal_scores, main
from defectfusion.pipeline import DefectFusion, NormalTrainingView


class LeaveOneOutThresholdCalibrationTest(unittest.TestCase):
    @staticmethod
    def _detector(extractor):
        return DefectFusion(
            extractor,
            anomaly_method="pca",
            image_score="mean",
            pixel_image_size=16,
            image_head_image_size=16,
        )

    def test_each_source_contributes_one_max_score_and_zero_fit_augment_works(self):
        class Extractor:
            image_size = 64
            positional_basis = None
            resize_mode = "direct"
            device = None

            def extract(self, image):
                value = float(np.asarray(image, dtype=np.float32)[0, 0, 0])
                return np.full((4, 1), value, dtype=np.float32), (2, 2)

        class Subspace:
            score_center = 0.0
            score_scale = 1.0

            def __init__(self, **_kwargs):
                pass

            def fit(self, _features):
                return self

            def score(self, features):
                return np.asarray(features)[:, 0]

        def augment(paths, count, _augmentations, _seed, *, include_original=True):
            if include_original:
                # This is the exact return type used by fit_augment_count=0.
                return list(paths)
            base = int(Path(paths[0]).stem)
            views = []
            for index in range(count):
                image = Image.new("RGB", (4, 4), (base + 10 * (index + 1), 0, 0))
                views.append(NormalTrainingView(image))
            return views

        progress = []
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for value in (1, 2, 3):
                path = Path(directory) / f"{value}.png"
                Image.new("RGB", (4, 4), (value, 0, 0)).save(path)
                paths.append(path)
            extractor = Extractor()
            with patch("defectfusion.model.NormalSubspace", Subspace), patch(
                "defectfusion.cli._augment_normal_images", side_effect=augment
            ):
                scores = _leave_one_out_normal_scores(
                    paths,
                    build_fusion=lambda: self._detector(extractor),
                    fit_augment_count=0,
                    decision_augment_count=2,
                    augmentations=["rotate"],
                    fit_seed=42,
                    decision_seed=142,
                    progress=lambda fold, total, _path, count: progress.append((fold, total, count)),
                )
                robust_scores = _leave_one_out_normal_scores(
                    paths,
                    build_fusion=lambda: self._detector(extractor),
                    fit_augment_count=0,
                    decision_augment_count=2,
                    decision_view_quantile=0.5,
                    augmentations=["rotate"],
                    fit_seed=42,
                    decision_seed=142,
                )

        np.testing.assert_allclose(scores, [21.0, 22.0, 23.0])
        np.testing.assert_allclose(robust_scores, [11.0, 12.0, 13.0])
        self.assertEqual(progress, [(1, 3, 3), (2, 3, 3), (3, 3, 3)])
        self.assertEqual(extractor.image_size, 16)

    def test_requires_at_least_two_independent_sources(self):
        with self.assertRaisesRegex(ValueError, "at least two normal shots"):
            _leave_one_out_normal_scores(
                ["only.png"],
                build_fusion=lambda: None,
                fit_augment_count=0,
                decision_augment_count=0,
                augmentations=[],
                fit_seed=42,
                decision_seed=142,
            )

        with self.assertRaisesRegex(ValueError, "view quantile"):
            _leave_one_out_normal_scores(
                ["first.png", "second.png"],
                build_fusion=lambda: None,
                fit_augment_count=0,
                decision_augment_count=0,
                decision_view_quantile=0.0,
                augmentations=[],
                fit_seed=42,
                decision_seed=142,
            )

    def test_json_config_rejects_invalid_calibration_choices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for key, value, expected in (
                ("normal_decision_calibration", "invalid", "must be augmentation or leave-one-out"),
                ("normal_decision_quantile_method", "invalid", "must be linear or higher"),
                ("normal_decision_view_quantile", 0, "must be in (0, 1]"),
            ):
                config = root / f"{key}.json"
                config.write_text(json.dumps({key: value}), encoding="utf-8")
                with patch("defectfusion.cli.DinoFeatureExtractor") as extractor, patch(
                    "sys.stderr"
                ) as stderr:
                    with self.assertRaisesRegex(SystemExit, "2"):
                        main([
                            "--config", str(config), "evaluate-mvtec",
                            "--data-root", str(root),
                        ])
                    message = "".join(str(call) for call in stderr.write.call_args_list)
                extractor.assert_not_called()
                self.assertIn(expected, message)


if __name__ == "__main__":
    unittest.main()
