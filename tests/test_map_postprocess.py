import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from defectfusion.pipeline import DefectFusion


class MapPostprocessTest(unittest.TestCase):
    def test_crf_uses_full_resolution_rgb_and_restores_patch_grid(self):
        calls = {}

        class FakeDenseCRF2D:
            def __init__(self, width, height, classes):
                calls["init"] = (width, height, classes)
                self.width = width
                self.height = height

            def setUnaryEnergy(self, unary):
                calls["unary"] = np.asarray(unary)

            def addPairwiseGaussian(self, **kwargs):
                calls["gaussian"] = kwargs

            def addPairwiseBilateral(self, **kwargs):
                calls["bilateral"] = kwargs

            def inference(self, iterations):
                calls["iterations"] = iterations
                count = self.width * self.height
                return [np.zeros(count), np.linspace(0.0, 1.0, count)]

        def unary_from_softmax(probabilities):
            calls["probabilities"] = np.asarray(probabilities)
            return -np.log(np.clip(probabilities, 1e-6, 1.0))

        package = types.ModuleType("pydensecrf")
        densecrf = types.ModuleType("pydensecrf.densecrf")
        utils = types.ModuleType("pydensecrf.utils")
        densecrf.DenseCRF2D = FakeDenseCRF2D
        utils.unary_from_softmax = unary_from_softmax
        package.densecrf = densecrf
        package.utils = utils

        fusion = object.__new__(DefectFusion)
        fusion.map_postprocess = "crf"
        anomaly_map = np.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]])
        image = Image.new("RGB", (6, 4), (10, 20, 30))

        modules = {
            "pydensecrf": package,
            "pydensecrf.densecrf": densecrf,
            "pydensecrf.utils": utils,
        }
        with patch.dict(sys.modules, modules):
            refined = fusion._postprocess_map(anomaly_map, image)

        self.assertEqual(refined.shape, anomaly_map.shape)
        self.assertEqual(calls["init"], (6, 4, 2))
        self.assertEqual(calls["probabilities"].shape, (2, 4, 6))
        np.testing.assert_allclose(calls["probabilities"].sum(axis=0), 1.0)
        self.assertEqual(calls["gaussian"], {"sxy": 3, "compat": 3})
        self.assertEqual(calls["bilateral"]["sxy"], 50)
        self.assertEqual(calls["bilateral"]["srgb"], 10)
        self.assertEqual(calls["bilateral"]["compat"], 5)
        self.assertEqual(calls["bilateral"]["rgbim"].shape, (4, 6, 3))
        self.assertEqual(calls["iterations"], 5)


if __name__ == "__main__":
    unittest.main()
