import unittest

from defectfusion.settings import image_size_overrides, unit_interval_overrides


class SettingsTest(unittest.TestCase):
    def test_image_size_overrides_parse_cli_values(self):
        self.assertEqual(
            image_size_overrides(["macaroni2=896", "pcb3=768"]),
            {"macaroni2": 896, "pcb3": 768},
        )

    def test_image_size_overrides_parse_config_mapping(self):
        self.assertEqual(image_size_overrides({"macaroni2": 896}), {"macaroni2": 896})

    def test_image_size_overrides_reject_invalid_values(self):
        for value in (["macaroni2"], ["macaroni2=0"], ["macaroni2=large"]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                image_size_overrides(value)

    def test_unit_interval_overrides_parse_cli_and_config_values(self):
        self.assertEqual(unit_interval_overrides(["cable=0.10", "transistor=0.15"]), {"cable": 0.1, "transistor": 0.15})
        self.assertEqual(unit_interval_overrides({"cable": 0.1}), {"cable": 0.1})

    def test_unit_interval_overrides_reject_invalid_values(self):
        for value in (["cable"], ["cable=-0.1"], ["cable=1.1"], ["cable=strong"]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                unit_interval_overrides(value, "Pixel ANoCo weight")


if __name__ == "__main__":
    unittest.main()
