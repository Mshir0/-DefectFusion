import tempfile
import unittest
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn

from distill_dinov3 import (
    LoRALinear,
    _mvtec_categories,
    _visa_categories,
    compute_distillation_losses,
    configure_student_trainable,
    dataset_record_groups,
    discover_records,
    _engine_metric_summary,
    _selected_category_records,
    _selected_defect_paths,
    evaluate_distilled_students,
    inject_lora,
    load_lora_adapter_into_backbone,
    masks_to_patch_weights,
    resolve_layer_pairs,
    save_lora_adapter,
    validate_args,
)


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, inputs):
        return self.value(self.key(self.query(inputs)))


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = Attention(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, inputs):
        return self.norm(inputs + self.attention(inputs))


class TinyBackbone(nn.Module):
    def __init__(self, depth=4, dim=8):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=depth, hidden_size=dim, patch_size=2)
        self.patch_embed = nn.Linear(dim, dim)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(Block(dim) for _ in range(depth))


class DistillationTests(unittest.TestCase):
    def test_engine_metric_summary_matches_cli_macro_convention(self):
        summary = _engine_metric_summary([
            {"image_auroc": 0.8, "pixel_auroc": 0.6},
            {"image_auroc": 1.0, "pixel_auroc": 0.9},
            {"image_aupr": 0.7},
        ])
        self.assertAlmostEqual(summary["image_auroc"], 0.9)
        self.assertAlmostEqual(summary["pixel_auroc"], 0.75)
        self.assertAlmostEqual(summary["image_aupr"], 0.7)

    def test_evaluation_category_and_defect_selection(self):
        categories = [
            SimpleNamespace(name="bottle"),
            SimpleNamespace(name="cable"),
        ]
        self.assertEqual([item.name for item in _selected_category_records(categories, ["cable"])], ["cable"])
        records = [
            SimpleNamespace(image_path="normal.png", is_anomaly=False),
            SimpleNamespace(image_path="used-defect.png", is_anomaly=True),
        ]
        self.assertEqual(_selected_defect_paths(records), ["used-defect.png"])

    def test_post_training_evaluation_uses_engine_output_layout(self):
        class FakeExtractor:
            instances = []

            def __init__(self, model_name, **kwargs):
                self.model_name = model_name
                self.device = kwargs["device"]
                self.model = TinyBackbone()
                self.__class__.instances.append(self)

        class FakeFusion:
            def __init__(self, extractor, **kwargs):
                self.extractor = extractor
                self.normal_paths = []

            def fit_normal(self, paths):
                self.normal_paths = list(paths)
                return self

            def memory_stats(self):
                return {"patch_count": 0, "bytes": 0}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            category = root / "data" / "bottle"
            normal_dir = category / "train" / "good"
            normal_dir.mkdir(parents=True)
            normal = normal_dir / "normal.png"
            Image.new("RGB", (8, 8)).save(normal)
            output = root / "output"
            adapter = output / "bottle" / "lora_adapter.pt"
            adapter.parent.mkdir(parents=True)
            adapter.write_bytes(b"adapter")
            used_defect = category / "test" / "crack" / "used.png"
            used_defect.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(used_defect)
            calls = {}

            def fake_evaluate_mvtec(fusion, category_dir, result_path, **kwargs):
                calls["normal_paths"] = fusion.normal_paths
                calls["excluded"] = kwargs["excluded_images"]
                calls["normal_reference_images"] = kwargs["normal_reference_images"]
                Path(result_path).write_text("[]\n", encoding="utf-8")
                return {
                    "category": Path(category_dir).name,
                    "images": 2,
                    "image_auroc": 0.8,
                    "image_aupr": 0.7,
                    "image_f1_max": 0.6,
                    "pixel_auroc": 0.5,
                    "pixel_aupr": 0.4,
                    "pixel_aupro": 0.3,
                    "pixel_f1_max": 0.2,
                    "timing_seconds": {"total": 0.1},
                }

            args = SimpleNamespace(
                dataset="mvtec", data_root=str(root / "data"), split_csv=None,
                categories=["bottle"], device="cpu", normal_shots=1,
                defect_shots=1, seed=42, student_model="base-vit-s", eval_image_size=448,
                eval_feature_layers="1,6,12", eval_resize_mode="direct",
                eval_layer_aggregation="mean", eval_layer_normalization="none",
                eval_top_k_ratio=0.05, eval_image_score="mtop1p",
                eval_image_top_ratio=0.01, eval_anomaly_method="pca",
                eval_pca_residual_metric="squared_l2", eval_knn_weight=0.5,
                eval_memory_max_patches=50000, eval_normal_fit_max_patches=0,
                eval_knn_chunk_size=256, eval_knn_backend="auto",
                eval_knn_dtype="float32", eval_dual_branch=False,
            )
            groups = {
                "bottle": [
                    ImageRecord(str(normal), None, False),
                    ImageRecord(str(used_defect), None, True, "crack"),
                ],
            }
            with patch("defectfusion.features.DinoFeatureExtractor", FakeExtractor), patch(
                "defectfusion.pipeline.DefectFusion", FakeFusion
            ), patch("defectfusion.mvtec.evaluate_mvtec", fake_evaluate_mvtec), patch(
                "distill_dinov3.load_lora_adapter_into_backbone"
            ) as load_adapter:
                summary = evaluate_distilled_students(args, output, groups)

            self.assertEqual(calls["normal_paths"], [str(normal)])
            self.assertEqual(calls["excluded"], [str(used_defect)])
            self.assertEqual(calls["normal_reference_images"], [str(normal)])
            self.assertEqual(FakeExtractor.instances[0].model_name, "base-vit-s")
            load_adapter.assert_called_once_with(FakeExtractor.instances[0].model, adapter, base_model="base-vit-s")
            self.assertAlmostEqual(summary["macro_average"]["pixel_aupro"], 0.3)
            self.assertTrue((output / "evaluation" / "results.json").is_file())
            self.assertTrue((output / "evaluation" / "summary.csv").is_file())

    def test_cli_validation_rejects_invalid_training_values(self):
        parser = ArgumentParser()
        args = SimpleNamespace(
            image_size=448, epochs=1, batch_size=1, centroid_batch_size=1,
            normal_shots=1, defect_shots=0, max_normal_images=0,
            max_defect_images=0, dataset="mvtec", data_root="datasets/mvtec",
            split_csv=None, normal_dir=None, defect_dir=None, mask_dir=None,
            categories=None, last_n_blocks=4, lora_rank=8, lora_alpha=16.0,
            lora_dropout=1.0, lr=1e-4, head_lr=1e-3, grad_clip=1.0,
            weight_decay=1e-4, lambda_feature=1.0, lambda_map=1.0,
            mask_alpha=2.0, margin=0.2, top_ratio=0.01, num_workers=0,
            save_every=0, feature_layers="1,6,12", teacher_layers=None,
            student_layers=None, evaluate=True, eval_image_size=None,
            eval_feature_layers=None, eval_resize_mode="direct",
            eval_layer_aggregation="mean", eval_layer_normalization="none",
            eval_anomaly_method="pca", eval_pca_residual_metric="squared_l2",
            eval_dual_branch=False, eval_knn_weight=0.5,
            eval_memory_max_patches=50000, eval_normal_fit_max_patches=0,
            eval_knn_chunk_size=256, eval_knn_backend="auto",
            eval_knn_dtype="float32", eval_top_k_ratio=0.05,
            eval_image_score="mtop1p", eval_image_top_ratio=0.01,
        )
        with self.assertRaises(SystemExit):
            validate_args(parser, args)

        args.lora_dropout = 0.05
        validate_args(parser, args)

    def test_lora_only_targets_final_blocks(self):
        model = TinyBackbone(depth=4)
        targets = inject_lora(model, rank=2, alpha=4, last_n_blocks=2)

        self.assertTrue(targets)
        self.assertTrue(all("layer.2" in name or "layer.3" in name for name in targets))
        self.assertIsInstance(model.encoder.layer[1].attention.query, nn.Linear)
        self.assertIsInstance(model.encoder.layer[2].attention.query, LoRALinear)

    def test_trainable_configuration_freezes_patch_embedding(self):
        model = TinyBackbone(depth=4)
        configure_student_trainable(model, adaptation="lora", last_n_blocks=1, lora_rank=2)

        self.assertFalse(model.patch_embed.weight.requires_grad)
        self.assertFalse(model.encoder.layer[0].norm.weight.requires_grad)
        self.assertTrue(model.encoder.layer[3].attention.query.lora_A.weight.requires_grad)
        self.assertFalse(model.encoder.layer[3].attention.query.base.weight.requires_grad)
        with self.assertRaises(ValueError):
            configure_student_trainable(TinyBackbone(depth=4), adaptation="local", last_n_blocks=1, lora_rank=2)

    def test_adapter_artifact_contains_only_lora_tensors(self):
        source = TinyBackbone(depth=4)
        targets = configure_student_trainable(source, adaptation="lora", last_n_blocks=1, lora_rank=2)
        for name, parameter in source.named_parameters():
            if "lora_" in name:
                parameter.data.fill_(0.125 if name.endswith("lora_A.weight") else -0.25)
        config = {
            "adaptation": "lora",
            "lora_rank": 2,
            "lora_alpha": 16.0,
            "lora_dropout": 0.05,
            "last_n_blocks": 1,
            "lora_targets": targets,
            "student_model": "tiny-vit-s",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = save_lora_adapter(root, SimpleNamespace(backbone=source), config)
            saved = torch.load(adapter, map_location="cpu")
            self.assertEqual(set(path.name for path in root.iterdir()), {"lora_adapter.pt", "training_config.json"})
            self.assertTrue(saved["lora_state"])
            self.assertTrue(all("lora_" in name for name in saved["lora_state"]))
            self.assertFalse(any("patch_embed" in name for name in saved["lora_state"]))

            restored = TinyBackbone(depth=4)
            load_lora_adapter_into_backbone(restored, adapter, base_model="tiny-vit-s")
            expected = dict(source.named_parameters())
            actual = dict(restored.named_parameters())
            for name, value in saved["lora_state"].items():
                self.assertTrue(torch.equal(actual[name], expected[name]))
            self.assertTrue(all(not parameter.requires_grad for parameter in restored.parameters()))
            with self.assertRaises(ValueError):
                load_lora_adapter_into_backbone(TinyBackbone(depth=4), adapter, base_model="wrong-vit-s")

    def test_layer_pair_validation_and_depth_mapping(self):
        teacher = TinyBackbone(depth=8)
        student = TinyBackbone(depth=4)
        pairs = resolve_layer_pairs(teacher, student, layers=(1, 4, 8))
        self.assertEqual(pairs, ((1, 1), (4, 4), (8, 4)))
        with self.assertRaises(ValueError):
            resolve_layer_pairs(teacher, student, layers=(9,))

    def test_weighted_losses_are_finite_and_differentiable(self):
        teacher_layers = [torch.randn(2, 4, 6), torch.randn(2, 4, 6)]
        student_layers = [torch.randn(2, 4, 6, requires_grad=True) for _ in range(2)]
        student_features = torch.randn(2, 4, 3, requires_grad=True)
        losses = compute_distillation_losses(
            teacher_layers,
            student_layers,
            student_features,
            teacher_centroid=torch.randn(6),
            student_centroid=torch.randn(3),
            labels=torch.tensor([False, True]),
            patch_weights=torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 3.0, 3.0, 1.0]]),
        )
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        self.assertIsNotNone(student_features.grad)

    def test_masks_and_record_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normal = root / "normal"
            defects = root / "synthetic" / "images" / "crack"
            masks = root / "synthetic" / "masks" / "crack"
            normal.mkdir(parents=True)
            defects.mkdir(parents=True)
            masks.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(normal / "good.png")
            Image.new("RGB", (8, 8)).save(defects / "000.png")
            mask = Image.new("L", (8, 8), color=0)
            for row in range(4):
                for column in range(4):
                    mask.putpixel((column, row), 255)
            mask.save(masks / "000_mask.png")

            records = discover_records(normal, defects, masks)
            self.assertEqual(len(records), 2)
            self.assertTrue(records[1].mask_path.endswith("000_mask.png"))
            weights = masks_to_patch_weights([mask, None], (8, 8), (2, 2), torch.device("cpu"), 2.0)
            self.assertEqual(tuple(weights.shape), (2, 4))
            self.assertEqual(weights[0, 0].item(), 3.0)
            self.assertTrue(torch.equal(weights[1], torch.ones(4)))

    def test_mvtec_loader_and_zero_defect_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            category = root / "bottle"
            normal = category / "train" / "good"
            defect = category / "test" / "crack"
            masks = category / "ground_truth" / "crack"
            normal.mkdir(parents=True)
            defect.mkdir(parents=True)
            masks.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(normal / "000.png")
            Image.new("RGB", (8, 8)).save(normal / "001.png")
            Image.new("RGB", (8, 8)).save(defect / "002.png")
            Image.new("L", (8, 8), color=255).save(masks / "002_mask.png")

            categories = _mvtec_categories(root)
            self.assertEqual(categories[0].name, "bottle")
            self.assertEqual(len(categories[0].normal_images), 2)
            self.assertTrue(categories[0].defect_samples[0].mask_path.endswith("002_mask.png"))

            args = SimpleNamespace(
                dataset="mvtec", data_root=str(root), split_csv=None,
                categories=["bottle"], normal_shots=1, defect_shots=0,
                seed=42, max_normal_images=0, max_defect_images=0,
            )
            groups, counts = dataset_record_groups(args)
            self.assertEqual(counts["bottle"], {"normal": 1, "defect": 0})
            self.assertFalse(any(record.is_anomaly for record in groups["bottle"]))
            args.defect_shots = 1
            groups, counts = dataset_record_groups(args)
            self.assertEqual(counts["bottle"]["defect"], 1)
            self.assertTrue(groups["bottle"][-1].is_anomaly)

    def test_visa_official_split_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_dir = root / "split_csv"
            images = root / "candle"
            split_dir.mkdir(parents=True)
            images.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(images / "train.png")
            Image.new("RGB", (8, 8)).save(images / "bad.png")
            Image.new("L", (8, 8), color=255).save(images / "mask.png")
            (split_dir / "1cls.csv").write_text(
                "object,split,label,image,mask,defect_type\n"
                "candle,train,normal,candle/train.png,,\n"
                "candle,test,anomaly,candle/bad.png,candle/mask.png,melted\n",
                encoding="utf-8",
            )

            categories = _visa_categories(root)
            self.assertEqual(categories[0].name, "candle")
            self.assertEqual(categories[0].normal_images, (images / "train.png",))
            self.assertEqual(categories[0].defect_samples[0].defect_type, "melted")
            self.assertEqual(categories[0].defect_samples[0].mask_path, str(images / "mask.png"))

    def test_visa_raw_layout_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normal = root / "candle" / "Data" / "Images" / "Normal"
            defect = root / "candle" / "Data" / "Images" / "Anomaly"
            masks = root / "candle" / "Data" / "Masks" / "Anomaly"
            normal.mkdir(parents=True)
            defect.mkdir(parents=True)
            masks.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(normal / "good.png")
            Image.new("RGB", (8, 8)).save(defect / "bad.png")
            Image.new("L", (8, 8), color=255).save(masks / "bad_mask.png")

            categories = _visa_categories(root)
            self.assertEqual(categories[0].name, "candle")
            self.assertEqual(len(categories[0].normal_images), 1)
            self.assertTrue(categories[0].defect_samples[0].mask_path.endswith("bad_mask.png"))


if __name__ == "__main__":
    unittest.main()
