import tempfile
import unittest
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

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
    inject_lora,
    masks_to_patch_weights,
    resolve_layer_pairs,
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
            student_layers=None,
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
        self.assertTrue(model.encoder.layer[0].norm.weight.requires_grad)
        self.assertTrue(model.encoder.layer[3].attention.query.lora_A.weight.requires_grad)
        self.assertFalse(model.encoder.layer[3].attention.query.base.weight.requires_grad)

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
