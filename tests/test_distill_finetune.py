import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch import nn

from defectfusion.distill_finetune import (
    LoRALinear,
    compute_distillation_losses,
    configure_student_trainable,
    discover_records,
    inject_lora,
    masks_to_patch_weights,
    resolve_layer_pairs,
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


if __name__ == "__main__":
    unittest.main()
