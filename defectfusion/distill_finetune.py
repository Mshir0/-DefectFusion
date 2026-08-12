"""Parameter-efficient DINOv3 teacher/student distillation for few-shot AD.

This module deliberately keeps the existing DefectFusion inference pipeline
unchanged.  It trains a deployable ViT-S backbone using a frozen ViT-B
teacher, multi-layer patch-token distillation, anomaly-map distillation, and a
small normal-compactness/margin objective.  Defect images and their masks are
optional; when they are supplied, the masks increase the feature-distillation
weight in the synthetic-defect region.

Example::

    python -m defectfusion.distill_finetune \
        --normal-dir data/train/good \
        --defect-dir data/synthetic/images \
        --mask-dir data/synthetic/masks \
        --output outputs/dinov3-vit-s-distilled \
        --epochs 10 --adaptation lora

The output directory contains ``student_base`` (the untouched HF model), an
adapter/projection checkpoint, processor files, a JSON training log, and
``student_merged``.  The latter is a standard Hugging Face model directory
that can be passed directly to DefectFusion's existing ``--model`` option.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MASK_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
DEFAULT_TEACHER = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DEFAULT_STUDENT = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEFAULT_FEATURE_LAYERS = (1, 6, 12)


@dataclass(frozen=True)
class ImageRecord:
    """One image and its optional pixel mask."""

    image_path: str
    mask_path: str | None
    is_anomaly: bool


def _iter_images(root: str | Path) -> list[Path]:
    """Return image files in deterministic order, excluding common masks."""

    path = Path(root)
    if not path.exists():
        raise FileNotFoundError(f"Image directory does not exist: {path}")
    result = []
    for item in path.rglob("*"):
        if not item.is_file() or item.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative_parts = item.relative_to(path).parts[:-1]
        if any(part.lower() in {"mask", "masks", "ground_truth", "gt"} for part in relative_parts):
            continue
        stem = item.stem.lower()
        if stem.endswith(("_mask", "_label", "_seg", "_gt")):
            continue
        result.append(item)
    return sorted(result, key=lambda p: p.as_posix().lower())


def _mask_index(mask_dir: str | Path | None) -> dict[str, Path]:
    if mask_dir is None:
        return {}
    root = Path(mask_dir)
    if not root.exists():
        raise FileNotFoundError(f"Mask directory does not exist: {root}")
    index: dict[str, Path] = {}
    for item in root.rglob("*"):
        if item.is_file() and item.suffix.lower() in MASK_SUFFIXES:
            # Keep the first deterministic match.  Index both ``000_mask``
            # and ``000`` so the common MVTec/synthetic naming schemes work.
            stem = item.stem.lower()
            index.setdefault(stem, item)
            for suffix in ("_mask", "_label", "_seg", "_gt"):
                if stem.endswith(suffix):
                    index.setdefault(stem[: -len(suffix)], item)
    return index


def _find_mask(
    image: Path,
    image_root: str | Path,
    mask_dir: str | Path | None,
    index: dict[str, Path],
) -> Path | None:
    """Resolve masks named ``stem``, ``stem_mask`` or in a sibling masks dir."""

    candidates: list[Path] = []
    if mask_dir is not None:
        root = Path(mask_dir)
        try:
            relative = image.relative_to(image_root)
        except ValueError:
            relative = Path(image.name)
        # The direct basename candidates cover the usual flat synthetic set.
        for stem in (image.stem, f"{image.stem}_mask", f"{image.stem}_label", f"{image.stem}_seg"):
            for suffix in MASK_SUFFIXES:
                candidates.append(root / f"{stem}{suffix}")
        # Also try a matching relative path when image and mask trees mirror.
        for suffix in MASK_SUFFIXES:
            candidates.append(root / relative.with_suffix(suffix))
            candidates.append(root / relative.with_name(f"{relative.stem}_mask{suffix}"))
    for stem in (f"{image.stem}_mask", f"{image.stem}_label", f"{image.stem}_seg", image.stem):
        for suffix in MASK_SUFFIXES:
            candidates.append(image.with_name(f"{stem}{suffix}"))
            candidates.append(image.parent / "masks" / f"{stem}{suffix}")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return index.get(image.stem.lower())


def discover_records(
    normal_dir: str | Path | None = None,
    defect_dir: str | Path | None = None,
    mask_dir: str | Path | None = None,
    data_root: str | Path | None = None,
    max_normal_images: int = 0,
    max_defect_images: int = 0,
) -> list[ImageRecord]:
    """Discover normal/defect images from explicit directories or a data root.

    ``data_root`` accepts the common layouts ``normal`` + ``defect`` and
    ``train/good`` + ``synthetic``.  A missing defect directory is valid and
    produces a feature-only normal training set.
    """

    if data_root is not None:
        root = Path(data_root)
        if normal_dir is None:
            for candidate in (root / "normal", root / "good", root / "train" / "good"):
                if candidate.is_dir():
                    normal_dir = candidate
                    break
        if defect_dir is None:
            for candidate in (root / "defect", root / "defects", root / "anomaly", root / "synthetic", root / "train" / "defect"):
                if candidate.is_dir():
                    defect_dir = candidate
                    break
        if mask_dir is None:
            for candidate in (root / "masks", root / "mask", root / "synthetic" / "masks"):
                if candidate.is_dir():
                    mask_dir = candidate
                    break
    if normal_dir is None:
        raise ValueError("A normal image directory is required (--normal-dir or --data-root)")

    normal_paths = _iter_images(normal_dir)
    defect_root = Path(defect_dir) if defect_dir is not None else None
    defect_paths = _iter_images(defect_root) if defect_root is not None and defect_root.exists() else []
    if max_normal_images > 0:
        normal_paths = normal_paths[:max_normal_images]
    if max_defect_images > 0:
        defect_paths = defect_paths[:max_defect_images]
    if not normal_paths:
        raise ValueError(f"No normal images found in {normal_dir}")
    mask_index = _mask_index(mask_dir)
    records = [ImageRecord(str(path), None, False) for path in normal_paths]
    records.extend(
        ImageRecord(
            str(path),
            str(mask) if (mask := _find_mask(path, defect_root or path.parent, mask_dir, mask_index)) else None,
            True,
        )
        for path in defect_paths
    )
    if defect_paths and not any(record.mask_path for record in records if record.is_anomaly):
        warnings.warn("Defect images were found but no masks resolved; mask weighting will be disabled.", RuntimeWarning)
    return records


class ImageMaskDataset(Dataset):
    """PIL-backed dataset kept intentionally light for few-shot training."""

    def __init__(self, records: Sequence[ImageRecord]):
        self.records = list(records)
        if not self.records:
            raise ValueError("The distillation dataset is empty")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.image_path) as source:
            image = source.convert("RGB").copy()
        mask = None
        if record.mask_path:
            with Image.open(record.mask_path) as source:
                mask = source.convert("L").copy()
        return image, mask, bool(record.is_anomaly), record.image_path


def _collate(batch):
    images, masks, labels, paths = zip(*batch)
    return list(images), list(masks), torch.tensor(labels, dtype=torch.bool), list(paths)


def _parse_layers(value: str | Sequence[int] | None) -> tuple[int, ...]:
    if value is None:
        return DEFAULT_FEATURE_LAYERS
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",") if part.strip()]
    else:
        values = list(value)
    layers = tuple(int(part) for part in values)
    if not layers:
        raise ValueError("At least one feature layer is required")
    return layers


def _state_count(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    count = getattr(config, "num_hidden_layers", None)
    if count is None:
        count = getattr(config, "num_layers", None)
    if count is None:
        raise ValueError("The DINO model config does not expose num_hidden_layers/num_layers")
    return int(count) + 1  # embedding output is hidden_states[0]


def _resolve_index(index: int, count: int, name: str = "layer") -> int:
    resolved = count + index if index < 0 else index
    if not 0 <= resolved < count:
        raise ValueError(f"{name} index {index} is invalid for {count} hidden states")
    return resolved


def resolve_layer_pairs(
    teacher: nn.Module,
    student: nn.Module,
    layers: Sequence[int] = DEFAULT_FEATURE_LAYERS,
    teacher_layers: Sequence[int] | None = None,
    student_layers: Sequence[int] | None = None,
) -> tuple[tuple[int, int], ...]:
    """Pair teacher/student hidden states, mapping depth if model depths differ."""

    teacher_count, student_count = _state_count(teacher), _state_count(student)
    if teacher_layers is not None or student_layers is not None:
        if teacher_layers is None or student_layers is None or len(teacher_layers) != len(student_layers):
            raise ValueError("teacher_layers and student_layers must be supplied together with equal length")
        return tuple(
            (
                _resolve_index(int(t), teacher_count, "teacher layer"),
                _resolve_index(int(s), student_count, "student layer"),
            )
            for t, s in zip(teacher_layers, student_layers)
        )
    pairs = []
    for requested in _parse_layers(layers):
        teacher_index = _resolve_index(requested, teacher_count, "feature layer")
        if requested < 0:
            student_index = _resolve_index(requested, student_count, "student feature layer")
        elif 0 <= requested < student_count:
            student_index = requested
        else:
            # Preserve relative depth for a teacher with more/fewer blocks.
            denominator = max(1, teacher_count - 1)
            student_index = round(teacher_index / denominator * max(1, student_count - 1))
            student_index = _resolve_index(student_index, student_count, "mapped student layer")
        pairs.append((teacher_index, student_index))
    return tuple(pairs)


def _patch_size(model: nn.Module) -> int:
    value = getattr(getattr(model, "config", None), "patch_size", 16)
    if isinstance(value, (tuple, list)):
        value = value[0]
    return int(value)


def _strip_special_tokens(state: Tensor, pixel_values: Tensor, model: nn.Module) -> tuple[Tensor, tuple[int, int]]:
    register_count = int(getattr(getattr(model, "config", None), "num_register_tokens", 0) or 0)
    patch = _patch_size(model)
    expected_grid = (max(1, pixel_values.shape[-2] // patch), max(1, pixel_values.shape[-1] // patch))
    expected = expected_grid[0] * expected_grid[1]
    for offset in (1 + register_count, 1, 0):
        if state.shape[1] - offset == expected:
            tokens = state[:, offset:]
            return tokens, expected_grid
    # Some wrappers omit/append special tokens.  Prefer the usual CLS offset
    # and retain exactly the expected number of patch tokens.
    offset = 1 + register_count if state.shape[1] > 1 + register_count else 0
    tokens = state[:, offset:]
    if tokens.shape[1] < expected:
        raise ValueError(f"Model returned {state.shape[1]} tokens, expected at least {expected} patch tokens")
    tokens = tokens[:, -expected:]
    return tokens, expected_grid


def extract_patch_features(model: nn.Module, pixel_values: Tensor, layer_indices: Sequence[int]) -> tuple[list[Tensor], tuple[int, int]]:
    """Run a DINO model and return L2-normalized patch tokens per layer."""

    output = model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
    states = getattr(output, "hidden_states", None)
    if states is None and isinstance(output, (tuple, list)):
        # A few custom HF wrappers return hidden states as the final tuple item.
        states = output[-1]
    if states is None:
        raise RuntimeError("The backbone did not return hidden_states; check the Transformers model implementation")
    selected: list[Tensor] = []
    grid: tuple[int, int] | None = None
    for index in layer_indices:
        if not (-len(states) <= index < len(states)):
            raise ValueError(f"Hidden-state index {index} is invalid for {len(states)} returned states")
        tokens, current_grid = _strip_special_tokens(states[index], pixel_values, model)
        if grid is None:
            grid = current_grid
        elif current_grid != grid:
            raise ValueError(f"Inconsistent patch grids in hidden states: {grid} vs {current_grid}")
        selected.append(F.normalize(tokens.float(), p=2, dim=-1))
    if not selected or grid is None:
        raise ValueError("No hidden states were selected")
    return selected, grid


def aggregate_layers(layers: Sequence[Tensor]) -> Tensor:
    """Mean normalized layer features followed by token-wise normalization."""

    return F.normalize(torch.stack(list(layers), dim=0).mean(dim=0), p=2, dim=-1)


class LoRALinear(nn.Module):
    """A bias-compatible linear layer with a zero-initialized LoRA branch."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, inputs: Tensor) -> Tensor:
        return self.base(inputs) + self.lora_B(self.dropout(self.lora_A(inputs))) * self.scaling

    def merge(self) -> None:
        """Merge the adapter into the base weight for zero-overhead inference."""

        with torch.no_grad():
            delta = self.lora_B.weight @ self.lora_A.weight
            self.base.weight.add_(delta.to(self.base.weight) * self.scaling)
            self.lora_B.weight.zero_()


_QKV_LEAF_NAMES = {"qkv", "q_proj", "k_proj", "v_proj", "query", "key", "value"}


def _module_block_index(name: str) -> int | None:
    for pattern in (r"(?:layers?|blocks?|h|encoder_layer|encoder_layers)\.(\d+)", r"\.layer\.(\d+)"):
        match = re.search(pattern, name.lower())
        if match:
            return int(match.group(1))
    return None


def _set_submodule(root: nn.Module, name: str, value: nn.Module) -> None:
    parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, value)


def inject_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    last_n_blocks: int = 4,
    target_names: Sequence[str] | None = None,
) -> list[str]:
    """Inject LoRA into QKV projections of the final transformer blocks."""

    if last_n_blocks <= 0:
        raise ValueError("last_n_blocks must be positive")
    if target_names is None:
        total_blocks = int(getattr(getattr(model, "config", None), "num_hidden_layers", last_n_blocks) or last_n_blocks)
        cutoff = max(0, total_blocks - int(last_n_blocks))
        candidates = []
        for name, module in model.named_modules():
            leaf = name.rsplit(".", 1)[-1].lower()
            block = _module_block_index(name)
            if isinstance(module, nn.Linear) and leaf in _QKV_LEAF_NAMES and block is not None and block >= cutoff:
                candidates.append(name)
    else:
        candidates = list(target_names)
    if not candidates:
        sample = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)][:20]
        raise RuntimeError("No QKV linear projections found for LoRA injection. Linear modules: " + ", ".join(sample))
    injected = []
    for name in candidates:
        module = model.get_submodule(name)
        if isinstance(module, LoRALinear):
            injected.append(name)
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"LoRA target {name!r} is not nn.Linear: {type(module).__name__}")
        _set_submodule(model, name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        injected.append(name)
    return injected


def _unfreeze_last_blocks(model: nn.Module, last_n_blocks: int) -> list[str]:
    total_blocks = int(getattr(getattr(model, "config", None), "num_hidden_layers", last_n_blocks) or last_n_blocks)
    cutoff = max(0, total_blocks - int(last_n_blocks))
    changed = []
    for name, parameter in model.named_parameters():
        block = _module_block_index(name)
        if block is not None and block >= cutoff:
            parameter.requires_grad_(True)
            changed.append(name)
    if not changed:
        raise RuntimeError("Could not identify transformer blocks for local fine-tuning")
    return changed


def configure_student_trainable(
    model: nn.Module,
    adaptation: str = "lora",
    last_n_blocks: int = 4,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
) -> list[str]:
    """Freeze the backbone and enable LoRA or final-block parameters + norms."""

    if adaptation not in {"lora", "local"}:
        raise ValueError("adaptation must be 'lora' or 'local'")
    model.requires_grad_(False)
    if adaptation == "lora":
        adapted = inject_lora(model, lora_rank, lora_alpha, lora_dropout, last_n_blocks)
    else:
        adapted = _unfreeze_last_blocks(model, last_n_blocks)
    # LayerNorm affine parameters are cheap and stabilize few-shot adaptation.
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    return adapted


class DistillationStudent(nn.Module):
    """Student backbone plus one projection head per distilled layer."""

    def __init__(self, backbone: nn.Module, layer_pairs: Sequence[tuple[int, int]], teacher_dim: int, student_dim: int):
        super().__init__()
        self.backbone = backbone
        self.layer_pairs = tuple((int(t), int(s)) for t, s in layer_pairs)
        self.projections = nn.ModuleList(nn.Linear(int(student_dim), int(teacher_dim)) for _ in self.layer_pairs)

    @property
    def student_layers(self) -> tuple[int, ...]:
        return tuple(pair[1] for pair in self.layer_pairs)

    def forward(self, pixel_values: Tensor) -> dict[str, object]:
        layers, grid = extract_patch_features(self.backbone, pixel_values, self.student_layers)
        projected = [F.normalize(head(tokens), p=2, dim=-1) for head, tokens in zip(self.projections, layers)]
        return {"layers": layers, "projected_layers": projected, "aggregate": aggregate_layers(layers), "grid": grid}


def _resize_tokens(tokens: Tensor, grid: tuple[int, int], target_grid: tuple[int, int]) -> Tensor:
    if grid == target_grid:
        return tokens
    batch, count, channels = tokens.shape
    if count != grid[0] * grid[1]:
        raise ValueError(f"Token count {count} does not match grid {grid}")
    image = tokens.transpose(1, 2).reshape(batch, channels, grid[0], grid[1])
    image = F.interpolate(image, size=target_grid, mode="bilinear", align_corners=False)
    return image.flatten(2).transpose(1, 2)


def masks_to_patch_weights(
    masks: Sequence[Image.Image | None],
    image_hw: tuple[int, int],
    grid: tuple[int, int],
    device: torch.device,
    mask_alpha: float = 2.0,
) -> Tensor:
    """Convert pixel masks into ``1 + alpha * mean(mask)`` patch weights."""

    height, width = image_hw
    values = []
    for mask in masks:
        if mask is None:
            values.append(torch.zeros((1, height, width), dtype=torch.float32))
            continue
        resized = mask.convert("L").resize((width, height), Image.Resampling.NEAREST)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        values.append(torch.from_numpy(array).unsqueeze(0))
    pixels = torch.stack(values, dim=0).to(device=device)
    patch_mask = F.adaptive_avg_pool2d(pixels, grid).flatten(1)
    return 1.0 + float(mask_alpha) * patch_mask


def anomaly_map_from_features(features: Tensor, normal_centroid: Tensor) -> Tensor:
    """Cosine-distance anomaly map against a frozen normal patch centroid."""

    centroid = F.normalize(normal_centroid.to(features), p=2, dim=-1)
    normalized = F.normalize(features.float(), p=2, dim=-1)
    return (1.0 - (normalized * centroid).sum(dim=-1)).clamp_min(0.0)


def _weighted_mean(values: Tensor, weights: Tensor | None = None) -> Tensor:
    if weights is None:
        return values.mean()
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def compute_distillation_losses(
    teacher_layers: Sequence[Tensor],
    student_projected_layers: Sequence[Tensor],
    student_features: Tensor,
    teacher_centroid: Tensor,
    student_centroid: Tensor,
    labels: Tensor,
    patch_weights: Tensor | None = None,
    lambda_feature: float = 1.0,
    lambda_map: float = 1.0,
    margin: float = 0.2,
    top_ratio: float = 0.01,
) -> dict[str, Tensor]:
    """Compute ``L_original``, ``L_feature`` and ``L_map`` plus total loss."""

    if len(teacher_layers) != len(student_projected_layers):
        raise ValueError("Teacher and student layer lists must have equal length")
    if patch_weights is not None and patch_weights.shape[:2] != student_features.shape[:2]:
        raise ValueError(f"Patch weights {tuple(patch_weights.shape)} do not match features {tuple(student_features.shape)}")
    feature_terms = []
    for teacher, student in zip(teacher_layers, student_projected_layers):
        if teacher.shape[:2] != student.shape[:2]:
            raise ValueError("Teacher/student patch grids differ; align them before computing distillation loss")
        cosine_error = 1.0 - F.cosine_similarity(student.float(), teacher.float(), dim=-1)
        feature_terms.append(_weighted_mean(cosine_error, patch_weights))
    feature_loss = torch.stack(feature_terms).mean()

    teacher_map = anomaly_map_from_features(aggregate_layers(teacher_layers), teacher_centroid).detach()
    student_map = anomaly_map_from_features(student_features, student_centroid)
    map_error = (student_map - teacher_map).abs()
    map_loss = _weighted_mean(map_error, patch_weights)

    normal = ~labels.bool()
    normal_loss = student_map[normal].mean() if normal.any() else student_map.new_zeros(())
    anomaly_loss = student_map.new_zeros(())
    if labels.any():
        patch_count = student_map.shape[1]
        top_k = max(1, min(patch_count, int(math.ceil(patch_count * float(top_ratio)))))
        top_scores = student_map[labels.bool()].topk(top_k, dim=1).values.mean(dim=1)
        anomaly_loss = F.relu(float(margin) - top_scores).mean()
    original_loss = normal_loss + anomaly_loss
    total = original_loss + float(lambda_feature) * feature_loss + float(lambda_map) * map_loss
    return {
        "loss": total,
        "original_loss": original_loss,
        "feature_loss": feature_loss,
        "map_loss": map_loss,
        "normal_compactness": normal_loss,
        "anomaly_margin": anomaly_loss,
    }


@torch.no_grad()
def estimate_normal_centroid(
    model: nn.Module,
    processor,
    paths: Sequence[str | Path],
    layer_indices: Sequence[int],
    image_size: int,
    device: torch.device,
    batch_size: int = 4,
) -> Tensor:
    """Estimate a normalized patch centroid from normal images."""

    if not paths:
        raise ValueError("At least one normal image is required to estimate a centroid")
    model.eval()
    feature_sum: Tensor | None = None
    feature_count = 0
    for start in range(0, len(paths), max(1, batch_size)):
        images = []
        for path in paths[start : start + max(1, batch_size)]:
            with Image.open(path) as source:
                images.append(source.convert("RGB").copy())
        inputs = prepare_batch(processor, images, image_size, device)
        layers, _ = extract_patch_features(model, inputs["pixel_values"], layer_indices)
        aggregate = aggregate_layers(layers)
        flattened = aggregate.reshape(-1, aggregate.shape[-1]).float()
        batch_sum = flattened.sum(dim=0)
        feature_sum = batch_sum if feature_sum is None else feature_sum + batch_sum
        feature_count += flattened.shape[0]
    if feature_sum is None or feature_count == 0:
        raise RuntimeError("Normal centroid estimation produced no patch features")
    centroid = feature_sum / feature_count
    return F.normalize(centroid, p=2, dim=-1)


def prepare_batch(processor, images: Sequence[Image.Image], image_size: int, device: torch.device) -> dict[str, Tensor]:
    """Apply a fixed square processor transform and move tensor fields to device."""

    kwargs = {
        "images": list(images),
        "return_tensors": "pt",
        "do_resize": True,
        "size": {"height": int(image_size), "width": int(image_size)},
        "do_center_crop": False,
    }
    try:
        encoded = processor(**kwargs)
    except (TypeError, ValueError):
        # Older/custom processors may not accept the explicit resize flags.
        encoded = processor(images=list(images), return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items() if torch.is_tensor(value)}


def _model_dim(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    for name in ("hidden_size", "embed_dim", "dim"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not infer model hidden size from config")


def _device(value: str | None) -> torch.device:
    resolved = value or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(resolved).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(resolved)


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def _trainable_parameter_groups(model: DistillationStudent, lr: float, head_lr: float, weight_decay: float):
    adapter, heads, other = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("projections"):
            heads.append(parameter)
        elif "lora_" in name:
            adapter.append(parameter)
        else:
            other.append(parameter)
    groups = []
    if adapter:
        groups.append({"params": adapter, "lr": float(lr), "weight_decay": float(weight_decay)})
    if other:
        groups.append({"params": other, "lr": float(lr) * 0.1, "weight_decay": float(weight_decay)})
    if heads:
        groups.append({"params": heads, "lr": float(head_lr), "weight_decay": float(weight_decay)})
    if not groups:
        raise RuntimeError("No trainable student parameters were configured")
    return groups


def _count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return int(total), int(trainable)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def save_student_checkpoint(
    output_dir: str | Path,
    student: DistillationStudent,
    processor,
    config: dict,
    teacher_centroid: Tensor,
    student_centroid: Tensor,
    epoch: int = 0,
    metrics: dict | None = None,
    save_base_model: bool = True,
) -> Path:
    """Save adapter/projection state, metadata, centroids, and optional base HF model."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    base_dir = root / "student_base"
    if save_base_model and not base_dir.exists():
        try:
            student.backbone.save_pretrained(base_dir)
        except Exception as exc:  # pragma: no cover - depends on HF wrapper
            warnings.warn(f"Could not save student_base via save_pretrained: {exc}", RuntimeWarning)
            torch.save({k: v.detach().cpu() for k, v in student.backbone.state_dict().items()}, root / "student_base_state.pt")
    try:
        processor.save_pretrained(root / "processor")
    except Exception as exc:  # pragma: no cover - custom processors
        warnings.warn(f"Could not save processor: {exc}", RuntimeWarning)
    trainable_state = {name: parameter.detach().cpu() for name, parameter in student.named_parameters() if parameter.requires_grad}
    checkpoint = {
        "format": 1,
        "epoch": int(epoch),
        "student_state": trainable_state,
        "teacher_centroid": teacher_centroid.detach().cpu(),
        "student_centroid": student_centroid.detach().cpu(),
        "config": config,
        "metrics": metrics or {},
    }
    path = root / "distill_checkpoint.pt"
    torch.save(checkpoint, path)
    (root / "training_config.json").write_text(json.dumps(config, ensure_ascii=True, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return path


def export_merged_student(
    output_dir: str | Path,
    student: DistillationStudent,
    processor,
) -> Path:
    """Merge LoRA branches and export a standard Hugging Face backbone."""

    targets = [name for name, module in student.backbone.named_modules() if isinstance(module, LoRALinear)]
    # Replace deepest children first so dotted parent lookups remain valid.
    for name in sorted(targets, key=lambda value: value.count("."), reverse=True):
        module = student.backbone.get_submodule(name)
        if not isinstance(module, LoRALinear):
            continue
        module.merge()
        _set_submodule(student.backbone, name, module.base)
    destination = Path(output_dir) / "student_merged"
    destination.mkdir(parents=True, exist_ok=True)
    student.backbone.save_pretrained(destination)
    processor.save_pretrained(destination)
    return destination


def load_student_checkpoint(checkpoint_dir: str | Path, device: str | torch.device | None = None) -> DistillationStudent:
    """Reconstruct a student from ``save_student_checkpoint`` output."""

    from transformers import AutoModel

    root = Path(checkpoint_dir)
    checkpoint = torch.load(root / "distill_checkpoint.pt", map_location="cpu")
    config = checkpoint["config"]
    base_dir = root / "student_base"
    if base_dir.is_dir():
        backbone = AutoModel.from_pretrained(base_dir)
    else:
        backbone = AutoModel.from_pretrained(config["student_model"])
        fallback = root / "student_base_state.pt"
        if fallback.is_file():
            backbone.load_state_dict(torch.load(fallback, map_location="cpu"))
    if config.get("adaptation") == "lora":
        inject_lora(
            backbone,
            rank=int(config["lora_rank"]),
            alpha=float(config["lora_alpha"]),
            dropout=float(config["lora_dropout"]),
            last_n_blocks=int(config["last_n_blocks"]),
            target_names=config.get("lora_targets"),
        )
    pairs = tuple(tuple(pair) for pair in config["layer_pairs"])
    student = DistillationStudent(backbone, pairs, int(config["teacher_dim"]), int(config["student_dim"]))
    student.load_state_dict(checkpoint["student_state"], strict=False)
    target = torch.device(device) if device is not None else _device(None)
    return student.to(target).eval()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> dict:
    """Run teacher/student distillation and return the final training summary."""

    from transformers import AutoImageProcessor, AutoModel

    set_seed(int(args.seed))
    device = _device(args.device)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    records = discover_records(
        normal_dir=args.normal_dir,
        defect_dir=args.defect_dir,
        mask_dir=args.mask_dir,
        data_root=args.data_root,
        max_normal_images=args.max_normal_images,
        max_defect_images=args.max_defect_images,
    )
    normal_paths = [record.image_path for record in records if not record.is_anomaly]
    teacher = AutoModel.from_pretrained(args.teacher_model).to(device).eval()
    student_backbone = AutoModel.from_pretrained(args.student_model)
    teacher.requires_grad_(False)
    teacher_processor = AutoImageProcessor.from_pretrained(args.teacher_model)
    student_processor = AutoImageProcessor.from_pretrained(args.student_model)

    # Save the pristine base before LoRA changes module names.  The compact
    # adapter checkpoint can then always be reconstructed without serializing
    # another complete trained backbone.
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    student_backbone.save_pretrained(output_dir / "student_base")
    student_processor.save_pretrained(output_dir / "processor")

    layer_pairs = resolve_layer_pairs(
        teacher,
        student_backbone,
        _parse_layers(args.feature_layers),
        _parse_layers(args.teacher_layers) if args.teacher_layers else None,
        _parse_layers(args.student_layers) if args.student_layers else None,
    )
    adapted_targets = configure_student_trainable(
        student_backbone,
        adaptation=args.adaptation,
        last_n_blocks=args.last_n_blocks,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    student_dim, teacher_dim = _model_dim(student_backbone), _model_dim(teacher)
    student = DistillationStudent(student_backbone, layer_pairs, teacher_dim, student_dim).to(device)

    teacher_layers_for_centroid = tuple(pair[0] for pair in layer_pairs)
    student_layers_for_centroid = tuple(pair[1] for pair in layer_pairs)
    teacher_centroid = estimate_normal_centroid(teacher, teacher_processor, normal_paths, teacher_layers_for_centroid, args.image_size, device, args.centroid_batch_size)
    student_centroid = estimate_normal_centroid(student.backbone, student_processor, normal_paths, student_layers_for_centroid, args.image_size, device, args.centroid_batch_size)
    # The centroid is a fixed pre-training reference; this avoids a moving
    # target that could make the normal-compactness loss degenerate.
    student_centroid = student_centroid.detach()
    teacher_centroid = teacher_centroid.detach()

    dataset = ImageMaskDataset(records)
    loader = DataLoader(dataset, batch_size=max(1, args.batch_size), shuffle=True, num_workers=max(0, args.num_workers), collate_fn=_collate, pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(_trainable_parameter_groups(student, args.lr, args.head_lr, args.weight_decay))
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    history = []
    total_params, trainable_params = _count_parameters(student)

    for epoch in range(1, int(args.epochs) + 1):
        student.train()
        running = {"loss": 0.0, "original_loss": 0.0, "feature_loss": 0.0, "map_loss": 0.0, "normal_compactness": 0.0, "anomaly_margin": 0.0}
        batches = 0
        for images, masks, labels, _paths in loader:
            labels = labels.to(device)
            teacher_inputs = prepare_batch(teacher_processor, images, args.image_size, device)
            student_inputs = prepare_batch(student_processor, images, args.image_size, device)
            if teacher_inputs["pixel_values"].shape[-2:] != student_inputs["pixel_values"].shape[-2:]:
                raise ValueError("Teacher and student processors produced different image sizes")
            with torch.no_grad(), _autocast(device, amp_enabled):
                teacher_layers, teacher_grid = extract_patch_features(teacher, teacher_inputs["pixel_values"], teacher_layers_for_centroid)
            with _autocast(device, amp_enabled):
                student_output = student(student_inputs["pixel_values"])
                student_layers = student_output["layers"]
                projected_layers = student_output["projected_layers"]
                student_grid = student_output["grid"]
                target_grid = teacher_grid
                if student_grid != target_grid:
                    student_layers = [_resize_tokens(layer, student_grid, target_grid) for layer in student_layers]
                    projected_layers = [_resize_tokens(layer, student_grid, target_grid) for layer in projected_layers]
                    student_features = _resize_tokens(student_output["aggregate"], student_grid, target_grid)
                else:
                    student_features = student_output["aggregate"]
                expected_patches = target_grid[0] * target_grid[1]
                if any(layer.shape[1] != expected_patches for layer in teacher_layers):
                    raise ValueError("Teacher hidden-state patch counts are inconsistent")
                patch_weights = masks_to_patch_weights(masks, teacher_inputs["pixel_values"].shape[-2:], target_grid, device, args.mask_alpha)
                losses = compute_distillation_losses(
                    teacher_layers,
                    projected_layers,
                    student_features,
                    teacher_centroid,
                    student_centroid,
                    labels,
                    patch_weights=patch_weights,
                    lambda_feature=args.lambda_feature,
                    lambda_map=args.lambda_map,
                    margin=args.margin,
                    top_ratio=args.top_ratio,
                )
            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                optimizer.step()
            batches += 1
            for key in running:
                running[key] += float(losses[key].detach().cpu())
        if not batches:
            raise RuntimeError("DataLoader produced no batches")
        epoch_metrics = {key: value / batches for key, value in running.items()}
        epoch_metrics["epoch"] = epoch
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, ensure_ascii=True, sort_keys=True))
        if args.save_every > 0 and epoch % args.save_every == 0:
            epoch_config = _training_config(args, layer_pairs, teacher_dim, student_dim, adapted_targets)
            save_student_checkpoint(
                output_dir / f"epoch-{epoch:04d}",
                student,
                student_processor,
                epoch_config,
                teacher_centroid,
                student_centroid,
                epoch=epoch,
                metrics=epoch_metrics,
                save_base_model=False,
            )

    config = _training_config(args, layer_pairs, teacher_dim, student_dim, adapted_targets)
    config["total_parameters"] = total_params
    config["trainable_parameters"] = trainable_params
    config["history"] = history
    checkpoint = save_student_checkpoint(
        output_dir,
        student,
        student_processor,
        config,
        teacher_centroid,
        student_centroid,
        epoch=int(args.epochs),
        metrics=history[-1],
    )
    merged_model = export_merged_student(output_dir, student, student_processor)
    summary = {
        "checkpoint": str(checkpoint),
        "merged_model": str(merged_model),
        "output": str(output_dir),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "history": history,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return summary


def _training_config(args, layer_pairs, teacher_dim, student_dim, adapted_targets):
    return {
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "image_size": int(args.image_size),
        "layer_pairs": [list(pair) for pair in layer_pairs],
        "teacher_dim": int(teacher_dim),
        "student_dim": int(student_dim),
        "adaptation": args.adaptation,
        "last_n_blocks": int(args.last_n_blocks),
        "lora_rank": int(args.lora_rank),
        "lora_alpha": float(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "lora_targets": list(adapted_targets) if args.adaptation == "lora" else [],
        "lambda_feature": float(args.lambda_feature),
        "lambda_map": float(args.lambda_map),
        "mask_alpha": float(args.mask_alpha),
        "margin": float(args.margin),
        "top_ratio": float(args.top_ratio),
        "seed": int(args.seed),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DINOv3 ViT-B teacher -> ViT-S LoRA/local distillation")
    parser.add_argument("--normal-dir")
    parser.add_argument("--defect-dir")
    parser.add_argument("--mask-dir")
    parser.add_argument("--data-root")
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER)
    parser.add_argument("--student-model", default=DEFAULT_STUDENT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--feature-layers", default=",".join(str(x) for x in DEFAULT_FEATURE_LAYERS))
    parser.add_argument("--teacher-layers")
    parser.add_argument("--student-layers")
    parser.add_argument("--adaptation", choices=("lora", "local"), default="lora")
    parser.add_argument("--last-n-blocks", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--centroid-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-feature", type=float, default=1.0)
    parser.add_argument("--lambda-map", type=float, default=1.0)
    parser.add_argument("--mask-alpha", type=float, default=2.0)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--top-ratio", type=float, default=0.01)
    parser.add_argument("--max-normal-images", type=int, default=0)
    parser.add_argument("--max-defect-images", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-every", type=int, default=0, help="also save epoch checkpoints; 0 saves only the final checkpoint")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.image_size <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        parser.error("image-size, epochs and batch-size must be positive")
    if args.last_n_blocks <= 0 or args.lora_rank <= 0:
        parser.error("last-n-blocks and lora-rank must be positive")
    if args.lambda_feature < 0 or args.lambda_map < 0 or args.mask_alpha < 0 or args.margin < 0:
        parser.error("loss weights and margin must be non-negative")
    if not 0 < args.top_ratio <= 1:
        parser.error("top-ratio must be in (0, 1]")
    summary = train(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
