"""Standalone DINOv3 teacher/student distillation for MVTec AD and VisA.

This module keeps distillation and dataset parsing in one standalone file. It
trains a deployable ViT-S+ backbone using a frozen ViT-B teacher, multi-layer
patch-token distillation, anomaly-map distillation, and a small
normal-compactness/margin objective. MVTec AD and VisA are parsed directly by
this file. Test defects are excluded by default; when explicitly selected,
their masks increase the feature-distillation weight in defect regions. The
optional post-training evaluation reuses DefectFusion's existing metric code
so its reported indicators match the project evaluator.

Example::

    python distill_dinov3.py \
        --dataset mvtec --data-root ./datasets/mvtec_anomaly \
        --categories bottle \
        --output outputs/dinov3-vit-s-plus-distilled \
        --epochs 10 --adaptation lora

Each selected category gets its own output directory containing a lightweight
LoRA adapter, a JSON training configuration, and a JSON training log. Original
ViT-S+ weights are never copied into the output directory. Automatic evaluation
reloads ``--student-model`` and attaches the saved adapter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import time
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
DEFAULT_STUDENT = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
DEFAULT_FEATURE_LAYERS = (1, 6, 12)
ENGINE_METRIC_FIELDS = (
    "image_auroc",
    "image_aupr",
    "image_f1_max",
    "pixel_auroc",
    "pixel_aupr",
    "pixel_aupro",
    "pixel_f1_max",
    "defect_type_accuracy",
    "defect_type_macro_precision",
    "defect_type_macro_recall",
    "defect_type_macro_f1",
    "defect_type_weighted_f1",
)


def _looks_like_explicit_local_path(reference: str) -> bool:
    """Return whether a model reference is explicitly intended as a local path."""

    return Path(reference).is_absolute() or reference.startswith(("./", "../", "~/", "~\\"))


def resolve_model_reference(reference: str, option_name: str) -> str:
    """Canonicalize usable local model directories and preserve Hub identifiers.

    ``transformers`` falls back to Hugging Face Hub parsing when a path is not
    recognized as a directory. For an absolute local path this produces a
    misleading repo-id error. Resolve valid directories before loading and
    reject inaccessible path-like references with an actionable message.
    """

    raw_reference = str(reference)
    if raw_reference != raw_reference.strip():
        raise ValueError(
            f"{option_name} contains leading or trailing whitespace: {raw_reference!r}. "
            "Remove the extra characters and pass the model directory again."
        )
    if not raw_reference:
        raise ValueError(f"{option_name} must be a Hugging Face identifier or local model directory")

    candidate = Path(raw_reference).expanduser()
    if candidate.is_dir():
        config = candidate / "config.json"
        if not config.is_file():
            raise ValueError(
                f"{option_name} local directory is missing config.json: {candidate}. "
                "Pass the Hugging Face model snapshot directory, not its parent directory."
            )
        return str(candidate.resolve())
    if _looks_like_explicit_local_path(raw_reference):
        state = "is not a directory" if candidate.exists() else "does not exist"
        raise ValueError(
            f"{option_name} local path {raw_reference!r} {state} from this Python process. "
            "Run `python -c \"from pathlib import Path; p = Path(...); print(p.exists(), p.is_dir())\"` "
            "with the same interpreter and environment."
        )
    return raw_reference


def _pretrained_load_kwargs(model_reference: str) -> dict[str, bool]:
    """Prevent local model directories from ever being interpreted as Hub IDs."""

    return {"local_files_only": True} if Path(model_reference).expanduser().is_dir() else {}


@dataclass(frozen=True)
class ImageRecord:
    """One image and its optional pixel mask."""

    image_path: str
    mask_path: str | None
    is_anomaly: bool
    defect_type: str | None = None


@dataclass(frozen=True)
class DatasetCategory:
    """Dataset-native normal references and optional labeled defect samples."""

    name: str
    normal_images: tuple[Path, ...]
    defect_samples: tuple[ImageRecord, ...]


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


def _stable_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(namespace.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") ^ int(seed)) % (2**32)


def _sample_paths(paths: Sequence[Path], shots: int, seed: int, namespace: str) -> list[Path]:
    """Deterministically select a requested few-shot subset without replacement."""

    values = sorted((Path(path) for path in paths), key=lambda path: path.as_posix().lower())
    if shots == -1 or shots >= len(values):
        return values
    if shots <= 0:
        return []
    return sorted(random.Random(_stable_seed(seed, namespace)).sample(values, shots), key=lambda path: path.as_posix().lower())


def _mvtec_categories(root: str | Path) -> list[DatasetCategory]:
    """Load standard MVTec AD category folders without importing project modules."""

    data_root = Path(root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"MVTec data root does not exist: {data_root}")
    categories: list[DatasetCategory] = []
    for category_dir in sorted(item for item in data_root.iterdir() if item.is_dir()):
        normal = _iter_images(category_dir / "train" / "good") if (category_dir / "train" / "good").is_dir() else []
        if not normal:
            continue
        defects = []
        test_dir = category_dir / "test"
        if test_dir.is_dir():
            for defect_dir in sorted(item for item in test_dir.iterdir() if item.is_dir() and item.name != "good"):
                for image in _iter_images(defect_dir):
                    mask = category_dir / "ground_truth" / defect_dir.name / f"{image.stem}_mask.png"
                    defects.append(ImageRecord(str(image), str(mask) if mask.is_file() else None, True, defect_dir.name))
        categories.append(DatasetCategory(category_dir.name, tuple(normal), tuple(defects)))
    if not categories:
        raise ValueError(f"No MVTec AD categories with train/good images found in {data_root}")
    return categories


def _csv_value(row: dict, *names: str) -> str:
    normalized = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        value = normalized.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _csv_path(root: Path, value: str) -> Path | None:
    if not value or value.lower() in {"none", "nan"}:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _visa_raw_categories(root: Path) -> list[DatasetCategory]:
    categories = []
    for category_dir in sorted(item for item in root.iterdir() if item.is_dir()):
        data_dir = category_dir / "Data"
        normal_dir = data_dir / "Images" / "Normal"
        anomaly_dir = data_dir / "Images" / "Anomaly"
        if not normal_dir.is_dir():
            continue
        normal = _iter_images(normal_dir)
        masks = _mask_index(data_dir / "Masks" / "Anomaly" if (data_dir / "Masks" / "Anomaly").is_dir() else None)
        defects = [
            ImageRecord(str(image), str(mask) if (mask := _find_mask(image, anomaly_dir, data_dir / "Masks" / "Anomaly", masks)) else None, True, "anomaly")
            for image in (_iter_images(anomaly_dir) if anomaly_dir.is_dir() else [])
        ]
        if normal:
            categories.append(DatasetCategory(category_dir.name, tuple(normal), tuple(defects)))
    if not categories:
        raise ValueError(f"No raw VisA categories with Data/Images/Normal found in {root}")
    return categories


def _visa_categories(root: str | Path, split_csv: str | Path | None = None) -> list[DatasetCategory]:
    """Load VisA official 1-class splits, with raw-release layout fallback."""

    data_root = Path(root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"VisA data root does not exist: {data_root}")
    split_path = Path(split_csv) if split_csv else data_root / "split_csv" / "1cls.csv"
    if not split_path.is_file():
        if split_csv is not None:
            raise FileNotFoundError(f"VisA split CSV does not exist: {split_path}")
        warnings.warn(
            "VisA 1cls.csv was not found; the raw layout fallback treats every "
            "Data/Images/Normal image as a training reference.",
            RuntimeWarning,
        )
        return _visa_raw_categories(data_root)
    groups: dict[str, dict[str, list]] = {}
    with split_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = {str(name).strip().lower() for name in (reader.fieldnames or [])}
        required = {"object", "split", "label", "image"}
        if not required.issubset(columns):
            raise ValueError(f"VisA split CSV requires columns {sorted(required)}; found {sorted(columns)}")
        for row in reader:
            category = _csv_value(row, "object")
            split = _csv_value(row, "split").lower()
            label = _csv_value(row, "label").lower()
            image = _csv_path(data_root, _csv_value(row, "image"))
            if not category or image is None or not image.is_file():
                continue
            group = groups.setdefault(category, {"normal": [], "defect": []})
            anomalous = label not in {"normal", "good", "0"}
            if split == "train" and not anomalous:
                group["normal"].append(image)
            elif split == "test" and anomalous:
                mask = _csv_path(data_root, _csv_value(row, "mask"))
                defect_type = _csv_value(row, "defect_type", "type") or "anomaly"
                group["defect"].append(ImageRecord(str(image), str(mask) if mask is not None and mask.is_file() else None, True, defect_type))
    categories = [
        DatasetCategory(name, tuple(sorted(group["normal"])), tuple(sorted(group["defect"], key=lambda record: record.image_path)))
        for name, group in sorted(groups.items())
        if group["normal"]
    ]
    if not categories:
        raise ValueError(f"No VisA training-normal records found in {split_path}")
    return categories


def dataset_record_groups(args: argparse.Namespace) -> tuple[dict[str, list[ImageRecord]], dict[str, dict[str, int]]]:
    """Build training records for MVTec or VisA from CLI paths only.

    Defect images always come from the test partition and are opt-in through
    ``--defect-shots``.  Their only purpose is the optional mask-weighted
    distillation term; no test defect is selected by default.
    """

    if args.dataset == "folder":
        records = discover_records(
            normal_dir=args.normal_dir,
            defect_dir=args.defect_dir,
            mask_dir=args.mask_dir,
            data_root=args.data_root,
            max_normal_images=args.max_normal_images,
            max_defect_images=args.max_defect_images,
        )
        return {"folder": records}, {"folder": {"normal": sum(not record.is_anomaly for record in records), "defect": sum(record.is_anomaly for record in records)}}
    if args.data_root is None:
        raise ValueError("--data-root is required for --dataset mvtec or --dataset visa")
    categories = _mvtec_categories(args.data_root) if args.dataset == "mvtec" else _visa_categories(args.data_root, args.split_csv)
    requested = set(args.categories or [])
    names = {category.name for category in categories}
    missing = sorted(requested - names)
    if missing:
        raise ValueError(f"Unknown {args.dataset} categories: {', '.join(missing)}")
    if requested:
        categories = [category for category in categories if category.name in requested]
    if not categories:
        raise ValueError("No categories selected for distillation")

    groups: dict[str, list[ImageRecord]] = {}
    counts: dict[str, dict[str, int]] = {}
    for category in categories:
        normal = _sample_paths(category.normal_images, args.normal_shots, args.seed, f"{args.dataset}:normal:{category.name}")
        if args.max_normal_images > 0:
            normal = normal[: args.max_normal_images]
        if not normal:
            raise ValueError(f"{args.dataset}/{category.name} has no selected normal images")
        selected_defects: list[ImageRecord] = []
        if args.defect_shots > 0:
            by_type: dict[str, list[ImageRecord]] = {}
            for record in category.defect_samples:
                by_type.setdefault(record.defect_type or "anomaly", []).append(record)
            for defect_type, candidates in sorted(by_type.items()):
                selected_paths = set(
                    str(path) for path in _sample_paths(
                        [Path(record.image_path) for record in candidates],
                        args.defect_shots,
                        args.seed,
                        f"{args.dataset}:defect:{category.name}:{defect_type}",
                    )
                )
                selected_defects.extend(record for record in candidates if record.image_path in selected_paths)
        if args.max_defect_images > 0:
            selected_defects = selected_defects[: args.max_defect_images]
        missing_masks = sum(record.mask_path is None for record in selected_defects)
        if missing_masks:
            warnings.warn(
                f"{args.dataset}/{category.name}: {missing_masks} selected defect images have no mask; "
                "they use only the anomaly-margin loss.",
                RuntimeWarning,
            )
        groups[category.name] = [ImageRecord(str(path), None, False, "good") for path in normal] + selected_defects
        counts[category.name] = {"normal": len(normal), "defect": len(selected_defects)}
    if not groups:
        raise ValueError("No images selected for distillation")
    return groups, counts


def dataset_records(args: argparse.Namespace) -> tuple[list[ImageRecord], dict[str, dict[str, int]]]:
    """Backward-compatible flattened view of the dataset-specific groups."""

    groups, counts = dataset_record_groups(args)
    return [record for records in groups.values() for record in records], counts


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


def configure_student_trainable(
    model: nn.Module,
    adaptation: str = "lora",
    last_n_blocks: int = 4,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
) -> list[str]:
    """Freeze ViT-S+ and enable only LoRA weights for adapter-only export."""

    if adaptation != "lora":
        raise ValueError("Only LoRA adaptation supports the adapter-only output format")
    model.requires_grad_(False)
    return inject_lora(model, lora_rank, lora_alpha, lora_dropout, last_n_blocks)


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


def _lora_parameter_count(backbone: nn.Module) -> int:
    return int(sum(parameter.numel() for name, parameter in backbone.named_parameters() if "lora_" in name))


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def save_lora_adapter(output_dir: str | Path, student: DistillationStudent, config: dict) -> Path:
    """Save only LoRA tensors; ViT-S+, LayerNorm, and projection weights are not exported."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    lora_state = {
        name: parameter.detach().cpu()
        for name, parameter in student.backbone.named_parameters()
        if "lora_" in name
    }
    if not lora_state:
        raise RuntimeError("No LoRA parameters were found to save")
    path = root / "lora_adapter.pt"
    torch.save({"format": 1, "adapter_type": "lora", "lora_state": lora_state}, path)
    (root / "training_config.json").write_text(
        json.dumps(config, ensure_ascii=True, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return path


def _adapter_config(adapter_path: str | Path) -> dict:
    config_path = Path(adapter_path).parent / "training_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"LoRA adapter configuration does not exist: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_lora_adapter_into_backbone(
    backbone: nn.Module,
    adapter_path: str | Path,
    config: dict | None = None,
    base_model: str | Path | None = None,
) -> dict:
    """Inject a saved LoRA adapter into a freshly loaded compatible ViT-S+."""

    path = Path(adapter_path)
    if not path.is_file():
        raise FileNotFoundError(f"LoRA adapter does not exist: {path}")
    adapter_config = config or _adapter_config(path)
    if adapter_config.get("adaptation") != "lora":
        raise ValueError("The adapter configuration is not a LoRA run")
    if base_model is not None and str(adapter_config.get("student_model")) != str(base_model):
        raise ValueError(
            "LoRA adapter was trained for a different student model: "
            f"expected {adapter_config.get('student_model')!r}, got {str(base_model)!r}"
        )
    targets = adapter_config.get("lora_targets")
    if not targets:
        raise ValueError("The adapter configuration does not list LoRA target modules")
    try:
        base_device = next(backbone.parameters()).device
    except StopIteration:  # pragma: no cover - Hugging Face backbones have parameters
        base_device = torch.device("cpu")
    inject_lora(
        backbone,
        rank=int(adapter_config["lora_rank"]),
        alpha=float(adapter_config["lora_alpha"]),
        dropout=float(adapter_config["lora_dropout"]),
        last_n_blocks=int(adapter_config["last_n_blocks"]),
        target_names=targets,
    )
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("adapter_type") != "lora" or not isinstance(checkpoint.get("lora_state"), dict):
        raise ValueError(f"Unsupported LoRA adapter format: {path}")
    lora_state = checkpoint["lora_state"]
    expected = {name for name, _parameter in backbone.named_parameters() if "lora_" in name}
    missing = sorted(expected - set(lora_state))
    unexpected = sorted(set(lora_state) - expected)
    if missing or unexpected:
        detail = []
        if missing:
            detail.append("missing=" + ", ".join(missing))
        if unexpected:
            detail.append("unexpected=" + ", ".join(unexpected))
        raise ValueError("LoRA adapter does not match the requested ViT-S+: " + "; ".join(detail))
    backbone.load_state_dict(lora_state, strict=False)
    # LoRA modules are constructed on CPU. Keep an already-loaded CUDA model
    # on its original device before the evaluator starts feature extraction.
    backbone.to(base_device)
    backbone.requires_grad_(False)
    return adapter_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluation_categories(args: argparse.Namespace) -> list[DatasetCategory]:
    """Load the complete test split used by the project's evaluator."""

    if args.dataset == "mvtec":
        return _mvtec_categories(args.data_root)
    if args.dataset == "visa":
        # The training parser intentionally keeps only normal-train and
        # anomalous-test records.  The project evaluator additionally needs
        # VisA test-normal rows, so use its canonical category loader here.
        from defectfusion.visa import load_visa_categories

        return load_visa_categories(args.data_root, args.split_csv)
    raise ValueError("Automatic evaluation is available only for --dataset mvtec or visa")


def _selected_category_records(
    categories: Sequence[DatasetCategory],
    requested_categories: Sequence[str] | None,
) -> list[DatasetCategory]:
    requested = set(requested_categories or [])
    if not requested:
        return list(categories)
    return [category for category in categories if category.name in requested]


def _engine_metric_summary(metrics: Sequence[dict]) -> dict[str, float]:
    """Use the same per-category macro averaging convention as the CLI."""

    summary: dict[str, float] = {}
    for name in ENGINE_METRIC_FIELDS:
        values = [float(item[name]) for item in metrics if name in item]
        if values:
            summary[name] = sum(values) / len(values)
    return summary


def _selected_defect_paths(records: Sequence[ImageRecord]) -> list[str]:
    """Return opt-in test defects so they are omitted from post-training scores."""

    return [record.image_path for record in records if record.is_anomaly]


def evaluate_distilled_students(
    args: argparse.Namespace,
    output_root: str | Path,
    groups: dict[str, list[ImageRecord]],
) -> dict:
    """Evaluate LoRA-adapted students with the existing DefectFusion metric pipeline.

    The evaluator intentionally delegates AUROC, AUPR, F1-max, and AUPRO to
    ``defectfusion.mvtec``.  This keeps the generated JSON/CSV layout and all
    metric definitions identical to ``evaluate-mvtec`` and ``evaluate-visa``.
    """

    from defectfusion.cli import _leave_one_out_normal_scores
    from defectfusion.features import DinoFeatureExtractor
    from defectfusion.mvtec import evaluate_mvtec, evaluate_samples
    from defectfusion.pipeline import DefectFusion
    from defectfusion.reporting import write_metrics_csv

    output_root = Path(output_root)
    evaluation_root = output_root / "evaluation"
    categories_dir = evaluation_root / "categories"
    categories_dir.mkdir(parents=True, exist_ok=True)
    categories = _selected_category_records(_evaluation_categories(args), args.categories)
    category_by_name = {category.name: category for category in categories}
    missing = sorted(set(groups) - set(category_by_name))
    if missing:
        raise ValueError(f"Could not resolve evaluation categories: {', '.join(missing)}")

    all_metrics: list[dict] = []
    for category_name, records in groups.items():
        category = category_by_name[category_name]
        normal_paths = [record.image_path for record in records if not record.is_anomaly]
        if not normal_paths:
            raise ValueError(f"{args.dataset}/{category_name} has no normal images for evaluation")
        adapter_path = output_root / category_name / "lora_adapter.pt"
        if not adapter_path.is_file():
            raise FileNotFoundError(f"LoRA adapter does not exist: {adapter_path}")
        print(
            f"[evaluate] {args.dataset}/{category_name}: normal={len(normal_paths)} "
            f"base={args.student_model} adapter={adapter_path}",
            flush=True,
        )
        extractor = DinoFeatureExtractor(
            str(args.student_model),
            image_size=args.eval_image_size,
            resize_mode=args.eval_resize_mode,
            device=args.device,
            feature_layers=_parse_layers(args.eval_feature_layers),
            layer_aggregation=args.eval_layer_aggregation,
            layer_normalization=args.eval_layer_normalization,
        )
        load_lora_adapter_into_backbone(extractor.model, adapter_path, base_model=args.student_model)
        extractor.model.eval()
        def build_fusion():
            return DefectFusion(
                extractor,
                top_k_ratio=args.eval_top_k_ratio,
                image_score=args.eval_image_score,
                image_top_ratio=args.eval_image_top_ratio,
                anomaly_method=args.eval_anomaly_method,
                pca_residual_metric=args.eval_pca_residual_metric,
                knn_weight=args.eval_knn_weight,
                memory_max_patches=args.eval_memory_max_patches,
                normal_fit_max_patches=args.eval_normal_fit_max_patches,
                knn_chunk_size=args.eval_knn_chunk_size,
                knn_backend=args.eval_knn_backend,
                knn_dtype=args.eval_knn_dtype,
                dual_branch=args.eval_dual_branch,
            )

        standardized_scores = None
        decision_reference_scores = None
        decision_reference_images = normal_paths
        decision_reference_seconds = 0.0
        decision_threshold_source = "normal_reference_max"
        decision_calibration = "training-reference"
        if args.eval_normal_decision_calibration == "training-reference" and args.eval_normal_decision_quantile < 1:
            decision_threshold_source = "normal_training_quantile"
        elif args.eval_normal_decision_calibration == "leave-one-out":
            if len(normal_paths) < 2:
                raise ValueError("distilled leave-one-out calibration requires at least two normal shots")
            started = time.perf_counter()
            standardized_scores = _leave_one_out_normal_scores(
                normal_paths,
                build_fusion=build_fusion,
                fit_augment_count=args.eval_normal_decision_fit_augment_count,
                decision_augment_count=args.eval_normal_decision_augment_count,
                augmentations=["rotate"],
                fit_seed=args.seed,
                decision_seed=args.eval_normal_decision_seed,
            )
            decision_reference_seconds = time.perf_counter() - started
            decision_reference_images = []
            decision_threshold_source = "normal_leave_one_out_quantile"
            decision_calibration = "leave-one-out"
        fusion = build_fusion().fit_normal(normal_paths)
        if standardized_scores is not None:
            decision_reference_scores = (
                standardized_scores * float(fusion.subspace.score_scale)
                + float(fusion.subspace.score_center)
            )
        memory_stats = fusion.memory_stats()
        excluded = _selected_defect_paths(records)
        result_path = categories_dir / f"{category_name}.json"
        if args.dataset == "mvtec":
            category_dir = Path(args.data_root) / category_name
            metrics = evaluate_mvtec(
                fusion,
                category_dir,
                result_path,
                excluded_images=excluded,
                normal_reference_images=decision_reference_images,
                normal_reference_scores=decision_reference_scores,
                normal_reference_seconds=decision_reference_seconds,
                decision_threshold_source=decision_threshold_source,
                decision_threshold_quantile=args.eval_normal_decision_quantile,
                decision_threshold_quantile_method=args.eval_normal_decision_quantile_method,
            )
        else:
            samples = [(sample.image, sample.defect_type, sample.anomalous, sample.mask) for sample in category.test_samples]
            metrics = evaluate_samples(
                fusion,
                category_name,
                samples,
                result_path,
                excluded_images=excluded,
                normal_reference_images=decision_reference_images,
                normal_reference_scores=decision_reference_scores,
                normal_reference_seconds=decision_reference_seconds,
                decision_threshold_source=decision_threshold_source,
                decision_threshold_quantile=args.eval_normal_decision_quantile,
                decision_threshold_quantile_method=args.eval_normal_decision_quantile_method,
            )
        metrics.update(
            {
                "dataset": args.dataset,
                "normal_shots": args.normal_shots,
                "normal_shot_images": [str(Path(path)) for path in normal_paths],
                "defect_shots": args.defect_shots,
                "defect_shot_images": [str(Path(path)) for path in excluded],
                "seed": args.seed,
                "normal_decision_calibration": decision_calibration,
                "normal_decision_quantile": args.eval_normal_decision_quantile,
                "normal_decision_quantile_method": args.eval_normal_decision_quantile_method,
                "normal_decision_augment_count": args.eval_normal_decision_augment_count if decision_calibration == "leave-one-out" else 0,
                "normal_decision_fit_augment_count": args.eval_normal_decision_fit_augment_count if decision_calibration == "leave-one-out" else 0,
                "normal_decision_folds": len(normal_paths) if decision_calibration == "leave-one-out" else 0,
                "normal_decision_seed": args.eval_normal_decision_seed if decision_calibration == "leave-one-out" else None,
                "model": str(args.student_model),
                "lora_adapter": str(adapter_path),
                "feature_layers": list(_parse_layers(args.eval_feature_layers)),
                "image_size": args.eval_image_size,
                "resize_mode": args.eval_resize_mode,
                "layer_aggregation": args.eval_layer_aggregation,
                "layer_normalization": args.eval_layer_normalization,
                "anomaly_method": args.eval_anomaly_method,
                "pca_residual_metric": args.eval_pca_residual_metric,
                "knn_weight": args.eval_knn_weight if args.eval_anomaly_method in {"pca_knn", "pca_knn_anoco"} else 0,
                "dual_branch": args.eval_dual_branch,
                "memory_max_patches": args.eval_memory_max_patches if args.eval_anomaly_method != "pca" else 0,
                "normal_fit_max_patches": args.eval_normal_fit_max_patches,
                "knn_chunk_size": args.eval_knn_chunk_size if args.eval_anomaly_method != "pca" else 0,
                "knn_backend": fusion.normal_memory.resolved_backend if args.eval_anomaly_method != "pca" else "none",
                "knn_dtype": args.eval_knn_dtype if args.eval_anomaly_method != "pca" else "none",
                "memory_patch_count": memory_stats["patch_count"] if args.eval_anomaly_method != "pca" else 0,
                "memory_bytes": memory_stats["bytes"] if args.eval_anomaly_method != "pca" else 0,
                "metrics_file": str(result_path),
            }
        )
        predictions = json.loads(result_path.read_text(encoding="utf-8"))
        result_path.write_text(
            json.dumps({"metrics": metrics, "predictions": predictions}, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        all_metrics.append(metrics)
        del fusion, extractor
        if str(args.device or "").startswith("cuda") or (args.device is None and torch.cuda.is_available()):
            torch.cuda.empty_cache()

    macro = _engine_metric_summary(all_metrics)
    summary_path = evaluation_root / "results.json"
    csv_path = evaluation_root / "summary.csv"
    summary = {
        "macro_average": macro,
        "categories": all_metrics,
        "summary_file": str(summary_path),
        "summary_csv": str(csv_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_metrics_csv(csv_path, all_metrics, macro)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def _train_records(
    args: argparse.Namespace,
    records: Sequence[ImageRecord],
    output_dir: str | Path,
    teacher: nn.Module,
    teacher_processor,
    device: torch.device,
) -> dict:
    """Train and export one independent student for one category."""

    from transformers import AutoImageProcessor, AutoModel

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normal_paths = [record.image_path for record in records if not record.is_anomaly]
    if not normal_paths:
        raise ValueError("Each category must contain at least one normal image")

    student_load_kwargs = _pretrained_load_kwargs(args.student_model)
    student_backbone = AutoModel.from_pretrained(args.student_model, **student_load_kwargs)
    student_processor = AutoImageProcessor.from_pretrained(args.student_model, **student_load_kwargs)
    for name, backbone in (("teacher", teacher), ("student", student_backbone)):
        patch_size = _patch_size(backbone)
        if args.image_size % patch_size:
            raise ValueError(
                f"--image-size ({args.image_size}) must be divisible by the {name} patch size ({patch_size})"
            )
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
    teacher_centroid = estimate_normal_centroid(teacher, teacher_processor, normal_paths, teacher_layers_for_centroid, args.image_size, device, args.centroid_batch_size).detach()
    student_centroid = estimate_normal_centroid(student.backbone, student_processor, normal_paths, student_layers_for_centroid, args.image_size, device, args.centroid_batch_size).detach()

    dataset = ImageMaskDataset(records)
    loader = DataLoader(
        dataset,
        batch_size=max(1, args.batch_size),
        shuffle=True,
        num_workers=max(0, args.num_workers),
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(_trainable_parameter_groups(student, args.lr, args.head_lr, args.weight_decay))
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    history = []
    total_params, trainable_params = _count_parameters(student)
    lora_params = _lora_parameter_count(student.backbone)

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
                projected_layers = student_output["projected_layers"]
                student_grid = student_output["grid"]
                if student_grid != teacher_grid:
                    projected_layers = [_resize_tokens(layer, student_grid, teacher_grid) for layer in projected_layers]
                    student_features = _resize_tokens(student_output["aggregate"], student_grid, teacher_grid)
                else:
                    student_features = student_output["aggregate"]
                expected_patches = teacher_grid[0] * teacher_grid[1]
                if any(layer.shape[1] != expected_patches for layer in teacher_layers):
                    raise ValueError("Teacher hidden-state patch counts are inconsistent")
                patch_weights = masks_to_patch_weights(masks, teacher_inputs["pixel_values"].shape[-2:], teacher_grid, device, args.mask_alpha)
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
        print(json.dumps(epoch_metrics, ensure_ascii=True, sort_keys=True), flush=True)
        if args.save_every > 0 and epoch % args.save_every == 0:
            epoch_config = _training_config(args, layer_pairs, teacher_dim, student_dim, adapted_targets)
            epoch_config.update({"epoch": epoch, "metrics": epoch_metrics})
            save_lora_adapter(output_dir / f"epoch-{epoch:04d}", student, epoch_config)

    config = _training_config(args, layer_pairs, teacher_dim, student_dim, adapted_targets)
    config.update(
        {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "lora_parameters": lora_params,
            "history": history,
        }
    )
    config.update({"epoch": int(args.epochs), "metrics": history[-1]})
    adapter = save_lora_adapter(output_dir, student, config)
    summary = {
        "lora_adapter": str(adapter),
        "output": str(output_dir),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "lora_parameters": lora_params,
        "history": history,
        "normal_images": len(normal_paths),
        "defect_images": sum(record.is_anomaly for record in records),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return summary


def train(args: argparse.Namespace) -> dict:
    """Train one independent student per selected MVTec/VisA category."""

    from transformers import AutoImageProcessor, AutoModel

    set_seed(int(args.seed))
    device = _device(args.device)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    groups, counts = dataset_record_groups(args)
    teacher_load_kwargs = _pretrained_load_kwargs(args.teacher_model)
    teacher = AutoModel.from_pretrained(args.teacher_model, **teacher_load_kwargs).to(device).eval()
    teacher.requires_grad_(False)
    teacher_processor = AutoImageProcessor.from_pretrained(args.teacher_model, **teacher_load_kwargs)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for category, records in groups.items():
        set_seed(_stable_seed(args.seed, f"{args.dataset}:train:{category}"))
        print(f"[distill] {args.dataset}/{category}: normal={counts[category]['normal']} defect={counts[category]['defect']}", flush=True)
        summaries[category] = _train_records(args, records, root / category, teacher, teacher_processor, device)
    evaluation = None
    if args.evaluate:
        del teacher, teacher_processor
        if device.type == "cuda":
            torch.cuda.empty_cache()
        evaluation = evaluate_distilled_students(args, root, groups)
    summary = {
        "dataset": args.dataset,
        "data_root": args.data_root,
        "categories": counts,
        "outputs": summaries,
        "evaluation": evaluation,
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return summary


def _training_config(args, layer_pairs, teacher_dim, student_dim, adapted_targets):
    return {
        "dataset": args.dataset,
        "data_root": args.data_root,
        "normal_shots": int(args.normal_shots),
        "defect_shots": int(args.defect_shots),
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
        "evaluate": bool(args.evaluate),
        "eval_image_size": int(args.eval_image_size),
        "eval_feature_layers": list(_parse_layers(args.eval_feature_layers)),
        "eval_resize_mode": args.eval_resize_mode,
        "eval_layer_aggregation": args.eval_layer_aggregation,
        "eval_layer_normalization": args.eval_layer_normalization,
        "eval_anomaly_method": args.eval_anomaly_method,
        "eval_pca_residual_metric": args.eval_pca_residual_metric,
        "eval_dual_branch": bool(args.eval_dual_branch),
        "eval_normal_decision_calibration": args.eval_normal_decision_calibration,
        "eval_normal_decision_quantile": float(args.eval_normal_decision_quantile),
        "eval_normal_decision_quantile_method": args.eval_normal_decision_quantile_method,
        "eval_normal_decision_augment_count": int(args.eval_normal_decision_augment_count),
        "eval_normal_decision_fit_augment_count": int(args.eval_normal_decision_fit_augment_count),
        "eval_normal_decision_seed": int(args.eval_normal_decision_seed),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone DINOv3 ViT-B -> ViT-S+ distillation for MVTec AD and VisA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = parser.add_argument_group("dataset")
    data.add_argument("--dataset", choices=("mvtec", "visa", "folder"), required=True)
    data.add_argument("--data-root", help="MVTec/VisA root; required for dataset=mvtec or visa")
    data.add_argument("--split-csv", help="VisA 1cls.csv; defaults to <data-root>/split_csv/1cls.csv")
    data.add_argument("--categories", nargs="+", help="categories to train; omit to train all categories independently")
    data.add_argument("--normal-shots", type=int, default=-1, help="normal train images per category; -1 uses all")
    data.add_argument("--defect-shots", type=int, default=0, help="opt-in test defects per defect type; 0 avoids test leakage")
    data.add_argument("--normal-dir", help="dataset=folder only")
    data.add_argument("--defect-dir", help="optional dataset=folder defect images")
    data.add_argument("--mask-dir", help="optional dataset=folder defect masks")
    data.add_argument("--max-normal-images", type=int, default=0, help="debug cap per category; 0 means unlimited")
    data.add_argument("--max-defect-images", type=int, default=0, help="debug cap per category; 0 means unlimited")

    model = parser.add_argument_group("models and output")
    model.add_argument("--teacher-model", default=DEFAULT_TEACHER, help="Hugging Face identifier or local model directory")
    model.add_argument("--student-model", default=DEFAULT_STUDENT, help="Hugging Face identifier or local model directory")
    model.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--feature-layers", default=",".join(str(x) for x in DEFAULT_FEATURE_LAYERS))
    parser.add_argument("--teacher-layers")
    parser.add_argument("--student-layers")
    parser.add_argument("--adaptation", choices=("lora",), default="lora", help="LoRA is required because only adapter weights are exported")
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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-every", type=int, default=0, help="also save epoch LoRA adapters; 0 saves only the final adapter")

    evaluation = parser.add_argument_group("post-training evaluation")
    evaluation.add_argument("--evaluate", action=argparse.BooleanOptionalAction, default=True, help="evaluate each LoRA-adapted student with DefectFusion after training")
    evaluation.add_argument("--eval-image-size", type=int, default=None, help="detector input size; defaults to --image-size")
    evaluation.add_argument("--eval-feature-layers", default=None, help="student hidden states used by the detector; defaults to --feature-layers")
    evaluation.add_argument("--eval-resize-mode", choices=("direct", "longest_pad"), default="direct")
    evaluation.add_argument("--eval-layer-aggregation", choices=("mean", "concat"), default="mean")
    evaluation.add_argument("--eval-layer-normalization", choices=("none", "l2"), default="none")
    evaluation.add_argument("--eval-anomaly-method", choices=("pca", "knn", "pca_knn"), default="pca")
    evaluation.add_argument("--eval-pca-residual-metric", choices=("squared_l2", "mahalanobis"), default="squared_l2")
    evaluation.add_argument("--eval-dual-branch", action=argparse.BooleanOptionalAction, default=False)
    evaluation.add_argument("--eval-knn-weight", type=float, default=0.5)
    evaluation.add_argument("--eval-memory-max-patches", type=int, default=50000)
    evaluation.add_argument("--eval-normal-fit-max-patches", type=int, default=0)
    evaluation.add_argument("--eval-knn-chunk-size", type=int, default=256)
    evaluation.add_argument("--eval-knn-backend", choices=("auto", "numpy", "torch"), default="auto")
    evaluation.add_argument("--eval-knn-dtype", choices=("float32", "float16"), default="float32")
    evaluation.add_argument("--eval-top-k-ratio", type=float, default=0.05)
    evaluation.add_argument("--eval-image-score", choices=("mtop1p", "mean", "max", "p99"), default="mtop1p")
    evaluation.add_argument("--eval-image-top-ratio", type=float, default=0.01)
    evaluation.add_argument("--eval-normal-decision-calibration", choices=("training-reference", "leave-one-out"), default="leave-one-out")
    evaluation.add_argument("--eval-normal-decision-quantile", type=float, default=0.995)
    evaluation.add_argument("--eval-normal-decision-quantile-method", choices=("linear", "higher"), default="higher")
    evaluation.add_argument("--eval-normal-decision-augment-count", type=int, default=30)
    evaluation.add_argument("--eval-normal-decision-fit-augment-count", type=int, default=4)
    evaluation.add_argument("--eval-normal-decision-seed", type=int, default=None)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject invalid combinations before loading either DINOv3 model."""

    for attribute, option_name in (("teacher_model", "--teacher-model"), ("student_model", "--student-model")):
        reference = getattr(args, attribute, None)
        if reference is None:
            continue
        try:
            setattr(args, attribute, resolve_model_reference(reference, option_name))
        except ValueError as exc:
            parser.error(str(exc))
    if args.image_size <= 0 or args.epochs <= 0 or args.batch_size <= 0 or args.centroid_batch_size <= 0:
        parser.error("image-size, epochs, batch-size and centroid-batch-size must be positive")
    if args.normal_shots == 0 or args.normal_shots < -1:
        parser.error("normal-shots must be -1 or a positive integer")
    if args.defect_shots < 0:
        parser.error("defect-shots must be non-negative")
    if args.max_normal_images < 0 or args.max_defect_images < 0:
        parser.error("max-normal-images and max-defect-images must be non-negative")
    if args.dataset in {"mvtec", "visa"} and not args.data_root:
        parser.error("--data-root is required for --dataset mvtec or visa")
    if args.dataset != "visa" and args.split_csv:
        parser.error("--split-csv is only valid for --dataset visa")
    if args.dataset == "folder" and not (args.normal_dir or args.data_root):
        parser.error("--normal-dir or --data-root is required for --dataset folder")
    if args.dataset != "folder" and any((args.normal_dir, args.defect_dir, args.mask_dir)):
        parser.error("--normal-dir, --defect-dir and --mask-dir are only valid for --dataset folder")
    if args.dataset == "folder" and args.categories:
        parser.error("--categories is only valid for --dataset mvtec or visa")
    if args.last_n_blocks <= 0 or args.lora_rank <= 0 or args.lora_alpha <= 0:
        parser.error("last-n-blocks, lora-rank and lora-alpha must be positive")
    if not 0 <= args.lora_dropout < 1:
        parser.error("lora-dropout must be in [0, 1)")
    if args.lr <= 0 or args.head_lr <= 0 or args.grad_clip <= 0:
        parser.error("lr, head-lr and grad-clip must be positive")
    if args.weight_decay < 0:
        parser.error("weight-decay must be non-negative")
    if args.lambda_feature < 0 or args.lambda_map < 0 or args.mask_alpha < 0 or args.margin < 0:
        parser.error("loss weights and margin must be non-negative")
    if not 0 < args.top_ratio <= 1:
        parser.error("top-ratio must be in (0, 1]")
    if args.num_workers < 0 or args.save_every < 0:
        parser.error("num-workers and save-every must be non-negative")
    if bool(args.teacher_layers) != bool(args.student_layers):
        parser.error("--teacher-layers and --student-layers must be supplied together")
    try:
        _parse_layers(args.feature_layers)
        if args.teacher_layers:
            teacher_layers = _parse_layers(args.teacher_layers)
            student_layers = _parse_layers(args.student_layers)
            if len(teacher_layers) != len(student_layers):
                parser.error("--teacher-layers and --student-layers must contain the same number of indices")
    except ValueError as exc:
        parser.error(str(exc))
    if args.eval_image_size is None:
        args.eval_image_size = args.image_size
    if args.eval_feature_layers is None:
        args.eval_feature_layers = args.feature_layers
    if args.eval_normal_decision_seed is None:
        args.eval_normal_decision_seed = args.seed + 100
    if args.eval_image_size <= 0:
        parser.error("eval-image-size must be positive")
    if not 0 <= args.eval_knn_weight <= 1:
        parser.error("eval-knn-weight must be in [0, 1]")
    if args.eval_memory_max_patches < 0:
        parser.error("eval-memory-max-patches must be non-negative")
    if args.eval_normal_fit_max_patches < 0:
        parser.error("eval-normal-fit-max-patches must be non-negative")
    if args.eval_knn_chunk_size <= 0:
        parser.error("eval-knn-chunk-size must be positive")
    if not 0 < args.eval_top_k_ratio <= 1 or not 0 < args.eval_image_top_ratio <= 1:
        parser.error("eval-top-k-ratio and eval-image-top-ratio must be in (0, 1]")
    if not 0 < args.eval_normal_decision_quantile <= 1:
        parser.error("eval-normal-decision-quantile must be in (0, 1]")
    if args.eval_normal_decision_augment_count < 0 or args.eval_normal_decision_fit_augment_count < 0:
        parser.error("eval normal decision augmentation counts must be non-negative")
    if args.evaluate and args.eval_normal_decision_calibration == "leave-one-out" and 0 < args.normal_shots < 2:
        parser.error("eval leave-one-out calibration requires at least two normal shots")
    if args.evaluate and args.eval_normal_decision_calibration == "leave-one-out" and (args.eval_anomaly_method != "pca" or args.eval_dual_branch):
        parser.error("eval leave-one-out calibration currently requires PCA without dual branch")
    try:
        _parse_layers(args.eval_feature_layers)
    except ValueError as exc:
        parser.error(str(exc))
    if args.evaluate and args.dataset == "folder":
        parser.error("automatic evaluation requires --dataset mvtec or visa; use --no-evaluate for --dataset folder")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    summary = train(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
