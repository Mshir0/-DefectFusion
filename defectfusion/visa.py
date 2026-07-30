from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VisaSample:
    image: Path
    defect_type: str
    anomalous: bool
    mask: Path | None


@dataclass(frozen=True)
class VisaCategory:
    name: str
    normal_images: tuple[Path, ...]
    test_samples: tuple[VisaSample, ...]


def _value(row, *names):
    normalized = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        value = normalized.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _resolve(root: Path, value: str) -> Path | None:
    if not value or value.lower() in {"none", "nan"}:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_visa_categories(data_root, split_csv=None):
    """Load the official VisA 1-class split without rearranging the dataset."""
    root = Path(data_root)
    split_path = Path(split_csv) if split_csv else root / "split_csv" / "1cls.csv"
    if not split_path.is_file():
        raise FileNotFoundError(f"VisA split CSV not found: {split_path}")

    groups = {}
    with split_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"object", "split", "label", "image"}
        columns = {str(name).strip().lower() for name in (reader.fieldnames or [])}
        if not required.issubset(columns):
            raise ValueError(f"VisA split CSV requires columns {sorted(required)}; found {sorted(columns)}")
        for row in reader:
            category = _value(row, "object")
            split = _value(row, "split").lower()
            label = _value(row, "label").lower()
            image = _resolve(root, _value(row, "image"))
            if not category or image is None:
                continue
            group = groups.setdefault(category, {"normal": [], "test": []})
            anomalous = label not in {"normal", "good", "0"}
            if split == "train" and not anomalous:
                group["normal"].append(image)
            elif split == "test":
                defect_type = _value(row, "defect_type", "type") or ("anomaly" if anomalous else "good")
                mask = _resolve(root, _value(row, "mask")) if anomalous else None
                group["test"].append(VisaSample(image, defect_type, anomalous, mask))

    categories = []
    for name, group in sorted(groups.items()):
        if not group["normal"]:
            raise ValueError(f"VisA category {name!r} has no normal training images")
        if not group["test"]:
            raise ValueError(f"VisA category {name!r} has no test images")
        categories.append(VisaCategory(name, tuple(sorted(group["normal"])), tuple(sorted(group["test"], key=lambda x: str(x.image)))))
    if not categories:
        raise ValueError(f"No VisA categories found in {split_path}")
    return categories
