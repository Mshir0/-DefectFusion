from __future__ import annotations


def image_size_overrides(values):
    if isinstance(values, dict):
        items = values.items()
    else:
        items = []
        for value in values or []:
            if "=" not in str(value):
                raise ValueError(f"Image-size override must be CATEGORY=SIZE: {value}")
            items.append(str(value).split("=", 1))
    overrides = {}
    for category, size in items:
        category = str(category).strip()
        try:
            size = int(size)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid image size for category {category!r}: {size}") from exc
        if not category or size <= 0:
            raise ValueError(f"Image-size override must have a category and positive size: {category}={size}")
        overrides[category] = size
    return overrides


def unit_interval_overrides(values, name="Weight"):
    if isinstance(values, dict):
        items = values.items()
    else:
        items = []
        for value in values or []:
            if "=" not in str(value):
                raise ValueError(f"{name} override must be CATEGORY=VALUE: {value}")
            items.append(str(value).split("=", 1))
    overrides = {}
    for category, value in items:
        category = str(category).strip()
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {name.lower()} for category {category!r}: {value}") from exc
        if not category or not 0 <= value <= 1:
            raise ValueError(f"{name} override must have a category and value in [0, 1]: {category}={value}")
        overrides[category] = value
    return overrides
