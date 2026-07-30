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
