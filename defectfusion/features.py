from __future__ import annotations
import numpy as np
import torch
from PIL import Image


class DinoFeatureExtractor:
    """Dense frozen DINOv2 extractor. The interface also accepts compatible DINOv3 wrappers."""
    def __init__(self, model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m", image_size=448, device=None):
        from transformers import AutoImageProcessor, AutoModel
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval().to(self.device)
        self.image_size = image_size

    @torch.inference_mode()
    def extract(self, image: Image.Image):
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model(**inputs).last_hidden_state
        # DINOv3 returns CLS + optional register tokens + spatial patch tokens.
        n_register = int(getattr(self.model.config, "num_register_tokens", 0) or 0)
        tokens = out[:, 1 + n_register :, :]
        n = tokens.shape[1]
        pixel_values = inputs.get("pixel_values")
        patch_size = getattr(getattr(self.model.config, "patch_size", None), "__int__", lambda: 16)()
        if pixel_values is not None and isinstance(patch_size, int):
            height, width = pixel_values.shape[-2:]
            grid = (height // patch_size, width // patch_size)
            if grid[0] * grid[1] == n:
                return tokens[0].float().cpu().numpy(), grid
        side = int(n ** 0.5)
        if side * side != n:
            raise ValueError(f"Backbone returned {n} patch tokens; cannot infer spatial grid")
        x = tokens[0].float().cpu().numpy()
        return x, (side, side)
