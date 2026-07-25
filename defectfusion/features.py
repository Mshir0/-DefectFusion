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
        tokens = out[:, 1:, :]
        n = tokens.shape[1]
        side = int(n ** 0.5)
        if side * side != n:
            raise ValueError(f"Backbone returned {n} tokens; expected a square patch grid")
        x = tokens[0].float().cpu().numpy()
        return x, (side, side)
