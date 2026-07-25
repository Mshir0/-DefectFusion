from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class DinoFeatureExtractor:
    """Dense frozen DINOv2 extractor. The interface also accepts compatible DINOv3 wrappers."""
    def __init__(self, model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m", image_size=448, device=None, debias=False, svd_components=20, feature_layers=(-1, -2, -3, -4), layer_aggregation="mean", multiscale_mode="overlap", crop_ratio=0.75, crop_overlap=0.5):
        from transformers import AutoImageProcessor, AutoModel
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval().to(self.device)
        self.image_size = image_size
        self.debias = debias
        self.svd_components = svd_components
        self.positional_basis = None
        self.feature_layers = tuple(feature_layers)
        if not self.feature_layers:
            raise ValueError("feature_layers cannot be empty")
        if layer_aggregation not in {"mean", "concat"}:
            raise ValueError("layer_aggregation must be mean or concat")
        self.layer_aggregation = layer_aggregation
        if multiscale_mode not in {"none", "overlap"}:
            raise ValueError("multiscale_mode must be none or overlap")
        if not 0 < crop_ratio <= 1 or not 0 <= crop_overlap < 1:
            raise ValueError("crop_ratio must be in (0,1] and crop_overlap in [0,1)")
        self.multiscale_mode = multiscale_mode
        self.crop_ratio = crop_ratio
        self.crop_overlap = crop_overlap

    def _patch_tokens(self, inputs):
        out = self.model(**inputs, output_hidden_states=True)
        n_register = int(getattr(self.model.config, "num_register_tokens", 0) or 0)
        try:
            selected = [out.hidden_states[index][:, 1 + n_register :, :] for index in self.feature_layers]
        except IndexError as exc:
            raise ValueError(f"Invalid feature layer in {self.feature_layers}; model returned {len(out.hidden_states)} states") from exc
        if self.layer_aggregation == "mean":
            tokens = torch.stack(selected, dim=0).mean(dim=0)
        else:
            tokens = torch.cat(selected, dim=-1)
        n = tokens.shape[1]
        pixel_values = inputs.get("pixel_values")
        patch_size = int(getattr(self.model.config, "patch_size", 16) or 16)
        if pixel_values is not None:
            height, width = pixel_values.shape[-2:]
            grid = (height // patch_size, width // patch_size)
            if grid[0] * grid[1] == n:
                return tokens, grid
        side = int(n ** 0.5)
        if side * side != n:
            raise ValueError(f"Backbone returned {n} patch tokens; cannot infer spatial grid")
        grid = (side, side)
        return tokens, grid

    @torch.inference_mode()
    def _build_positional_basis(self):
        # INSID3 estimates positional directions from a zero-content image.
        image = Image.new("RGB", (self.image_size, self.image_size), color=0)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        tokens, _ = self._patch_tokens(inputs)
        features = F.normalize(tokens[0].float(), p=2, dim=-1).T
        features = features - features.mean(dim=1, keepdim=True)
        basis, _, _ = torch.linalg.svd(features, full_matrices=False)
        keep = min(self.svd_components, basis.shape[1])
        self.positional_basis = basis[:, :keep].contiguous()

    def _debias(self, tokens):
        if self.positional_basis is None:
            self._build_positional_basis()
        features = F.normalize(tokens.float(), p=2, dim=-1)
        basis = self.positional_basis.to(device=features.device, dtype=features.dtype)
        features = features - (features @ basis) @ basis.T
        return F.normalize(features, p=2, dim=-1)

    @torch.inference_mode()
    def extract(self, image: Image.Image):
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        tokens, grid = self._patch_tokens(inputs)
        tokens = self._debias(tokens) if self.debias else tokens.float()
        return tokens[0].cpu().numpy(), grid

    def extract_views(self, image: Image.Image):
        image = image.convert("RGB")
        width, height = image.size
        views = [(image, (0, 0, width, height))]
        if self.multiscale_mode == "overlap" and self.crop_ratio < 1:
            crop_w = max(1, round(width * self.crop_ratio))
            crop_h = max(1, round(height * self.crop_ratio))
            stride_x = max(1, round(crop_w * (1 - self.crop_overlap)))
            stride_y = max(1, round(crop_h * (1 - self.crop_overlap)))
            xs = list(range(0, max(1, width - crop_w + 1), stride_x))
            ys = list(range(0, max(1, height - crop_h + 1), stride_y))
            if xs[-1] != width - crop_w: xs.append(width - crop_w)
            if ys[-1] != height - crop_h: ys.append(height - crop_h)
            for y in ys:
                for x in xs:
                    box = (x, y, x + crop_w, y + crop_h)
                    views.append((image.crop(box), box))
        return [(*self.extract(view), box) for view, box in views]
