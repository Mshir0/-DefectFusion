from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class DinoFeatureExtractor:
    """Dense frozen DINOv2 extractor. The interface also accepts compatible DINOv3 wrappers."""
    def __init__(self, model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m", image_size=448, device=None, debias=False, svd_components=20, feature_layers=(-1, -3, -5, -7), layer_aggregation="mean"):
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
        state_count = int(getattr(self.model.config, "num_hidden_layers", 0) or 0) + 1
        invalid = [index for index in self.feature_layers if index >= state_count or index < -state_count]
        if invalid:
            raise ValueError(
                f"Feature layers {invalid} are invalid for a model with "
                f"{state_count} hidden states (embedding output plus transformer blocks)"
            )
        if layer_aggregation not in {"mean", "concat"}:
            raise ValueError("layer_aggregation must be mean or concat")
        self.layer_aggregation = layer_aggregation

    def _prepare(self, image):
        return self.processor(
            images=image,
            return_tensors="pt",
            do_resize=True,
            size={"height": self.image_size, "width": self.image_size},
            do_center_crop=False,
        ).to(self.device)

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
        inputs = self._prepare(image)
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
        inputs = self._prepare(image)
        tokens, grid = self._patch_tokens(inputs)
        tokens = self._debias(tokens) if self.debias else tokens.float()
        return tokens[0].cpu().numpy(), grid
