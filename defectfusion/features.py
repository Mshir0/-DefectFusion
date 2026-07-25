from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class DinoFeatureExtractor:
    """Dense frozen DINOv2 extractor. The interface also accepts compatible DINOv3 wrappers."""
    def __init__(self, model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m", image_size=448, device=None, debias=False, svd_components=20, feature_layers=(-1, -2, -3, -4), layer_aggregation="mean", foreground_mask="dino_saliency", foreground_percentile=0.15, saliency_layer=6):
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
        if foreground_mask not in {"none", "dino_saliency"}:
            raise ValueError("foreground_mask must be none or dino_saliency")
        self.foreground_mask = foreground_mask
        if not 0 <= foreground_percentile < 1:
            raise ValueError("foreground_percentile must be in [0, 1)")
        self.foreground_percentile = foreground_percentile
        self.saliency_layer = saliency_layer
        if self.foreground_mask == "dino_saliency":
            try:
                self.model.set_attn_implementation("eager")
            except AttributeError:
                pass

    def _patch_tokens(self, inputs):
        need_attention = self.foreground_mask == "dino_saliency"
        out = self.model(**inputs, output_hidden_states=True, output_attentions=need_attention)
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
                return tokens, grid, self._foreground_from_attention(out.attentions, n_register, n, grid) if need_attention else np.ones(n, dtype=bool)
        side = int(n ** 0.5)
        if side * side != n:
            raise ValueError(f"Backbone returned {n} patch tokens; cannot infer spatial grid")
        grid = (side, side)
        return tokens, grid, self._foreground_from_attention(out.attentions, n_register, n, grid) if need_attention else np.ones(n, dtype=bool)

    def _foreground_from_attention(self, attentions, n_register, n_patches, grid):
        if attentions is None:
            raise RuntimeError("DINO attentions are unavailable; use --foreground-mask none or an eager attention implementation")
        layer = self.saliency_layer if self.saliency_layer >= 0 else len(attentions) + self.saliency_layer
        if not 0 <= layer < len(attentions):
            raise ValueError(f"saliency_layer {self.saliency_layer} is outside 0..{len(attentions) - 1}")
        attention = attentions[layer]
        start = 1 + n_register
        if n_register:
            saliency = attention[:, :, 1:start, start : start + n_patches].mean(dim=(1, 2))[0]
        else:
            saliency = attention[:, :, 0, start : start + n_patches].mean(dim=1)[0]
        values = saliency.float().cpu().numpy()
        threshold = np.quantile(values, self.foreground_percentile)
        return values >= threshold

    @torch.inference_mode()
    def _build_positional_basis(self):
        # INSID3 estimates positional directions from a zero-content image.
        image = Image.new("RGB", (self.image_size, self.image_size), color=0)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        tokens, _, _ = self._patch_tokens(inputs)
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
    def extract(self, image: Image.Image, return_mask=False):
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        tokens, grid, foreground = self._patch_tokens(inputs)
        tokens = self._debias(tokens) if self.debias else tokens.float()
        result = (tokens[0].cpu().numpy(), grid)
        return (*result, foreground) if return_mask else result
