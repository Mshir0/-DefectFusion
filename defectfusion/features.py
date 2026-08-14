from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class DinoFeatureExtractor:
    """Dense frozen DINOv2 extractor. The interface also accepts compatible DINOv3 wrappers."""
    def __init__(self, model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m", image_size=448, resize_mode="direct", device=None, debias=False, svd_components=20, feature_layers=(1, 17, 21, 23), layer_aggregation="mean", layer_normalization="none"):
        from transformers import AutoImageProcessor, AutoModel
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_path = Path(str(model_name)).expanduser()
        if model_path.is_dir():
            model_name = str(model_path.resolve())
            load_kwargs = {"local_files_only": True}
        else:
            load_kwargs = {}
        self.processor = AutoImageProcessor.from_pretrained(model_name, **load_kwargs)
        self.model = AutoModel.from_pretrained(model_name, **load_kwargs).eval().to(self.device)
        self.image_size = image_size
        if resize_mode not in {"direct", "longest_pad"}:
            raise ValueError("resize_mode must be direct or longest_pad")
        self.resize_mode = resize_mode
        self._content_bounds = None
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
        if layer_normalization not in {"none", "l2"}:
            raise ValueError("layer_normalization must be none or l2")
        self.layer_normalization = layer_normalization

    def _prepare(self, image):
        self._content_bounds = None
        if self.resize_mode == "longest_pad":
            width, height = image.size
            scale = self.image_size / max(width, height)
            resized = image.resize(
                (max(1, round(width * scale)), max(1, round(height * scale))),
                Image.Resampling.BICUBIC,
            )
            mean = getattr(self.processor, "image_mean", (0.0, 0.0, 0.0))
            fill = tuple(round(float(value) * 255) for value in mean)
            padded = Image.new("RGB", (self.image_size, self.image_size), color=fill)
            offset = ((self.image_size - resized.width) // 2, (self.image_size - resized.height) // 2)
            padded.paste(resized, offset)
            self._content_bounds = (offset[0], offset[1], offset[0] + resized.width, offset[1] + resized.height)
            image = padded
        return self.processor(
            images=image,
            return_tensors="pt",
            do_resize=self.resize_mode == "direct",
            size={"height": self.image_size, "width": self.image_size},
            do_center_crop=False,
        ).to(self.device)

    def _patch_tokens(self, inputs, layer_normalization=None, output=None, feature_layers=None):
        out = self.model(**inputs, output_hidden_states=True) if output is None else output
        n_register = int(getattr(self.model.config, "num_register_tokens", 0) or 0)
        layers = self.feature_layers if feature_layers is None else tuple(feature_layers)
        try:
            selected = [out.hidden_states[index][:, 1 + n_register :, :] for index in layers]
        except IndexError as exc:
            raise ValueError(f"Invalid feature layer in {layers}; model returned {len(out.hidden_states)} states") from exc
        normalization = self.layer_normalization if layer_normalization is None else layer_normalization
        if normalization == "l2":
            selected = [F.normalize(tokens.float(), p=2, dim=-1) for tokens in selected]
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
                if self._content_bounds is not None:
                    left, top, right, bottom = self._content_bounds
                    row_centers = (torch.arange(grid[0], device=tokens.device) + 0.5) * patch_size
                    column_centers = (torch.arange(grid[1], device=tokens.device) + 0.5) * patch_size
                    rows = torch.where((row_centers >= top) & (row_centers < bottom))[0]
                    columns = torch.where((column_centers >= left) & (column_centers < right))[0]
                    if not len(rows):
                        rows = torch.argmin(torch.abs(row_centers - (top + bottom) / 2)).reshape(1)
                    if not len(columns):
                        columns = torch.argmin(torch.abs(column_centers - (left + right) / 2)).reshape(1)
                    tokens = tokens.reshape(tokens.shape[0], grid[0], grid[1], tokens.shape[-1])
                    tokens = tokens[:, rows][:, :, columns].reshape(tokens.shape[0], len(rows) * len(columns), tokens.shape[-1])
                    grid = (len(rows), len(columns))
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

    @torch.inference_mode()
    def extract_dual(self, image: Image.Image):
        """Return raw and per-layer-L2 features from one backbone forward pass."""
        image = image.convert("RGB")
        inputs = self._prepare(image)
        output = self.model(**inputs, output_hidden_states=True)
        raw, grid = self._patch_tokens(inputs, layer_normalization="none", output=output)
        normalized, normalized_grid = self._patch_tokens(inputs, layer_normalization="l2", output=output)
        if normalized_grid != grid:
            raise ValueError(f"Dual feature grids differ: raw={grid}, l2={normalized_grid}")
        if self.debias:
            raw = self._debias(raw)
            normalized = self._debias(normalized)
        return raw[0].float().cpu().numpy(), normalized[0].float().cpu().numpy(), grid

    @torch.inference_mode()
    def extract_dual_layers(self, image: Image.Image):
        """Return dual aggregate features plus individual L2-normalized layers."""
        image = image.convert("RGB")
        inputs = self._prepare(image)
        output = self.model(**inputs, output_hidden_states=True)
        raw, grid = self._patch_tokens(inputs, layer_normalization="none", output=output)
        normalized, normalized_grid = self._patch_tokens(inputs, layer_normalization="l2", output=output)
        layers = []
        for index in self.feature_layers:
            layer, layer_grid = self._patch_tokens(
                inputs, layer_normalization="l2", output=output, feature_layers=(index,),
            )
            if layer_grid != grid:
                raise ValueError(f"Per-layer feature grid differs: aggregate={grid}, layer={layer_grid}")
            layers.append(layer)
        if normalized_grid != grid:
            raise ValueError(f"Dual feature grids differ: raw={grid}, l2={normalized_grid}")
        if self.debias:
            raw = self._debias(raw)
            normalized = self._debias(normalized)
            layers = [self._debias(layer) for layer in layers]
        return (
            raw[0].float().cpu().numpy(), normalized[0].float().cpu().numpy(),
            np.stack([layer[0].float().cpu().numpy() for layer in layers], axis=0), grid,
        )
