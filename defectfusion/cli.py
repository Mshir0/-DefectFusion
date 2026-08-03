from __future__ import annotations

import argparse
import glob
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

from .features import DinoFeatureExtractor
from .pipeline import DefectFusion, NormalTrainingView
from .mvtec import evaluate_mvtec, evaluate_samples
from .reporting import experiment_output_dir, write_metrics_csv
from .settings import image_size_overrides as parse_image_size_overrides
from .visa import load_visa_categories


def _images(root: str, recursive: bool = True) -> list[str]:
    pattern = "**/*" if recursive else "*"
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(str(p) for p in Path(root).glob(pattern) if p.is_file() and p.suffix.lower() in allowed)


def _config(path: str | None) -> dict:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _layers(value) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(x) for x in value)
    return tuple(int(x.strip()) for x in str(value).split(",") if x.strip())


FEATURE_LAYER_PRESETS = {
    "cross4": (1, 17, 21, 23),
    "last4": (-1, -2, -3, -4),
    # SubspaceAD uses these seven intermediate hidden states with mean fusion.
    "middle7": (-12, -13, -14, -15, -16, -17, -18),
}


def _feature_layers(args, cfg) -> tuple[tuple[int, ...], str | None]:
    explicit = getattr(args, "feature_layers", None)
    preset = getattr(args, "feature_layer_preset", None)
    if explicit and preset:
        raise ValueError("--feature-layers and --feature-layer-preset are mutually exclusive")
    if explicit:
        return _layers(explicit), None
    if preset:
        return FEATURE_LAYER_PRESETS[preset], preset
    if "feature_layers" in cfg:
        return _layers(cfg["feature_layers"]), None
    config_preset = cfg.get("feature_layer_preset")
    if config_preset:
        if config_preset not in FEATURE_LAYER_PRESETS:
            raise ValueError(f"Unknown feature_layer_preset: {config_preset}")
        return FEATURE_LAYER_PRESETS[config_preset], config_preset
    return FEATURE_LAYER_PRESETS["cross4"], "cross4"


def _augment_normal_images(paths, count, augmentations, seed):
    if count <= 0:
        return paths
    from torchvision.transforms import functional as TF
    rng = random.Random(seed)
    result = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        result.append(NormalTrainingView(image))
        for _ in range(count):
            augmented = image.copy()
            inverse_position_matrix = np.eye(3, dtype=np.float64)
            for name in augmentations:
                if name == "rotate":
                    angle = rng.uniform(0, 345)
                    augmented = TF.rotate(augmented, angle)
                    radians = math.radians(angle)
                    cosine, sine = math.cos(radians), math.sin(radians)
                    inverse_rotation = np.array([[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]])
                    inverse_rotation[:2, 2] = 0.5 - inverse_rotation[:2, :2] @ np.array([0.5, 0.5])
                    inverse_position_matrix = inverse_position_matrix @ inverse_rotation
                elif name == "hflip" and rng.random() < 0.5:
                    augmented = TF.hflip(augmented)
                    inverse_position_matrix = inverse_position_matrix @ np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, 1.0]])
                elif name == "vflip" and rng.random() < 0.5:
                    augmented = TF.vflip(augmented)
                    inverse_position_matrix = inverse_position_matrix @ np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
                elif name == "color_jitter":
                    augmented = TF.adjust_brightness(augmented, rng.uniform(0.8, 1.2))
                    augmented = TF.adjust_contrast(augmented, rng.uniform(0.8, 1.2))
                    augmented = TF.adjust_saturation(augmented, rng.uniform(0.8, 1.2))
                    augmented = TF.adjust_hue(augmented, rng.uniform(-0.1, 0.1))
                elif name == "affine":
                    translate = [round(rng.uniform(-0.15, 0.15) * image.width), round(rng.uniform(-0.15, 0.15) * image.height)]
                    augmented = TF.affine(augmented, angle=0, translate=translate, scale=1.0, shear=rng.uniform(-10, 10))
                    inverse_position_matrix = None
                elif name not in {"hflip", "vflip"}:
                    raise ValueError(f"Unknown normal augmentation: {name}")
            result.append(NormalTrainingView(augmented, inverse_position_matrix))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description="Few-shot / zero-shot defect detection")
    p.add_argument("--config", help="JSON config; CLI flags override it")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit normal subspace and optional defect prototypes")
    f.add_argument("--normal-dir"); f.add_argument("--prototype-dir", help="subdirectories are defect labels")
    f.add_argument("--model", default=None); f.add_argument("--output", default=None)
    f.add_argument("--image-size", type=int, default=None)
    f.add_argument("--resize-mode", choices=["direct", "longest_pad"], default=None)
    f.add_argument("--alpha", type=float, default=None); f.add_argument("--unknown-threshold", type=float, default=None)
    f.add_argument("--device", default=None); f.add_argument("--non-recursive", action="store_true")
    f.add_argument("--debias", action="store_true"); f.add_argument("--svd-components", type=int, default=20)
    f.add_argument("--top-k-ratio", type=float, default=None, help="highest PCA-residual patch ratio for typing")
    f.add_argument("--image-score", choices=["mtop1p", "mean", "max", "p99"], default=None)
    f.add_argument("--image-top-ratio", type=float, default=None)
    f.add_argument("--image-fusion-stage", choices=["patch", "score"], default=None)
    f.add_argument("--image-spatial-weight", type=float, default=None)
    f.add_argument("--image-min-component-size", type=int, default=None)
    f.add_argument("--type-matching", choices=["prototype_mean", "bidirectional_patch", "rbf_svm"], default=None)
    f.add_argument("--anomaly-method", choices=["pca", "knn", "pca_knn", "anoco", "pca_anoco", "pca_knn_anoco"], default=None); f.add_argument("--knn-weight", type=float, default=None)
    f.add_argument("--anoco-neighbors", type=int, default=None); f.add_argument("--anoco-query-weight", type=float, default=None); f.add_argument("--anoco-temperature", type=float, default=None); f.add_argument("--anoco-affinity", choices=["softmax", "cosine"], default=None); f.add_argument("--anoco-anchor-ranking", choices=["mean", "minimum"], default=None); f.add_argument("--anoco-norm-compatibility", action="store_true"); f.add_argument("--anoco-weight", type=float, default=None); f.add_argument("--anoco-layer-consensus", action="store_true")
    f.add_argument("--pca-residual-metric", choices=["squared_l2", "mahalanobis"], default=None)
    f.add_argument("--fusion-mode", choices=["fixed", "gated"], default=None); f.add_argument("--gate-temperature", type=float, default=None)
    f.add_argument("--memory-max-patches", type=int, default=None); f.add_argument("--knn-chunk-size", type=int, default=None)
    f.add_argument("--knn-spatial-radius", type=float, default=None)
    f.add_argument("--align-training-positions", action="store_true", help="map augmented normal patches back to reference coordinates before spatial kNN")
    f.add_argument("--dual-branch", action="store_true", help="use L2 features for image score and raw features for pixel map")
    f.add_argument("--test-augmentations", nargs="*", choices=["hflip", "vflip"], default=None)
    f.add_argument("--knn-backend", choices=["auto", "numpy", "torch"], default=None); f.add_argument("--knn-dtype", choices=["float32", "float16"], default=None)
    f.add_argument("--feature-layers", default=None); f.add_argument("--feature-layer-preset", choices=FEATURE_LAYER_PRESETS, default=None); f.add_argument("--layer-aggregation", choices=["mean", "concat"], default=None); f.add_argument("--layer-normalization", choices=["none", "l2"], default=None)
    f.add_argument("--map-postprocess", choices=["none", "gaussian", "crf"], default=None); f.add_argument("--gaussian-sigma", type=float, default=None)
    q = sub.add_parser("predict", help="score one image or a directory")
    q.add_argument("--model-state", required=True); q.add_argument("--image", required=True)
    q.add_argument("--model", default=None); q.add_argument("--device", default=None)
    q.add_argument("--image-size", type=int, default=None)
    q.add_argument("--resize-mode", choices=["direct", "longest_pad"], default=None)
    q.add_argument("--output", help="write JSON results to a file")
    q.add_argument("--debias", action="store_true"); q.add_argument("--svd-components", type=int, default=20)
    q.add_argument("--feature-layers", default=None); q.add_argument("--feature-layer-preset", choices=FEATURE_LAYER_PRESETS, default=None); q.add_argument("--layer-aggregation", choices=["mean", "concat"], default=None); q.add_argument("--layer-normalization", choices=["none", "l2"], default=None)
    e = sub.add_parser("evaluate-mvtec", aliases=["evaluate-visa"], help="evaluate MVTec AD or official VisA splits")
    e.add_argument("--data-dir", help="MVTec category directory")
    e.add_argument("--data-root", help="MVTec root containing all 15 category directories")
    e.add_argument("--split-csv", help="VisA split CSV; defaults to <data-root>/split_csv/1cls.csv")
    e.add_argument("--categories", nargs="+", help="evaluate only these category names")
    e.add_argument("--prototype-dir", help="optional defect prototypes; subdirectories are defect labels")
    e.add_argument("--normal-shots", type=int, default=-1, help="normal train/good references per category; -1 uses all")
    e.add_argument("--defect-shots", "--few-shot", dest="defect_shots", type=int, default=0, help="labeled defect exemplars per defect type")
    e.add_argument("--seed", type=int, default=42, help="seed used for reproducible normal and defect sampling")
    e.add_argument("--normal-augment-count", type=int, default=None, help="augmented views per normal shot; defaults to 30 in few-shot mode")
    e.add_argument("--normal-augmentations", nargs="+", choices=["rotate", "hflip", "vflip", "color_jitter", "affine"], default=None)
    e.add_argument("--affine-categories", nargs="+", default=None, help="append affine augmentation only for these categories")
    e.add_argument("--no-augment-categories", nargs="+", default=None)
    e.add_argument("--model", default=None); e.add_argument("--device", default=None)
    e.add_argument("--image-size", type=int, default=None)
    e.add_argument("--image-size-override", action="append", default=None, metavar="CATEGORY=SIZE", help="override input size for one category; repeatable")
    e.add_argument("--pixel-image-size-override", action="append", default=None, metavar="CATEGORY=SIZE", help="override only the pixel-head input size; repeatable")
    e.add_argument("--image-head-size-override", action="append", default=None, metavar="CATEGORY=SIZE", help="override only the image-head input size; repeatable")
    e.add_argument("--pixel-multiscale-size-override", action="append", default=None, metavar="CATEGORY=SIZE", help="add a second pixel PCA/kNN resolution; repeatable")
    e.add_argument("--pixel-multiscale-weight", type=float, default=None, help="second pixel-resolution contribution after grid alignment")
    e.add_argument("--resize-mode", choices=["direct", "longest_pad"], default=None)
    e.add_argument("--output", default="outputs/mvtec-results", help="experiment output directory; a filename is converted to a same-stem directory")
    e.add_argument("--debias", action="store_true", help="apply INSID3 positional debiasing")
    e.add_argument("--svd-components", type=int, default=20, help="INSID3 positional basis rank")
    e.add_argument("--top-k-ratio", type=float, default=None, help="highest PCA-residual patch ratio for typing")
    e.add_argument("--image-score", choices=["mtop1p", "mean", "max", "p99"], default=None)
    e.add_argument("--image-top-ratio", type=float, default=None, help="top patch fraction used by mtop1p")
    e.add_argument("--image-fusion-stage", choices=["patch", "score"], default=None, help="fuse PCA/kNN before or after image aggregation")
    e.add_argument("--image-spatial-weight", type=float, default=None, help="relative image-score boost from connected Top-K patches")
    e.add_argument("--image-min-component-size", type=int, default=None, help="reject Top-K candidate regions smaller than this many patches")
    e.add_argument("--component-reject-categories", nargs="+", default=None, help="apply image-min-component-size only to these categories")
    e.add_argument("--type-matching", choices=["prototype_mean", "bidirectional_patch", "rbf_svm"], default=None)
    e.add_argument("--anomaly-method", choices=["pca", "knn", "pca_knn", "anoco", "pca_anoco", "pca_knn_anoco"], default=None, help="normal anomaly detector; pca_knn_anoco uses PCA+kNN for pixels and PCA+ANoCo for images")
    e.add_argument("--pca-residual-metric", choices=["squared_l2", "mahalanobis"], default=None, help="PCA orthogonal-residual metric")
    e.add_argument("--knn-weight", type=float, default=None, help="kNN contribution in calibrated pca_knn fusion")
    e.add_argument("--anoco-neighbors", type=int, default=None, help="anchor-consistent normal neighbors")
    e.add_argument("--anoco-query-weight", type=float, default=None, help="query fidelity weight in closed-form manifold pull")
    e.add_argument("--anoco-temperature", type=float, default=None, help="normal-neighbor softmax temperature")
    e.add_argument("--anoco-affinity", choices=["softmax", "cosine"], default=None, help="ANoCo edge affinity; cosine is an experimental normalized mode")
    e.add_argument("--anoco-anchor-ranking", choices=["mean", "minimum"], default=None, help="combine query and anchor similarity when selecting normal neighbors")
    e.add_argument("--anoco-norm-compatibility", action="store_true", help="multiply ANoCo affinities by normalized feature-norm compatibility")
    e.add_argument("--anoco-weight", type=float, default=None, help="ANoCo contribution in calibrated pca_anoco fusion")
    e.add_argument("--pixel-anoco-weight", type=float, default=None, help="auxiliary ANoCo contribution in the calibrated pixel PCA+kNN branch")
    e.add_argument("--anoco-layer-consensus", action="store_true", help="replace aggregate-layer ANoCo with the median calibrated drift across selected layers")
    e.add_argument("--fusion-mode", choices=["fixed", "gated"], default=None, help="fixed weight or normal-tail-calibrated patch gate")
    e.add_argument("--gate-temperature", type=float, default=None, help="soft gate temperature; lower values select one expert more strongly")
    e.add_argument("--memory-max-patches", type=int, default=None, help="maximum normal patches retained for kNN; 0 keeps all")
    e.add_argument("--normal-fit-max-patches", type=int, default=None, help="maximum normal patches used to fit each branch; 0 keeps all")
    e.add_argument("--knn-chunk-size", type=int, default=None, help="query patches per kNN matrix chunk")
    e.add_argument("--knn-spatial-radius", type=float, default=None, help="normalized local kNN radius; -1 searches globally")
    e.add_argument("--knn-spatial-categories", nargs="+", default=None, help="apply the local kNN radius only to these categories")
    e.add_argument("--align-training-positions", action="store_true", help="map rotate/flip normal augmentation positions back to canonical coordinates")
    e.add_argument("--dual-branch", action="store_true", help="use L2 features for image score and raw features for pixel map")
    e.add_argument("--test-augmentations", nargs="*", choices=["hflip", "vflip"], default=None, help="flip TTA views; identity is always included")
    e.add_argument("--knn-backend", choices=["auto", "numpy", "torch"], default=None, help="auto uses Torch when the extractor is on CUDA")
    e.add_argument("--knn-dtype", choices=["float32", "float16"], default=None, help="CUDA matrix precision; float16 is faster and uses half the memory")
    e.add_argument("--feature-layers", default=None, help="comma-separated hidden-state indices")
    e.add_argument("--feature-layer-preset", choices=FEATURE_LAYER_PRESETS, default=None, help="named layer selection; middle7 matches the SubspaceAD indices")
    e.add_argument("--layer-aggregation", choices=["mean", "concat"], default=None)
    e.add_argument("--layer-normalization", choices=["none", "l2"], default=None, help="normalize each hidden layer before fusion")
    e.add_argument("--map-postprocess", choices=["none", "gaussian", "crf"], default=None); e.add_argument("--gaussian-sigma", type=float, default=None)
    a = p.parse_args(argv); cfg = _config(a.config)
    model_name = getattr(a, "model", None) or cfg.get("model", "facebook/dinov3-vit7b16-pretrain-lvd1689m")
    image_size = getattr(a, "image_size", None) or cfg.get("image_size", 448)
    if image_size <= 0: p.error("--image-size must be positive")
    resize_mode = getattr(a, "resize_mode", None) or cfg.get("resize_mode", "direct")
    top_k_ratio = getattr(a, "top_k_ratio", None) or cfg.get("top_k_ratio", 0.05)
    image_score = getattr(a, "image_score", None) or cfg.get("image_score", "mtop1p")
    image_top_ratio = getattr(a, "image_top_ratio", None); image_top_ratio = image_top_ratio if image_top_ratio is not None else cfg.get("image_top_ratio", 0.01)
    image_fusion_stage = getattr(a, "image_fusion_stage", None) or cfg.get("image_fusion_stage", "patch")
    image_spatial_weight = getattr(a, "image_spatial_weight", None); image_spatial_weight = image_spatial_weight if image_spatial_weight is not None else cfg.get("image_spatial_weight", 0.0)
    pixel_multiscale_weight = getattr(a, "pixel_multiscale_weight", None); pixel_multiscale_weight = pixel_multiscale_weight if pixel_multiscale_weight is not None else cfg.get("pixel_multiscale_weight", 0.5)
    image_min_component_size = getattr(a, "image_min_component_size", None); image_min_component_size = image_min_component_size if image_min_component_size is not None else cfg.get("image_min_component_size", 1)
    type_matching = getattr(a, "type_matching", None) or cfg.get("type_matching", "bidirectional_patch")
    anomaly_method = getattr(a, "anomaly_method", None) or cfg.get("anomaly_method", "pca")
    pca_residual_metric = getattr(a, "pca_residual_metric", None) or cfg.get("pca_residual_metric", "squared_l2")
    knn_weight = getattr(a, "knn_weight", None); knn_weight = knn_weight if knn_weight is not None else cfg.get("knn_weight", 0.5)
    anoco_neighbors = getattr(a, "anoco_neighbors", None); anoco_neighbors = anoco_neighbors if anoco_neighbors is not None else cfg.get("anoco_neighbors", 16)
    anoco_query_weight = getattr(a, "anoco_query_weight", None); anoco_query_weight = anoco_query_weight if anoco_query_weight is not None else cfg.get("anoco_query_weight", 1.0)
    anoco_temperature = getattr(a, "anoco_temperature", None); anoco_temperature = anoco_temperature if anoco_temperature is not None else cfg.get("anoco_temperature", 0.07)
    anoco_affinity = getattr(a, "anoco_affinity", None); anoco_affinity = anoco_affinity if anoco_affinity is not None else cfg.get("anoco_affinity", "softmax")
    anoco_anchor_ranking = getattr(a, "anoco_anchor_ranking", None); anoco_anchor_ranking = anoco_anchor_ranking if anoco_anchor_ranking is not None else cfg.get("anoco_anchor_ranking", "mean")
    anoco_norm_compatibility = bool(getattr(a, "anoco_norm_compatibility", False) or cfg.get("anoco_norm_compatibility", False))
    anoco_weight = getattr(a, "anoco_weight", None); anoco_weight = anoco_weight if anoco_weight is not None else cfg.get("anoco_weight", 0.5)
    pixel_anoco_weight = getattr(a, "pixel_anoco_weight", None); pixel_anoco_weight = pixel_anoco_weight if pixel_anoco_weight is not None else cfg.get("pixel_anoco_weight", 0.0)
    anoco_layer_consensus = bool(getattr(a, "anoco_layer_consensus", False) or cfg.get("anoco_layer_consensus", False))
    fusion_mode = getattr(a, "fusion_mode", None) or cfg.get("fusion_mode", "fixed")
    gate_temperature = getattr(a, "gate_temperature", None); gate_temperature = gate_temperature if gate_temperature is not None else cfg.get("gate_temperature", 1.0)
    memory_max_patches = getattr(a, "memory_max_patches", None); memory_max_patches = memory_max_patches if memory_max_patches is not None else cfg.get("memory_max_patches", 50000)
    normal_fit_max_patches = getattr(a, "normal_fit_max_patches", None); normal_fit_max_patches = normal_fit_max_patches if normal_fit_max_patches is not None else cfg.get("normal_fit_max_patches", 0)
    knn_chunk_size = getattr(a, "knn_chunk_size", None); knn_chunk_size = knn_chunk_size if knn_chunk_size is not None else cfg.get("knn_chunk_size", 256)
    knn_spatial_radius = getattr(a, "knn_spatial_radius", None); knn_spatial_radius = knn_spatial_radius if knn_spatial_radius is not None else cfg.get("knn_spatial_radius", -1.0)
    align_training_positions = bool(getattr(a, "align_training_positions", False) or cfg.get("align_training_positions", False))
    dual_branch = bool(getattr(a, "dual_branch", False) or cfg.get("dual_branch", False))
    if anomaly_method == "pca_knn_anoco" and not dual_branch: p.error("--anomaly-method pca_knn_anoco requires --dual-branch")
    if anoco_layer_consensus and (not dual_branch or anomaly_method not in {"pca_anoco", "pca_knn_anoco"} or image_fusion_stage != "patch"): p.error("--anoco-layer-consensus requires --dual-branch, a PCA+ANoCo method, and --image-fusion-stage patch")
    test_augmentations = getattr(a, "test_augmentations", None); test_augmentations = test_augmentations if test_augmentations is not None else cfg.get("test_augmentations", [])
    knn_backend = getattr(a, "knn_backend", None) or cfg.get("knn_backend", "auto")
    knn_dtype = getattr(a, "knn_dtype", None) or cfg.get("knn_dtype", "float32")
    if not 0 <= knn_weight <= 1: p.error("--knn-weight must be in [0, 1]")
    if anoco_neighbors <= 0: p.error("--anoco-neighbors must be positive")
    if anoco_query_weight <= 0: p.error("--anoco-query-weight must be positive")
    if anoco_temperature <= 0: p.error("--anoco-temperature must be positive")
    if not 0 <= anoco_weight <= 1: p.error("--anoco-weight must be in [0, 1]")
    if not 0 <= pixel_anoco_weight <= 1: p.error("--pixel-anoco-weight must be in [0, 1]")
    if pixel_anoco_weight > 0 and anomaly_method not in {"pca_knn", "pca_knn_anoco"}: p.error("--pixel-anoco-weight requires pca_knn or pca_knn_anoco")
    if pixel_anoco_weight > 0 and fusion_mode != "fixed": p.error("--pixel-anoco-weight currently requires --fusion-mode fixed")
    if not 0 < image_top_ratio <= 1: p.error("--image-top-ratio must be in (0, 1]")
    if image_spatial_weight < 0: p.error("--image-spatial-weight must be non-negative")
    if not 0 <= pixel_multiscale_weight <= 1: p.error("--pixel-multiscale-weight must be in [0, 1]")
    if image_min_component_size <= 0: p.error("--image-min-component-size must be positive")
    if gate_temperature <= 0: p.error("--gate-temperature must be positive")
    if memory_max_patches < 0: p.error("--memory-max-patches must be non-negative")
    if normal_fit_max_patches < 0: p.error("--normal-fit-max-patches must be non-negative")
    if knn_chunk_size <= 0: p.error("--knn-chunk-size must be positive")
    if knn_spatial_radius != -1 and not 0 <= knn_spatial_radius <= 1: p.error("--knn-spatial-radius must be -1 or in [0, 1]")
    if align_training_positions and knn_spatial_radius < 0: p.error("--align-training-positions requires --knn-spatial-radius in [0, 1]")
    if align_training_positions and resize_mode != "direct": p.error("--align-training-positions currently requires --resize-mode direct")
    try:
        feature_layers, feature_layer_preset = _feature_layers(a, cfg)
    except ValueError as exc:
        p.error(str(exc))
    layer_aggregation = getattr(a, "layer_aggregation", None) or cfg.get("layer_aggregation", "mean")
    layer_normalization = getattr(a, "layer_normalization", None) or cfg.get("layer_normalization", "none")
    map_postprocess = getattr(a, "map_postprocess", None) or cfg.get("map_postprocess", "none")
    gaussian_sigma = getattr(a, "gaussian_sigma", None)
    gaussian_sigma = gaussian_sigma if gaussian_sigma is not None else cfg.get("gaussian_sigma", 1.0)
    extractor = DinoFeatureExtractor(
        model_name, image_size=image_size, resize_mode=resize_mode, device=getattr(a, "device", None) or cfg.get("device"),
        debias=getattr(a, "debias", False), svd_components=getattr(a, "svd_components", 20),
        feature_layers=feature_layers, layer_aggregation=layer_aggregation, layer_normalization=layer_normalization,
    )
    if a.cmd == "fit":
        normal_dir = a.normal_dir or cfg.get("normal_dir")
        if not normal_dir: p.error("fit requires --normal-dir or config normal_dir")
        output = a.output or cfg.get("output", "outputs/model.json")
        alpha = a.alpha if a.alpha is not None else cfg.get("alpha", 0.5)
        threshold = a.unknown_threshold if a.unknown_threshold is not None else cfg.get("unknown_threshold", 0.35)
        paths = _images(normal_dir, not a.non_recursive)
        fusion = DefectFusion(extractor, alpha=alpha, unknown_threshold=threshold, top_k_ratio=top_k_ratio, image_score=image_score, image_top_ratio=image_top_ratio, image_fusion_stage=image_fusion_stage, image_spatial_weight=image_spatial_weight, image_min_component_size=image_min_component_size, type_matching=type_matching, map_postprocess=map_postprocess, gaussian_sigma=gaussian_sigma, anomaly_method=anomaly_method, pca_residual_metric=pca_residual_metric, knn_weight=knn_weight, anoco_neighbors=anoco_neighbors, anoco_query_weight=anoco_query_weight, anoco_temperature=anoco_temperature, anoco_affinity=anoco_affinity, anoco_anchor_ranking=anoco_anchor_ranking, anoco_norm_compatibility=anoco_norm_compatibility, anoco_weight=anoco_weight, pixel_anoco_weight=pixel_anoco_weight, anoco_layer_consensus=anoco_layer_consensus, memory_max_patches=memory_max_patches, knn_chunk_size=knn_chunk_size, knn_backend=knn_backend, knn_dtype=knn_dtype, knn_spatial_radius=knn_spatial_radius, align_training_positions=align_training_positions, dual_branch=dual_branch, fusion_mode=fusion_mode, gate_temperature=gate_temperature, test_augmentations=test_augmentations).fit_normal(paths)
        proto_dir = a.prototype_dir or cfg.get("prototype_dir")
        if proto_dir:
            for label_dir in sorted(Path(proto_dir).iterdir()):
                if label_dir.is_dir():
                    for image in _images(str(label_dir)):
                        fusion.add_prototype(label_dir.name, image)
        fusion.save(output); print(output)
    elif a.cmd == "predict":
        result = DefectFusion.load(a.model_state, extractor).predict(a.image)
        payload = json.dumps(result, ensure_ascii=False, indent=2)
        if a.output: Path(a.output).write_text(payload + "\n", encoding="utf-8")
        print(payload)
    else:
        is_visa = a.cmd == "evaluate-visa"
        if is_visa and not a.data_root: p.error("evaluate-visa requires --data-root")
        if not is_visa and not a.data_dir and not a.data_root: p.error("evaluate-mvtec requires --data-dir or --data-root")
        if a.normal_shots == 0 or a.normal_shots < -1: p.error("--normal-shots must be -1 or a positive integer")
        if a.defect_shots < 0: p.error("--defect-shots must be non-negative")
        if a.normal_augment_count is not None and a.normal_augment_count < 0: p.error("--normal-augment-count must be non-negative")
        normal_augmentations = a.normal_augmentations or cfg.get("normal_augmentations", ["rotate"])
        affine_categories = set(a.affine_categories or cfg.get("affine_categories", []))
        component_reject_categories = set(a.component_reject_categories or cfg.get("component_reject_categories", []))
        knn_spatial_categories = set(a.knn_spatial_categories or cfg.get("knn_spatial_categories", []))
        try:
            image_size_overrides = parse_image_size_overrides(a.image_size_override if a.image_size_override is not None else cfg.get("image_size_overrides", {}))
            pixel_image_size_overrides = parse_image_size_overrides(a.pixel_image_size_override if a.pixel_image_size_override is not None else cfg.get("pixel_image_size_overrides", {}))
            image_head_size_overrides = parse_image_size_overrides(a.image_head_size_override if a.image_head_size_override is not None else cfg.get("image_head_size_overrides", {}))
            pixel_multiscale_size_overrides = parse_image_size_overrides(a.pixel_multiscale_size_override if a.pixel_multiscale_size_override is not None else cfg.get("pixel_multiscale_size_overrides", {}))
        except ValueError as exc:
            p.error(str(exc))
        if (pixel_image_size_overrides or image_head_size_overrides) and not dual_branch:
            p.error("branch-specific image-size overrides require --dual-branch")
        if align_training_positions and not knn_spatial_categories and "affine" in normal_augmentations:
            p.error("--align-training-positions does not yet support affine normal augmentation")
        if align_training_positions and not knn_spatial_categories and affine_categories:
            p.error("--align-training-positions does not yet support --affine-categories")
        no_augment_categories = a.no_augment_categories or cfg.get("no_augment_categories", ["transistor"])
        if is_visa:
            categories = load_visa_categories(a.data_root, a.split_csv)
        else:
            categories = [Path(a.data_dir)] if a.data_dir else sorted(x for x in Path(a.data_root).iterdir() if (x / "train" / "good").is_dir())
        available_categories = {category.name for category in categories}
        override_categories = set(image_size_overrides) | set(pixel_image_size_overrides) | set(image_head_size_overrides) | set(pixel_multiscale_size_overrides)
        unknown_overrides = sorted((affine_categories | component_reject_categories | knn_spatial_categories | override_categories) - available_categories)
        if unknown_overrides: p.error(f"unknown override categories: {', '.join(unknown_overrides)}")
        if knn_spatial_categories and knn_spatial_radius < 0:
            p.error("--knn-spatial-categories requires --knn-spatial-radius in [0, 1]")
        unsupported_spatial_affine = sorted(knn_spatial_categories & affine_categories)
        if unsupported_spatial_affine:
            p.error(f"spatial kNN position alignment does not support affine categories: {', '.join(unsupported_spatial_affine)}")
        if a.categories:
            requested_categories = set(a.categories)
            categories = [category for category in categories if category.name in requested_categories]
            found_categories = {category.name for category in categories}
            missing_categories = sorted(requested_categories - found_categories)
            if missing_categories: p.error(f"unknown categories: {', '.join(missing_categories)}")
        output_dir = experiment_output_dir(a.output)
        category_output_dir = output_dir / "categories"
        category_output_dir.mkdir(parents=True, exist_ok=True)
        all_metrics = []
        for category in categories:
            category_name = category.name
            category_image_size = image_size_overrides.get(category_name, image_size)
            category_pixel_image_size = pixel_image_size_overrides.get(category_name, category_image_size)
            category_image_head_size = image_head_size_overrides.get(category_name, category_image_size)
            category_secondary_pixel_size = pixel_multiscale_size_overrides.get(category_name)
            if category_secondary_pixel_size == category_pixel_image_size:
                p.error(f"secondary pixel size must differ from primary pixel size for {category_name}")
            if extractor.image_size != category_pixel_image_size:
                extractor.image_size = category_pixel_image_size
                extractor.positional_basis = None
            normal_candidates = [str(path) for path in category.normal_images] if is_visa else _images(str(category / "train" / "good"))
            if a.normal_shots == -1:
                normal_selected = normal_candidates
            else:
                normal_rng = random.Random(a.seed)
                normal_selected = sorted(normal_rng.sample(normal_candidates, min(a.normal_shots, len(normal_candidates))))
            print(f"[normal-shots] {category_name}: {len(normal_selected)}/{len(normal_candidates)}", flush=True)
            augment_count = a.normal_augment_count if a.normal_augment_count is not None else (30 if a.normal_shots != -1 else 0)
            if category_name in no_augment_categories: augment_count = 0
            category_augmentations = list(normal_augmentations)
            if category_name in affine_categories and "affine" not in category_augmentations:
                category_augmentations.append("affine")
            category_component_size = image_min_component_size if not component_reject_categories or category_name in component_reject_categories else 1
            category_spatial_enabled = not knn_spatial_categories or category_name in knn_spatial_categories
            category_knn_spatial_radius = knn_spatial_radius if category_spatial_enabled else -1.0
            category_align_training_positions = align_training_positions or (category_name in knn_spatial_categories)
            normal_training_images = _augment_normal_images(normal_selected, augment_count, category_augmentations, a.seed)
            print(f"[normal-augment] {category_name}: {len(normal_training_images)} views", flush=True)
            print(f"[category-config] {category_name}: pixel_size={category_pixel_image_size} secondary_pixel_size={category_secondary_pixel_size or 'none'} image_head_size={category_image_head_size} augmentations={category_augmentations} component_size={category_component_size} fit_max_patches={normal_fit_max_patches or 'all'} spatial_radius={category_knn_spatial_radius}", flush=True)
            fusion = DefectFusion(extractor, top_k_ratio=top_k_ratio, image_score=image_score, image_top_ratio=image_top_ratio, image_fusion_stage=image_fusion_stage, image_spatial_weight=image_spatial_weight, image_min_component_size=category_component_size, type_matching=type_matching, map_postprocess=map_postprocess, gaussian_sigma=gaussian_sigma, anomaly_method=anomaly_method, pca_residual_metric=pca_residual_metric, knn_weight=knn_weight, anoco_neighbors=anoco_neighbors, anoco_query_weight=anoco_query_weight, anoco_temperature=anoco_temperature, anoco_affinity=anoco_affinity, anoco_anchor_ranking=anoco_anchor_ranking, anoco_norm_compatibility=anoco_norm_compatibility, anoco_weight=anoco_weight, pixel_anoco_weight=pixel_anoco_weight, anoco_layer_consensus=anoco_layer_consensus, memory_max_patches=memory_max_patches, normal_fit_max_patches=normal_fit_max_patches, knn_chunk_size=knn_chunk_size, knn_backend=knn_backend, knn_dtype=knn_dtype, knn_spatial_radius=category_knn_spatial_radius, image_knn_spatial_radius=-1.0 if knn_spatial_categories else None, align_training_positions=category_align_training_positions, dual_branch=dual_branch, fusion_mode=fusion_mode, gate_temperature=gate_temperature, test_augmentations=test_augmentations, pixel_image_size=category_pixel_image_size, image_head_image_size=category_image_head_size, secondary_pixel_image_size=category_secondary_pixel_size, pixel_multiscale_weight=pixel_multiscale_weight).fit_normal(normal_training_images)
            if anomaly_method != "pca":
                print(
                    f"[knn] {category_name}: backend={fusion.normal_memory.resolved_backend} "
                    f"device={fusion.normal_memory.device or 'cpu'} dtype={knn_dtype} "
                    f"patches={len(fusion.normal_memory.features)} chunk={knn_chunk_size}",
                    flush=True,
                )
            if a.prototype_dir:
                for label_dir in sorted(Path(a.prototype_dir).iterdir()):
                    if label_dir.is_dir():
                        for image in _images(str(label_dir)):
                            fusion.add_prototype(label_dir.name, image)
            selected = []
            if a.defect_shots > 0:
                rng = random.Random(a.seed)
                if is_visa:
                    defect_groups = {}
                    for sample in category.test_samples:
                        if sample.anomalous:
                            defect_groups.setdefault(sample.defect_type, []).append(str(sample.image))
                else:
                    defect_groups = {x.name: _images(str(x)) for x in sorted((category / "test").iterdir()) if x.is_dir() and x.name != "good"}
                for defect_name, candidates in sorted(defect_groups.items()):
                    chosen = rng.sample(candidates, min(a.defect_shots, len(candidates)))
                    for image in chosen:
                        fusion.add_prototype(defect_name, image)
                        selected.append(image)
                        print(f"[defect-shot] {category_name}/{defect_name}: {Path(image).name}", flush=True)
            result_path = category_output_dir / f"{category_name}.json"
            if is_visa:
                samples = [(x.image, x.defect_type, x.anomalous, x.mask) for x in category.test_samples]
                metrics = evaluate_samples(fusion, category_name, samples, result_path, excluded_type_images=selected)
            else:
                metrics = evaluate_mvtec(fusion, category, result_path, excluded_type_images=selected)
            metrics["dataset"] = "visa" if is_visa else "mvtec"
            metrics["normal_shots"] = a.normal_shots
            metrics["normal_shot_images"] = [str(Path(x)) for x in normal_selected]
            metrics["normal_augment_count"] = augment_count
            metrics["normal_augmentations"] = category_augmentations
            metrics["normal_training_views"] = len(normal_training_images)
            metrics["defect_shots"] = a.defect_shots
            metrics["seed"] = a.seed
            metrics["defect_shot_images"] = [str(Path(x)) for x in selected]
            metrics["defect_type_excluded_images"] = [str(Path(x)) for x in selected]
            metrics["debias"] = a.debias
            metrics["svd_components"] = a.svd_components if a.debias else 0
            metrics["top_k_ratio"] = top_k_ratio
            metrics["image_score"] = image_score
            metrics["image_top_ratio"] = image_top_ratio if image_score == "mtop1p" else 0
            metrics["image_fusion_stage"] = image_fusion_stage
            metrics["image_spatial_weight"] = image_spatial_weight
            metrics["image_min_component_size"] = category_component_size
            metrics["affine_category_override"] = category_name in affine_categories
            metrics["component_reject_category_override"] = category_name in component_reject_categories
            metrics["feature_layers"] = list(feature_layers)
            metrics["feature_layer_preset"] = feature_layer_preset
            metrics["image_size"] = category_image_size
            metrics["image_size_override"] = category_name in image_size_overrides
            metrics["pixel_image_size"] = category_pixel_image_size
            metrics["image_head_image_size"] = category_image_head_size
            metrics["pixel_image_size_override"] = category_name in pixel_image_size_overrides
            metrics["image_head_size_override"] = category_name in image_head_size_overrides
            metrics["secondary_pixel_image_size"] = category_secondary_pixel_size
            metrics["pixel_multiscale_size_override"] = category_name in pixel_multiscale_size_overrides
            metrics["pixel_multiscale_weight"] = pixel_multiscale_weight if category_secondary_pixel_size is not None else 0
            metrics["resize_mode"] = resize_mode
            metrics["layer_aggregation"] = layer_aggregation
            metrics["layer_normalization"] = layer_normalization
            metrics["type_matching"] = type_matching
            metrics["anomaly_method"] = anomaly_method
            metrics["pca_residual_metric"] = pca_residual_metric
            metrics["knn_weight"] = knn_weight if anomaly_method in {"pca_knn", "pca_knn_anoco"} else 0
            metrics["anoco_neighbors"] = anoco_neighbors if anomaly_method in {"anoco", "pca_anoco", "pca_knn_anoco"} else 0
            metrics["anoco_query_weight"] = anoco_query_weight if anomaly_method in {"anoco", "pca_anoco", "pca_knn_anoco"} else 0
            metrics["anoco_temperature"] = anoco_temperature if anomaly_method in {"anoco", "pca_anoco", "pca_knn_anoco"} else 0
            metrics["anoco_affinity"] = anoco_affinity if anomaly_method in {"anoco", "pca_anoco", "pca_knn_anoco"} else "none"
            metrics["anoco_anchor_ranking"] = anoco_anchor_ranking if anomaly_method in {"anoco", "pca_anoco", "pca_knn_anoco"} else "none"
            metrics["anoco_norm_compatibility"] = anoco_norm_compatibility if anomaly_method in {"anoco", "pca_anoco", "pca_knn_anoco"} else False
            memory_stats = fusion.memory_stats()
            metrics["memory_patch_count"] = memory_stats["patch_count"] if anomaly_method != "pca" else 0
            metrics["memory_bytes"] = memory_stats["bytes"] if anomaly_method != "pca" else 0
            metrics["anoco_weight"] = anoco_weight if anomaly_method in {"pca_anoco", "pca_knn_anoco"} else 0
            metrics["pixel_anoco_weight"] = pixel_anoco_weight if anomaly_method in {"pca_knn", "pca_knn_anoco"} else 0
            metrics["anoco_layer_consensus"] = anoco_layer_consensus if anomaly_method in {"pca_anoco", "pca_knn_anoco"} else False
            metrics["fusion_mode"] = fusion_mode if anomaly_method in {"pca_knn", "pca_anoco", "pca_knn_anoco"} else "none"
            metrics["gate_temperature"] = gate_temperature if anomaly_method in {"pca_knn", "pca_anoco", "pca_knn_anoco"} and fusion_mode == "gated" else 0
            metrics["memory_max_patches"] = memory_max_patches if anomaly_method != "pca" else 0
            metrics["normal_fit_max_patches"] = normal_fit_max_patches
            metrics["knn_chunk_size"] = knn_chunk_size if anomaly_method != "pca" else 0
            metrics["knn_backend"] = fusion.normal_memory.resolved_backend if anomaly_method != "pca" else "none"
            metrics["knn_dtype"] = knn_dtype if anomaly_method != "pca" else "none"
            metrics["knn_spatial_radius"] = category_knn_spatial_radius if anomaly_method != "pca" else -1
            metrics["knn_spatial_category_override"] = category_name in knn_spatial_categories
            metrics["align_training_positions"] = align_training_positions
            metrics["dual_branch"] = dual_branch
            metrics["test_augmentations"] = list(test_augmentations)
            metrics["map_postprocess"] = map_postprocess
            metrics["gaussian_sigma"] = gaussian_sigma if map_postprocess == "gaussian" else 0
            metrics["metrics_file"] = str(result_path)
            predictions = json.loads(result_path.read_text(encoding="utf-8"))
            category_payload = {"metrics": metrics, "predictions": predictions}
            result_path.write_text(json.dumps(category_payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
            all_metrics.append(metrics)
        metric_names = (
            "image_auroc", "image_aupr", "image_f1_max", "pixel_auroc", "pixel_aupr",
            "pixel_aupro", "pixel_f1_max",
            "defect_type_accuracy", "defect_type_macro_precision", "defect_type_macro_recall",
            "defect_type_macro_f1", "defect_type_weighted_f1",
        )
        macro = {}
        for name in metric_names:
            values = [float(item[name]) for item in all_metrics if name in item]
            if values:
                macro[name] = sum(values) / len(values)
        summary_path = output_dir / "results.json"
        csv_path = output_dir / "summary.csv"
        summary = {
            "macro_average": macro,
            "categories": all_metrics,
            "summary_file": str(summary_path),
            "summary_csv": str(csv_path),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_metrics_csv(csv_path, all_metrics, macro)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
