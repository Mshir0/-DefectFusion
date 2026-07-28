from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path
from PIL import Image

from .features import DinoFeatureExtractor
from .pipeline import DefectFusion
from .mvtec import evaluate_mvtec


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
    "cross4": (-1, -3, -5, -7),
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
        result.append(image)
        for _ in range(count):
            augmented = image.copy()
            for name in augmentations:
                if name == "rotate": augmented = TF.rotate(augmented, rng.uniform(0, 345))
                elif name == "hflip" and rng.random() < 0.5: augmented = TF.hflip(augmented)
                elif name == "vflip" and rng.random() < 0.5: augmented = TF.vflip(augmented)
                elif name == "color_jitter":
                    augmented = TF.adjust_brightness(augmented, rng.uniform(0.8, 1.2))
                    augmented = TF.adjust_contrast(augmented, rng.uniform(0.8, 1.2))
                    augmented = TF.adjust_saturation(augmented, rng.uniform(0.8, 1.2))
                    augmented = TF.adjust_hue(augmented, rng.uniform(-0.1, 0.1))
                elif name == "affine":
                    translate = [round(rng.uniform(-0.15, 0.15) * image.width), round(rng.uniform(-0.15, 0.15) * image.height)]
                    augmented = TF.affine(augmented, angle=0, translate=translate, scale=1.0, shear=rng.uniform(-10, 10))
                elif name not in {"hflip", "vflip"}:
                    raise ValueError(f"Unknown normal augmentation: {name}")
            result.append(augmented)
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
    f.add_argument("--type-matching", choices=["prototype_mean", "bidirectional_patch", "rbf_svm"], default=None)
    f.add_argument("--anomaly-method", choices=["pca", "knn", "pca_knn"], default=None); f.add_argument("--knn-weight", type=float, default=None)
    f.add_argument("--fusion-mode", choices=["fixed", "gated"], default=None); f.add_argument("--gate-temperature", type=float, default=None)
    f.add_argument("--memory-max-patches", type=int, default=None); f.add_argument("--knn-chunk-size", type=int, default=None)
    f.add_argument("--knn-spatial-radius", type=float, default=None)
    f.add_argument("--dual-branch", action="store_true", help="use L2 features for image score and raw features for pixel map")
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
    e = sub.add_parser("evaluate-mvtec", help="fit on train/good and evaluate one MVTec category")
    e.add_argument("--data-dir", help="MVTec category directory")
    e.add_argument("--data-root", help="MVTec root containing all 15 category directories")
    e.add_argument("--prototype-dir", help="optional defect prototypes; subdirectories are defect labels")
    e.add_argument("--normal-shots", type=int, default=-1, help="normal train/good references per category; -1 uses all")
    e.add_argument("--defect-shots", "--few-shot", dest="defect_shots", type=int, default=0, help="labeled defect exemplars per defect type")
    e.add_argument("--seed", type=int, default=42, help="seed used for reproducible normal and defect sampling")
    e.add_argument("--normal-augment-count", type=int, default=None, help="augmented views per normal shot; defaults to 30 in few-shot mode")
    e.add_argument("--normal-augmentations", nargs="+", choices=["rotate", "hflip", "vflip", "color_jitter", "affine"], default=None)
    e.add_argument("--no-augment-categories", nargs="+", default=None)
    e.add_argument("--model", default=None); e.add_argument("--device", default=None)
    e.add_argument("--image-size", type=int, default=None)
    e.add_argument("--resize-mode", choices=["direct", "longest_pad"], default=None)
    e.add_argument("--output", default="outputs/mvtec-results.jsonl")
    e.add_argument("--debias", action="store_true", help="apply INSID3 positional debiasing")
    e.add_argument("--svd-components", type=int, default=20, help="INSID3 positional basis rank")
    e.add_argument("--top-k-ratio", type=float, default=None, help="highest PCA-residual patch ratio for typing")
    e.add_argument("--image-score", choices=["mtop1p", "mean", "max", "p99"], default=None)
    e.add_argument("--image-top-ratio", type=float, default=None, help="top patch fraction used by mtop1p")
    e.add_argument("--image-fusion-stage", choices=["patch", "score"], default=None, help="fuse PCA/kNN before or after image aggregation")
    e.add_argument("--type-matching", choices=["prototype_mean", "bidirectional_patch", "rbf_svm"], default=None)
    e.add_argument("--anomaly-method", choices=["pca", "knn", "pca_knn"], default=None, help="normal anomaly detector")
    e.add_argument("--knn-weight", type=float, default=None, help="kNN contribution in calibrated pca_knn fusion")
    e.add_argument("--fusion-mode", choices=["fixed", "gated"], default=None, help="fixed weight or normal-tail-calibrated patch gate")
    e.add_argument("--gate-temperature", type=float, default=None, help="soft gate temperature; lower values select one expert more strongly")
    e.add_argument("--memory-max-patches", type=int, default=None, help="maximum normal patches retained for kNN; 0 keeps all")
    e.add_argument("--knn-chunk-size", type=int, default=None, help="query patches per kNN matrix chunk")
    e.add_argument("--knn-spatial-radius", type=float, default=None, help="normalized local kNN radius; -1 searches globally")
    e.add_argument("--dual-branch", action="store_true", help="use L2 features for image score and raw features for pixel map")
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
    image_top_ratio = getattr(a, "image_top_ratio", None); image_top_ratio = image_top_ratio if image_top_ratio is not None else cfg.get("image_top_ratio", 0.05)
    image_fusion_stage = getattr(a, "image_fusion_stage", None) or cfg.get("image_fusion_stage", "patch")
    type_matching = getattr(a, "type_matching", None) or cfg.get("type_matching", "bidirectional_patch")
    anomaly_method = getattr(a, "anomaly_method", None) or cfg.get("anomaly_method", "pca")
    knn_weight = getattr(a, "knn_weight", None); knn_weight = knn_weight if knn_weight is not None else cfg.get("knn_weight", 0.5)
    fusion_mode = getattr(a, "fusion_mode", None) or cfg.get("fusion_mode", "fixed")
    gate_temperature = getattr(a, "gate_temperature", None); gate_temperature = gate_temperature if gate_temperature is not None else cfg.get("gate_temperature", 1.0)
    memory_max_patches = getattr(a, "memory_max_patches", None); memory_max_patches = memory_max_patches if memory_max_patches is not None else cfg.get("memory_max_patches", 50000)
    knn_chunk_size = getattr(a, "knn_chunk_size", None); knn_chunk_size = knn_chunk_size if knn_chunk_size is not None else cfg.get("knn_chunk_size", 256)
    knn_spatial_radius = getattr(a, "knn_spatial_radius", None); knn_spatial_radius = knn_spatial_radius if knn_spatial_radius is not None else cfg.get("knn_spatial_radius", -1.0)
    dual_branch = bool(getattr(a, "dual_branch", False) or cfg.get("dual_branch", False))
    knn_backend = getattr(a, "knn_backend", None) or cfg.get("knn_backend", "auto")
    knn_dtype = getattr(a, "knn_dtype", None) or cfg.get("knn_dtype", "float32")
    if not 0 <= knn_weight <= 1: p.error("--knn-weight must be in [0, 1]")
    if not 0 < image_top_ratio <= 1: p.error("--image-top-ratio must be in (0, 1]")
    if gate_temperature <= 0: p.error("--gate-temperature must be positive")
    if memory_max_patches < 0: p.error("--memory-max-patches must be non-negative")
    if knn_chunk_size <= 0: p.error("--knn-chunk-size must be positive")
    if knn_spatial_radius != -1 and not 0 <= knn_spatial_radius <= 1: p.error("--knn-spatial-radius must be -1 or in [0, 1]")
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
        fusion = DefectFusion(extractor, alpha=alpha, unknown_threshold=threshold, top_k_ratio=top_k_ratio, image_score=image_score, image_top_ratio=image_top_ratio, image_fusion_stage=image_fusion_stage, type_matching=type_matching, map_postprocess=map_postprocess, gaussian_sigma=gaussian_sigma, anomaly_method=anomaly_method, knn_weight=knn_weight, memory_max_patches=memory_max_patches, knn_chunk_size=knn_chunk_size, knn_backend=knn_backend, knn_dtype=knn_dtype, knn_spatial_radius=knn_spatial_radius, dual_branch=dual_branch, fusion_mode=fusion_mode, gate_temperature=gate_temperature).fit_normal(paths)
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
        if not a.data_dir and not a.data_root: p.error("evaluate-mvtec requires --data-dir or --data-root")
        if a.normal_shots == 0 or a.normal_shots < -1: p.error("--normal-shots must be -1 or a positive integer")
        if a.defect_shots < 0: p.error("--defect-shots must be non-negative")
        if a.normal_augment_count is not None and a.normal_augment_count < 0: p.error("--normal-augment-count must be non-negative")
        normal_augmentations = a.normal_augmentations or cfg.get("normal_augmentations", ["rotate"])
        no_augment_categories = a.no_augment_categories or cfg.get("no_augment_categories", ["transistor"])
        categories = [Path(a.data_dir)] if a.data_dir else sorted(x for x in Path(a.data_root).iterdir() if (x / "train" / "good").is_dir())
        all_metrics = []
        for category in categories:
            normal_dir = str(category / "train" / "good")
            normal_candidates = _images(normal_dir)
            if a.normal_shots == -1:
                normal_selected = normal_candidates
            else:
                normal_rng = random.Random(a.seed)
                normal_selected = sorted(normal_rng.sample(normal_candidates, min(a.normal_shots, len(normal_candidates))))
            print(f"[normal-shots] {category.name}: {len(normal_selected)}/{len(normal_candidates)}", flush=True)
            augment_count = a.normal_augment_count if a.normal_augment_count is not None else (30 if a.normal_shots != -1 else 0)
            if category.name in no_augment_categories: augment_count = 0
            normal_training_images = _augment_normal_images(normal_selected, augment_count, normal_augmentations, a.seed)
            print(f"[normal-augment] {category.name}: {len(normal_training_images)} views", flush=True)
            fusion = DefectFusion(extractor, top_k_ratio=top_k_ratio, image_score=image_score, image_top_ratio=image_top_ratio, image_fusion_stage=image_fusion_stage, type_matching=type_matching, map_postprocess=map_postprocess, gaussian_sigma=gaussian_sigma, anomaly_method=anomaly_method, knn_weight=knn_weight, memory_max_patches=memory_max_patches, knn_chunk_size=knn_chunk_size, knn_backend=knn_backend, knn_dtype=knn_dtype, knn_spatial_radius=knn_spatial_radius, dual_branch=dual_branch, fusion_mode=fusion_mode, gate_temperature=gate_temperature).fit_normal(normal_training_images)
            if anomaly_method != "pca":
                print(
                    f"[knn] {category.name}: backend={fusion.normal_memory.resolved_backend} "
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
                for defect_dir in sorted(x for x in (category / "test").iterdir() if x.is_dir() and x.name != "good"):
                    candidates = _images(str(defect_dir))
                    chosen = rng.sample(candidates, min(a.defect_shots, len(candidates)))
                    for image in chosen:
                        fusion.add_prototype(defect_dir.name, image)
                        selected.append(image)
                        print(f"[defect-shot] {category.name}/{defect_dir.name}: {Path(image).name}", flush=True)
            result_path = a.output if len(categories) == 1 else str(Path(a.output).with_name(f"{Path(a.output).stem}-{category.name}.jsonl"))
            metrics = evaluate_mvtec(fusion, category, result_path, excluded_images=selected)
            metrics["normal_shots"] = a.normal_shots
            metrics["normal_shot_images"] = [str(Path(x)) for x in normal_selected]
            metrics["normal_augment_count"] = augment_count
            metrics["normal_augmentations"] = list(normal_augmentations)
            metrics["normal_training_views"] = len(normal_training_images)
            metrics["defect_shots"] = a.defect_shots
            metrics["seed"] = a.seed
            metrics["defect_shot_images"] = [str(Path(x)) for x in selected]
            metrics["debias"] = a.debias
            metrics["svd_components"] = a.svd_components if a.debias else 0
            metrics["top_k_ratio"] = top_k_ratio
            metrics["image_score"] = image_score
            metrics["image_top_ratio"] = image_top_ratio if image_score == "mtop1p" else 0
            metrics["image_fusion_stage"] = image_fusion_stage
            metrics["feature_layers"] = list(feature_layers)
            metrics["feature_layer_preset"] = feature_layer_preset
            metrics["image_size"] = image_size
            metrics["resize_mode"] = resize_mode
            metrics["layer_aggregation"] = layer_aggregation
            metrics["layer_normalization"] = layer_normalization
            metrics["type_matching"] = type_matching
            metrics["anomaly_method"] = anomaly_method
            metrics["knn_weight"] = knn_weight if anomaly_method == "pca_knn" else 0
            metrics["fusion_mode"] = fusion_mode if anomaly_method == "pca_knn" else "none"
            metrics["gate_temperature"] = gate_temperature if anomaly_method == "pca_knn" and fusion_mode == "gated" else 0
            metrics["memory_max_patches"] = memory_max_patches if anomaly_method != "pca" else 0
            metrics["knn_chunk_size"] = knn_chunk_size if anomaly_method != "pca" else 0
            metrics["knn_backend"] = fusion.normal_memory.resolved_backend if anomaly_method != "pca" else "none"
            metrics["knn_dtype"] = knn_dtype if anomaly_method != "pca" else "none"
            metrics["knn_spatial_radius"] = knn_spatial_radius if anomaly_method != "pca" else -1
            metrics["dual_branch"] = dual_branch
            metrics["map_postprocess"] = map_postprocess
            metrics["gaussian_sigma"] = gaussian_sigma if map_postprocess == "gaussian" else 0
            all_metrics.append(metrics)
        if len(all_metrics) == 1:
            summary = all_metrics[0]
        else:
            metric_names = (
                "image_auroc", "image_aupr", "pixel_auroc", "pixel_aupr", "pixel_aupro",
                "defect_type_accuracy", "defect_type_macro_f1",
            )
            macro = {}
            for name in metric_names:
                values = [float(item[name]) for item in all_metrics if name in item]
                if values:
                    macro[name] = sum(values) / len(values)
            summary = {"macro_average": macro, "categories": all_metrics}
            summary_path = Path(a.output)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary["summary_file"] = str(summary_path)
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
