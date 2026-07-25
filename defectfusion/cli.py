from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

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


def main(argv=None):
    p = argparse.ArgumentParser(description="Few-shot / zero-shot defect detection")
    p.add_argument("--config", help="JSON config; CLI flags override it")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit normal subspace and optional defect prototypes")
    f.add_argument("--normal-dir"); f.add_argument("--prototype-dir", help="subdirectories are defect labels")
    f.add_argument("--model", default=None); f.add_argument("--output", default=None)
    f.add_argument("--alpha", type=float, default=None); f.add_argument("--unknown-threshold", type=float, default=None)
    f.add_argument("--device", default=None); f.add_argument("--non-recursive", action="store_true")
    f.add_argument("--debias", action="store_true"); f.add_argument("--svd-components", type=int, default=20)
    f.add_argument("--top-k-ratio", type=float, default=None, help="highest PCA-residual patch ratio for typing")
    f.add_argument("--image-score", choices=["mtop1p", "mean", "max", "p99"], default=None)
    f.add_argument("--feature-layers", default=None); f.add_argument("--layer-aggregation", choices=["mean", "concat"], default=None)
    q = sub.add_parser("predict", help="score one image or a directory")
    q.add_argument("--model-state", required=True); q.add_argument("--image", required=True)
    q.add_argument("--model", default=None); q.add_argument("--device", default=None)
    q.add_argument("--output", help="write JSON results to a file")
    q.add_argument("--debias", action="store_true"); q.add_argument("--svd-components", type=int, default=20)
    q.add_argument("--feature-layers", default=None); q.add_argument("--layer-aggregation", choices=["mean", "concat"], default=None)
    e = sub.add_parser("evaluate-mvtec", help="fit on train/good and evaluate one MVTec category")
    e.add_argument("--data-dir", help="MVTec category directory")
    e.add_argument("--data-root", help="MVTec root containing all 15 category directories")
    e.add_argument("--prototype-dir", help="optional few-shot prototypes; subdirectories are defect labels")
    e.add_argument("--few-shot", type=int, default=0, help="random exemplars per MVTec defect type")
    e.add_argument("--seed", type=int, default=42, help="seed used for reproducible few-shot selection")
    e.add_argument("--model", default=None); e.add_argument("--device", default=None)
    e.add_argument("--output", default="outputs/mvtec-results.jsonl")
    e.add_argument("--debias", action="store_true", help="apply INSID3 positional debiasing")
    e.add_argument("--svd-components", type=int, default=20, help="INSID3 positional basis rank")
    e.add_argument("--top-k-ratio", type=float, default=None, help="highest PCA-residual patch ratio for typing")
    e.add_argument("--image-score", choices=["mtop1p", "mean", "max", "p99"], default=None)
    e.add_argument("--feature-layers", default=None, help="comma-separated hidden-state indices")
    e.add_argument("--layer-aggregation", choices=["mean", "concat"], default=None)
    a = p.parse_args(argv); cfg = _config(a.config)
    model_name = getattr(a, "model", None) or cfg.get("model", "facebook/dinov3-vit7b16-pretrain-lvd1689m")
    top_k_ratio = getattr(a, "top_k_ratio", None) or cfg.get("top_k_ratio", 0.05)
    image_score = getattr(a, "image_score", None) or cfg.get("image_score", "mtop1p")
    feature_layers = _layers(getattr(a, "feature_layers", None) or cfg.get("feature_layers", [-1, -2, -3, -4]))
    layer_aggregation = getattr(a, "layer_aggregation", None) or cfg.get("layer_aggregation", "mean")
    extractor = DinoFeatureExtractor(
        model_name, device=getattr(a, "device", None) or cfg.get("device"),
        debias=getattr(a, "debias", False), svd_components=getattr(a, "svd_components", 20),
        feature_layers=feature_layers, layer_aggregation=layer_aggregation,
    )
    if a.cmd == "fit":
        normal_dir = a.normal_dir or cfg.get("normal_dir")
        if not normal_dir: p.error("fit requires --normal-dir or config normal_dir")
        output = a.output or cfg.get("output", "outputs/model.json")
        alpha = a.alpha if a.alpha is not None else cfg.get("alpha", 0.5)
        threshold = a.unknown_threshold if a.unknown_threshold is not None else cfg.get("unknown_threshold", 0.35)
        paths = _images(normal_dir, not a.non_recursive)
        fusion = DefectFusion(extractor, alpha=alpha, unknown_threshold=threshold, top_k_ratio=top_k_ratio, image_score=image_score).fit_normal(paths)
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
        categories = [Path(a.data_dir)] if a.data_dir else sorted(x for x in Path(a.data_root).iterdir() if (x / "train" / "good").is_dir())
        all_metrics = []
        for category in categories:
            normal_dir = str(category / "train" / "good")
            fusion = DefectFusion(extractor, top_k_ratio=top_k_ratio, image_score=image_score).fit_normal(_images(normal_dir))
            if a.prototype_dir:
                for label_dir in sorted(Path(a.prototype_dir).iterdir()):
                    if label_dir.is_dir():
                        for image in _images(str(label_dir)):
                            fusion.add_prototype(label_dir.name, image)
            selected = []
            if a.few_shot > 0:
                rng = random.Random(a.seed)
                for defect_dir in sorted(x for x in (category / "test").iterdir() if x.is_dir() and x.name != "good"):
                    candidates = _images(str(defect_dir))
                    chosen = rng.sample(candidates, min(a.few_shot, len(candidates)))
                    for image in chosen:
                        fusion.add_prototype(defect_dir.name, image)
                        selected.append(image)
                        print(f"[few-shot] {category.name}/{defect_dir.name}: {Path(image).name}", flush=True)
            result_path = a.output if len(categories) == 1 else str(Path(a.output).with_name(f"{Path(a.output).stem}-{category.name}.jsonl"))
            metrics = evaluate_mvtec(fusion, category, result_path, excluded_images=selected)
            metrics["few_shot"] = a.few_shot
            metrics["seed"] = a.seed
            metrics["few_shot_images"] = [str(Path(x)) for x in selected]
            metrics["debias"] = a.debias
            metrics["svd_components"] = a.svd_components if a.debias else 0
            metrics["top_k_ratio"] = top_k_ratio
            metrics["image_score"] = image_score
            metrics["feature_layers"] = list(feature_layers)
            metrics["layer_aggregation"] = layer_aggregation
            all_metrics.append(metrics)
        print(json.dumps(all_metrics[0] if len(all_metrics) == 1 else {"categories": all_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
