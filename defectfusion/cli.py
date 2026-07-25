from __future__ import annotations

import argparse
import glob
import json
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


def main(argv=None):
    p = argparse.ArgumentParser(description="Few-shot / zero-shot defect detection")
    p.add_argument("--config", help="JSON config; CLI flags override it")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit normal subspace and optional defect prototypes")
    f.add_argument("--normal-dir"); f.add_argument("--prototype-dir", help="subdirectories are defect labels")
    f.add_argument("--model", default=None); f.add_argument("--output", default=None)
    f.add_argument("--alpha", type=float, default=None); f.add_argument("--unknown-threshold", type=float, default=None)
    f.add_argument("--device", default=None); f.add_argument("--non-recursive", action="store_true")
    q = sub.add_parser("predict", help="score one image or a directory")
    q.add_argument("--model-state", required=True); q.add_argument("--image", required=True)
    q.add_argument("--model", default=None); q.add_argument("--device", default=None)
    q.add_argument("--output", help="write JSON results to a file")
    e = sub.add_parser("evaluate-mvtec", help="fit on train/good and evaluate one MVTec category")
    e.add_argument("--data-dir", required=True, help="MVTec category directory")
    e.add_argument("--model", default=None); e.add_argument("--device", default=None)
    e.add_argument("--output", default="outputs/mvtec-results.jsonl")
    a = p.parse_args(argv); cfg = _config(a.config)
    model_name = getattr(a, "model", None) or cfg.get("model", "facebook/dinov2-small")
    extractor = DinoFeatureExtractor(model_name, device=getattr(a, "device", None) or cfg.get("device"))
    if a.cmd == "fit":
        normal_dir = a.normal_dir or cfg.get("normal_dir")
        if not normal_dir: p.error("fit requires --normal-dir or config normal_dir")
        output = a.output or cfg.get("output", "outputs/model.json")
        alpha = a.alpha if a.alpha is not None else cfg.get("alpha", 0.5)
        threshold = a.unknown_threshold if a.unknown_threshold is not None else cfg.get("unknown_threshold", 0.35)
        paths = _images(normal_dir, not a.non_recursive)
        fusion = DefectFusion(extractor, alpha=alpha, unknown_threshold=threshold).fit_normal(paths)
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
        normal_dir = str(Path(a.data_dir) / "train" / "good")
        fusion = DefectFusion(extractor).fit_normal(_images(normal_dir))
        print(json.dumps(evaluate_mvtec(fusion, a.data_dir, a.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
