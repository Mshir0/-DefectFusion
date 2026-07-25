from __future__ import annotations
import argparse, glob, json
from .features import DinoFeatureExtractor
from .pipeline import DefectFusion

def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); f.add_argument("--normal-dir", required=True); f.add_argument("--model", default="facebook/dinov2-small"); f.add_argument("--output", default="outputs/model.json")
    q = sub.add_parser("predict"); q.add_argument("--model-state", required=True); q.add_argument("--image", required=True); q.add_argument("--model", default="facebook/dinov2-small")
    a = p.parse_args(); extractor = DinoFeatureExtractor(a.model)
    if a.cmd == "fit":
        paths = sorted(glob.glob(a.normal_dir + "/*")); DefectFusion(extractor).fit_normal(paths).save(a.output); print(a.output)
    else: print(json.dumps(DefectFusion.load(a.model_state, extractor).predict(a.image), ensure_ascii=False))

if __name__ == "__main__": main()
