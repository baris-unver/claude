#!/usr/bin/env python
"""Export the query encoder + cell database for in-browser inference (onnxruntime-web).

GitHub Pages cannot run the model, so the model goes to the visitor instead. Two artefacts:

  encoder.onnx  model.ground (backbone + projection): image -> 512-d unit embedding.
  cells.bin     ground + alpha*aerial pre-combined into ONE matrix, so scoring in the browser is a
                single matmul instead of two. fp16 halves it at no measurable cost to the result.

The image-refine step is NOT shipped: it needs the 97,834-image index (191 MB fp32). The browser
therefore runs "prototypes + aerial + refine", which the evaluation measured at R@100m 22.9 vs 23.8
for the full method on test_seen -- about one point. That is disclosed in the UI rather than hidden.

    python webapp/export_web.py -c configs/ankara.yaml --out site/model
"""
import argparse, json, shutil, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from geoloc_tr.config import load_config          # noqa: E402
from geoloc_tr.database import CellDatabase       # noqa: E402
from geoloc_tr.model import load_checkpoint       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/ankara.yaml")
    ap.add_argument("--out", default="site/model")
    a = ap.parse_args()
    cfg = load_config(a.config, [])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    model, ck, _, _ = load_checkpoint(cfg.out_dir / "best.pt", torch.device("cpu"))
    model.eval()
    size = ck.model.image_size

    class Q(torch.nn.Module):
        """Just the query path. The heads and the aerial encoder are not needed at inference:
        the classifier prototypes are already baked into cells.npz."""
        def __init__(self, m): super().__init__(); self.g = m.ground
        def forward(self, x): return self.g(x)

    q = Q(model).eval()
    dummy = torch.zeros(1, 3, size, size)
    with torch.no_grad():
        ref = q(dummy).numpy()
    f32 = out / "encoder_fp32.onnx"
    torch.onnx.export(q, dummy, f32, input_names=["image"], output_names=["embedding"],
                      opset_version=17, dynamo=False,
                      dynamic_axes={"image": {0: "n"}, "embedding": {0: "n"}})
    print(f"encoder fp32: {f32.stat().st_size/1048576:.1f} MB")

    # Shipped at fp32 (86 MB), deliberately. Both cheaper options were measured and rejected:
    #   int8 dynamic (22.9 MB): cosine vs torch fell to 0.944 on real images, only 72.9% of
    #     queries kept the same top-1 cell, and the p90 position shift was 1.4 km. That is not a
    #     smaller demo, it is a different model.
    #   fp16 (43 MB): onnxconverter-common rewrites timm's attention Cast nodes into a graph
    #     onnxruntime refuses to load, and blocking those ops makes the conversion hang.
    # fp32 reproduces the server exactly (cosine 1.00000) and runs on both the wasm and webgpu
    # backends, so the demo's answers are the same ones the evaluation measured.
    enc = out / "encoder.onnx"
    shutil.copyfile(f32, enc)

    # numerical check: int8 must still produce the same embedding direction
    import onnxruntime as ort
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 3, size, size)).astype(np.float32)
    with torch.no_grad():
        ptt = q(torch.from_numpy(x)).numpy()
    for name, p in (("shipped fp32", enc),):
        o = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]).run(None, {"image": x})[0]
        cos = (o * ptt).sum(1) / (np.linalg.norm(o, axis=1) * np.linalg.norm(ptt, axis=1))
        print(f"  {name} vs torch: cosine {cos.min():.5f}..{cos.max():.5f}")

    db = CellDatabase.load(cfg.out_dir / "cells.npz")
    alpha = json.load(open(cfg.out_dir / "calibration.json")).get("alpha", 0.0)
    C = db.ground.astype(np.float32).copy()
    if alpha > 0 and db.aerial is not None:
        C += alpha * db.aerial.astype(np.float32)
    C.astype(np.float16).tofile(out / "cells.bin")
    np.stack([db.centers[:, 0], db.centers[:, 1]], 1).astype(np.float32).tofile(out / "centers.bin")
    json.dump({"image_size": size, "dim": int(C.shape[1]), "n_cells": int(C.shape[0]),
               "alpha": alpha, "top_k": cfg.retrieval.top_k,
               "refine_radius_m": cfg.retrieval.refine_radius_m,
               "refine_temperature": cfg.retrieval.refine_temperature,
               "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
               "image_refine": False},
              open(out / "meta.json", "w"))
    tot = sum(p.stat().st_size for p in out.iterdir() if p.name != "encoder_fp32.onnx")
    print(f"cells.bin fp16: {(out/'cells.bin').stat().st_size/1048576:.1f} MB  "
          f"({C.shape[0]}x{C.shape[1]}, alpha={alpha})")
    print(f"TOTAL shipped: {tot/1048576:.1f} MB")


if __name__ == "__main__":
    main()
