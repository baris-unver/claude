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
    python webapp/export_web.py -c configs/ankara_overhead.yaml --overhead --out site/model_overhead

--overhead exports the overhead-query model (geoloc_tr/overhead.py) the same way, except that its
278,932-cell database (272 MB at fp16, over GitHub's 100 MB file limit) is compressed: an uncentred
PCA to 128 dims (proj.bin, applied to the query in the browser) and int8 rows with a per-row scale.
Measured on the 2000-query test sets (R@100m, prototypes + codes + refine, TTA 4):
fp16-512 99.3 / 86.7 / 62.0, int8-128 98.4 / 84.2 / 58.1 (current / Wayback 2023 / 2017) -- 35 MB.
"""
import argparse, json, shutil, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from geoloc_tr.config import load_config          # noqa: E402
from geoloc_tr.database import CellDatabase       # noqa: E402
from geoloc_tr.geo import haversine_m             # noqa: E402
from geoloc_tr.model import load_checkpoint       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/ankara.yaml")
    ap.add_argument("--out", default="site/model")
    ap.add_argument("--overhead", action="store_true", help="overhead-query model: PCA + int8 database")
    ap.add_argument("--pca", type=int, default=128, help="(--overhead) projected dimension")
    a = ap.parse_args()
    cfg = load_config(a.config, [])
    if a.overhead:
        from geoloc_tr.overhead import configure
        configure(cfg)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    model, ck, _, _ = load_checkpoint(cfg.out_dir / "best.pt", torch.device("cpu"))
    model.eval()
    size = ck.model.image_size

    class Q(torch.nn.Module):
        """Just the query path. The heads and the aerial encoder are not needed at inference:
        the classifier prototypes are already baked into cells.npz."""
        def __init__(self, m): super().__init__(); self.g = m.ground
        def forward(self, x): return self.g(x)

    class QS(torch.nn.Module):
        """Query path + the scale head (overhead pyramid): embedding and log2(extent / reference)."""
        def __init__(self, m): super().__init__(); self.g = m.ground; self.s = m.scale_head
        def forward(self, x):
            f, e = self.g.features(x)
            return e, self.s(f).squeeze(-1)

    pyramid = a.overhead and model.scale_head is not None
    q = (QS(model) if pyramid else Q(model)).eval()
    dummy = torch.zeros(1, 3, size, size)
    f32 = out / "encoder_fp32.onnx"
    outputs = ["embedding", "log_scale"] if pyramid else ["embedding"]
    torch.onnx.export(q, dummy, f32, input_names=["image"], output_names=outputs,
                      opset_version=17, dynamo=False,
                      dynamic_axes={"image": {0: "n"}, **{o: {0: "n"} for o in outputs}})
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
    f32.rename(enc)

    # numerical check: int8 must still produce the same embedding direction
    import onnxruntime as ort
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 3, size, size)).astype(np.float32)
    with torch.no_grad():
        ptt = q(torch.from_numpy(x))
        ptt = ptt[0].numpy() if pyramid else ptt.numpy()
    for name, p in (("shipped fp32", enc),):
        o = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]).run(None, {"image": x})[0]
        cos = (o * ptt).sum(1) / (np.linalg.norm(o, axis=1) * np.linalg.norm(ptt, axis=1))
        print(f"  {name} vs torch: cosine {cos.min():.5f}..{cos.max():.5f}")

    db = CellDatabase.load(cfg.out_dir / "cells.npz")
    alpha = json.load(open(cfg.out_dir / "calibration.json")).get("alpha", 0.0)
    C = db.ground.astype(np.float32).copy()
    if alpha > 0 and db.aerial is not None:
        C += alpha * db.aerial.astype(np.float32)
    meta = {"image_size": size, "dim": int(C.shape[1]), "n_cells": int(C.shape[0]),
            "alpha": alpha, "top_k": cfg.retrieval.top_k,
            "refine_radius_m": cfg.retrieval.refine_radius_m,
            "refine_temperature": cfg.retrieval.refine_temperature,
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "image_refine": False, "dtype": "fp16", "pca_dim": None}
    if a.overhead:
        from geoloc_tr.aerial import meters_per_pixel

        def compress(M, stem):
            """uncentred PCA keeps dot products: q.c ~ (qP).(cP); per-row int8 then costs nothing measurable"""
            _, V = np.linalg.eigh(M.T @ M)
            P = V[:, ::-1][:, :a.pca].astype(np.float32)          # (512, d)
            R = M @ P
            scale = (np.abs(R).max(1, keepdims=True) / 127.0).astype(np.float32)
            np.round(R / scale).astype(np.int8).tofile(out / f"{stem}.bin")
            scale.ravel().tofile(out / f"{stem}_scales.bin")
            P.tofile(out / f"{stem}_proj.bin")
            return P, scale

        P, scale = compress(C, "cells")
        mpp = meters_per_pixel(cfg.bbox.center[0], cfg.overhead.eval_zoom)
        meta.update({"dtype": "int8", "pca_dim": int(a.pca), "tta": 4, "target_mpp": mpp,
                     "extent_m": mpp * size, "mode": "overhead"})
        if pyramid:
            # coarse-to-fine: the 560 m class prototypes, each cell's coarse class, the extra code sets
            from geoloc_tr.geo import cell_parent
            from geoloc_tr.overhead import reference_extent_m
            hier = load_checkpoint(cfg.out_dir / "best.pt", torch.device("cpu"))[2]
            ci = hier.level_values.index(cfg.overhead.coarse_level)
            lc = hier.levels[ci]
            model.heads[ci].prototypes.detach().float().numpy().astype(np.float16).tofile(out / "coarse.bin")

            def coarse_index(cells):
                par = np.fromiter((cell_parent(int(c), lc.level) for c in cells), dtype=np.uint64, count=len(cells))
                return lc.index_of(par).astype(np.int32)

            coarse_index(db.cells).tofile(out / "cells_coarse.bin")
            sets = [{"name": "cells", "extent_m": mpp * size, "centers": "centers.bin", "coarse": "cells_coarse.bin",
                     "n": int(db.size)}]
            for z in cfg.overhead.code_zooms:
                pth = cfg.out_dir / f"codes_z{z}.npy"
                if z == cfg.overhead.eval_zoom or not pth.exists():
                    continue
                compress(db.ground.astype(np.float32) + alpha * np.load(pth).astype(np.float32), f"cells_z{z}")
                sets.append({"name": f"cells_z{z}", "extent_m": meters_per_pixel(cfg.bbox.center[0], z) * size,
                             "centers": "centers.bin", "coarse": "cells_coarse.bin", "n": int(db.size)})
            fp = cfg.out_dir / "cells_fine.npz"
            if fp.exists():
                fine = CellDatabase.load(fp)
                compress(fine.ground.astype(np.float32) + alpha * fine.aerial.astype(np.float32), "cells_fine")
                np.stack([fine.centers[:, 0], fine.centers[:, 1]], 1).astype(np.float32).tofile(out / "centers_fine.bin")
                coarse_index(fine.cells).tofile(out / "cells_fine_coarse.bin")
                sets.append({"name": "cells_fine", "extent_m": meters_per_pixel(cfg.bbox.center[0], cfg.overhead.fine_zoom) * size,
                             "centers": "centers_fine.bin", "coarse": "cells_fine_coarse.bin", "n": int(fine.size)})
            meta["pyramid"] = {"ref_extent_m": reference_extent_m(cfg), "coarse_classes": int(lc.num_classes),
                               "coarse_topk": cfg.overhead.coarse_topk, "small_topk": cfg.overhead.small_topk,
                               "small_extent_frac": cfg.overhead.small_extent_frac, "crop_above": 1.4, "sets": sets}
            print("  pyramid sets:", [(x["name"], round(x["extent_m"]), x["n"]) for x in sets])
        # numerical check of the shipped database: a sample of exact cell vectors used as queries must
        # still retrieve themselves (or a same-prototype sibling within one cell) through the int8/PCA path
        rng = np.random.default_rng(1)
        pick = rng.choice(C.shape[0], 2000, replace=False)
        qv = C[pick] / np.linalg.norm(C[pick], axis=1, keepdims=True)
        Cq = np.fromfile(out / "cells.bin", dtype=np.int8).reshape(-1, a.pca).astype(np.float32) * scale
        approx = ((qv @ P) @ Cq.T).argmax(1)
        d_m = haversine_m(db.centers[pick, 0], db.centers[pick, 1], db.centers[approx, 0], db.centers[approx, 1])
        print(f"  int8/pca-{a.pca} database self-retrieval: {(d_m <= 100).mean() * 100:.1f}% within 100 m, "
              f"median {np.median(d_m):.1f} m (exact path: 100%)")
    else:
        C.astype(np.float16).tofile(out / "cells.bin")
    np.stack([db.centers[:, 0], db.centers[:, 1]], 1).astype(np.float32).tofile(out / "centers.bin")
    json.dump(meta, open(out / "meta.json", "w"))
    tot = sum(p.stat().st_size for p in out.iterdir() if p.name != "encoder_fp32.onnx")
    for old_name in ("scales.bin", "proj.bin"):  # names from the single-set export
        if (out / old_name).exists() and (out / f"cells_{old_name}").exists():
            (out / old_name).unlink()
    print(f"cells.bin {meta['dtype']}: {(out/'cells.bin').stat().st_size/1048576:.1f} MB  "
          f"({C.shape[0]}x{meta['pca_dim'] or C.shape[1]}, alpha={alpha})")
    print(f"TOTAL shipped: {tot/1048576:.1f} MB")


if __name__ == "__main__":
    main()
