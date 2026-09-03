#!/usr/bin/env python
"""Add the aerial mode's pre-selected queries to the static site: site/overhead.json + site/img_overhead/.

Takes the twelve views chosen by webapp/pick_overhead_samples.py (four per source at error percentiles:
current Esri imagery, Wayback 2023-08-31, Wayback 2017-11-16), copies the JPEGs and bakes the FULL
method's answer (prototypes + per-cell codes + image refine, 4-rotation TTA) for each, computed here
on the server side exactly as the Flask app does. Uploads on the static site run the browser pipeline
from webapp/export_web.py --overhead instead, and the page labels the two differently.

    python webapp/build_static_overhead.py -c configs/ankara_overhead.yaml [--out site]
"""
import argparse, json, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from geoloc_tr.aerial import meters_per_pixel                 # noqa: E402
from geoloc_tr.config import load_config                      # noqa: E402
from geoloc_tr.database import CellDatabase, ImageIndex       # noqa: E402
from geoloc_tr.geo import haversine_m                         # noqa: E402
from geoloc_tr.localize import localize                       # noqa: E402
from geoloc_tr.model import load_checkpoint                   # noqa: E402
from geoloc_tr.overhead import Pyramid, configure, embed_photos  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/ankara_overhead.yaml")
    ap.add_argument("--out", default="site")
    ap.add_argument("--tta", type=int, default=4)
    a = ap.parse_args()
    cfg = configure(load_config(a.config, []))
    dev = torch.device("cpu")
    model, ck, h, _ = load_checkpoint(cfg.out_dir / "best.pt", dev)
    cfg.model = ck.model
    db = CellDatabase.load(cfg.out_dir / "cells.npz")
    idx = ImageIndex.load(cfg.out_dir / "image_index.npz")
    alpha = json.load(open(cfg.out_dir / "calibration.json")).get("alpha", 0.0)
    samples = json.loads((Path(__file__).parent / "overhead_samples.json").read_text())
    pyr = None
    if model.scale_head is not None:
        codes = {z: np.load(cfg.out_dir / f"codes_z{z}.npy") for z in cfg.overhead.code_zooms
                 if z != cfg.overhead.eval_zoom and (cfg.out_dir / f"codes_z{z}.npy").exists()}
        fine = CellDatabase.load(cfg.out_dir / "cells_fine.npz") if (cfg.out_dir / "cells_fine.npz").exists() else None
        pyr = Pyramid(model, h, db, idx, codes, cfg, alpha, dev, fine_db=fine)

    out = Path(a.out); imgs = out / "img_overhead"
    imgs.mkdir(parents=True, exist_ok=True)
    recs = []
    for s in samples:
        src = ROOT / s["path"]
        shutil.copyfile(src, imgs / f"{s['id']}.jpg")
        if pyr is not None:
            r = pyr.localize(Image.open(src), gsd=None, rotations=a.tta)
            lat, lon, sc, cells, cdb = r["lat"], r["lon"], r["top_scores"], r["top_cells"], r["db"]
            extra = {"pyramid": {"extent_m": round(r["extent_m"]), "cropped": r["cropped"], "db_level": r["db_level"],
                                 "code_zoom": r["code_zoom"], "region_cells": r["region_cells"], "picked": r["picked"]}}
        else:
            q = embed_photos(model, [Image.open(src)], cfg.model.image_size, dev, rotations=a.tta)
            res = localize(q, db, cfg.retrieval, alpha, idx)
            lat, lon = (float(v) for v in res.latlon[0])
            sc, cells, cdb, extra = res.top_scores[0], res.top_cells[0], db, {}
        w = np.exp((sc - sc.max()) / cfg.retrieval.refine_temperature)
        recs.append({
            "id": s["id"], "img": f"img_overhead/{s['id']}.jpg", "source": s["source"], "label": s["label"],
            "rotation_deg": s["rotation_deg"], "extent_m": s["extent_m"],
            "lat": lat, "lon": lon, "gt_lat": s["gt_lat"], "gt_lon": s["gt_lon"],
            "error_m": float(haversine_m(np.array([lat]), np.array([lon]), np.array([s["gt_lat"]]), np.array([s["gt_lon"]]))[0]),
            "recorded_err_m": s["recorded_err_m"], "top_score": float(sc[0]),
            "candidates": [{"lat": round(float(cdb.centers[j, 0]), 6), "lon": round(float(cdb.centers[j, 1]), 6),
                            "score": round(float(x), 4), "weight": round(float(v), 4)}
                           for j, x, v in zip(cells[:20], sc[:20], w[:20])],
            **extra,
        })
        print(f"  {s['source']:9s} {recs[-1]['error_m']:9.1f} m (picker recorded {s['recorded_err_m']:.1f})")

    results = json.loads((cfg.out_dir / "results.json").read_text()) if (cfg.out_dir / "results.json").exists() else None
    mpp = meters_per_pixel(cfg.bbox.center[0], cfg.overhead.eval_zoom)
    scale_md = cfg.out_dir / "results_scale.csv"
    scale_rows = pd.read_csv(scale_md).to_dict(orient="records") if scale_md.exists() else None
    json.dump({"cells": int(db.size), "database_level": cfg.overhead.database_level, "alpha": alpha, "tta": a.tta,
               "pyramid": pyr is not None, "fine_cells": int(pyr.fine_db.size) if pyr is not None and pyr.fine_db is not None else 0,
               "scale_results": scale_rows,
               "target_mpp": mpp, "extent_m": mpp * cfg.model.image_size,
               "train_dates": 1 + len(cfg.overhead.train_releases), "results": results, "queries": recs},
              open(out / "overhead.json", "w"))
    print(f"wrote {out}/overhead.json + {len(recs)} images")


if __name__ == "__main__":
    main()
