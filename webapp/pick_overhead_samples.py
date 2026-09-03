#!/usr/bin/env python
"""Choose the demo's pre-selected OVERHEAD queries: webapp/overhead_samples.json + webapp/overhead_samples/*.jpg.

Four per source at the 10/40/70/95th percentiles of the error (spanning the distribution, not a best-of):
the current Esri imagery (what the cell database is built from) and the two Esri Wayback dates the model
never trained on. Views are the evaluation's seeded `test_urban` points, rendered at 448 px from z18 --
the same ~205 m extent as the evaluation's 224 px z17 views, twice the pixels for display. The recorded
error is measured through the webapp's own inference path (resize to 224, 4-rotation TTA).
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from geoloc_tr.config import load_config                                   # noqa: E402
from geoloc_tr.database import CellDatabase, ImageIndex                    # noqa: E402
from geoloc_tr.evaluate import errors_m                                    # noqa: E402
from geoloc_tr.localize import localize                                    # noqa: E402
from geoloc_tr.model import load_checkpoint                                # noqa: E402
from geoloc_tr.overhead import (configure, embed_photos, prefetch_tiles, render_view,   # noqa: E402
                                tile_caches, tiles_for_views)

CONFIG = "configs/ankara_overhead.yaml"
N_CAND = 120
QUANTILES = [0.10, 0.40, 0.70, 0.95]
DISPLAY_PX, DISPLAY_ZOOM, TTA = 448, 18, 4
SOURCES = [("current", None, "Esri current imagery"),
           ("wb64776", 64776, "Wayback 2023-08-31 (unseen date)"),
           ("wb25521", 25521, "Wayback 2017-11-16 (unseen date)")]


def main():
    cfg = configure(load_config(ROOT / CONFIG))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_cfg, h, _ = load_checkpoint(cfg.out_dir / "best.pt", dev)
    cfg.model = ckpt_cfg.model
    db = CellDatabase.load(cfg.out_dir / "cells.npz")
    idx = ImageIndex.load(cfg.out_dir / "image_index.npz")
    alpha = json.load(open(cfg.out_dir / "calibration.json"))["alpha"]
    query_sets = importlib.import_module("09_overhead").query_sets
    test = query_sets(cfg, tile_caches(cfg))["test_urban"]
    lat, lon, rot = test.lat[:N_CAND], test.lon[:N_CAND], test.rotation[:N_CAND]

    out_dir = Path(__file__).parent / "overhead_samples"
    out_dir.mkdir(exist_ok=True)
    out = []
    for key, release, label in SOURCES:
        cache = tile_caches(cfg, release, [DISPLAY_ZOOM])[DISPLAY_ZOOM]
        prefetch_tiles(cache, tiles_for_views(cache, lat, lon, DISPLAY_PX), cfg.overhead.tile_workers, progress=False)
        imgs = [render_view(cache, float(a), float(b), DISPLAY_PX, float(r)) for a, b, r in zip(lat, lon, rot)]
        q = embed_photos(model, imgs, cfg.model.image_size, dev, rotations=TTA)
        res = localize(q, db, cfg.retrieval, alpha, idx)
        err = errors_m(res.latlon, np.stack([lat, lon], 1))
        # a Wayback release can lack a z18 tile (404 -> grey patch): never pick those as demo queries
        blank = np.array([np.asarray(im).std() < 4 for im in imgs])
        err = np.where(blank, np.nan, err)
        order = np.argsort(err)[: int((~blank).sum())]
        for qt in QUANTILES:
            i = int(order[min(int(qt * len(order)), len(order) - 1)])
            sid = f"{key}_{i:03d}"
            imgs[i].save(out_dir / f"{sid}.jpg", "JPEG", quality=90)
            out.append({"id": sid, "source": key, "label": label, "path": f"webapp/overhead_samples/{sid}.jpg",
                        "gt_lat": float(lat[i]), "gt_lon": float(lon[i]), "rotation_deg": float(rot[i]),
                        "extent_m": round(DISPLAY_PX * 156543.03 * np.cos(np.radians(float(lat[i]))) / 2 ** DISPLAY_ZOOM),
                        "recorded_err_m": float(err[i]), "quantile": qt})
        ok = err[~blank]
        print(f"{label}: {len(ok)} candidates, median error {np.median(ok):.0f} m, R@100m {(ok <= 100).mean() * 100:.0f}%")
    for f in out_dir.glob("*.jpg"):
        if f.stem not in {s["id"] for s in out}:
            f.unlink()
    (Path(__file__).parent / "overhead_samples.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {len(out)} samples")
    for s in out:
        print(f"  {s['source']:9s} q{s['quantile']:.2f}  recorded error {s['recorded_err_m']:8.1f} m")


if __name__ == "__main__":
    main()
