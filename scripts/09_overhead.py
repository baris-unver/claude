#!/usr/bin/env python
"""Step 9 (optional, independent of 03-08): overhead queries, i.e. localise satellite / aerial photos.

    python scripts/09_overhead.py fetch    -c configs/ankara_overhead.yaml   # tile pyramid over the bbox (+ Wayback test tiles)
    python scripts/09_overhead.py train    -c configs/ankara_overhead.yaml   # query encoder on random overhead views
    python scripts/09_overhead.py build    -c configs/ankara_overhead.yaml   # cell database, per-cell codes, calibration
    python scripts/09_overhead.py evaluate -c configs/ankara_overhead.yaml   # recall@d: current imagery + Wayback dates
    python scripts/09_overhead.py predict  -c configs/ankara_overhead.yaml photo.jpg [--gsd 0.5] [--tta 8]
"""
import importlib
import json
from pathlib import Path

import numpy as np
import torch
from common import device, parser, setup
from PIL import Image

from geoloc_tr.aerial import meters_per_pixel
from geoloc_tr.data import eval_transform
from geoloc_tr.database import CellDatabase, ImageIndex
from geoloc_tr.evaluate import calibrate_alpha, errors_m, evaluate_queries, to_markdown
from geoloc_tr.localize import localize
from geoloc_tr.model import load_checkpoint
from geoloc_tr.overhead import (OverheadQueryDataset, OverheadTrainDataset, PointSampler, build_overhead_database,
                                configure, embed_views, overhead_hierarchy, prefetch_bbox, prefetch_tiles, tile_caches,
                                tile_change_fraction, training_sources, urban_rects)
from geoloc_tr.train import fit

SEEDS = {"val": 11, "test_urban": 22, "test_bbox": 33}


def sampler(cfg, urban_frac):
    return PointSampler(cfg.bbox, urban_rects(cfg), urban_frac)


def query_sets(cfg, caches):
    """Fixed (seeded) query sets, identical across build/evaluate runs. `val` and `test_urban` are centred in
    built-up cells, `test_bbox` uniformly over the whole bbox."""
    oc = cfg.overhead
    return {
        "val": OverheadQueryDataset.sample(cfg, caches, sampler(cfg, 1.0), oc.val_queries, SEEDS["val"]),
        "test_urban": OverheadQueryDataset.sample(cfg, caches, sampler(cfg, 1.0), oc.eval_queries, SEEDS["test_urban"]),
        "test_bbox": OverheadQueryDataset.sample(cfg, caches, sampler(cfg, 0.0), oc.eval_queries, SEEDS["test_bbox"]),
    }


def gt_of(ds):
    return np.stack([ds.lat, ds.lon], 1)


def cmd_fetch(cfg, args):
    caches = tile_caches(cfg)
    for z, c in sorted(caches.items()):
        got, failed = prefetch_bbox(c, cfg.bbox, cfg.overhead.tile_workers)
        print(f"z{z}: fetched {got} new tiles, {failed} failed -> {c.root}")
    for rel in cfg.overhead.train_releases:
        for z, c in sorted(tile_caches(cfg, rel, cfg.overhead.release_zooms).items()):
            got, failed = prefetch_bbox(c, cfg.bbox, cfg.overhead.tile_workers)
            print(f"train release {rel} z{z}: fetched {got} new tiles, {failed} failed -> {c.root}")
    test = query_sets(cfg, caches)["test_urban"]
    for rel in cfg.overhead.eval_releases:
        wb = tile_caches(cfg, rel)
        for z, tiles in test.with_caches(wb).tiles().items():
            got, failed = prefetch_tiles(wb[z], tiles, cfg.overhead.tile_workers)
            print(f"wayback {rel} z{z}: {len(tiles)} tiles, fetched {got}, {failed} failed")


def cmd_train(cfg, args):
    h, db_cells = overhead_hierarchy(cfg.bbox, cfg.overhead.levels, cfg.overhead.database_level)
    print("classes per level:", {lc.level: lc.num_classes for lc in h.levels},
          f"database cells (L{cfg.overhead.database_level}): {len(db_cells)}")
    caches = tile_caches(cfg)
    rects = urban_rects(cfg)
    print("built-up cells:", 0 if rects is None else len(rects), "| views/epoch:", cfg.overhead.samples_per_epoch)
    sources = training_sources(cfg)
    print("training sources:", len(sources), "(current imagery + Wayback releases", cfg.overhead.train_releases, ")")
    train_ds = OverheadTrainDataset(cfg, h, sources, PointSampler(cfg.bbox, rects, cfg.overhead.urban_frac))
    val_ds = query_sets(cfg, caches)["val"]
    _, ckpt = fit(cfg, train_ds, val_ds, h, device(), pretrained=not args.no_pretrained, use_aerial=False)
    print("best checkpoint:", ckpt)


def cmd_build(cfg, args):
    dev = device()
    model, ckpt_cfg, h, _ = load_checkpoint(args.checkpoint or cfg.out_dir / "best.pt", dev)
    cfg.model = ckpt_cfg.model
    caches = tile_caches(cfg)
    db, idx = build_overhead_database(model, h, cfg, caches, dev)
    db.save(cfg.out_dir / "cells.npz")
    idx.save(cfg.out_dir / "image_index.npz")
    print(f"database: {db.size} cells at S2 level {db.level}, codes from z{cfg.overhead.eval_zoom} views")
    val = query_sets(cfg, caches)["val"]
    q, _ = embed_views(model, val, cfg, dev)
    alpha = calibrate_alpha(q, gt_of(val), db, cfg)
    json.dump({"alpha": alpha}, open(cfg.out_dir / "calibration.json", "w"))
    print(f"cell-code weight alpha={alpha}")


def cmd_evaluate(cfg, args):
    dev = device()
    model, ckpt_cfg, h, _ = load_checkpoint(args.checkpoint or cfg.out_dir / "best.pt", dev)
    cfg.model = ckpt_cfg.model
    db = CellDatabase.load(cfg.out_dir / "cells.npz")
    idx = ImageIndex.load(cfg.out_dir / "image_index.npz")
    cal = cfg.out_dir / "calibration.json"
    alpha = json.load(open(cal)).get("alpha", 0.0) if cal.exists() else 0.0
    caches = tile_caches(cfg)
    sets = query_sets(cfg, caches)
    notes = {"test_urban": "built-up cells, current imagery", "test_bbox": "uniform over the bbox, current imagery"}
    for rel in cfg.overhead.eval_releases:
        wb = tile_caches(cfg, rel)
        name = f"test_urban_wayback{rel}"
        sets[name] = sets["test_urban"].with_caches(wb)
        z = cfg.overhead.eval_zoom
        frac = tile_change_fraction(caches[z], wb[z], sets[name].tiles()[z])
        notes[name] = f"same points, Esri Wayback release {rel}; {frac * 100:.0f}% of its tiles differ from the current imagery"
    del sets["val"]
    report, results = [], {}
    for name, ds in sets.items():
        q, lg = embed_views(model, ds, cfg, dev, rotations=args.tta)
        table, errs = evaluate_queries(q, lg, gt_of(ds), h, db, cfg, alpha, idx)
        report.append(f"### {name} ({len(ds)} queries, z{cfg.overhead.eval_zoom}, random rotation; {notes[name]})\n\n{to_markdown(table)}\n")
        results[name] = table.to_dict(orient="records")
    md = (f"Overhead queries, TTA rotations={args.tta}. 'aerial' = per-cell codes of the database imagery, "
          f"'image refine' = snap to the best-matching cells.\n\n" + "\n".join(report))
    (cfg.out_dir / "results.md").write_text(md)
    json.dump(results, open(cfg.out_dir / "results.json", "w"), indent=2)
    print(md)


@torch.no_grad()
def cmd_predict(cfg, args):
    dev = device()
    model, ckpt_cfg, h, _ = load_checkpoint(args.checkpoint or cfg.out_dir / "best.pt", dev)
    cfg.model = ckpt_cfg.model
    db = CellDatabase.load(cfg.out_dir / "cells.npz")
    idx_path = cfg.out_dir / "image_index.npz"
    idx = ImageIndex.load(idx_path) if idx_path.exists() else None
    cal = cfg.out_dir / "calibration.json"
    alpha = json.load(open(cal)).get("alpha", 0.0) if cal.exists() else 0.0
    px = cfg.model.image_size
    tf = eval_transform(px)
    target_mpp = meters_per_pixel(cfg.bbox.center[0], cfg.overhead.eval_zoom)
    paths = [q for pat in args.images for q in (sorted(Path().glob(pat)) or [Path(pat)])]
    embs = []
    for pth in paths:
        img = Image.open(pth).convert("RGB")
        if args.gsd:  # rescale so one pixel covers the same ground as the training views at eval_zoom
            f = args.gsd / target_mpp
            img = img.resize((max(px, round(img.width * f)), max(px, round(img.height * f))), Image.BILINEAR)
        views = [img.rotate(360.0 * k / args.tta, resample=Image.BILINEAR) if k else img for k in range(args.tta)]
        e = model.ground(torch.stack([tf(v) for v in views]).to(dev)).float().cpu().numpy().sum(0)
        embs.append(e / np.linalg.norm(e).clip(1e-12))
    q = np.stack(embs).astype(np.float32)
    res = localize(q, db, cfg.retrieval, alpha, idx)
    pts = []
    for i, pth in enumerate(paths):
        lat, lon = res.latlon[i]
        sc = res.top_scores[i]
        w = np.exp((sc - sc.max()) / cfg.retrieval.refine_temperature)
        cands = [[float(db.centers[j, 0]), float(db.centers[j, 1]), float(v)] for j, v in zip(res.top_cells[i][:10], w[:10])]
        print(f"{pth}\t{lat:.6f}\t{lon:.6f}\ttop-cell {int(db.cells[res.top_cells[i, 0]])}\tscore {sc[0]:.3f}")
        pts.append({"name": str(pth), "lat": float(lat), "lon": float(lon), "cands": cands})
    Path(args.map).write_text(importlib.import_module("08_predict").MAP_HTML % json.dumps(pts))
    print("map:", args.map)


def main():
    p = parser(__doc__)
    p.add_argument("command", choices=["fetch", "train", "build", "evaluate", "predict"])
    p.add_argument("images", nargs="*", help="(predict) image paths / globs")
    p.add_argument("--checkpoint", default=None, help="defaults to <out_dir>/best.pt")
    p.add_argument("--no-pretrained", action="store_true", help="(train) random-init backbone (offline)")
    p.add_argument("--tta", type=int, default=1, help="(evaluate/predict) average over this many rotations")
    p.add_argument("--gsd", type=float, default=None, help="(predict) metres per pixel of the input images, if known")
    p.add_argument("--map", default="overhead_predictions.html")
    args = p.parse_args()
    cfg = configure(setup(args))
    {"fetch": cmd_fetch, "train": cmd_train, "build": cmd_build, "evaluate": cmd_evaluate, "predict": cmd_predict}[args.command](cfg, args)


if __name__ == "__main__":
    main()
