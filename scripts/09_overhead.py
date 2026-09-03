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
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from common import device, parser, setup
from PIL import Image

from geoloc_tr.aerial import meters_per_pixel
from geoloc_tr.database import CellDatabase, ImageIndex
from geoloc_tr.evaluate import calibrate_alpha, errors_m, evaluate_queries, recall_at, to_markdown
from geoloc_tr.localize import localize
from geoloc_tr.model import load_checkpoint
from geoloc_tr.aerial import TileCache  # noqa: F401
from geoloc_tr.overhead import (OverheadQueryDataset, OverheadTrainDataset, PointSampler, Pyramid, build_fine_database,
                                build_overhead_database, cell_codes, configure, embed_photos, embed_views,
                                reference_extent_m, render_photo, overhead_hierarchy, prefetch_bbox, prefetch_tiles, tile_caches,
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
        if z in cfg.overhead.urban_only_zooms:
            continue  # fetched over the built-up area only, below
        got, failed = prefetch_bbox(c, cfg.bbox, cfg.overhead.tile_workers)
        print(f"z{z}: fetched {got} new tiles, {failed} failed -> {c.root}")
    urban = urban_tiles(cfg)
    for z in cfg.overhead.urban_only_zooms:
        c = tile_caches(cfg, None, [z])[z]
        got, failed = prefetch_tiles(c, urban(z), cfg.overhead.tile_workers)
        print(f"z{z} (built-up area only): fetched {got} new tiles, {failed} failed -> {c.root}")
    for rel in cfg.overhead.train_releases:
        for z, c in sorted(tile_caches(cfg, rel, cfg.overhead.release_zooms).items()):
            if z in cfg.overhead.release_urban_only_zooms:
                got, failed = prefetch_tiles(c, urban(z), cfg.overhead.tile_workers)
            else:
                got, failed = prefetch_bbox(c, cfg.bbox, cfg.overhead.tile_workers)
            print(f"train release {rel} z{z}: fetched {got} new tiles, {failed} failed -> {c.root}")
    test = query_sets(cfg, caches)["test_urban"]
    for rel in cfg.overhead.eval_releases:
        wb = tile_caches(cfg, rel)
        for z, tiles in test.with_caches(wb).tiles().items():
            got, failed = prefetch_tiles(wb[z], tiles, cfg.overhead.tile_workers)
            print(f"wayback {rel} z{z}: {len(tiles)} tiles, fetched {got}, {failed} failed")
        # wide-extent test photos (evaluate-scale): z17 around the first SCALE_N points, up to the widest extent
        n = SCALE_N
        wide = max(EXTENT_ZOOMS_M(cfg).values())
        src = int(wide / meters_per_pixel(cfg.bbox.center[0], 17)) + 8
        need = set()
        for a, b in zip(test.lat[:n], test.lon[:n]):
            need |= wb[17].tiles_for_patch(float(a), float(b), src)
        got, failed = prefetch_tiles(wb[17], sorted(need), cfg.overhead.tile_workers)
        print(f"wayback {rel} z17 wide-extent test tiles: {len(need)}, fetched {got}, {failed} failed")
        z18 = tile_caches(cfg, rel, [18])[18]
        need = set()
        for a, b in zip(test.lat[:n], test.lon[:n]):
            need |= z18.tiles_for_patch(float(a), float(b), 2 * cfg.model.image_size + 8)
        got, failed = prefetch_tiles(z18, sorted(need), cfg.overhead.tile_workers)
        print(f"wayback {rel} z18 small-extent test tiles: {len(need)}, fetched {got}, {failed} failed")


def urban_tiles(cfg):
    """zoom -> tiles covering the built-up (Mapillary-covered) level-15 cells, one tile of margin."""
    from geoloc_tr.config import BBox
    from geoloc_tr.overhead import bbox_tile_range
    rects = urban_rects(cfg)

    def f(zoom):
        tiles = set()
        for r in (rects if rects is not None else []):
            xs, ys = bbox_tile_range(BBox(r[2], r[0], r[3], r[1]), zoom, margin=1)
            tiles |= {(x, y) for x in xs for y in ys}
        return sorted(tiles)
    return f


SCALE_N = 500  # test photos per extent in evaluate-scale


def EXTENT_ZOOMS_M(cfg):
    """extent label -> metres, one per training zoom (224 px at the bbox centre latitude)."""
    lat = cfg.bbox.center[0]
    return {f"z{z}": meters_per_pixel(lat, z) * cfg.model.image_size for z in cfg.overhead.zooms}


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
    for z in cfg.overhead.code_zooms:
        if z == cfg.overhead.eval_zoom:
            continue
        c = tile_caches(cfg, None, [z])[z]
        np.save(cfg.out_dir / f"codes_z{z}.npy", cell_codes(model, db.centers, c, cfg, dev))
        print(f"per-cell codes at z{z} -> codes_z{z}.npy")
    if cfg.overhead.fine_level > cfg.overhead.database_level:
        c = tile_caches(cfg, None, [cfg.overhead.fine_zoom])[cfg.overhead.fine_zoom]
        fine = build_fine_database(model, db, cfg, c, dev, urban_rects(cfg))
        fine.save(cfg.out_dir / "cells_fine.npz")
        print(f"fine database: {fine.size} cells at S2 level {fine.level} with z{cfg.overhead.fine_zoom} codes -> cells_fine.npz")
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


def load_pyramid(cfg, args, dev):
    model, ckpt_cfg, h, _ = load_checkpoint(args.checkpoint or cfg.out_dir / "best.pt", dev)
    cfg.model = ckpt_cfg.model
    db = CellDatabase.load(cfg.out_dir / "cells.npz")
    idx_path = cfg.out_dir / "image_index.npz"
    idx = ImageIndex.load(idx_path) if idx_path.exists() else None
    cal = cfg.out_dir / "calibration.json"
    alpha = json.load(open(cal)).get("alpha", 0.0) if cal.exists() else 0.0
    codes = {z: np.load(cfg.out_dir / f"codes_z{z}.npy") for z in cfg.overhead.code_zooms
             if z != cfg.overhead.eval_zoom and (cfg.out_dir / f"codes_z{z}.npy").exists()}
    fine = CellDatabase.load(cfg.out_dir / "cells_fine.npz") if (cfg.out_dir / "cells_fine.npz").exists() else None
    return model, h, db, idx, alpha, Pyramid(model, h, db, idx, codes, cfg, alpha, dev, fine_db=fine)


def cmd_evaluate_scale(cfg, args):
    """Recall vs ground extent of the photo, for the plain single pass and the pyramid (known / estimated
    scale), on the current imagery and the first held-out Wayback date."""
    dev = device()
    model, h, db, idx, alpha, pyr = load_pyramid(cfg, args, dev)
    caches = tile_caches(cfg, None, cfg.overhead.zooms)
    test = query_sets(cfg, caches)["test_urban"]
    n = min(SCALE_N, args.n or SCALE_N)
    lat, lon = test.lat[:n], test.lon[:n]
    rng = np.random.default_rng(5)
    rot = rng.uniform(0, 360, n) if cfg.overhead.rotate else np.zeros(n)
    sources = {"current": caches}
    if cfg.overhead.eval_releases:
        rel = cfg.overhead.eval_releases[0]
        sources[f"wayback{rel}"] = tile_caches(cfg, rel, [17, 18])
    th = cfg.retrieval.eval_thresholds_m
    px = cfg.model.image_size
    lines = [f"Recall vs photo extent ({n} built-up test points, random heading, TTA {args.tta}). "
             f"plain = whole photo through the single pass; pyramid = coarse region + ~{reference_extent_m(cfg):.0f} m crop, "
             f"scale from --gsd (known) or from the scale head (estimated).\n"]
    rows_all = []
    want = set(args.extents.split(",")) if args.extents else None
    for sname, cb in sources.items():
        for label, extent in EXTENT_ZOOMS_M(cfg).items():
            if want and label not in want:
                continue
            mpp_max = min(1.0, extent / px)  # native ~0.9 m/px for wide photos, finer for narrow ones
            if not any(meters_per_pixel(lat[0], z) <= mpp_max for z in cb):
                continue  # this source has no tiles fine enough for that extent
            photos = [render_photo(cb, float(a), float(b), extent, float(r), mpp_max) for a, b, r in zip(lat, lon, rot)]
            gt = np.stack([lat, lon], 1)
            out = {}
            q = embed_photos(model, photos, px, dev, rotations=args.tta)
            out["plain"] = localize(q, db, cfg.retrieval, alpha, idx).latlon
            est, known, scale_err, ranks, picked = [], [], [], [], []
            for ph, a, b in zip(photos, lat, lon):
                gsd = extent / min(ph.width, ph.height)
                rk = pyr.localize(ph, gsd=gsd, rotations=args.tta); known.append([rk["lat"], rk["lon"]])
                ranks.append(pyr.coarse_rank(rk["coarse_logits"], float(a), float(b))); picked.append(rk["picked"])
                re_ = pyr.localize(ph, gsd=None, rotations=args.tta); est.append([re_["lat"], re_["lon"]])
                scale_err.append(abs(math.log2(re_["extent_m"] / extent)))
            out["pyramid, known scale"] = np.asarray(known)
            out["pyramid, estimated scale"] = np.asarray(est)
            se, rk_ = np.asarray(scale_err), np.asarray(ranks)
            diag = (f"true 560 m cell in top-{cfg.overhead.coarse_topk}: {(rk_ < cfg.overhead.coarse_topk).mean() * 100:.0f}%, "
                    f"in top-{cfg.overhead.small_topk}: {(rk_ < cfg.overhead.small_topk).mean() * 100:.0f}%; "
                    f"global pass chosen: {np.mean([p == 'global' for p in picked]) * 100:.0f}%; db level {rk['db_level']}, codes z{rk['code_zoom']}")
            for method, pred in out.items():
                r = recall_at(errors_m(pred, gt), th)
                rows_all.append({"source": sname, "extent_m": round(extent), "photo_px": photos[0].width, "method": method, **r})
            lines.append(f"{sname:14s} {label:>4s} {extent:6.0f} m ({photos[0].width:4d} px): " +
                         " | ".join(f"{m}: R@100 {rows_all[-3 + i]['R@100m']:5.1f} med {rows_all[-3 + i]['median_m']:6.0f}" for i, m in enumerate(out)) +
                         f" | scale est. within 2x: {(se <= 1).mean() * 100:.0f}%, median |log2 err| {np.median(se):.2f} | {diag}")
            print(lines[-1], flush=True)
    md = "\n".join(lines)
    csv, mdp = cfg.out_dir / "results_scale.csv", cfg.out_dir / "results_scale.md"
    if want and csv.exists():  # partial rerun: replace those rows, keep the rest
        old = pd.read_csv(csv)
        new = pd.DataFrame(rows_all)
        keep = ~old.set_index(["source", "extent_m", "method"]).index.isin(new.set_index(["source", "extent_m", "method"]).index)
        pd.concat([old[keep], new]).sort_values(["source", "extent_m", "method"], ascending=[True, False, True]).to_csv(csv, index=False)
        mdp.write_text(mdp.read_text() + "\n\nRerun (" + args.extents + "):\n" + md if mdp.exists() else md)
    else:
        pd.DataFrame(rows_all).to_csv(csv, index=False)
        mdp.write_text(md)
    print(f"wrote {mdp}")


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
    target_mpp = meters_per_pixel(cfg.bbox.center[0], cfg.overhead.eval_zoom)
    paths = [q for pat in args.images for q in (sorted(Path().glob(pat)) or [Path(pat)])]
    pts = []
    if args.pyramid:
        pyr = load_pyramid(cfg, args, dev)[-1]
        for pth in paths:
            r = pyr.localize(Image.open(pth), gsd=args.gsd, rotations=args.tta)
            w = np.exp((r["top_scores"] - r["top_scores"].max()) / cfg.retrieval.refine_temperature)
            cands = [[float(r["db"].centers[j, 0]), float(r["db"].centers[j, 1]), float(v)] for j, v in zip(r["top_cells"][:10], w[:10])]
            print(f"{pth}\t{r['lat']:.6f}\t{r['lon']:.6f}\textent {r['extent_m']:.0f} m ({r['extent_from']})"
                  f"\t{'cropped' if r['cropped'] else 'whole'}, L{r['db_level']} cells, codes z{r['code_zoom']}, "
                  f"{r['region_cells']} region cells, {r['picked']} pass\tscore {r['top_score']:.3f}")
            pts.append({"name": str(pth), "lat": r["lat"], "lon": r["lon"], "cands": cands})
        Path(args.map).write_text(importlib.import_module("08_predict").MAP_HTML % json.dumps(pts))
        print("map:", args.map)
        return
    q = embed_photos(model, [Image.open(p) for p in paths], cfg.model.image_size, dev, rotations=args.tta,
                     gsd=args.gsd, target_mpp=target_mpp)
    res = localize(q, db, cfg.retrieval, alpha, idx)
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
    p.add_argument("command", choices=["fetch", "train", "build", "evaluate", "evaluate-scale", "predict"])
    p.add_argument("images", nargs="*", help="(predict) image paths / globs")
    p.add_argument("--checkpoint", default=None, help="defaults to <out_dir>/best.pt")
    p.add_argument("--no-pretrained", action="store_true", help="(train) random-init backbone (offline)")
    p.add_argument("--tta", type=int, default=1, help="(evaluate/predict) average over this many rotations")
    p.add_argument("--gsd", type=float, default=None, help="(predict) metres per pixel of the input images, if known")
    p.add_argument("--map", default="overhead_predictions.html")
    p.add_argument("--pyramid", action="store_true", help="(predict) coarse-to-fine, any photo extent")
    p.add_argument("--n", type=int, default=None, help="(evaluate-scale) photos per extent")
    p.add_argument("--extents", default=None, help="(evaluate-scale) comma-separated zoom labels to (re)run, e.g. z18,z19")
    args = p.parse_args()
    cfg = configure(setup(args))
    {"fetch": cmd_fetch, "train": cmd_train, "build": cmd_build, "evaluate": cmd_evaluate,
     "evaluate-scale": cmd_evaluate_scale, "predict": cmd_predict}[args.command](cfg, args)


if __name__ == "__main__":
    main()
