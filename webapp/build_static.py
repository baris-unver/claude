#!/usr/bin/env python
"""Build a static GitHub Pages demo: precomputed predictions + images + a no-server UI.

GitHub Pages serves static files only, so inference cannot run there. This bakes the model's
answers for a set of held-out queries into one JSON and ships the map UI against it. Upload is
not possible on the static site and the page says so, pointing at the local Flask app instead.

Attribution: Mapillary imagery is CC BY-SA 4.0, so republishing the photos requires naming the
contributor and linking the source. Creator usernames are fetched from the Graph API here and
rendered next to every image.

    python webapp/build_static.py -c configs/ankara.yaml [--n 48] [--out site]
"""
import argparse, json, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from geoloc_tr.config import load_config, mapillary_token   # noqa: E402
from geoloc_tr.data import eval_transform                   # noqa: E402
from geoloc_tr.database import CellDatabase, ImageIndex     # noqa: E402
from geoloc_tr.geo import haversine_m                       # noqa: E402
from geoloc_tr.localize import localize                     # noqa: E402
from geoloc_tr.model import load_checkpoint                 # noqa: E402

METHOD = "full: proto + aerial + image refine"


def creators(ids):
    """id -> (username, user_id) from the Graph API; attribution is a licence obligation."""
    import requests
    out, tok, s = {}, mapillary_token(), requests.Session()
    for i in range(0, len(ids), 50):
        b = ids[i:i + 50]
        try:
            r = s.get("https://graph.mapillary.com/", timeout=60,
                      params={"access_token": tok, "fields": "id,creator", "ids": ",".join(b)})
            for k, v in (r.json() or {}).items():
                c = (v or {}).get("creator") or {}
                out[str(k)] = (c.get("username"), c.get("id"))
        except Exception as e:
            print(f"  attribution fetch failed for a batch: {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/ankara.yaml")
    ap.add_argument("--n", type=int, default=48, help="queries per split pair (half from each)")
    ap.add_argument("--out", default="site")
    ap.add_argument("--max-side", type=int, default=640, help="re-encode demo images to this")
    a = ap.parse_args()

    cfg = load_config(a.config, [])
    dev = torch.device("cpu")
    model, ck, _, _ = load_checkpoint(cfg.out_dir / "best.pt", dev); model.eval()
    db = CellDatabase.load(cfg.out_dir / "cells.npz")
    idx = ImageIndex.load(cfg.out_dir / "image_index.npz") \
        if (cfg.out_dir / "image_index.npz").exists() else None
    alpha = json.load(open(cfg.out_dir / "calibration.json")).get("alpha", 0.0)
    tf = eval_transform(ck.model.image_size)

    splits = pd.read_parquet(ROOT / "data" / cfg.data.city / "splits.parquet")
    rows = []
    for split in ("test_seen", "test_unseen"):
        err = pd.read_parquet(cfg.out_dir / f"errors_{split}.parquet")[["id", METHOD]] \
            .rename(columns={METHOD: "err_m"})
        df = splits[splits["split"] == split].merge(err, on="id").sort_values("err_m")
        # even sweep of the error distribution, so the demo shows the real spread of outcomes
        for q in np.linspace(0.02, 0.98, a.n // 2):
            rows.append(df.iloc[min(int(q * len(df)), len(df) - 1)])
    seen = set(); picks = []
    for r in rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"]); picks.append(r)
    print(f"{len(picks)} unique queries")

    out = Path(a.out); imgs = out / "img"
    imgs.mkdir(parents=True, exist_ok=True)
    who = creators([str(int(r["id"])) for r in picks])
    print(f"attribution resolved for {sum(1 for v in who.values() if v[0])}/{len(picks)}")

    recs = []
    for r in picks:
        src = ROOT / r["path"]
        im = Image.open(src).convert("RGB")
        im.thumbnail((a.max_side, a.max_side))
        name = f"{int(r['id'])}.jpg"
        im.save(imgs / name, quality=86, optimize=True)

        with torch.no_grad():
            q = model(tf(Image.open(src).convert("RGB")).unsqueeze(0))[0].numpy()
        res = localize(q, db, cfg.retrieval, alpha, idx)
        lat, lon = (float(v) for v in res.latlon[0])
        sc = res.top_scores[0]
        w = np.exp((sc - sc.max()) / cfg.retrieval.refine_temperature)
        user, uid = who.get(str(int(r["id"])), (None, None))
        recs.append({
            "id": str(int(r["id"])), "img": f"img/{name}", "split": r["split"],
            "lat": lat, "lon": lon,
            "gt_lat": float(r["lat"]), "gt_lon": float(r["lon"]),
            "error_m": float(haversine_m(np.array([lat]), np.array([lon]),
                                         np.array([r["lat"]]), np.array([r["lon"]]))[0]),
            "top_score": float(sc[0]),
            "candidates": [{"lat": round(float(db.centers[j, 0]), 6),
                            "lon": round(float(db.centers[j, 1]), 6),
                            "score": round(float(s), 4), "weight": round(float(v), 4)}
                           for j, s, v in zip(res.top_cells[0][:20], sc[:20], w[:20])],
            "creator": user, "creator_id": uid,
        })
        print(f"  {r['split']:12s} {recs[-1]['error_m']:9.1f} m  {user or '(unknown)'}")

    for f in (Path(__file__).parent / "static_site").iterdir():   # page + browser inference module
        shutil.copyfile(f, out / f.name)

    results = json.loads((cfg.out_dir / "results.json").read_text())
    json.dump({"city": cfg.data.city, "cells": int(db.size),
               "database_level": cfg.cells.database_level, "alpha": alpha,
               "results": results, "queries": recs},
              open(out / "data.json", "w"))
    print(f"\nwrote {out}/data.json ({(out/'data.json').stat().st_size/1048576:.2f} MB) "
          f"+ {len(recs)} images ({sum(p.stat().st_size for p in imgs.iterdir())/1048576:.1f} MB)")


if __name__ == "__main__":
    main()
