#!/usr/bin/env python
"""Local web app for testing the trained geoloc-tr models: pick a pre-selected query or upload your
own, and see where the model thinks it was taken, on a map, against the ground truth. Two modes:
street-level photos (configs/ankara.yaml) and overhead satellite / aerial photos (aerial mode,
configs/ankara_overhead.yaml, see geoloc_tr/overhead.py).

    python webapp/app.py -c configs/ankara.yaml [--overhead-config configs/ankara_overhead.yaml] [--device cpu] [--port 8000]

Defaults to CPU: a single 224 px ViT-S/14 forward takes ~0.2 s there, and it keeps the GPU free.
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from geoloc_tr.config import load_config                       # noqa: E402
from geoloc_tr.data import eval_transform                      # noqa: E402
from geoloc_tr.database import CellDatabase, ImageIndex        # noqa: E402
from geoloc_tr.geo import haversine_m                          # noqa: E402
from geoloc_tr.localize import localize                        # noqa: E402
from geoloc_tr.model import load_checkpoint                    # noqa: E402
from geoloc_tr.aerial import meters_per_pixel                  # noqa: E402
from geoloc_tr.overhead import configure, embed_photos         # noqa: E402

app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"))
STATE = {}


def _embed(img: Image.Image) -> np.ndarray:
    x = STATE["tf"](img.convert("RGB")).unsqueeze(0).to(STATE["dev"])
    with torch.no_grad():
        return STATE["model"](x)[0].cpu().numpy()


def _predict(img: Image.Image) -> dict:
    cfg, db = STATE["cfg"], STATE["db"]
    res = localize(_embed(img), db, cfg.retrieval, STATE["alpha"], STATE["idx"])
    lat, lon = (float(v) for v in res.latlon[0])
    sc = res.top_scores[0]
    w = np.exp((sc - sc.max()) / cfg.retrieval.refine_temperature)
    cands = [{"lat": float(db.centers[j, 0]), "lon": float(db.centers[j, 1]),
              "score": float(s), "weight": float(v)}
             for j, s, v in zip(res.top_cells[0][:20], sc[:20], w[:20])]
    return {"lat": lat, "lon": lon, "candidates": cands,
            "top_score": float(sc[0]), "alpha": STATE["alpha"]}


# ---------------------------------------------------------------------------------------------
# overhead (satellite / aerial photo) queries -- a second model + database, see geoloc_tr/overhead.py
# ---------------------------------------------------------------------------------------------
def _predict_overhead(img: Image.Image, gsd: float | None, tta: int) -> dict:
    oh = STATE["oh"]
    cfg, db = oh["cfg"], oh["db"]
    q = embed_photos(oh["model"], [img], cfg.model.image_size, oh["dev"], rotations=tta, gsd=gsd,
                     target_mpp=oh["target_mpp"])
    res = localize(q, db, cfg.retrieval, oh["alpha"], oh["idx"])
    lat, lon = (float(v) for v in res.latlon[0])
    sc = res.top_scores[0]
    w = np.exp((sc - sc.max()) / cfg.retrieval.refine_temperature)
    cands = [{"lat": float(db.centers[j, 0]), "lon": float(db.centers[j, 1]),
              "score": float(s), "weight": float(v)}
             for j, s, v in zip(res.top_cells[0][:20], sc[:20], w[:20])]
    return {"lat": lat, "lon": lon, "candidates": cands, "top_score": float(sc[0]),
            "alpha": oh["alpha"], "tta": tta, "gsd": gsd, "mode": "overhead"}


@app.get("/api/overhead/samples")
def overhead_samples():
    if "oh" not in STATE:
        return jsonify([])
    return jsonify([{k: s[k] for k in ("id", "source", "label", "gt_lat", "gt_lon", "rotation_deg",
                                       "extent_m", "recorded_err_m", "quantile")}
                    for s in STATE["oh"]["samples"]])


@app.get("/api/overhead/image/<sid>")
def overhead_image(sid):
    s = STATE.get("oh", {}).get("by_id", {}).get(sid)
    if s is None:
        return jsonify({"error": "unknown sample"}), 404
    return send_file(s["path"], mimetype="image/jpeg")


@app.post("/api/overhead/predict")
def overhead_predict():
    if "oh" not in STATE:
        return jsonify({"error": "overhead model not loaded (run scripts/09_overhead.py first)"}), 503

    def _f(v, default=None, cast=float):
        try:
            return cast(v) if v not in (None, "", "null") else default
        except ValueError:
            return default

    if "file" in request.files:
        f = request.files["file"]
        try:
            img = Image.open(io.BytesIO(f.read()))
        except Exception as e:
            return jsonify({"error": f"could not read image: {e}"}), 400
        tta = max(1, min(_f(request.form.get("tta"), 4, int), 16))
        out = _predict_overhead(img, _f(request.form.get("gsd")), tta)
        out["source"] = f.filename or "upload"
        return jsonify(out)

    body = request.json or {}
    s = STATE["oh"]["by_id"].get(str(body.get("sample_id")))
    if s is None:
        return jsonify({"error": "unknown sample"}), 404
    out = _predict_overhead(Image.open(s["path"]), None, 4)
    out["source"] = f"{s['label']} · {s['id']}"
    out["gt_lat"], out["gt_lon"] = s["gt_lat"], s["gt_lon"]
    out["error_m"] = float(haversine_m(np.array([out["lat"]]), np.array([out["lon"]]),
                                       np.array([s["gt_lat"]]), np.array([s["gt_lon"]]))[0])
    out["recorded_err_m"] = s["recorded_err_m"]
    return jsonify(out)


def _load_overhead(path: str, dev: torch.device) -> None:
    """Second model for overhead queries; silently absent when its run directory does not exist."""
    cfg = configure(load_config(path, []))
    if not (cfg.out_dir / "best.pt").exists() or not (cfg.out_dir / "cells.npz").exists():
        print(f"overhead model not found under {cfg.out_dir}; aerial mode disabled")
        return
    model, ckpt_cfg, _, _ = load_checkpoint(cfg.out_dir / "best.pt", dev)
    cfg.model = ckpt_cfg.model
    sp = Path(__file__).parent / "overhead_samples.json"
    samples = [{**x, "path": str((ROOT / x["path"]).resolve())} for x in json.loads(sp.read_text())] if sp.exists() else []
    STATE["oh"] = dict(
        cfg=cfg, dev=dev, model=model,
        db=CellDatabase.load(cfg.out_dir / "cells.npz"),
        idx=(ImageIndex.load(cfg.out_dir / "image_index.npz")
             if (cfg.out_dir / "image_index.npz").exists() else None),
        alpha=(json.load(open(cfg.out_dir / "calibration.json")).get("alpha", 0.0)
               if (cfg.out_dir / "calibration.json").exists() else 0.0),
        target_mpp=meters_per_pixel(cfg.bbox.center[0], cfg.overhead.eval_zoom),
        samples=samples, by_id={s["id"]: s for s in samples},
        results=(json.loads((cfg.out_dir / "results.json").read_text())
                 if (cfg.out_dir / "results.json").exists() else None),
    )
    print(f"overhead model on {dev} · {STATE['oh']['db'].size} cells · alpha={STATE['oh']['alpha']} · "
          f"{len(samples)} pre-selected overhead queries")


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/<name>.png")
def png(name):
    # index.html references the brand mark by a bare relative path so the same file works in the static
    # GitHub Pages build; Flask would otherwise only serve it under /static/.
    return send_from_directory(app.static_folder, f"{name}.png")


@app.get("/api/samples")
def samples():
    return jsonify([{k: s[k] for k in
                     ("id", "split", "gt_lat", "gt_lon", "recorded_err_m", "quantile")}
                    for s in STATE["samples"]])


@app.get("/api/image/<sid>")
def image(sid):
    s = STATE["by_id"].get(sid)
    if s is None:
        return jsonify({"error": "unknown sample"}), 404
    return send_file(s["path"], mimetype="image/jpeg")


@app.post("/api/predict")
def predict():
    if "file" in request.files:
        f = request.files["file"]
        try:
            img = Image.open(io.BytesIO(f.read()))
        except Exception as e:
            return jsonify({"error": f"could not read image: {e}"}), 400
        out = _predict(img)
        out["source"] = f.filename or "upload"
        return jsonify(out)

    sid = (request.json or {}).get("sample_id")
    s = STATE["by_id"].get(str(sid))
    if s is None:
        return jsonify({"error": "unknown sample"}), 404
    out = _predict(Image.open(s["path"]))
    out["source"] = f"{s['split']} · {s['id']}"
    out["gt_lat"], out["gt_lon"] = s["gt_lat"], s["gt_lon"]
    out["error_m"] = float(haversine_m(np.array([out["lat"]]), np.array([out["lon"]]),
                                       np.array([s["gt_lat"]]), np.array([s["gt_lon"]]))[0])
    out["recorded_err_m"] = s["recorded_err_m"]
    return jsonify(out)


@app.get("/api/meta")
def meta():
    cfg = STATE["cfg"]
    out = {"city": cfg.data.city, "cells": int(STATE["db"].size),
           "database_level": cfg.cells.database_level, "alpha": STATE["alpha"],
           "device": str(STATE["dev"]), "results": STATE["results"], "overhead": None}
    if "oh" in STATE:
        oh = STATE["oh"]
        out["overhead"] = {"cells": int(oh["db"].size), "database_level": oh["cfg"].overhead.database_level,
                           "alpha": oh["alpha"], "target_mpp": oh["target_mpp"],
                           "extent_m": oh["target_mpp"] * oh["cfg"].model.image_size,
                           "train_releases": oh["cfg"].overhead.train_releases, "results": oh["results"]}
    return jsonify(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/ankara.yaml")
    ap.add_argument("--overhead-config", default="configs/ankara_overhead.yaml",
                    help="overhead-query model (aerial mode); skipped when its run directory is missing")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()

    cfg = load_config(a.config, [])
    dev = torch.device(a.device)
    model, ckpt_cfg, _, _ = load_checkpoint(cfg.out_dir / "best.pt", dev)
    model.eval()
    STATE.update(
        cfg=cfg, dev=dev, model=model,
        tf=eval_transform(ckpt_cfg.model.image_size),
        db=CellDatabase.load(cfg.out_dir / "cells.npz"),
        idx=(ImageIndex.load(cfg.out_dir / "image_index.npz")
             if (cfg.out_dir / "image_index.npz").exists() else None),
        alpha=(json.load(open(cfg.out_dir / "calibration.json")).get("alpha", 0.0)
               if (cfg.out_dir / "calibration.json").exists() else 0.0),
        # samples.json stores repo-relative paths; Flask's send_file resolves relative paths
        # against the app root (webapp/), not the cwd, so make them absolute at load time.
        samples=[{**x, "path": str((ROOT / x["path"]).resolve())}
                 for x in json.loads((Path(__file__).parent / "samples.json").read_text())],
        results=(json.loads((cfg.out_dir / "results.json").read_text())
                 if (cfg.out_dir / "results.json").exists() else None),
    )
    STATE["by_id"] = {s["id"]: s for s in STATE["samples"]}
    if a.overhead_config:
        _load_overhead(a.overhead_config, dev)
    print(f"model on {dev} · {STATE['db'].size} cells · alpha={STATE['alpha']} · "
          f"{len(STATE['samples'])} pre-selected queries")
    print(f"open http://{a.host}:{a.port}")
    app.run(host=a.host, port=a.port, threaded=True)


if __name__ == "__main__":
    main()
