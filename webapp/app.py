#!/usr/bin/env python
"""Local web app for testing the trained geoloc-tr model: pick a pre-selected query or upload your
own, and see where the model thinks it was taken, on a map, against the ground truth.

    python webapp/app.py -c configs/ankara.yaml [--device cpu] [--port 8000]

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


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


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
    return jsonify({"city": cfg.data.city, "cells": int(STATE["db"].size),
                    "database_level": cfg.cells.database_level, "alpha": STATE["alpha"],
                    "device": str(STATE["dev"]), "results": STATE["results"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/ankara.yaml")
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
    print(f"model on {dev} · {STATE['db'].size} cells · alpha={STATE['alpha']} · "
          f"{len(STATE['samples'])} pre-selected queries")
    print(f"open http://{a.host}:{a.port}")
    app.run(host=a.host, port=a.port, threaded=True)


if __name__ == "__main__":
    main()
