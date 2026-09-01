#!/usr/bin/env python
"""Step 8: localise arbitrary images; prints lat/lon and writes an HTML map."""
import json
from pathlib import Path

import numpy as np
from common import device, parser, setup
from torch.utils.data import DataLoader

from geoloc_tr.data import ImageListDataset
from geoloc_tr.database import CellDatabase, ImageIndex
from geoloc_tr.localize import localize
from geoloc_tr.model import load_checkpoint

MAP_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>geoloc-tr predictions</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{height:100%%;margin:0}</style></head><body><div id="map"></div><script>
const pts=%s;const map=L.map('map');L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'&copy; OpenStreetMap'}).addTo(map);
const g=L.featureGroup();pts.forEach(p=>{L.marker([p.lat,p.lon]).bindPopup(p.name+'<br>'+p.lat.toFixed(6)+', '+p.lon.toFixed(6)).addTo(g);
p.cands.forEach(c=>L.circleMarker([c[0],c[1]],{radius:4,color:'#e33',opacity:c[2]}).addTo(g));});g.addTo(map);map.fitBounds(g.getBounds().pad(0.3));
</script></body></html>"""


def main():
    p = parser(__doc__)
    p.add_argument("images", nargs="+")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--map", default="predictions.html")
    args = p.parse_args()
    cfg = setup(args)
    dev = device()
    model, ckpt_cfg, h, _ = load_checkpoint(args.checkpoint or cfg.out_dir / "best.pt", dev)
    db = CellDatabase.load(cfg.out_dir / "cells.npz")
    idx_path = cfg.out_dir / "image_index.npz"
    idx = ImageIndex.load(idx_path) if idx_path.exists() else None
    cal = cfg.out_dir / "calibration.json"
    alpha = json.load(open(cal)).get("alpha", 0.0) if cal.exists() else 0.0

    paths = [q for pat in args.images for q in (sorted(Path().glob(pat)) or [Path(pat)])]
    dl = DataLoader(ImageListDataset(paths, ckpt_cfg.model.image_size), batch_size=32)
    import torch
    with torch.no_grad():
        q = np.concatenate([model(b["image"].to(dev))[0].cpu().numpy() for b in dl])
    res = localize(q, db, cfg.retrieval, alpha, idx)
    pts = []
    for i, pth in enumerate(paths):
        lat, lon = res.latlon[i]
        sc = res.top_scores[i]
        w = np.exp((sc - sc.max()) / cfg.retrieval.refine_temperature)
        cands = [[float(db.centers[j, 0]), float(db.centers[j, 1]), float(v)] for j, v in zip(res.top_cells[i][:10], w[:10])]
        print(f"{pth}\t{lat:.6f}\t{lon:.6f}\ttop-cell {int(db.cells[res.top_cells[i, 0]])}\tscore {sc[0]:.3f}")
        pts.append({"name": str(pth), "lat": float(lat), "lon": float(lon), "cands": cands})
    Path(args.map).write_text(MAP_HTML % json.dumps(pts))
    print("map:", args.map)


if __name__ == "__main__":
    main()
