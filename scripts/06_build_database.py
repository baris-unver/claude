#!/usr/bin/env python
"""Step 6: prototypes -> S2-upsampled cell database (+ aerial codes), training-image index, aerial calibration."""
import json

import numpy as np
from common import device, parser, setup

from geoloc_tr.aerial import make_aerial_store
from geoloc_tr.data import GroundDataset, load_table
from geoloc_tr.database import build_database, build_image_index, embed_images
from geoloc_tr.evaluate import calibrate_alpha
from geoloc_tr.model import load_checkpoint


def main():
    p = parser(__doc__)
    p.add_argument("--checkpoint", default=None, help="defaults to <out_dir>/best.pt")
    args = p.parse_args()
    cfg = setup(args)
    dev = device()
    ckpt = args.checkpoint or cfg.out_dir / "best.pt"
    model, ckpt_cfg, h, _ = load_checkpoint(ckpt, dev)
    cfg.model = ckpt_cfg.model
    store = make_aerial_store(cfg) if model.aerial is not None else None
    db = build_database(model, h, cfg, store, dev)
    db.save(cfg.out_dir / "cells.npz")
    print(f"database: {db.size} cells at S2 level {db.level}, aerial={'yes' if db.aerial is not None else 'no'}")
    idx = build_image_index(model, load_table(cfg, ["train"]), h, cfg, dev)
    idx.save(cfg.out_dir / "image_index.npz")
    val = load_table(cfg, ["val"])
    alpha = 0.0
    if len(val) and db.aerial is not None:
        ds = GroundDataset(val, h, cfg.model.image_size, train=False)
        q = embed_images(model, ds, cfg.train.batch_size, cfg.train.num_workers, dev)
        alpha = calibrate_alpha(q, np.stack([ds.lat, ds.lon], 1), db, cfg)
    json.dump({"alpha": alpha}, open(cfg.out_dir / "calibration.json", "w"))
    print(f"aerial calibration factor alpha={alpha}")


if __name__ == "__main__":
    main()
