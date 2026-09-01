#!/usr/bin/env python
"""Step 7: recall@{25..5000 m} on the held-out sequences (test_seen) and held-out blocks (test_unseen)."""
import json

import numpy as np
import pandas as pd
from common import device, parser, setup

from geoloc_tr.data import GroundDataset, load_table
from geoloc_tr.database import CellDatabase, ImageIndex, embed_images
from geoloc_tr.evaluate import evaluate_queries, to_markdown
from geoloc_tr.model import load_checkpoint


def main():
    p = parser(__doc__)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--splits", nargs="+", default=["test_seen", "test_unseen"])
    args = p.parse_args()
    cfg = setup(args)
    dev = device()
    model, ckpt_cfg, h, _ = load_checkpoint(args.checkpoint or cfg.out_dir / "best.pt", dev)
    cfg.model = ckpt_cfg.model
    db = CellDatabase.load(cfg.out_dir / "cells.npz")
    idx_path = cfg.out_dir / "image_index.npz"
    idx = ImageIndex.load(idx_path) if idx_path.exists() else None
    alpha = json.load(open(cfg.out_dir / "calibration.json")).get("alpha", 0.0) if (cfg.out_dir / "calibration.json").exists() else 0.0
    report, results = [], {}
    for split in args.splits:
        df = load_table(cfg, [split])
        if not len(df):
            continue
        ds = GroundDataset(df, h, cfg.model.image_size, train=False)
        q, lg = embed_images(model, ds, cfg.train.batch_size, cfg.train.num_workers, dev, return_logits=True)
        table, errs = evaluate_queries(q, lg, np.stack([ds.lat, ds.lon], 1), h, db, cfg, alpha, idx)
        report.append(f"### {split} ({len(df)} queries)\n\n{to_markdown(table)}\n")
        results[split] = table.to_dict(orient="records")
        pd.DataFrame({"id": df["id"], **{k: v for k, v in errs.items()}}).to_parquet(cfg.out_dir / f"errors_{split}.parquet", index=False)
    md = "\n".join(report)
    (cfg.out_dir / "results.md").write_text(md)
    json.dump(results, open(cfg.out_dir / "results.json", "w"), indent=2)
    print(md)


if __name__ == "__main__":
    main()
