#!/usr/bin/env python
"""Step 2: fetch Graph-API metadata (SfM-corrected positions, thumbnails) and download images -> metadata.parquet."""
import numpy as np
import pandas as pd
from common import parser, setup

from geoloc_tr.config import mapillary_token
from geoloc_tr.mapillary import download_thumbnails, fetch_metadata


def main():
    args = parser(__doc__).parse_args()
    cfg = setup(args)
    cov = pd.read_parquet(cfg.coverage_path)
    n0 = len(cov)
    if cfg.data.min_capture_year:
        cov = cov[cov["captured_at"] >= pd.Timestamp(f"{cfg.data.min_capture_year}-01-01").value // 10 ** 6]
    if not cfg.data.include_pano:
        cov = cov[~cov["is_pano"].astype(bool)]
    if cfg.data.max_images and len(cov) > cfg.data.max_images:
        cov = cov.sample(cfg.data.max_images, random_state=0)
    print(f"{n0} coverage rows -> {len(cov)} after filters")

    tok = mapillary_token()
    meta = fetch_metadata(cov["id"].tolist(), tok, thumb_field=cfg.data.thumb_field, batch=cfg.data.graph_batch,
                          workers=cfg.data.download_workers)
    # keep tile-derived attributes the entity endpoint may omit
    meta = meta.merge(cov[["id", "sequence", "captured_at", "compass_angle", "is_pano"]].rename(
        columns={"sequence": "sequence_tile", "captured_at": "captured_at_tile", "compass_angle": "compass_tile",
                 "is_pano": "is_pano_tile"}), on="id", how="left")
    for a, b in (("sequence", "sequence_tile"), ("captured_at", "captured_at_tile"), ("compass_angle", "compass_tile"),
                 ("is_pano", "is_pano_tile")):
        if a not in meta:
            meta[a] = meta[b]
        else:
            meta[a] = meta[a].where(meta[a].notna(), meta[b])
    meta = meta.drop(columns=[c for c in meta.columns if c.endswith("_tile")])
    meta = meta.dropna(subset=["lat", "lon"])

    out = download_thumbnails(meta, cfg.images_dir, thumb_field=cfg.data.thumb_field, max_side=cfg.data.image_max_side,
                              include_pano=cfg.data.include_pano, n_pano_crops=cfg.data.pano_crops,
                              workers=cfg.data.thumb_workers)
    if "compass_angle" in out and "yaw_offset" in out:
        out["heading"] = (out["compass_angle"].astype(float).fillna(0) + out["yaw_offset"]) % 360
    out["sequence"] = out["sequence"].astype(str)
    out.to_parquet(cfg.metadata_path, index=False)
    print(f"{len(out)} image files stored under {cfg.images_dir} -> {cfg.metadata_path}")


if __name__ == "__main__":
    main()
