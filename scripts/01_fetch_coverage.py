#!/usr/bin/env python
"""Step 1: enumerate all Mapillary images in the city bbox (coverage vector tiles) -> coverage.parquet."""
from common import parser, setup

from geoloc_tr.config import mapillary_token
from geoloc_tr.mapillary import fetch_coverage, fetch_coverage_bbox_search


def main():
    p = parser(__doc__)
    p.add_argument("--method", choices=["tiles", "bbox"], default="tiles")
    args = p.parse_args()
    cfg = setup(args)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    tok = mapillary_token()
    if args.method == "tiles":
        df = fetch_coverage(cfg.bbox, tok, z=cfg.data.tile_zoom, workers=cfg.data.download_workers)
    else:
        df = fetch_coverage_bbox_search(cfg.bbox, tok)
    df.to_parquet(cfg.coverage_path, index=False)
    print(f"{len(df)} images, {df['sequence'].nunique() if len(df) else 0} sequences, "
          f"{int(df['is_pano'].sum()) if len(df) else 0} panoramas -> {cfg.coverage_path}")


if __name__ == "__main__":
    main()
