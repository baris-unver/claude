#!/usr/bin/env python
"""Step 4 (optional): download aerial tiles and pre-cut one patch per database cell."""
import numpy as np
from common import parser, setup

from geoloc_tr.aerial import make_aerial_store
from geoloc_tr.data import load_table
from geoloc_tr.geo import CellHierarchy, cell_ids, cells_centers, upsample_cells


def main():
    args = parser(__doc__).parse_args()
    cfg = setup(args)
    store = make_aerial_store(cfg)
    if store is None:
        print("aerial.enabled is false; nothing to do")
        return
    train = load_table(cfg, ["train"])
    h = CellHierarchy.build(train["lat"].to_numpy(), train["lon"].to_numpy(), cfg.cells.hierarchy_levels,
                            cfg.cells.min_images_per_class)
    cells, _ = upsample_cells(h.finest.cell_ids, cfg.cells.database_level)
    # also the cells of every training image (some fall in dropped classes) so training never stalls on I/O
    all_imgs = load_table(cfg)
    img_cells = np.unique(cell_ids(all_imgs["lat"].to_numpy(), all_imgs["lon"].to_numpy(), cfg.cells.database_level))
    cells = np.unique(np.concatenate([cells, img_cells]))
    centers = cells_centers(cells)
    n = store.cache.prefetch(centers.tolist(), cfg.aerial.patch_px, workers=cfg.aerial.workers)
    print(f"{len(cells)} database cells at S2 level {cfg.cells.database_level}; fetched {n} new tiles")
    store.build(cells, centers)
    print(f"patches under {store.patch_dir}")


if __name__ == "__main__":
    main()
