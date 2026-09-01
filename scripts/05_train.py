#!/usr/bin/env python
"""Step 5: train the hierarchical proxy classifier (+ aerial encoder)."""
from common import device, parser, setup

from geoloc_tr.aerial import make_aerial_store
from geoloc_tr.data import load_table
from geoloc_tr.geo import CellHierarchy
from geoloc_tr.train import train


def main():
    p = parser(__doc__)
    p.add_argument("--no-pretrained", action="store_true", help="random-init backbone (offline)")
    args = p.parse_args()
    cfg = setup(args)
    train_df = load_table(cfg, ["train"])
    val_df = load_table(cfg, ["val"])
    h = CellHierarchy.build(train_df["lat"].to_numpy(), train_df["lon"].to_numpy(), cfg.cells.hierarchy_levels,
                            cfg.cells.min_images_per_class)
    print("classes per level:", {lc.level: lc.num_classes for lc in h.levels}, f"train={len(train_df)} val={len(val_df)}")
    store = make_aerial_store(cfg) if cfg.model.aerial_enabled else None
    _, ckpt = train(cfg, train_df, val_df, h, store, device(), pretrained=not args.no_pretrained)
    print("best checkpoint:", ckpt)


if __name__ == "__main__":
    main()
