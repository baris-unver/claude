#!/usr/bin/env python
"""Generate the offline synthetic city used by the tests and the smoke run (configs/synthetic.yaml)."""
from common import parser, setup

from geoloc_tr.synthetic import make_synthetic_city


def main():
    p = parser(__doc__)
    p.add_argument("--streets", type=int, default=40)
    p.add_argument("--points", type=int, default=60)
    p.add_argument("--passes", type=int, default=4)
    args = p.parse_args()
    cfg = setup(args)
    df = make_synthetic_city(cfg.data_dir, cfg.bbox, n_streets=args.streets, points_per_street=args.points,
                             image_size=cfg.model.image_size, seed=cfg.split.seed, passes_per_street=args.passes)
    print(f"{len(df)} synthetic images -> {cfg.metadata_path}")


if __name__ == "__main__":
    main()
