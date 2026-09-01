#!/usr/bin/env python
"""Step 3: spatial-block + sequence-aware train/val/test splits -> splits.parquet."""
import pandas as pd
from common import parser, setup

from geoloc_tr.splits import make_splits, split_summary


def main():
    args = parser(__doc__).parse_args()
    cfg = setup(args)
    df = pd.read_parquet(cfg.metadata_path)
    df = make_splits(df, cfg)
    df.to_parquet(cfg.splits_path, index=False)
    print(split_summary(df).to_string(index=False))


if __name__ == "__main__":
    main()
