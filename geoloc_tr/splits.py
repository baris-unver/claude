"""Train / val / test splits that respect the structure of street-level sequences.

* `test_unseen`: whole S2 blocks (default level 13, ~1 km) are held out -> measures generalisation to
  areas without ground-level training data (this is where aerial codes help).
* `test_seen` / `val`: whole *sequences* held out inside trained blocks -> the main protocol
  (a different drive through a known area, like the paper's held-out query sets).
* `train`: the rest, thinned so consecutive frames are at least `min_train_spacing_m` apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .geo import cell_ids, haversine_m


def thin_sequences(df: pd.DataFrame, spacing_m: float) -> np.ndarray:
    """Boolean keep-mask: greedy per-sequence thinning by capture order."""
    keep = np.zeros(len(df), dtype=bool)
    if spacing_m <= 0:
        keep[:] = True
        return keep
    order = np.lexsort((df["captured_at"].to_numpy(), df["sequence"].to_numpy()))
    seq = df["sequence"].to_numpy()[order]
    lat = df["lat"].to_numpy()[order]
    lon = df["lon"].to_numpy()[order]
    last_lat = last_lon = None
    last_seq = None
    for k in range(len(order)):
        if seq[k] != last_seq:
            last_seq, last_lat, last_lon = seq[k], lat[k], lon[k]
            keep[order[k]] = True
            continue
        if haversine_m(last_lat, last_lon, lat[k], lon[k]) >= spacing_m:
            keep[order[k]] = True
            last_lat, last_lon = lat[k], lon[k]
    return keep


def make_splits(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    sc = cfg.split
    rng = np.random.default_rng(sc.seed)
    df = df.reset_index(drop=True).copy()
    df["sequence"] = df["sequence"].astype(str)

    blocks = cell_ids(df["lat"].to_numpy(), df["lon"].to_numpy(), sc.block_level)
    uniq_blocks = np.unique(blocks)
    n_unseen = int(round(len(uniq_blocks) * sc.unseen_block_frac))
    unseen = set(rng.choice(uniq_blocks, size=n_unseen, replace=False).tolist()) if n_unseen else set()
    split = np.full(len(df), "train", dtype=object)
    in_unseen = np.array([b in unseen for b in blocks.tolist()])
    split[in_unseen] = "test_unseen"

    seqs = np.unique(df["sequence"].to_numpy()[~in_unseen])
    seqs = rng.permutation(seqs)
    n_val = int(round(len(seqs) * sc.seq_val_frac))
    n_test = int(round(len(seqs) * sc.seq_test_frac))
    val_seqs = set(seqs[:n_val].tolist())
    test_seqs = set(seqs[n_val:n_val + n_test].tolist())
    for i, s in enumerate(df["sequence"].tolist()):
        if in_unseen[i]:
            continue
        if s in val_seqs:
            split[i] = "val"
        elif s in test_seqs:
            split[i] = "test_seen"
    df["split"] = split
    df["block"] = blocks

    keep = np.ones(len(df), dtype=bool)
    tr = df["split"] == "train"
    keep[tr.to_numpy()] = thin_sequences(df[tr], cfg.data.min_train_spacing_m)
    for name in ("val", "test_seen", "test_unseen"):
        m = (df["split"] == name).to_numpy()
        if m.any():
            keep[m] = thin_sequences(df[m], sc.eval_spacing_m)
    df["keep"] = keep
    return df


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["keep"]]
    return d.groupby("split").agg(images=("id", "size"), sequences=("sequence", "nunique")).reset_index()
