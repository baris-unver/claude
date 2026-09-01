"""End-to-end offline run: synthetic city -> splits -> train tiny model -> database -> evaluation."""
import numpy as np
import pandas as pd
import pytest
import torch

from geoloc_tr.aerial import make_aerial_store
from geoloc_tr.config import load_config
from geoloc_tr.data import GroundDataset
from geoloc_tr.database import build_database, build_image_index, embed_images
from geoloc_tr.evaluate import calibrate_alpha, evaluate_queries
from geoloc_tr.geo import CellHierarchy
from geoloc_tr.model import load_checkpoint
from geoloc_tr.splits import make_splits
from geoloc_tr.synthetic import make_synthetic_city
from geoloc_tr.train import train


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    cfg = load_config("configs/synthetic.yaml", {"data.root": str(root / "data"), "train.out_dir": str(root / "run"),
                                                 "train.epochs": 4})
    make_synthetic_city(cfg.data_dir, cfg.bbox, n_streets=24, points_per_street=40, image_size=cfg.model.image_size)
    df = make_splits(pd.read_parquet(cfg.metadata_path), cfg)
    df.to_parquet(cfg.splits_path, index=False)
    return cfg


def test_splits_structure(cfg):
    df = pd.read_parquet(cfg.splits_path)
    assert set(df["split"]) >= {"train", "val", "test_seen", "test_unseen"}
    tr = df[df.split == "train"]
    for name in ("val", "test_seen"):
        assert not set(df[df.split == name]["sequence"]) & set(tr["sequence"])
    assert not set(df[df.split == "test_unseen"]["block"]) & set(tr["block"])


def test_train_database_evaluate(cfg):
    df = pd.read_parquet(cfg.splits_path)
    df = df[df["keep"]]
    train_df, val_df = df[df.split == "train"], df[df.split == "val"]
    h = CellHierarchy.build(train_df["lat"].to_numpy(), train_df["lon"].to_numpy(), cfg.cells.hierarchy_levels,
                            cfg.cells.min_images_per_class)
    store = make_aerial_store(cfg)
    dev = torch.device("cpu")
    _, ckpt = train(cfg, train_df, val_df, h, store, dev, pretrained=False)
    model, cfg2, h2, extra = load_checkpoint(ckpt, dev)
    assert h2.level_values == h.level_values

    db = build_database(model, h2, cfg, store, dev, progress=False)
    assert db.level == cfg.cells.database_level and db.aerial is not None and db.aerial.shape == db.ground.shape
    assert db.size == 4 * h2.finest.num_classes
    idx = build_image_index(model, train_df, h2, cfg, dev, progress=False)

    test_df = df[df.split == "test_seen"]
    ds = GroundDataset(test_df, h2, cfg.model.image_size, train=False)
    q, lg = embed_images(model, ds, 64, 0, dev, return_logits=True, progress=False)
    vds = GroundDataset(val_df, h2, cfg.model.image_size, train=False)
    alpha = calibrate_alpha(embed_images(model, vds, 64, 0, dev, progress=False), np.stack([vds.lat, vds.lon], 1), db, cfg)
    table, errs = evaluate_queries(q, lg, np.stack([ds.lat, ds.lon], 1), h2, db, cfg, alpha, idx)
    print("\n", table.to_string(index=False))
    by = table.set_index("method")
    full = by.loc["full: proto + aerial + image refine"]
    cls = by.loc[f"classification@L{h2.finest.level}"]
    # A random guess inside a 2.5 x 2.2 km box lands within 100 m of the truth ~0.5% of the time.
    assert cls["R@100m"] > 40
    assert full["R@100m"] > 50
    assert full["median_m"] < 100
    # image-level refinement should sharpen the cell estimate at the finest threshold
    assert full["R@25m"] >= by.loc["prototypes + top-k refine"]["R@25m"]
    assert 0.0 <= alpha <= max(cfg.retrieval.calibration_grid)
