"""Offline end-to-end run of the overhead-query pipeline on synthetic tiles."""
import numpy as np
import pytest
import torch

from geoloc_tr.config import load_config
from geoloc_tr.database import ImageIndex
from geoloc_tr.evaluate import calibrate_alpha, evaluate_queries
from geoloc_tr.geo import cell_edge_m, cell_ids, haversine_m
from geoloc_tr.model import load_checkpoint
from geoloc_tr.overhead import (OverheadQueryDataset, OverheadTrainDataset, PointSampler, bbox_cells,
                                build_overhead_database, configure, embed_views, overhead_hierarchy, prefetch_bbox,
                                render_view, tile_caches, tile_change_fraction, training_sources, urban_rects)
from geoloc_tr.train import fit


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("overhead")
    return configure(load_config("configs/synthetic.yaml", {
        "data.root": str(root / "data"), "train.out_dir": str(root / "run"), "train.epochs": 2, "train.batch_size": 32,
        "model.aerial_enabled": False, "overhead.levels": [14, 16], "overhead.database_level": 17,
        "overhead.zooms": [16, 17, 18], "overhead.eval_zoom": 17, "overhead.samples_per_epoch": 96,
        "overhead.val_queries": 24, "overhead.eval_queries": 40, "overhead.eval_releases": [7],
        "overhead.train_releases": [3], "overhead.release_zooms": [16, 17],
    }))


def test_bbox_cells_and_hierarchy(cfg):
    cells = bbox_cells(cfg.bbox, 17)
    b = cfg.bbox
    area = haversine_m(b.south, b.west, b.south, b.east) * haversine_m(b.south, b.west, b.north, b.west)
    expected = area / cell_edge_m(17) ** 2
    assert 0.5 * expected < len(cells) < 2.0 * expected
    h, db = overhead_hierarchy(cfg.bbox, [14, 16], 17)
    assert h.level_values == [14, 16] and len(db) == len(cells)
    assert h.levels[-1].counts.sum() == len(db)
    lat, lon = PointSampler(cfg.bbox, None, 0.0).sample_many(200, 0)
    labels = h.labels(lat, lon)
    assert (labels >= 0).mean() > 0.95  # only cells straddling the bbox edge are missing
    fine = cell_ids(lat, lon, 16)
    ok = labels[:, 1] >= 0
    assert np.array_equal(h.levels[-1].cell_ids[labels[ok, 1]], fine[ok])


def test_render_view_geometry(cfg):
    caches = tile_caches(cfg)
    lat, lon = cfg.bbox.center
    plain = render_view(caches[17], lat, lon, 64)
    assert plain.size == (64, 64)
    rot = np.asarray(render_view(caches[17], lat, lon, 64, rotation_deg=90.0))
    # a 90-degree rotation of a north-up view is (up to resampling) the rotated north-up view
    ref = np.rot90(np.asarray(plain))
    assert np.abs(rot[8:-8, 8:-8].astype(float) - ref[8:-8, 8:-8].astype(float)).mean() < 12
    assert render_view(caches[17], lat, lon, 64, rotation_deg=33.0, scale=1.2).size == (64, 64)


def test_overhead_pipeline(cfg):
    caches = tile_caches(cfg)
    for c in caches.values():
        got, failed = prefetch_bbox(c, cfg.bbox, workers=4, progress=False)
        assert failed == 0
    h, _ = overhead_hierarchy(cfg.bbox, cfg.overhead.levels, cfg.overhead.database_level)
    assert urban_rects(cfg) is None  # no image table in this fixture -> uniform sampling only
    sampler = PointSampler(cfg.bbox, None, cfg.overhead.urban_frac)
    sources = training_sources(cfg)
    assert len(sources) == 2 and sorted(sources[1]) == [16, 17]
    train_ds = OverheadTrainDataset(cfg, h, sources, sampler)
    item = train_ds[0]
    assert item["image"].shape == (3, cfg.model.image_size, cfg.model.image_size) and item["labels"].shape == (2,)
    val_ds = OverheadQueryDataset.sample(cfg, caches, sampler, cfg.overhead.val_queries, 11)
    dev = torch.device("cpu")
    _, ckpt = fit(cfg, train_ds, val_ds, h, dev, pretrained=False, use_aerial=False)
    model, cfg2, h2, _ = load_checkpoint(ckpt, dev)
    assert h2.level_values == h.level_values and model.aerial is None

    db, idx = build_overhead_database(model, h2, cfg, caches, dev, progress=False)
    assert db.level == 17 and db.aerial is not None and db.aerial.shape == db.ground.shape
    assert isinstance(idx, ImageIndex) and len(idx.emb) == db.size
    assert np.allclose(np.linalg.norm(db.aerial, axis=1), 1.0, atol=1e-4)

    test_ds = OverheadQueryDataset.sample(cfg, caches, sampler, cfg.overhead.eval_queries, 22)
    q, lg = embed_views(model, test_ds, cfg, dev, progress=False)
    assert q.shape == (len(test_ds), cfg.model.embed_dim) and lg.shape == (len(test_ds), h2.finest.num_classes)
    q4, _ = embed_views(model, test_ds, cfg, dev, rotations=4, progress=False)
    assert q4.shape == q.shape and np.allclose(np.linalg.norm(q4, axis=1), 1.0, atol=1e-4)
    gt = np.stack([test_ds.lat, test_ds.lon], 1)
    alpha = calibrate_alpha(*embed_views(model, val_ds, cfg, dev, progress=False)[:1], np.stack([val_ds.lat, val_ds.lon], 1), db, cfg)
    table, errs = evaluate_queries(q, lg, gt, h2, db, cfg, alpha, idx)
    assert set(table["method"]) >= {"image retrieval (NN)", "prototypes + top-k refine", "full: proto + aerial + image refine"}
    assert all(np.isfinite(e).all() for e in errs.values())

    # a "wayback" source with the synthetic template renders identical tiles -> nothing changed
    wb = tile_caches(cfg, cfg.overhead.eval_releases[0])
    wb_ds = test_ds.with_caches(wb)
    assert wb[17].root != caches[17].root
    for z, tiles in wb_ds.tiles().items():
        assert wb[z].zoom == z and len(tiles) > 0
        for x, y in tiles:
            wb[z].fetch(x, y)
    assert tile_change_fraction(caches[17], wb[17], wb_ds.tiles()[17]) == 0.0
