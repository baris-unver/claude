import numpy as np

from geoloc_tr import geo


def test_haversine_known_distance():
    # Ankara (Kızılay) -> Istanbul (Taksim) is ~350 km
    d = geo.haversine_m(39.9208, 32.8541, 41.0370, 28.9850)
    assert 340_000 < d < 360_000
    assert geo.haversine_m(39.9, 32.8, 39.9, 32.8) == 0


def test_unit_roundtrip():
    lat = np.array([39.92, -10.0, 60.5])
    lon = np.array([32.85, 170.0, -120.0])
    la, lo = geo.unit_to_latlon(geo.latlon_to_unit(lat, lon))
    assert np.allclose(la, lat) and np.allclose(lo, lon)


def test_cell_hierarchy_and_upsampling():
    rng = np.random.default_rng(0)
    lat = 39.92 + rng.normal(0, 0.005, 500)
    lon = 32.85 + rng.normal(0, 0.005, 500)
    h = geo.CellHierarchy.build(lat, lon, [13, 15, 17], min_images_per_class=1)
    assert h.level_values == [13, 15, 17]
    labels = h.labels(lat, lon)
    assert labels.shape == (500, 3)
    assert (labels >= 0).all()
    # with a minimum count, sparsely populated fine cells are dropped and their images get -1
    h_min = geo.CellHierarchy.build(lat, lon, [13, 15, 17], min_images_per_class=2)
    assert h_min.finest.num_classes < h.finest.num_classes
    assert (h_min.labels(lat, lon)[:, 2] >= 0).sum() == h_min.finest.counts.sum()
    for li, lc in enumerate(h.levels):
        valid = labels[:, li] >= 0
        assert (labels[valid, li] < lc.num_classes).all()
        # class centres lie inside the cell they describe
        cids = geo.cell_ids(lc.centers[:, 0], lc.centers[:, 1], lc.level)
        assert (cids == lc.cell_ids).all()
    cells, parent = geo.upsample_cells(h.finest.cell_ids, 19)
    assert len(cells) == 16 * h.finest.num_classes
    assert (np.bincount(parent) == 16).all()
    for c, p in zip(cells[:40], parent[:40]):
        assert geo.cell_parent(int(c), 17) == int(h.finest.cell_ids[p])
    # roundtrip through dict
    h2 = geo.CellHierarchy.from_dict(h.to_dict())
    assert (h2.labels(lat, lon) == labels).all()


def test_cell_edge_scale():
    assert 60 < geo.cell_edge_m(17) < 80
    assert abs(geo.cell_edge_m(16) / geo.cell_edge_m(17) - 2) < 1e-9
