import math

from geoloc_tr.config import BBox
from geoloc_tr.mapillary import lonlat_to_tile, tile_to_lonlat, tiles_covering


def test_tile_roundtrip():
    lon, lat = 32.85, 39.92
    x, y = lonlat_to_tile(lon, lat, 14)
    lon2, lat2 = tile_to_lonlat(x, y, 14)
    assert math.isclose(lon, lon2, abs_tol=1e-9) and math.isclose(lat, lat2, abs_tol=1e-9)


def test_tiles_covering_bbox():
    bb = BBox(32.80, 39.90, 32.83, 39.92)
    tiles = tiles_covering(bb, 14)
    assert len(tiles) >= 2
    for z, x, y in tiles:
        assert z == 14
        w, n = tile_to_lonlat(x, y, z)
        e, s = tile_to_lonlat(x + 1, y + 1, z)
        # each tile overlaps the bbox
        assert e >= bb.west and w <= bb.east and n >= bb.south and s <= bb.north
