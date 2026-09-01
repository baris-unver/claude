"""Geodesy helpers and the S2-cell class hierarchy used for proxy classification."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import s2sphere

EARTH_RADIUS_M = 6_371_008.8
# S2 average edge length of a level-0 cell in radians (from the reference S2 metrics table).
_AVG_EDGE_L0_RAD = 1.459


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres; broadcasts over numpy arrays."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(a, dtype=np.float64)) for a in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def latlon_to_unit(lat, lon) -> np.ndarray:
    """(lat, lon) degrees -> unit vectors, shape (..., 3)."""
    lat = np.radians(np.asarray(lat, dtype=np.float64))
    lon = np.radians(np.asarray(lon, dtype=np.float64))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def unit_to_latlon(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, dtype=np.float64)
    n = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz = xyz / np.clip(n, 1e-12, None)
    lat = np.degrees(np.arcsin(np.clip(xyz[..., 2], -1, 1)))
    lon = np.degrees(np.arctan2(xyz[..., 1], xyz[..., 0]))
    return lat, lon


def cell_edge_m(level: int) -> float:
    """Approximate average edge length of an S2 cell at `level`."""
    return _AVG_EDGE_L0_RAD / (2 ** level) * EARTH_RADIUS_M


def cell_id(lat: float, lon: float, level: int) -> int:
    ll = s2sphere.LatLng.from_degrees(float(lat), float(lon))
    return s2sphere.CellId.from_lat_lng(ll).parent(level).id()


def cell_ids(lat, lon, level: int) -> np.ndarray:
    lat = np.asarray(lat, dtype=np.float64).ravel()
    lon = np.asarray(lon, dtype=np.float64).ravel()
    return np.fromiter((cell_id(a, b, level) for a, b in zip(lat, lon)), dtype=np.uint64, count=len(lat))


def cell_level(cid: int) -> int:
    return s2sphere.CellId(int(cid)).level()


def cell_center(cid: int) -> tuple[float, float]:
    ll = s2sphere.CellId(int(cid)).to_lat_lng()
    return ll.lat().degrees, ll.lng().degrees


def cell_parent(cid: int, level: int) -> int:
    c = s2sphere.CellId(int(cid))
    if c.level() == level:
        return int(cid)
    if c.level() < level:
        raise ValueError(f"cell level {c.level()} is coarser than requested parent level {level}")
    return c.parent(level).id()


def cell_children(cid: int, level: int) -> list[int]:
    c = s2sphere.CellId(int(cid))
    if c.level() == level:
        return [int(cid)]
    if c.level() > level:
        raise ValueError("children level must be finer than the cell level")
    return [ch.id() for ch in c.children(level)]


def cell_vertices(cid: int) -> list[tuple[float, float]]:
    """Corner (lat, lon) list of a cell, useful for plotting."""
    cell = s2sphere.Cell(s2sphere.CellId(int(cid)))
    out = []
    for k in range(4):
        ll = s2sphere.LatLng.from_point(cell.get_vertex(k))
        out.append((ll.lat().degrees, ll.lng().degrees))
    return out


@dataclass
class LevelClasses:
    """Classes (S2 cells) of one hierarchy level."""

    level: int
    cell_ids: np.ndarray  # (C,) uint64, sorted
    centers: np.ndarray  # (C, 2) lat/lon: mean position of training images in the cell
    counts: np.ndarray  # (C,)

    @property
    def num_classes(self) -> int:
        return len(self.cell_ids)

    def index_of(self, cids: np.ndarray) -> np.ndarray:
        """Map cell ids -> class index (or -1 when not a class)."""
        cids = np.asarray(cids, dtype=np.uint64)
        pos = np.searchsorted(self.cell_ids, cids)
        pos = np.clip(pos, 0, max(len(self.cell_ids) - 1, 0))
        ok = (len(self.cell_ids) > 0) & (self.cell_ids[pos] == cids)
        return np.where(ok, pos, -1).astype(np.int64)

    @property
    def sigma_m(self) -> float:
        return cell_edge_m(self.level)


@dataclass
class CellHierarchy:
    levels: list[LevelClasses]

    @classmethod
    def build(cls, lat: np.ndarray, lon: np.ndarray, levels: list[int], min_images_per_class: int = 1) -> "CellHierarchy":
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        out = []
        for level in sorted(levels):
            cids = cell_ids(lat, lon, level)
            uniq, inv, counts = np.unique(cids, return_inverse=True, return_counts=True)
            keep = counts >= min_images_per_class
            sums = np.zeros((len(uniq), 2))
            np.add.at(sums, inv, np.stack([lat, lon], 1))
            centers = sums / counts[:, None]
            out.append(LevelClasses(level, uniq[keep], centers[keep], counts[keep]))
        return cls(out)

    @property
    def level_values(self) -> list[int]:
        return [lc.level for lc in self.levels]

    @property
    def finest(self) -> LevelClasses:
        return self.levels[-1]

    def labels(self, lat, lon) -> np.ndarray:
        """(N, L) class indices per level, -1 when the image falls in a dropped cell."""
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        cols = [lc.index_of(cell_ids(lat, lon, lc.level)) for lc in self.levels]
        return np.stack(cols, axis=1)

    def to_dict(self) -> dict:
        return {
            "levels": [
                {"level": lc.level, "cell_ids": lc.cell_ids.astype(np.uint64), "centers": lc.centers,
                 "counts": lc.counts}
                for lc in self.levels
            ]
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CellHierarchy":
        return cls([LevelClasses(int(x["level"]), np.asarray(x["cell_ids"], dtype=np.uint64),
                                 np.asarray(x["centers"], dtype=np.float64), np.asarray(x["counts"]))
                    for x in d["levels"]])


def upsample_cells(class_cells: np.ndarray, target_level: int) -> tuple[np.ndarray, np.ndarray]:
    """Expand class cells to all their descendants at `target_level`.

    Returns (target_cell_ids, parent_index) where parent_index[i] is the row in `class_cells` that
    target cell i descends from. This is the S2-hierarchy "prototype upsampling" step: every fine cell
    inherits the prototype of the class cell it lies in.
    """
    tgt, parent = [], []
    for i, cid in enumerate(class_cells):
        ch = cell_children(int(cid), target_level)
        tgt.extend(ch)
        parent.extend([i] * len(ch))
    return np.asarray(tgt, dtype=np.uint64), np.asarray(parent, dtype=np.int64)


def cells_centers(cids: np.ndarray) -> np.ndarray:
    return np.asarray([cell_center(int(c)) for c in cids], dtype=np.float64).reshape(-1, 2)
