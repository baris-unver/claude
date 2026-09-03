"""Overhead-query localisation: satellite / aerial photos as *queries*.

The paper (and the ground pipeline in this repo) only use aerial tiles on the database side. This module
trains the same proxy-classification model on overhead views instead, so a nadir satellite or aircraft
photo can be localised inside the city:

* Training samples are random crops of the aerial tile pyramid at random positions inside the bbox, from
  a random acquisition date (current tiles + Esri Wayback snapshots), at a random native zoom (scale),
  random rotation and colour jitter, labelled by the S2 cells that contain the crop centre. Every cell
  of the bbox is a class, so there are no coverage gaps.
* The database is the finest-level prototypes upsampled to the database level, plus one encoder code per
  database cell (stored in the ``aerial`` slot of :class:`CellDatabase`). The same codes are also the
  ``ImageIndex`` used for the nearest-cell baseline and for refinement.
* Held-out evaluation uses (a) points never seen as crop centres, and (b) the same points rendered from
  historical Esri Wayback releases, i.e. imagery acquired on a different date than the training tiles.
"""
from __future__ import annotations

import hashlib
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import s2sphere
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .aerial import TILE_PX, TileCache, meters_per_pixel
from .config import BBox, Config
from .data import IMAGENET_MEAN, IMAGENET_STD, eval_transform
from .database import CellDatabase, ImageIndex, build_database
from .localize import localize
from .geo import (CellHierarchy, LevelClasses, cell_center, cell_children, cell_edge_m, cell_ids, cell_parent,
                  cell_vertices, cells_centers, latlon_to_unit, upsample_cells)
from .mapillary import lonlat_to_tile
from .model import GeoLocModel

log = logging.getLogger(__name__)


def configure(cfg: Config) -> Config:
    """Make the shared cell/model settings consistent with the overhead ones (single query encoder, overhead
    class levels and database level) so that train/database/evaluate code reading cfg.cells works unchanged."""
    cfg.model.aerial_enabled = False
    cfg.cells.hierarchy_levels = list(cfg.overhead.levels)
    cfg.cells.database_level = cfg.overhead.database_level
    return cfg


# ---------------------------------------------------------------------------------------------
# cells of a bbox
# ---------------------------------------------------------------------------------------------
def bbox_cells(bbox: BBox, level: int) -> np.ndarray:
    """All S2 cells at `level` whose centre lies inside the bbox (sorted uint64)."""
    top = min(level, 10)
    cov = s2sphere.RegionCoverer()
    cov.min_level = cov.max_level = top
    cov.max_cells = 100_000
    rect = s2sphere.LatLngRect(s2sphere.LatLng.from_degrees(bbox.south, bbox.west),
                               s2sphere.LatLng.from_degrees(bbox.north, bbox.east))
    out = []
    for c in cov.get_covering(rect):
        for ch in cell_children(c.id(), level):
            lat, lon = cell_center(ch)
            if bbox.contains(lon, lat):
                out.append(ch)
    return np.unique(np.asarray(out, dtype=np.uint64))


def overhead_hierarchy(bbox: BBox, levels: list[int], db_level: int) -> tuple[CellHierarchy, np.ndarray]:
    """Class hierarchy over *every* cell of the bbox (counts = number of database cells per class)."""
    db = bbox_cells(bbox, db_level)
    lcs = []
    for lv in sorted(levels):
        if lv > db_level:
            raise ValueError("overhead.levels must be <= overhead.database_level")
        parents = np.fromiter((cell_parent(int(c), lv) for c in db), dtype=np.uint64, count=len(db))
        uniq, counts = np.unique(parents, return_counts=True)
        lcs.append(LevelClasses(lv, uniq, cells_centers(uniq), counts))
    return CellHierarchy(lcs), db


def urban_rects(cfg: Config) -> np.ndarray | None:
    """(K,4) [lat_min, lat_max, lon_min, lon_max] of the level-`urban_level` cells that contain Mapillary
    images, i.e. the built-up part of the bbox. None when no image table exists yet."""
    path = cfg.splits_path if cfg.splits_path.exists() else cfg.metadata_path
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["lat", "lon"])
    cids = np.unique(cell_ids(df["lat"].to_numpy(), df["lon"].to_numpy(), cfg.overhead.urban_level))
    rects = []
    for c in cids:
        v = np.asarray(cell_vertices(int(c)))
        rects.append([v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max()])
    return np.asarray(rects, dtype=np.float64)


class PointSampler:
    """Random query positions: with probability `urban_frac` inside a built-up cell, else uniform in bbox."""

    def __init__(self, bbox: BBox, rects: np.ndarray | None, urban_frac: float):
        self.bbox = bbox
        self.rects = rects if rects is not None and len(rects) else None
        self.urban_frac = urban_frac if self.rects is not None else 0.0

    def sample(self, rng: np.random.Generator) -> tuple[float, float]:
        return self.sample_flag(rng)[:2]

    def sample_flag(self, rng: np.random.Generator) -> tuple[float, float, bool]:
        """(lat, lon, in a built-up cell)"""
        if self.rects is not None and rng.random() < self.urban_frac:
            r = self.rects[rng.integers(len(self.rects))]
            return float(rng.uniform(r[0], r[1])), float(rng.uniform(r[2], r[3])), True
        b = self.bbox
        return float(rng.uniform(b.south, b.north)), float(rng.uniform(b.west, b.east)), False

    def sample_many(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        pts = np.asarray([self.sample(rng) for _ in range(n)], dtype=np.float64).reshape(-1, 2)
        return pts[:, 0], pts[:, 1]


# ---------------------------------------------------------------------------------------------
# tiles and views
# ---------------------------------------------------------------------------------------------
def tile_caches(cfg: Config, release: int | None = None, zooms: list[int] | None = None) -> dict[int, TileCache]:
    """One TileCache per zoom of the pyramid. `release` selects an Esri Wayback release (a dated snapshot of
    World Imagery) instead of the live tiles; the synthetic template renders offline in both cases."""
    zooms = sorted(set(zooms) if zooms is not None else set(cfg.overhead.zooms) | {cfg.overhead.eval_zoom})
    synthetic = cfg.aerial.url_template == "synthetic"
    if release is None:
        root, tmpl = cfg.aerial_dir / "tiles", cfg.aerial.url_template
    else:
        root = cfg.aerial_dir / f"wayback_{release}"
        tmpl = "synthetic" if synthetic else cfg.overhead.wayback_template.replace("{release}", str(release))
    return {z: TileCache(root, tmpl, z, synthetic=synthetic) for z in zooms}


def training_sources(cfg: Config) -> list[dict[int, TileCache]]:
    """Tile sources drawn from at training time: the current imagery plus every `train_releases` snapshot."""
    return [tile_caches(cfg)] + [tile_caches(cfg, r, cfg.overhead.release_zooms) for r in cfg.overhead.train_releases]


def bbox_tile_range(bbox: BBox, zoom: int, margin: int = 1) -> tuple[range, range]:
    x0, y1 = lonlat_to_tile(bbox.west, bbox.south, zoom)  # tile y grows southwards
    x1, y0 = lonlat_to_tile(bbox.east, bbox.north, zoom)
    return range(int(x0) - margin, int(x1) + margin + 1), range(int(y0) - margin, int(y1) + margin + 1)


def prefetch_tiles(cache: TileCache, tiles: list[tuple[int, int]], workers: int, progress: bool = True) -> tuple[int, int]:
    """Download the listed tiles that are not cached yet. Returns (fetched, failed)."""
    need = [t for t in tiles if not cache._path(*t).exists()]
    if not need:
        return 0, 0
    failed = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(cache.fetch, x, y) for x, y in need]
        it = as_completed(futs)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, total=len(futs), desc=f"tiles z{cache.zoom}")
        for f in it:
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                failed += 1
                log.warning("tile fetch failed: %s", e)
    return len(need) - failed, failed


def prefetch_bbox(cache: TileCache, bbox: BBox, workers: int, progress: bool = True) -> tuple[int, int]:
    xs, ys = bbox_tile_range(bbox, cache.zoom)
    return prefetch_tiles(cache, [(x, y) for x in xs for y in ys], workers, progress)


def tiles_for_views(cache: TileCache, lat: np.ndarray, lon: np.ndarray, px: int) -> list[tuple[int, int]]:
    src = view_source_px(px, scale=1.25, rotated=True)
    need: set[tuple[int, int]] = set()
    for a, b in zip(lat, lon):
        need |= cache.tiles_for_patch(float(a), float(b), src)
    return sorted(need)


def view_source_px(px: int, scale: float, rotated: bool) -> int:
    return int(math.ceil(px * scale * (math.sqrt(2.0) if rotated else 1.0))) + 4


def render_view(cache: TileCache, lat: float, lon: float, px: int, rotation_deg: float = 0.0,
                scale: float = 1.0) -> Image.Image:
    """`px`-pixel square view centred on (lat, lon): north-up mosaic -> rotate -> crop `px*scale` -> resize.
    Rotation is counter-clockwise in degrees; scale > 1 shows more ground per pixel."""
    src = view_source_px(px, scale, rotation_deg != 0.0)
    try:
        img = cache.patch(lat, lon, src)
    except Exception as e:  # noqa: BLE001  (a missing/undownloadable tile must not kill a training run)
        log.warning("view at %.5f,%.5f z%d failed (%s); using a blank patch", lat, lon, cache.zoom, e)
        img = Image.new("RGB", (src, src), (128, 128, 128))
    if rotation_deg:
        img = img.rotate(rotation_deg, resample=Image.BILINEAR)
    side = int(round(px * scale))
    o = (src - side) // 2
    img = img.crop((o, o, o + side, o + side))
    if side != px:
        img = img.resize((px, px), Image.BILINEAR)
    return img


def _norm() -> T.Compose:
    return T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def reference_extent_m(cfg: Config) -> float:
    """Ground extent of one model input at eval_zoom at the bbox centre (~205 m at z17, 40 N)."""
    return meters_per_pixel(cfg.bbox.center[0], cfg.overhead.eval_zoom) * cfg.model.image_size


# ---------------------------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------------------------
class OverheadTrainDataset(Dataset):
    """Infinite stream of random overhead views; `samples_per_epoch` defines an epoch. Randomness comes
    from torch's per-worker RNG, so every worker and every epoch draws different views."""

    def __init__(self, cfg: Config, hierarchy: CellHierarchy, sources: dict[int, TileCache] | list[dict[int, TileCache]],
                 sampler: PointSampler):
        oc = cfg.overhead
        self.n = oc.samples_per_epoch
        self.px = cfg.model.image_size
        self.sources = [sources] if isinstance(sources, dict) else list(sources)
        self.zooms = [[z for z in oc.zooms if z in src] for src in self.sources]  # per source
        assert all(self.zooms), "every training source needs at least one of overhead.zooms"
        w = dict(zip(oc.zooms, oc.zoom_weights if len(oc.zoom_weights) == len(oc.zooms) else [1.0] * len(oc.zooms)))
        self.zoom_w = [np.array([w[z] for z in zs], dtype=np.float64) for zs in self.zooms]
        # zooms only fetched over the built-up area: never drawn for a point outside it (the first
        # source is the current imagery, the others are Wayback releases)
        self.urban_only = [set(oc.urban_only_zooms)] + [set(oc.release_urban_only_zooms)] * (len(self.sources) - 1)
        self.rotate = oc.rotate
        self.scale_range = tuple(oc.scale_jitter)
        self.hierarchy, self.sampler = hierarchy, sampler
        self.mask_factor = oc.level_mask_factor
        self.level_edges = np.array([cell_edge_m(lc.level) for lc in hierarchy.levels])
        self.ref_extent = reference_extent_m(cfg)
        self.color = T.ColorJitter(0.3, 0.3, 0.25, 0.03)
        self.norm = _norm()

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        lat, lon, urban = self.sampler.sample_flag(rng)
        si = int(rng.integers(len(self.sources)))  # acquisition date, uniformly
        ok = np.array([urban or z not in self.urban_only[si] for z in self.zooms[si]])
        rot = float(rng.uniform(0.0, 360.0)) if self.rotate else 0.0
        scale = float(rng.uniform(*self.scale_range))
        src_px = view_source_px(self.px, scale, rot != 0.0)
        # draw a zoom; if its tiles are not all on disk (a Wayback release without high-res coverage
        # there: 404s at fetch time), drop it and redraw rather than render a blank view or hit the network
        while True:
            pw = self.zoom_w[si] * ok
            zi = int(rng.choice(len(pw), p=pw / pw.sum()))
            z = self.zooms[si][zi]
            cache = self.sources[si][z]
            if all(cache.has(x, y) for x, y in cache.tiles_for_patch(lat, lon, src_px)) or pw.sum() - pw[zi] <= 0:
                break
            ok[zi] = False
        img = self.color(render_view(cache, lat, lon, self.px, rot, scale))
        labels = self.hierarchy.labels(np.array([lat]), np.array([lon]))[0]
        extent = meters_per_pixel(lat, z) * self.px * scale
        labels = np.where(extent > self.mask_factor * self.level_edges, -1, labels)  # see level_mask_factor
        return {"image": self.norm(img), "labels": torch.from_numpy(labels),
                "latlon": torch.tensor([lat, lon], dtype=torch.float64), "index": i,
                "log_scale": torch.tensor(math.log2(extent / self.ref_extent), dtype=torch.float32)}


class OverheadQueryDataset(Dataset):
    """A fixed, reproducible set of query views (position, zoom, rotation) rendered from one tile source."""

    def __init__(self, caches: dict[int, TileCache], lat: np.ndarray, lon: np.ndarray, zoom: np.ndarray,
                 rotation: np.ndarray, px: int):
        self.caches = caches
        self.lat, self.lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
        self.zoom, self.rotation = np.asarray(zoom, np.int64), np.asarray(rotation, np.float64)
        self.px = px
        self.norm = _norm()

    @classmethod
    def sample(cls, cfg: Config, caches: dict[int, TileCache], sampler: PointSampler, n: int, seed: int,
               zoom: int | None = None, rotate: bool | None = None) -> "OverheadQueryDataset":
        lat, lon = sampler.sample_many(n, seed)
        rng = np.random.default_rng(seed + 1)
        rotate = cfg.overhead.rotate if rotate is None else rotate
        rot = rng.uniform(0.0, 360.0, n) if rotate else np.zeros(n)
        z = np.full(n, cfg.overhead.eval_zoom if zoom is None else zoom)
        return cls(caches, lat, lon, z, rot, cfg.model.image_size)

    def with_caches(self, caches: dict[int, TileCache]) -> "OverheadQueryDataset":
        return OverheadQueryDataset(caches, self.lat, self.lon, self.zoom, self.rotation, self.px)

    def tiles(self) -> dict[int, list[tuple[int, int]]]:
        out = {}
        for z in np.unique(self.zoom):
            m = self.zoom == z
            out[int(z)] = tiles_for_views(self.caches[int(z)], self.lat[m], self.lon[m], self.px)
        return out

    def __len__(self) -> int:
        return len(self.lat)

    def __getitem__(self, i: int) -> dict:
        img = render_view(self.caches[int(self.zoom[i])], float(self.lat[i]), float(self.lon[i]), self.px,
                          float(self.rotation[i]), 1.0)
        return {"image": self.norm(img), "latlon": torch.tensor([self.lat[i], self.lon[i]], dtype=torch.float64),
                "index": i}


class OverheadCellDataset(Dataset):
    """North-up view centred on every database cell (the per-cell codes of the database)."""

    def __init__(self, centers: np.ndarray, cache: TileCache, px: int):
        self.centers, self.cache, self.px = centers, cache, px
        self.norm = _norm()

    def __len__(self) -> int:
        return len(self.centers)

    def __getitem__(self, i: int) -> dict:
        img = render_view(self.cache, float(self.centers[i, 0]), float(self.centers[i, 1]), self.px)
        return {"image": self.norm(img), "index": i}


# ---------------------------------------------------------------------------------------------
# database
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def build_overhead_database(model: GeoLocModel, hierarchy: CellHierarchy, cfg: Config, caches: dict[int, TileCache],
                            device, progress: bool = True) -> tuple[CellDatabase, ImageIndex]:
    """Prototypes upsampled to the database level + one query-encoder code per database cell. The codes
    fill the `aerial` slot of the database (weighted by alpha at query time) and double as the ImageIndex."""
    configure(cfg)
    db = build_database(model, hierarchy, cfg, aerial_store=None, device=device, progress=False)
    codes = cell_codes(model, db.centers, caches[cfg.overhead.eval_zoom], cfg, device, progress)
    db.aerial = codes
    return db, ImageIndex(codes, db.centers.copy(), db.xyz.copy())


@torch.no_grad()
def cell_codes(model: GeoLocModel, centers: np.ndarray, cache: TileCache, cfg: Config, device,
               progress: bool = True) -> np.ndarray:
    """One query-encoder code per cell: the north-up view centred on the cell from `cache`'s zoom."""
    ds = OverheadCellDataset(centers, cache, cfg.model.image_size)
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)
    it = dl
    if progress:
        from tqdm import tqdm
        it = tqdm(dl, desc=f"cell codes z{cache.zoom}", leave=False)
    model.eval()
    return torch.cat([model.ground(b["image"].to(device)).float().cpu() for b in it]).numpy().astype(np.float32)


@torch.no_grad()
def embed_views(model: GeoLocModel, ds: Dataset, cfg: Config, device, rotations: int = 1, progress: bool = True):
    """Query embeddings (+ finest-level logits). `rotations` > 1 averages the embedding over that many
    rotations of each view (test-time augmentation for queries of unknown heading)."""
    from .database import embed_images

    if rotations <= 1 or not isinstance(ds, OverheadQueryDataset):
        return embed_images(model, ds, cfg.train.batch_size, cfg.train.num_workers, device, return_logits=True,
                            progress=progress)
    acc = None
    for k in range(rotations):
        rot = OverheadQueryDataset(ds.caches, ds.lat, ds.lon, ds.zoom, (ds.rotation + 360.0 * k / rotations) % 360.0, ds.px)
        e = embed_images(model, rot, cfg.train.batch_size, cfg.train.num_workers, device, progress=progress)
        acc = e if acc is None else acc + e
    emb = acc / np.linalg.norm(acc, axis=1, keepdims=True).clip(1e-12)
    logits = emb @ model.heads[-1].prototypes.detach().float().cpu().numpy().T / model.heads[-1].temperature
    return emb.astype(np.float32), logits.astype(np.float32)


def rescale_photo(img: Image.Image, px: int, gsd: float | None, target_mpp: float | None) -> Image.Image:
    """When the photo's ground sampling distance `gsd` (m/px) is known, centre-crop the square that covers
    the model's ground extent (`px` * `target_mpp` metres, ~205 m at z17) so the subsequent resize to `px`
    puts one pixel on the same ground as the training views. A photo that covers less than the extent is
    used whole (the encoder saw a 4x range of zooms in training). Unknown gsd: unchanged."""
    if gsd and target_mpp:
        side = min(min(img.width, img.height), int(round(px * target_mpp / gsd)))
        left, top = (img.width - side) // 2, (img.height - side) // 2
        img = img.crop((left, top, left + side, top + side))
    return img


@torch.no_grad()
def embed_photos(model: GeoLocModel, images: list[Image.Image], px: int, device, rotations: int = 1,
                 gsd: float | None = None, target_mpp: float | None = None, return_scale: bool = False):
    """Query embeddings of arbitrary overhead photos (shortest side resized to `px`, centre crop), averaged
    over `rotations` in-plane rotations. Shared by scripts/09_overhead.py predict and the webapp.
    `return_scale` also returns the scale head's log2(extent / reference extent), averaged the same way."""
    tf = eval_transform(px)
    model.eval()
    out, scales = [], []
    for img in images:
        img = rescale_photo(img.convert("RGB"), px, gsd, target_mpp)
        views = [img.rotate(360.0 * k / rotations, resample=Image.BILINEAR) if k else img for k in range(rotations)]
        x = torch.stack([tf(v) for v in views]).to(device)
        feat, emb = model.ground.features(x)
        e = emb.float().cpu().numpy().sum(0)
        out.append(e / np.linalg.norm(e).clip(1e-12))
        if return_scale:
            scales.append(float(model.scale_head(feat).float().mean()) if model.scale_head is not None else float("nan"))
    q = np.stack(out).astype(np.float32)
    return (q, np.asarray(scales, dtype=np.float32)) if return_scale else q


def tile_change_fraction(a: TileCache, b: TileCache, tiles: list[tuple[int, int]]) -> float:
    """Fraction of `tiles` whose cached bytes differ between two sources (are the Wayback tiles really
    different imagery, or the same acquisition re-served?)."""
    diff = tot = 0
    for x, y in tiles:
        pa, pb = a._path(x, y), b._path(x, y)
        if not (pa.exists() and pb.exists()):
            continue
        tot += 1
        diff += hashlib.md5(pa.read_bytes()).digest() != hashlib.md5(pb.read_bytes()).digest()
    return diff / tot if tot else float("nan")


def in_rects(lat: np.ndarray, lon: np.ndarray, rects: np.ndarray | None) -> np.ndarray:
    """Boolean mask of points inside any of the (lat_min, lat_max, lon_min, lon_max) rectangles."""
    if rects is None or not len(rects):
        return np.zeros(len(lat), dtype=bool)
    out = np.zeros(len(lat), dtype=bool)
    order = np.argsort(lat)
    lat_s = lat[order]
    for r in rects:
        lo, hi = np.searchsorted(lat_s, r[0]), np.searchsorted(lat_s, r[1], side="right")
        idx = order[lo:hi]
        out[idx[(lon[idx] >= r[2]) & (lon[idx] <= r[3])]] = True
    return out


@torch.no_grad()
def build_fine_database(model: GeoLocModel, db: CellDatabase, cfg: Config, cache: TileCache, device,
                        rects: np.ndarray | None, progress: bool = True) -> CellDatabase:
    """Finer grid for narrow photos: the `fine_level` children of the database cells (built-up area only
    when `fine_urban_only`), each with the parent's prototype and a code from a `fine_zoom` view."""
    oc = cfg.overhead
    keep = in_rects(db.centers[:, 0], db.centers[:, 1], rects) if oc.fine_urban_only else np.ones(db.size, dtype=bool)
    parents = np.flatnonzero(keep)
    cells, parent_row = upsample_cells(db.cells[parents], oc.fine_level)
    parent_row = parents[parent_row]
    centers = cells_centers(cells)
    codes = cell_codes(model, centers, cache, cfg, device, progress)
    return CellDatabase(cells=cells, centers=centers, xyz=latlon_to_unit(centers[:, 0], centers[:, 1]),
                        ground=db.ground[parent_row], aerial=codes, parent_class=db.parent_class[parent_row],
                        level=oc.fine_level)


# ---------------------------------------------------------------------------------------------
# pyramid inference: coarse pass on the whole photo, fine pass on a ~205 m crop inside the region
# ---------------------------------------------------------------------------------------------
class Pyramid:
    """Scale-robust localisation for photos of any ground extent.

    1. Whole photo -> embedding; the scale head estimates its extent (or `gsd` gives it exactly); the
       class head at `coarse_level` (560 m cells) gives the top-k region cells.
    2. Photos wider than the reference extent are centre-cropped to it; narrower ones are used whole and
       matched against the per-cell codes of the nearest zoom. The fine pass scores only the database
       cells inside the region, then refines exactly like `localize`.
    """

    def __init__(self, model: GeoLocModel, hierarchy: CellHierarchy, db: CellDatabase, idx: ImageIndex | None,
                 codes: dict[int, np.ndarray], cfg: Config, alpha: float, device, fine_db: CellDatabase | None = None):
        self.model, self.h, self.db, self.idx, self.cfg, self.alpha, self.dev = model, hierarchy, db, idx, cfg, alpha, device
        self.codes = dict(codes)  # zoom -> (M, D) codes; eval_zoom is db.aerial
        self.codes.setdefault(cfg.overhead.eval_zoom, db.aerial)
        self.fine_db = fine_db  # optional finer grid whose codes come from fine_zoom views
        self.ref_extent = reference_extent_m(cfg)
        self.ref_mpp = self.ref_extent / cfg.model.image_size
        self.coarse_idx = hierarchy.level_values.index(cfg.overhead.coarse_level)
        self.lc = hierarchy.levels[self.coarse_idx]
        self.db_coarse = self._coarse_index(db)  # (M,) coarse class index of every database cell
        self.fine_coarse = self._coarse_index(fine_db) if fine_db is not None else None

    def _coarse_index(self, db: CellDatabase) -> np.ndarray:
        parents = np.fromiter((cell_parent(int(c), self.lc.level) for c in db.cells), dtype=np.uint64, count=db.size)
        return self.lc.index_of(parents)

    def coarse_rank(self, coarse_logits: np.ndarray, lat: float, lon: float) -> int:
        """Diagnostic: rank of the true coarse cell in the coarse pass (0 = top)."""
        true = self.lc.index_of(cell_ids([lat], [lon], self.lc.level))[0]
        return int((coarse_logits > coarse_logits[true]).sum()) if true >= 0 else -1

    def extent_of(self, img: Image.Image, gsd: float | None, log_scale: float) -> tuple[float, str]:
        if gsd:
            return gsd * min(img.width, img.height), "gsd"
        return self.ref_extent * 2.0 ** float(log_scale), "estimated"

    def region(self, coarse_logits: np.ndarray, coarse_of_cells: np.ndarray, k: int) -> np.ndarray:
        top = np.argsort(-coarse_logits)[:k]
        return np.isin(coarse_of_cells, top)

    def _choose(self, fine_extent: float):
        """(database, codes, code zoom) whose code views are nearest in scale to the fine-pass extent."""
        px = self.cfg.model.image_size
        cands = [(self.db, self.codes[z], z, self.db_coarse) for z in self.codes]
        if self.fine_db is not None:
            cands.append((self.fine_db, self.fine_db.aerial, self.cfg.overhead.fine_zoom, self.fine_coarse))
        lat = self.cfg.bbox.center[0]
        return min(cands, key=lambda c: abs(math.log2(max(fine_extent, 1.0) / (meters_per_pixel(lat, c[2]) * px))))

    def _pass(self, q: np.ndarray, db: CellDatabase, codes: np.ndarray, sub: np.ndarray):
        sub_db = CellDatabase(db.cells[sub], db.centers[sub], db.xyz[sub], db.ground[sub], codes[sub],
                              db.parent_class[sub], db.level)
        sub_idx = ImageIndex(codes[sub], db.centers[sub], db.xyz[sub]) if self.idx is not None else None
        return localize(q, sub_db, self.cfg.retrieval, self.alpha, sub_idx, refine=True)

    @torch.no_grad()
    def localize(self, img: Image.Image, gsd: float | None = None, rotations: int = 4) -> dict:
        cfg, px, oc = self.cfg, self.cfg.model.image_size, self.cfg.overhead
        img = img.convert("RGB")
        q0, ls = embed_photos(self.model, [img], px, self.dev, rotations=rotations, return_scale=True)
        extent, how = self.extent_of(img, gsd, ls[0])
        coarse = (torch.from_numpy(q0).to(self.dev) @ self.model.heads[self.coarse_idx].prototypes.t()).float().cpu().numpy()[0]
        # fine pass: crop wide photos to the reference extent; narrow photos are used whole
        cropped = extent >= 1.4 * self.ref_extent
        if cropped:
            crop_gsd = extent / min(img.width, img.height)
            q1 = embed_photos(self.model, [img], px, self.dev, rotations=rotations, gsd=crop_gsd, target_mpp=self.ref_mpp)
            fine_extent = self.ref_extent
        else:
            q1, fine_extent = q0, extent
        db, codes, zoom, coarse_of = self._choose(fine_extent)
        small = extent < oc.small_extent_frac * self.ref_extent
        k = oc.small_topk if small else oc.coarse_topk
        mask = self.region(coarse, coarse_of, k)
        res = self._pass(q1, db, codes, np.flatnonzero(mask))
        sub = np.flatnonzero(mask)
        picked = "region"
        if small:  # the 560 m head is unreliable on narrow photos: also search everywhere, keep the better score
            allres = self._pass(q1, db, codes, np.arange(db.size))
            if allres.top_scores[0, 0] > res.top_scores[0, 0]:
                res, sub, picked = allres, np.arange(db.size), "global"
        return {"lat": float(res.latlon[0, 0]), "lon": float(res.latlon[0, 1]), "extent_m": float(extent),
                "extent_from": how, "code_zoom": int(zoom), "cropped": cropped, "db_level": int(db.level),
                "region_cells": int(mask.sum()), "picked": picked, "coarse_logits": coarse,
                "top_score": float(res.top_scores[0, 0]), "top_cells": sub[res.top_cells[0]], "top_scores": res.top_scores[0],
                "db": db}


def render_photo(cache_by_zoom: dict[int, TileCache], lat: float, lon: float, extent_m: float, rotation_deg: float = 0.0,
                 mpp_max: float = 1.0) -> Image.Image:
    """A synthetic 'photo' covering `extent_m` metres at native resolution: rendered from the finest cached
    zoom whose m/px is <= mpp_max (so a 1.6 km photo is ~1800 px from z17, not 224 px), then rotated."""
    zooms = sorted(z for z in cache_by_zoom if meters_per_pixel(lat, z) <= mpp_max) or [max(cache_by_zoom)]
    z = zooms[0]
    px = int(round(extent_m / meters_per_pixel(lat, z)))
    return render_view(cache_by_zoom[z], lat, lon, px, rotation_deg, 1.0)
