"""Aerial (satellite) imagery patches per S2 cell from any XYZ tile source, with an on-disk tile cache.

A synthetic mode renders deterministic tiles from the tile coordinates, so the whole pipeline can be
exercised offline (tests) without contacting a tile server.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .mapillary import lonlat_to_tile, make_session

log = logging.getLogger(__name__)
TILE_PX = 256


def meters_per_pixel(lat: float, zoom: int) -> float:
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)


def _synthetic_tile(z: int, x: int, y: int, size: int = TILE_PX) -> Image.Image:
    """Smooth, location-dependent pattern so a model can learn *something* from it in tests."""
    rng = np.random.default_rng(int(hashlib.md5(f"{z}/{x}/{y}".encode()).hexdigest()[:8], 16))
    yy, xx = np.mgrid[0:size, 0:size] / size
    gx, gy = x + xx, y + yy  # continuous global tile coords -> continuous across tile borders
    r = 0.5 + 0.5 * np.sin(gx * 2.1) * np.cos(gy * 1.7)
    g = 0.5 + 0.5 * np.sin(gx * 0.9 + gy * 1.3)
    b = 0.5 + 0.5 * np.cos(gx * 1.5 - gy * 0.7)
    arr = np.stack([r, g, b], -1) * 255 + rng.normal(0, 8, (size, size, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


class TileCache:
    def __init__(self, root: Path, url_template: str, zoom: int, synthetic: bool = False, session=None):
        self.root = Path(root)
        self.url_template = url_template
        self.zoom = zoom
        self.synthetic = synthetic
        self.session = session
        self._mem: dict[tuple[int, int], Image.Image] = {}

    def _path(self, x: int, y: int) -> Path:
        return self.root / str(self.zoom) / str(x) / f"{y}.jpg"

    def has(self, x: int, y: int) -> bool:
        return (x, y) in self._mem or self._path(x, y).exists()

    def fetch(self, x: int, y: int) -> Image.Image:
        p = self._path(x, y)
        if p.exists():
            return Image.open(p).convert("RGB")
        if self.synthetic:
            img = _synthetic_tile(self.zoom, x, y)
        else:
            self.session = self.session or make_session()
            url = self.url_template.format(z=self.zoom, x=x, y=y, token=os.environ.get("MAPBOX_TOKEN", ""))
            r = self.session.get(url, timeout=60)
            r.raise_for_status()
            import io
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
        p.parent.mkdir(parents=True, exist_ok=True)
        img.save(p, "JPEG", quality=92)
        return img

    def get(self, x: int, y: int) -> Image.Image:
        key = (x, y)
        if key not in self._mem:
            if len(self._mem) > 512:
                self._mem.clear()
            self._mem[key] = self.fetch(x, y)
        return self._mem[key]

    # ------------------------------------------------------------------------------------------
    def tiles_for_patch(self, lat: float, lon: float, patch_px: int) -> set[tuple[int, int]]:
        fx, fy = lonlat_to_tile(lon, lat, self.zoom)
        cx, cy = fx * TILE_PX, fy * TILE_PX
        half = patch_px / 2
        xs = range(int(math.floor((cx - half) / TILE_PX)), int(math.floor((cx + half - 1) / TILE_PX)) + 1)
        ys = range(int(math.floor((cy - half) / TILE_PX)), int(math.floor((cy + half - 1) / TILE_PX)) + 1)
        return {(x, y) for x in xs for y in ys}

    def prefetch(self, points: Iterable[tuple[float, float]], patch_px: int, workers: int = 6,
                 progress: bool = True) -> int:
        need: set[tuple[int, int]] = set()
        for lat, lon in points:
            need |= self.tiles_for_patch(lat, lon, patch_px)
        need = {t for t in need if not self._path(*t).exists()}
        if not need:
            return 0
        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(self.fetch, x, y) for x, y in need]
            it = as_completed(futs)
            if progress:
                from tqdm import tqdm
                it = tqdm(it, total=len(futs), desc="aerial tiles")
            for f in it:
                try:
                    f.result()
                except Exception as e:
                    log.warning("tile fetch failed: %s", e)
        return len(need)

    def patch(self, lat: float, lon: float, patch_px: int) -> Image.Image:
        """Square crop of `patch_px` pixels centred on (lat, lon), north-up, stitched from tiles."""
        fx, fy = lonlat_to_tile(lon, lat, self.zoom)
        cx, cy = fx * TILE_PX, fy * TILE_PX
        x0, y0 = int(round(cx - patch_px / 2)), int(round(cy - patch_px / 2))
        canvas = Image.new("RGB", (patch_px, patch_px))
        for tx, ty in self.tiles_for_patch(lat, lon, patch_px):
            tile = self.get(tx, ty)
            canvas.paste(tile, (tx * TILE_PX - x0, ty * TILE_PX - y0))
        return canvas


class AerialPatchStore:
    """Caches per-cell patches as JPEGs (aerial/patches/<cell_id>.jpg) so training does not re-stitch."""

    def __init__(self, cache: TileCache, patch_dir: Path, patch_px: int):
        self.cache = cache
        self.patch_dir = Path(patch_dir)
        self.patch_px = patch_px
        self.patch_dir.mkdir(parents=True, exist_ok=True)

    def path(self, cell: int) -> Path:
        return self.patch_dir / f"{int(cell)}.jpg"

    def get(self, cell: int, lat: float, lon: float) -> Image.Image:
        p = self.path(cell)
        if p.exists():
            return Image.open(p).convert("RGB")
        img = self.cache.patch(lat, lon, self.patch_px)
        img.save(p, "JPEG", quality=90)
        return img

    def build(self, cells: np.ndarray, centers: np.ndarray, progress: bool = True) -> None:
        it = zip(cells, centers)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, total=len(cells), desc="aerial patches")
        for c, (lat, lon) in it:
            self.get(int(c), float(lat), float(lon))


def make_aerial_store(cfg) -> "AerialPatchStore | None":
    """Build the patch store from a Config (None when aerial is disabled). url_template='synthetic'
    renders deterministic tiles offline."""
    if not cfg.aerial.enabled:
        return None
    synthetic = cfg.aerial.url_template == "synthetic"
    cache = TileCache(cfg.aerial_dir / "tiles", cfg.aerial.url_template, cfg.aerial.zoom, synthetic=synthetic)
    return AerialPatchStore(cache, cfg.aerial_dir / f"patches_z{cfg.aerial.zoom}_{cfg.aerial.patch_px}px", cfg.aerial.patch_px)
