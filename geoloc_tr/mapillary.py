"""Mapillary API v4 access: coverage enumeration via vector tiles, metadata via the Graph API, thumbnails.

Endpoints (https://www.mapillary.com/developer/api-documentation):
  coverage tiles : https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}?access_token=TOKEN
                   z=14 tiles carry one point per image in layer "image" with properties
                   id, sequence_id, captured_at (ms), compass_angle, is_pano, creator_id, organization_id.
  entity         : https://graph.mapillary.com/{image_id}?fields=...&access_token=TOKEN
  batch entities : https://graph.mapillary.com/?ids=a,b,c&fields=...
  bbox search    : https://graph.mapillary.com/images?bbox=w,s,e,n&fields=...&limit=2000  (paged)
Thumbnail URLs (thumb_256/1024/2048/original_url) expire after a short time, so download immediately.
"""
from __future__ import annotations

import io
import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import BBox

log = logging.getLogger(__name__)

TILE_URL = "https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}"
GRAPH_URL = "https://graph.mapillary.com"
IMAGE_FIELDS = [
    "id", "geometry", "computed_geometry", "captured_at", "compass_angle", "computed_compass_angle",
    "is_pano", "camera_type", "sequence", "width", "height",
]


def make_session(retries: int = 5, backoff: float = 1.0) -> requests.Session:
    s = requests.Session()
    r = Retry(total=retries, backoff_factor=backoff, status_forcelist=(429, 500, 502, 503, 504),
              allowed_methods=frozenset(["GET"]))
    s.mount("https://", HTTPAdapter(max_retries=r, pool_maxsize=32))
    s.headers["User-Agent"] = "geoloc-tr/0.1 (research)"
    return s


# ---------------------------------------------------------------------------------------------
# Web-Mercator tile maths
# ---------------------------------------------------------------------------------------------
def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[float, float]:
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return x, y


def tile_to_lonlat(x: float, y: float, z: int) -> tuple[float, float]:
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def tiles_covering(bbox: BBox, z: int) -> list[tuple[int, int, int]]:
    x0, y1 = lonlat_to_tile(bbox.west, bbox.south, z)
    x1, y0 = lonlat_to_tile(bbox.east, bbox.north, z)
    out = []
    for x in range(int(math.floor(x0)), int(math.floor(x1)) + 1):
        for y in range(int(math.floor(y0)), int(math.floor(y1)) + 1):
            out.append((z, x, y))
    return out


# ---------------------------------------------------------------------------------------------
# Coverage via vector tiles
# ---------------------------------------------------------------------------------------------
def _decode_mvt(data: bytes) -> dict:
    import mapbox_vector_tile  # noqa: WPS433 (optional heavy import)

    try:  # mapbox-vector-tile >= 2.0
        return mapbox_vector_tile.decode(data, default_options={"y_coord_down": True})
    except TypeError:  # 1.x API
        return mapbox_vector_tile.decode(data, y_coord_down=True)


def decode_image_points(data: bytes, z: int, x: int, y: int) -> list[dict]:
    """Decode the `image` layer of a coverage tile into rows with lon/lat."""
    tile = _decode_mvt(data)
    layer = tile.get("image")
    if not layer:
        return []
    extent = float(layer.get("extent", 4096))
    rows = []
    for f in layer["features"]:
        geom = f["geometry"]
        if geom.get("type") != "Point":
            continue
        px, py = geom["coordinates"]
        lon, lat = tile_to_lonlat(x + px / extent, y + py / extent, z)
        p = f.get("properties", {})
        rows.append({
            "id": int(p["id"]),
            "lon": lon,
            "lat": lat,
            "sequence": str(p.get("sequence_id", "")),
            "captured_at": int(p.get("captured_at", 0) or 0),
            "compass_angle": float(p.get("compass_angle", float("nan")) or float("nan")),
            "is_pano": bool(p.get("is_pano", False)),
            "creator_id": int(p.get("creator_id", 0) or 0),
            "organization_id": int(p.get("organization_id", 0) or 0),
        })
    return rows


def fetch_coverage(bbox: BBox, token: str, z: int = 14, workers: int = 8,
                   session: requests.Session | None = None, progress: bool = True) -> pd.DataFrame:
    """Enumerate every Mapillary image inside `bbox` (one row per image; positions are the *original*
    GPS, the Graph API later provides SfM-corrected `computed_geometry`)."""
    session = session or make_session()
    tiles = tiles_covering(bbox, z)
    log.info("fetching %d coverage tiles at z=%d", len(tiles), z)

    def one(t):
        z_, x, y = t
        r = session.get(TILE_URL.format(z=z_, x=x, y=y), params={"access_token": token}, timeout=60)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return decode_image_points(r.content, z_, x, y)

    rows: list[dict] = []
    it: Iterable
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, t) for t in tiles]
        it = as_completed(futs)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, total=len(futs), desc="coverage tiles")
        for fut in it:
            rows.extend(fut.result())
    df = pd.DataFrame(rows, columns=["id", "lon", "lat", "sequence", "captured_at", "compass_angle", "is_pano",
                                     "creator_id", "organization_id"])
    if len(df):
        df = df[(df.lon >= bbox.west) & (df.lon <= bbox.east) & (df.lat >= bbox.south) & (df.lat <= bbox.north)]
        df = df.drop_duplicates("id").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------------------------
# Coverage via Graph API bbox search (fallback; bbox must be small, so we grid it)
# ---------------------------------------------------------------------------------------------
def fetch_coverage_bbox_search(bbox: BBox, token: str, step_deg: float = 0.02, limit: int = 2000,
                               session: requests.Session | None = None, fields: list[str] | None = None) -> pd.DataFrame:
    session = session or make_session()
    fields = fields or IMAGE_FIELDS
    rows = []
    lons = np.arange(bbox.west, bbox.east, step_deg)
    lats = np.arange(bbox.south, bbox.north, step_deg)
    for w in lons:
        for s in lats:
            url = f"{GRAPH_URL}/images"
            params = {"access_token": token, "fields": ",".join(fields), "limit": limit,
                      "bbox": f"{w},{s},{min(w + step_deg, bbox.east)},{min(s + step_deg, bbox.north)}"}
            while True:
                r = session.get(url, params=params, timeout=60)
                r.raise_for_status()
                js = r.json()
                rows.extend(_flatten_entity(e) for e in js.get("data", []))
                nxt = js.get("paging", {}).get("next")
                if not nxt:
                    break
                url, params = nxt, {}
    return pd.DataFrame(rows).drop_duplicates("id").reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Metadata via Graph API
# ---------------------------------------------------------------------------------------------
def _flatten_entity(e: dict) -> dict:
    out = {"id": int(e["id"])}
    g = e.get("computed_geometry") or e.get("geometry")
    if g:
        out["lon"], out["lat"] = float(g["coordinates"][0]), float(g["coordinates"][1])
    g0 = e.get("geometry")
    if g0:
        out["lon_raw"], out["lat_raw"] = float(g0["coordinates"][0]), float(g0["coordinates"][1])
    for k in ("captured_at", "compass_angle", "computed_compass_angle", "is_pano", "camera_type", "width", "height"):
        if k in e:
            out[k] = e[k]
    if "sequence" in e:
        out["sequence"] = str(e["sequence"])
    for k, v in e.items():
        if k.startswith("thumb_") and k.endswith("_url"):
            out[k] = v
    return out


def fetch_metadata(ids: Iterable[int], token: str, fields: list[str] | None = None, thumb_field: str = "thumb_1024_url",
                   batch: int = 50, workers: int = 8, session: requests.Session | None = None,
                   progress: bool = True) -> pd.DataFrame:
    """Fetch entity fields (incl. an expiring thumbnail URL) for many image ids."""
    session = session or make_session()
    fields = list(dict.fromkeys((fields or IMAGE_FIELDS) + [thumb_field]))
    ids = [int(i) for i in ids]
    batches = [ids[i:i + batch] for i in range(0, len(ids), batch)]

    def one(b: list[int]) -> list[dict]:
        params = {"access_token": token, "fields": ",".join(fields), "ids": ",".join(map(str, b))}
        r = session.get(f"{GRAPH_URL}/", params=params, timeout=60)
        if r.ok:
            js = r.json()
            if isinstance(js, dict) and "data" in js and isinstance(js["data"], list):
                return [_flatten_entity(e) for e in js["data"]]
            return [_flatten_entity(e) for e in js.values() if isinstance(e, dict) and "id" in e]
        # batch endpoint refused -> fall back to single-entity requests
        out = []
        for i in b:
            r1 = session.get(f"{GRAPH_URL}/{i}", params={"access_token": token, "fields": ",".join(fields)}, timeout=60)
            if r1.ok:
                out.append(_flatten_entity(r1.json()))
            else:
                log.warning("metadata failed for %s: %s", i, r1.status_code)
        return out

    rows: list[dict] = []
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, b) for b in batches]
        it = as_completed(futs)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, total=len(futs), desc="metadata")
        for fut in it:
            rows.extend(fut.result())
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Thumbnails
# ---------------------------------------------------------------------------------------------
def _save_resized(img: Image.Image, path: Path, max_side: int) -> tuple[int, int]:
    img = img.convert("RGB")
    w, h = img.size
    s = max(w, h)
    if max_side and s > max_side:
        img = img.resize((round(w * max_side / s), round(h * max_side / s)), Image.BICUBIC)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=90)
    return img.size


def pano_crops(img: Image.Image, n: int) -> list[tuple[Image.Image, float]]:
    """Cut an equirectangular panorama into `n` horizontal crops (yaw offset in degrees each).

    This is a cheap approximation of perspective re-projection, good enough for training crops.
    """
    w, h = img.size
    cw = w // n
    top, bottom = int(h * 0.2), int(h * 0.8)  # drop sky / car hood
    out = []
    for k in range(n):
        crop = img.crop((k * cw, top, (k + 1) * cw, bottom))
        out.append((crop, 360.0 * (k + 0.5) / n - 180.0))
    return out


def download_thumbnails(meta: pd.DataFrame, out_dir: Path, thumb_field: str = "thumb_1024_url", max_side: int = 768,
                        include_pano: bool = False, n_pano_crops: int = 4, workers: int = 8,
                        session: requests.Session | None = None, progress: bool = True) -> pd.DataFrame:
    """Download thumbnails; returns one row per stored image file (panoramas may expand into crops)."""
    session = session or make_session()
    out_dir = Path(out_dir)

    def one(row) -> list[dict]:
        url = row.get(thumb_field)
        if not isinstance(url, str) or not url:
            return []
        is_pano = bool(row.get("is_pano", False))
        base = {k: row[k] for k in row.index if not str(k).startswith("thumb_")}
        results = []
        if is_pano and not include_pano:
            return []
        dest = out_dir / f"{int(row['id'])}.jpg"
        if not is_pano and dest.exists():
            w, h = Image.open(dest).size
            return [{**base, "path": str(dest), "w": w, "h": h, "yaw_offset": 0.0}]
        r = session.get(url, timeout=60)
        if not r.ok:
            log.warning("thumb failed for %s: %s", row["id"], r.status_code)
            return []
        img = Image.open(io.BytesIO(r.content))
        if is_pano:
            for k, (crop, yaw) in enumerate(pano_crops(img, n_pano_crops)):
                p = out_dir / f"{int(row['id'])}_{k}.jpg"
                w, h = _save_resized(crop, p, max_side)
                results.append({**base, "path": str(p), "w": w, "h": h, "yaw_offset": yaw})
        else:
            w, h = _save_resized(img, dest, max_side)
            results.append({**base, "path": str(dest), "w": w, "h": h, "yaw_offset": 0.0})
        return results

    rows: list[dict] = []
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, row) for _, row in meta.iterrows()]
        it = as_completed(futs)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, total=len(futs), desc="thumbnails")
        for fut in it:
            try:
                rows.extend(fut.result())
            except Exception as e:  # keep going on corrupt files
                log.warning("download error: %s", e)
    return pd.DataFrame(rows)
