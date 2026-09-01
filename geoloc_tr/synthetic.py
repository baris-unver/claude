"""A synthetic 'city' so the whole pipeline can be run and tested offline.

Images are rendered from the camera position/heading with a deterministic, location-dependent
appearance (per-cell colours + a within-cell gradient + heading-dependent stripes + noise), so a tiny
model can genuinely learn to localise them. Metadata columns match what the Mapillary downloader emits.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import BBox


_N_BEACONS = 8


def _beacons(bbox: BBox) -> np.ndarray:
    rng = np.random.default_rng(1234)
    return np.stack([rng.uniform(bbox.south, bbox.north, _N_BEACONS), rng.uniform(bbox.west, bbox.east, _N_BEACONS)], 1)


def render_image(lat: float, lon: float, heading: float, bbox: BBox, size: int, rng: np.random.Generator) -> Image.Image:
    """Position is encoded as the amplitudes of fixed oriented gratings ("beacons"): a coarse, smooth term
    (distance decay) and a fine, periodic term per beacon. Grating amplitudes survive crops and mild colour
    jitter, so a small CNN with global pooling can learn them; heading only adds a weak overlay."""
    from .geo import haversine_m

    b = _beacons(bbox)
    d = haversine_m(lat, lon, b[:, 0], b[:, 1])  # (K,)
    coarse = np.exp(-d / 700.0)
    fine = 0.5 + 0.5 * np.sin(2 * np.pi * d / 250.0)
    yy, xx = np.mgrid[0:size, 0:size] / size
    img = np.full((size, size, 3), 128.0)
    for k in range(_N_BEACONS):
        ang = np.pi * k / _N_BEACONS
        g1 = np.sin(2 * np.pi * (3 + k) * (xx * np.cos(ang) + yy * np.sin(ang)))
        g2 = np.sin(2 * np.pi * (3 + k) * (xx * np.cos(ang + np.pi / 2) + yy * np.sin(ang + np.pi / 2)))
        col = np.array([1.0 if k % 3 == 0 else 0.3, 1.0 if k % 3 == 1 else 0.3, 1.0 if k % 3 == 2 else 0.3])
        img += (45 * coarse[k] * g1 + 35 * fine[k] * g2)[..., None] * col
    ang = np.radians(heading)
    img += 8 * np.sin(2 * np.pi * 1.5 * (xx * np.cos(ang) + yy * np.sin(ang)))[..., None]
    img += rng.normal(0, 4, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def make_synthetic_city(root: Path, bbox: BBox, n_streets: int = 40, points_per_street: int = 60,
                        image_size: int = 64, seed: int = 0, passes_per_street: int = 4) -> pd.DataFrame:
    root = Path(root)
    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    iid = 10_000_000
    for s in range(n_streets):
        a = np.array([rng.uniform(bbox.south, bbox.north), rng.uniform(bbox.west, bbox.east)])
        ang = rng.uniform(0, 2 * np.pi)
        length = rng.uniform(0.004, 0.012)  # degrees (~0.4-1.3 km)
        b = a + length * np.array([np.sin(ang), np.cos(ang) / np.cos(np.radians(a[0]))])
        b = np.clip(b, [bbox.south, bbox.west], [bbox.north, bbox.east])
        heading = (np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])) + 360) % 360
        for pass_k in range(passes_per_street):  # several drives per street -> held-out sequences overlap train
            seq = f"seq_{s}_{pass_k}"
            t = np.sort(rng.uniform(0, 1, points_per_street))
            hd = heading if pass_k % 2 == 0 else (heading + 180) % 360
            for j, tj in enumerate(t):
                lat, lon = a + tj * (b - a) + rng.normal(0, 2e-5, 2)
                img = render_image(lat, lon, hd + rng.normal(0, 5), bbox, image_size, rng)
                p = img_dir / f"{iid}.jpg"
                img.save(p, "JPEG", quality=90)
                rows.append({"id": iid, "lat": float(lat), "lon": float(lon), "sequence": seq,
                             "captured_at": 1_600_000_000_000 + s * 10_000_000 + pass_k * 1_000_000 + j * 1000,
                             "compass_angle": float(hd), "is_pano": False, "camera_type": "perspective",
                             "path": str(p), "w": image_size, "h": image_size, "yaw_offset": 0.0})
                iid += 1
    df = pd.DataFrame(rows)
    df.to_parquet(root / "metadata.parquet", index=False)
    return df
