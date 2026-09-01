"""Retrieval over the cell database + local refinement -> a lat/lon estimate per query."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .config import RetrievalConfig
from .database import CellDatabase, ImageIndex
from .geo import EARTH_RADIUS_M, unit_to_latlon


@dataclass
class LocalizationResult:
    latlon: np.ndarray  # (N,2) final estimate
    top_cells: np.ndarray  # (N,k) indices into the database
    top_scores: np.ndarray  # (N,k)
    cell_latlon: np.ndarray  # (N,2) estimate before image-level refinement


def score_cells(q: np.ndarray, db: CellDatabase, alpha: float) -> np.ndarray:
    s = q @ db.ground.T
    if alpha > 0 and db.aerial is not None:
        s = s + alpha * (q @ db.aerial.T)
    return s


def _softmax(x: np.ndarray, temp: float) -> np.ndarray:
    z = (x - x.max(axis=-1, keepdims=True)) / temp
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def localize(q: np.ndarray, db: CellDatabase, rcfg: RetrievalConfig, alpha: float = 0.0,
             image_index: ImageIndex | None = None, chunk: int = 256, refine: bool = True) -> LocalizationResult:
    q = np.asarray(q, dtype=np.float32)
    N, k = len(q), min(rcfg.top_k, db.size)
    top_idx = np.zeros((N, k), dtype=np.int64)
    top_sc = np.zeros((N, k), dtype=np.float32)
    for s in range(0, N, chunk):
        sc = score_cells(q[s:s + chunk], db, alpha)
        part = np.argpartition(-sc, k - 1, axis=1)[:, :k]
        psc = np.take_along_axis(sc, part, axis=1)
        order = np.argsort(-psc, axis=1)
        top_idx[s:s + chunk] = np.take_along_axis(part, order, axis=1)
        top_sc[s:s + chunk] = np.take_along_axis(psc, order, axis=1)

    # cell-level estimate: score-weighted centroid of the top-k cells near the best one
    best_xyz = db.xyz[top_idx[:, 0]]  # (N,3)
    cand_xyz = db.xyz[top_idx]  # (N,k,3)
    d = np.linalg.norm(cand_xyz - best_xyz[:, None, :], axis=-1) * EARTH_RADIUS_M
    radius = rcfg.refine_radius_m if refine else 0.0
    mask = d <= max(radius, 1e-6)
    w = _softmax(np.where(mask, top_sc, -np.inf), rcfg.refine_temperature)
    est_xyz = (w[..., None] * cand_xyz).sum(1)
    cell_latlon = np.stack(unit_to_latlon(est_xyz), 1)
    final = cell_latlon.copy()

    if refine and rcfg.image_refine and image_index is not None and len(image_index.emb):
        tree = cKDTree(image_index.xyz)
        r = radius / EARTH_RADIUS_M  # chord ~ arc for small angles
        neigh = tree.query_ball_point(est_xyz, r)
        for i, nb in enumerate(neigh):
            if not nb:
                continue
            nb = np.asarray(nb)
            sims = image_index.emb[nb] @ q[i]
            top = nb[np.argsort(-sims)[:5]]
            ws = _softmax(image_index.emb[top] @ q[i], rcfg.refine_temperature)
            xyz = (ws[:, None] * image_index.xyz[top]).sum(0)
            final[i] = np.stack(unit_to_latlon(xyz[None]), 1)[0]
    return LocalizationResult(final, top_idx, top_sc, cell_latlon)


def nearest_image_baseline(q: np.ndarray, image_index: ImageIndex) -> np.ndarray:
    """Plain image retrieval: position of the most similar training image."""
    out = np.zeros((len(q), 2))
    for s in range(0, len(q), 256):
        sims = q[s:s + 256] @ image_index.emb.T
        out[s:s + 256] = image_index.latlon[sims.argmax(1)]
    return out
