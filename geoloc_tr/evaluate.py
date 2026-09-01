"""Recall@distance evaluation of the different localisation variants, and aerial-weight calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .database import CellDatabase, ImageIndex
from .geo import CellHierarchy, haversine_m
from .localize import localize, nearest_image_baseline


def errors_m(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return haversine_m(pred[:, 0], pred[:, 1], gt[:, 0], gt[:, 1])


def recall_at(err: np.ndarray, thresholds_m: list[float]) -> dict[str, float]:
    out = {f"R@{int(t)}m": float((err <= t).mean() * 100) for t in thresholds_m}
    out["median_m"] = float(np.median(err)) if len(err) else float("nan")
    return out


def calibrate_alpha(q: np.ndarray, gt: np.ndarray, db: CellDatabase, cfg: Config, target_m: float = 100.0) -> float:
    """Pick the aerial weight that maximises recall@target on the validation queries."""
    if db.aerial is None:
        return 0.0
    best_a, best_r = 0.0, -1.0
    for a in cfg.retrieval.calibration_grid:
        res = localize(q, db, cfg.retrieval, alpha=a, image_index=None, refine=True)
        r = float((errors_m(res.latlon, gt) <= target_m).mean())
        if r > best_r:
            best_a, best_r = a, r
    return best_a


def evaluate_queries(q: np.ndarray, finest_logits: np.ndarray | None, gt: np.ndarray, hierarchy: CellHierarchy,
                     db: CellDatabase, cfg: Config, alpha: float, image_index: ImageIndex | None) -> tuple[pd.DataFrame, dict]:
    th = cfg.retrieval.eval_thresholds_m
    rows, errs = [], {}

    def add(name, pred):
        e = errors_m(pred, gt)
        errs[name] = e
        rows.append({"method": name, **recall_at(e, th)})

    if finest_logits is not None:
        add(f"classification@L{hierarchy.finest.level}", hierarchy.finest.centers[finest_logits.argmax(1)])
    if image_index is not None:
        add("image retrieval (NN)", nearest_image_baseline(q, image_index))
    add("prototypes, top-1 cell", localize(q, db, cfg.retrieval, 0.0, None, refine=False).latlon)
    add("prototypes + top-k refine", localize(q, db, cfg.retrieval, 0.0, None, refine=True).latlon)
    if db.aerial is not None and alpha > 0:
        add(f"prototypes + aerial(a={alpha:g}) + refine", localize(q, db, cfg.retrieval, alpha, None, refine=True).latlon)
    if image_index is not None:
        add("full: proto + aerial + image refine", localize(q, db, cfg.retrieval, alpha, image_index, refine=True).latlon)
    return pd.DataFrame(rows), errs


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = [str(r[c]) if c == "method" else f"{r[c]:.1f}" for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
