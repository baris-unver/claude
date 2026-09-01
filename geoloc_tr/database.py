"""Cell database: prototypes from the finest classifier, upsampled through the S2 hierarchy to the
database level, plus (optional) aerial codes per cell and an index of training-image embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .aerial import AerialPatchStore
from .config import Config
from .data import AerialCellDataset, GroundDataset
from .geo import CellHierarchy, cells_centers, latlon_to_unit, upsample_cells
from .model import GeoLocModel


@dataclass
class CellDatabase:
    cells: np.ndarray  # (M,) uint64 S2 ids at database level
    centers: np.ndarray  # (M,2) lat/lon
    xyz: np.ndarray  # (M,3) unit vectors
    ground: np.ndarray  # (M,D) float32, L2-normalised prototypes
    aerial: np.ndarray | None  # (M,D) float32 or None
    parent_class: np.ndarray  # (M,) index into the finest LevelClasses
    level: int

    def save(self, path: Path) -> None:
        np.savez_compressed(path, cells=self.cells, centers=self.centers, xyz=self.xyz, ground=self.ground,
                            aerial=self.aerial if self.aerial is not None else np.zeros((0,)),
                            parent_class=self.parent_class, level=self.level)

    @classmethod
    def load(cls, path: Path) -> "CellDatabase":
        z = np.load(path)
        aer = z["aerial"]
        return cls(z["cells"], z["centers"], z["xyz"], z["ground"], aer if aer.size else None, z["parent_class"],
                   int(z["level"]))

    @property
    def size(self) -> int:
        return len(self.cells)


@dataclass
class ImageIndex:
    """Embeddings + positions of training images (for image-level refinement and a retrieval baseline)."""

    emb: np.ndarray  # (N,D)
    latlon: np.ndarray  # (N,2)
    xyz: np.ndarray  # (N,3)

    def save(self, path: Path) -> None:
        np.savez_compressed(path, emb=self.emb, latlon=self.latlon, xyz=self.xyz)

    @classmethod
    def load(cls, path: Path) -> "ImageIndex":
        z = np.load(path)
        return cls(z["emb"], z["latlon"], z["xyz"])


@torch.no_grad()
def embed_images(model: GeoLocModel, ds: GroundDataset, batch_size: int, num_workers: int, device,
                 return_logits: bool = False, progress: bool = True):
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    it = dl
    if progress:
        from tqdm import tqdm
        it = tqdm(dl, desc="embed", leave=False)
    embs, logits = [], []
    model.eval()
    for batch in it:
        e, lg = model(batch["image"].to(device, non_blocking=True))
        embs.append(e.float().cpu())
        if return_logits:
            logits.append(lg[-1].float().cpu())
    emb = torch.cat(embs).numpy()
    if return_logits:
        return emb, torch.cat(logits).numpy()
    return emb


@torch.no_grad()
def build_database(model: GeoLocModel, hierarchy: CellHierarchy, cfg: Config, aerial_store: AerialPatchStore | None,
                   device, progress: bool = True) -> CellDatabase:
    finest = hierarchy.finest
    db_level = cfg.cells.database_level
    if db_level < finest.level:
        raise ValueError("database_level must be >= finest hierarchy level")
    protos = model.heads[-1].prototypes.detach().float().cpu().numpy()  # (C,D)
    cells, parent = upsample_cells(finest.cell_ids, db_level)
    centers = cells_centers(cells) if db_level > finest.level else finest.centers.copy()
    ground = protos[parent]
    aerial = None
    if aerial_store is not None and model.aerial is not None:
        ds = AerialCellDataset(cells, centers, aerial_store, cfg.model.image_size)
        dl = DataLoader(ds, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)
        it = dl
        if progress:
            from tqdm import tqdm
            it = tqdm(dl, desc="aerial codes", leave=False)
        codes = [model.encode_aerial(b["aerial"].to(device)).float().cpu() for b in it]
        aerial = torch.cat(codes).numpy().astype(np.float32)
    return CellDatabase(cells=cells, centers=centers, xyz=latlon_to_unit(centers[:, 0], centers[:, 1]),
                        ground=ground.astype(np.float32), aerial=aerial, parent_class=parent, level=db_level)


def build_image_index(model: GeoLocModel, df: pd.DataFrame, hierarchy: CellHierarchy, cfg: Config, device,
                      progress: bool = True) -> ImageIndex:
    ds = GroundDataset(df, hierarchy, cfg.model.image_size, train=False)
    emb = embed_images(model, ds, cfg.train.batch_size, cfg.train.num_workers, device, progress=progress)
    latlon = np.stack([ds.lat, ds.lon], 1)
    return ImageIndex(emb.astype(np.float32), latlon, latlon_to_unit(ds.lat, ds.lon))
