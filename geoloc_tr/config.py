"""Configuration: city presets and the experiment config dataclasses."""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BBox:
    """Geographic bounding box, WGS84 degrees."""

    west: float
    south: float
    east: float
    north: float

    def contains(self, lon: float, lat: float) -> bool:
        return self.west <= lon <= self.east and self.south <= lat <= self.north

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.south + self.north) / 2, (self.west + self.east) / 2)


# Rough urban-core bounding boxes for large Turkish cities (west, south, east, north).
# They are deliberately generous; the actual dataset extent is whatever Mapillary covers inside them.
CITY_PRESETS: dict[str, BBox] = {
    "ankara": BBox(32.60, 39.80, 33.05, 40.05),
    "istanbul": BBox(28.55, 40.85, 29.45, 41.25),
    "izmir": BBox(26.95, 38.30, 27.30, 38.55),
    "bursa": BBox(28.90, 40.13, 29.20, 40.28),
    "antalya": BBox(30.55, 36.82, 30.85, 36.95),
    "eskisehir": BBox(30.40, 39.72, 30.62, 39.82),
    "konya": BBox(32.40, 37.80, 32.60, 37.95),
    "adana": BBox(35.20, 36.95, 35.40, 37.08),
}


@dataclass
class DataConfig:
    city: str = "ankara"
    bbox: BBox | None = None
    root: str = "data"
    tile_zoom: int = 14  # Mapillary coverage vector tiles carry individual image points at z=14
    include_pano: bool = False
    pano_crops: int = 4
    min_capture_year: int = 2015
    image_max_side: int = 768
    thumb_field: str = "thumb_1024_url"
    min_train_spacing_m: float = 3.0  # thin dense sequences so consecutive frames are not near-duplicates
    max_images: int | None = None
    download_workers: int = 8  # graph.mapillary.com concurrency; >=32 gets mass 403s from the API
    # Thumbnails are served by a different host (scontent.*.fbcdn.net) with a much higher tolerance:
    # measured 29 img/s at 8 workers, 112 img/s at 24, and 81 img/s at 48 (saturation), all with zero
    # failures. Keeping one knob for both phases means the Graph API's limit caps the whole download.
    thumb_workers: int = 24
    graph_batch: int = 50  # ids per Graph API batch request


@dataclass
class AerialConfig:
    enabled: bool = True
    # Any XYZ tile template. Default is Esri World Imagery (check its terms of use for your project);
    # alternatives: Mapbox satellite (needs MAPBOX_TOKEN), or your own tile server.
    url_template: str = (
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    )
    zoom: int = 18
    patch_px: int = 224  # square crop centred on the cell centre, cut from the tile mosaic
    workers: int = 6


@dataclass
class OverheadConfig:
    """Overhead (satellite / aerial) *queries*; see geoloc_tr/overhead.py and scripts/09_overhead.py."""

    levels: list[int] = field(default_factory=lambda: [12, 14, 16])  # class levels: ~2.2 km / 560 m / 140 m
    database_level: int = 17  # ~70 m cells; prototypes upsampled here, one encoder code per cell
    # native tile zooms drawn at training time and their sampling weights. 224 px at 40 N covers
    # z14 1.6 km / z15 820 m / z16 410 m / z17 205 m / z18 103 m / z19 51 m; the weights keep the
    # 100-400 m band (where the fine cells are learnable) as the bulk of the views.
    zooms: list[int] = field(default_factory=lambda: [14, 15, 16, 17, 18, 19])
    zoom_weights: list[float] = field(default_factory=lambda: [0.08, 0.12, 0.2, 0.3, 0.2, 0.1])
    urban_only_zooms: list[int] = field(default_factory=lambda: [19])  # fetched for the built-up area only
    release_urban_only_zooms: list[int] = field(default_factory=lambda: [18, 19])
    # a class level is not supervised for views wider than this many cell edges: a 1.6 km view cannot
    # tell which 140 m cell its centre is in, and asking it to only pollutes the fine prototypes
    level_mask_factor: float = 3.0
    scale_loss_weight: float = 0.1  # log2-extent regression head (see ModelConfig.scale_head)
    eval_zoom: int = 17  # z17 at 40 N: 224 px ~ 205 m of ground; z18 ~ 103 m, z16 ~ 410 m
    code_zooms: list[int] = field(default_factory=lambda: [17, 18])  # per-cell codes rendered at these zooms
    # pyramid inference: coarse pass on the whole photo -> top-k cells at this class level -> fine pass
    # on a ~205 m crop restricted to those cells
    coarse_level: int = 14
    coarse_topk: int = 12
    # narrow photos (< small_extent_frac x reference extent): a finer database over the built-up area
    # (fine_level cells with codes from fine_zoom views), a wider coarse region, and a second, unrestricted
    # fine pass -- whichever of the two scores higher wins, because the 560 m region head is weakest on them
    fine_level: int = 18
    fine_zoom: int = 19
    fine_urban_only: bool = True
    small_extent_frac: float = 0.7
    small_topk: int = 40
    scale_jitter: list[float] = field(default_factory=lambda: [0.8, 1.25])
    rotate: bool = True  # random in-plane rotation (queries of unknown heading)
    samples_per_epoch: int = 80000
    urban_frac: float = 0.5  # share of training views centred in built-up cells (those with Mapillary images)
    urban_level: int = 15  # ~280 m cells define "built-up"
    eval_queries: int = 2000
    val_queries: int = 1000
    # Esri Wayback: dated snapshots of World Imagery, used as different-date test sources. Release ids
    # come from https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json
    # (64776 = 2023-08-31, 25521 = 2017-11-16).
    eval_releases: list[int] = field(default_factory=lambda: [64776, 25521])
    # Extra *training* sources: Wayback releases whose imagery differs from the current tiles. Training on
    # several acquisition dates is what makes the encoder recognise places rather than memorise one
    # rendering; a model trained on the current imagery alone scores ~2-25% R@100m on other dates.
    # Ankara epochs (z17, all different from each other and from the test releases above):
    # 34007=2025-02-27, 37965=2024-02-08, 10321=2022-03-16, 9812=2021-02-24, 4756=2019-12-12,
    # 18966=2016-12-20, 15084=2015-03-18.
    train_releases: list[int] = field(default_factory=lambda: [34007, 37965, 10321, 9812, 4756, 18966, 15084])
    release_zooms: list[int] = field(default_factory=lambda: [15, 16, 17, 18])  # zooms fetched for the extra sources
    wayback_template: str = ("https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/"
                             "default028mm/MapServer/tile/{release}/{z}/{y}/{x}")
    tile_workers: int = 16


@dataclass
class CellConfig:
    # S2 levels of the classification heads (coarse -> fine). Approximate edge lengths at 40 N:
    # 11 ~ 4.5 km, 13 ~ 1.1 km, 15 ~ 280 m, 17 ~ 70 m, 18 ~ 35 m.
    hierarchy_levels: list[int] = field(default_factory=lambda: [11, 13, 15, 17])
    database_level: int = 18  # prototypes are upsampled to this level for retrieval
    min_images_per_class: int = 5  # cells with fewer training images are merged into their parent class
    label_smoothing_sigma_scale: float = 1.0  # sigma_l = scale * mean cell edge at level l


@dataclass
class SplitConfig:
    block_level: int = 13  # spatial blocks held out entirely (unseen-area test)
    unseen_block_frac: float = 0.08
    seq_val_frac: float = 0.05
    seq_test_frac: float = 0.10
    eval_spacing_m: float = 10.0  # thin val/test sequences so queries are not near-duplicates
    seed: int = 0


@dataclass
class ModelConfig:
    backbone: str = "timm:vit_small_patch14_dinov2.lvd142m"  # or "tiny" (offline smoke tests), "timm:resnet50"
    image_size: int = 224
    embed_dim: int = 512
    freeze_backbone_blocks: int = 0  # freeze the first N transformer blocks (0 = train all)
    temperature: float = 0.05  # cosine-classifier temperature
    aerial_enabled: bool = True
    aerial_backbone: str | None = None  # defaults to `backbone`
    aerial_loss_weight: float = 0.5
    aerial_temperature: float = 0.07
    scale_head: bool = False  # regress log2(ground extent / reference extent) from backbone features


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 3e-4
    backbone_lr_mult: float = 0.1
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    num_workers: int = 8
    amp: bool = True
    seed: int = 0
    out_dir: str = "runs/default"
    eval_every: int = 1
    max_steps_per_epoch: int | None = None  # for smoke tests
    augment: str = "strong"  # "strong" for real imagery, "light" for the synthetic city


@dataclass
class RetrievalConfig:
    top_k: int = 32
    refine_radius_m: float = 150.0  # top-k cells within this radius of the best cell vote for the final estimate
    refine_temperature: float = 0.02
    image_refine: bool = True  # snap to positions of nearest training images inside the winning neighbourhood
    calibration_grid: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    eval_thresholds_m: list[float] = field(default_factory=lambda: [25, 50, 100, 200, 500, 1000, 5000])


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    aerial: AerialConfig = field(default_factory=AerialConfig)
    cells: CellConfig = field(default_factory=CellConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    overhead: OverheadConfig = field(default_factory=OverheadConfig)

    # ---- derived paths -------------------------------------------------------------------------
    @property
    def bbox(self) -> BBox:
        if self.data.bbox is not None:
            return self.data.bbox
        try:
            return CITY_PRESETS[self.data.city.lower()]
        except KeyError as e:
            raise KeyError(f"Unknown city {self.data.city!r}; add a bbox to the config or pick one of "
                           f"{sorted(CITY_PRESETS)}") from e

    @property
    def data_dir(self) -> Path:
        return Path(self.data.root) / self.data.city.lower()

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def aerial_dir(self) -> Path:
        return self.data_dir / "aerial"

    @property
    def coverage_path(self) -> Path:
        return self.data_dir / "coverage.parquet"

    @property
    def metadata_path(self) -> Path:
        return self.data_dir / "metadata.parquet"

    @property
    def splits_path(self) -> Path:
        return self.data_dir / "splits.parquet"

    @property
    def out_dir(self) -> Path:
        return Path(self.train.out_dir)


def _from_dict(cls, d: dict[str, Any]):
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        if f.name == "bbox" and isinstance(v, (list, tuple)):
            v = BBox(*v)
        elif f.name == "bbox" and isinstance(v, dict):
            v = BBox(**v)
        elif dataclasses.is_dataclass(f.type) or f.name in {"data", "aerial", "cells", "split", "model",
                                                             "train", "retrieval", "overhead"}:
            sub = {"data": DataConfig, "aerial": AerialConfig, "cells": CellConfig, "split": SplitConfig,
                   "model": ModelConfig, "train": TrainConfig, "retrieval": RetrievalConfig,
                   "overhead": OverheadConfig}[f.name]
            v = _from_dict(sub, v or {})
        kwargs[f.name] = v
    return cls(**kwargs)


def load_config(path: str | os.PathLike | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config (or defaults) and apply dotted overrides such as {"train.epochs": 3}."""
    d: dict[str, Any] = {}
    if path is not None:
        with open(path) as fh:
            d = yaml.safe_load(fh) or {}
    for k, v in (overrides or {}).items():
        cur = d
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return _from_dict(Config, d)


def to_dict(cfg: Config) -> dict[str, Any]:
    d = dataclasses.asdict(cfg)
    if d["data"].get("bbox") is not None:
        d["data"]["bbox"] = list(cfg.data.bbox.as_tuple())
    return d


def parse_override(s: str) -> tuple[str, Any]:
    """Parse "key.sub=value" from the command line, YAML-typing the value."""
    k, _, v = s.partition("=")
    return k.strip(), yaml.safe_load(v)


def mapillary_token() -> str:
    tok = os.environ.get("MAPILLARY_TOKEN")
    if not tok:
        env = Path(".env")
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("MAPILLARY_TOKEN="):
                    tok = line.split("=", 1)[1].strip()
    if not tok:
        raise RuntimeError("Set MAPILLARY_TOKEN (see .env.example)")
    return tok
