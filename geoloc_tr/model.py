"""Ground encoder + hierarchical cosine classifiers (prototypes) + optional aerial encoder."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config, ModelConfig, load_config, to_dict
from .geo import CellHierarchy


class TinyBackbone(nn.Module):
    """Small conv net for offline smoke tests / tiny synthetic data. Not meant for real accuracy."""

    def __init__(self, width: int = 32, out_dim: int = 128):
        super().__init__()
        c = width
        self.net = nn.Sequential(
            nn.Conv2d(3, c, 3, 2, 1), nn.BatchNorm2d(c), nn.GELU(),
            nn.Conv2d(c, 2 * c, 3, 2, 1), nn.BatchNorm2d(2 * c), nn.GELU(),
            nn.Conv2d(2 * c, 4 * c, 3, 2, 1), nn.BatchNorm2d(4 * c), nn.GELU(),
            nn.Conv2d(4 * c, out_dim, 3, 2, 1), nn.BatchNorm2d(out_dim), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.num_features = out_dim

    def forward(self, x):
        return self.net(x)


def build_backbone(name: str, image_size: int, pretrained: bool = True) -> tuple[nn.Module, int]:
    if name == "tiny":
        m = TinyBackbone()
        return m, m.num_features
    if name.startswith("timm:"):
        import timm

        arch = name.split(":", 1)[1]
        kwargs: dict[str, Any] = {"pretrained": pretrained, "num_classes": 0}
        if "vit" in arch or "dinov2" in arch or "eva" in arch:
            kwargs["img_size"] = image_size
            kwargs["dynamic_img_size"] = True
        m = timm.create_model(arch, **kwargs)
        return m, m.num_features
    raise ValueError(f"unknown backbone {name!r} (use 'tiny' or 'timm:<model_name>')")


def freeze_vit_blocks(backbone: nn.Module, n: int) -> None:
    if n <= 0:
        return
    for attr in ("patch_embed", "pos_embed", "cls_token", "pos_drop", "norm_pre"):
        mod = getattr(backbone, attr, None)
        if isinstance(mod, nn.Module):
            for p in mod.parameters():
                p.requires_grad_(False)
        elif isinstance(mod, torch.Tensor):
            mod.requires_grad_(False)
    blocks = getattr(backbone, "blocks", None)
    if blocks is not None:
        for b in list(blocks)[:n]:
            for p in b.parameters():
                p.requires_grad_(False)


class Encoder(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, embed_dim: int):
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Sequential(nn.Linear(feat_dim, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)[1]

    def features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(backbone features, unit embedding)"""
        f = self.backbone(x)
        return f, F.normalize(self.proj(f), dim=-1)


class CosineHead(nn.Module):
    """Linear classifier without bias whose (normalised) rows are the class prototypes."""

    def __init__(self, embed_dim: int, num_classes: int, temperature: float):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)
        self.temperature = temperature

    @property
    def prototypes(self) -> torch.Tensor:
        return F.normalize(self.weight, dim=-1)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return emb @ self.prototypes.t() / self.temperature


class GeoLocModel(nn.Module):
    def __init__(self, mcfg: ModelConfig, num_classes: list[int], pretrained: bool = True):
        super().__init__()
        self.mcfg = mcfg
        bb, fd = build_backbone(mcfg.backbone, mcfg.image_size, pretrained)
        freeze_vit_blocks(bb, mcfg.freeze_backbone_blocks)
        self.ground = Encoder(bb, fd, mcfg.embed_dim)
        self.heads = nn.ModuleList([CosineHead(mcfg.embed_dim, c, mcfg.temperature) for c in num_classes])
        self.scale_head: nn.Module | None = None
        if mcfg.scale_head:
            self.scale_head = nn.Sequential(nn.Linear(fd, 256), nn.GELU(), nn.Linear(256, 1))
        self.aerial: Encoder | None = None
        if mcfg.aerial_enabled:
            abb, afd = build_backbone(mcfg.aerial_backbone or mcfg.backbone, mcfg.image_size, pretrained)
            freeze_vit_blocks(abb, mcfg.freeze_backbone_blocks)
            self.aerial = Encoder(abb, afd, mcfg.embed_dim)

    def forward(self, images: torch.Tensor, return_extra: bool = False):
        feat, emb = self.ground.features(images)
        logits = [h(emb) for h in self.heads]
        if not return_extra:
            return emb, logits
        extra = {}
        if self.scale_head is not None:
            extra["log_scale"] = self.scale_head(feat).squeeze(-1)
        return emb, logits, extra

    def encode_aerial(self, patches: torch.Tensor) -> torch.Tensor:
        assert self.aerial is not None
        return self.aerial(patches)

    def param_groups(self, lr: float, backbone_mult: float, weight_decay: float) -> list[dict]:
        bb, rest = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (bb if ".backbone." in n else rest).append(p)
        return [{"params": bb, "lr": lr * backbone_mult, "weight_decay": weight_decay},
                {"params": rest, "lr": lr, "weight_decay": weight_decay}]


# ---------------------------------------------------------------------------------------------
# checkpoint IO
# ---------------------------------------------------------------------------------------------
def save_checkpoint(path: Path, model: GeoLocModel, cfg: Config, hierarchy: CellHierarchy, extra: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": to_dict(cfg),
        "hierarchy": hierarchy.to_dict(),
        "extra": extra or {},
    }, path)


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[GeoLocModel, Config, CellHierarchy, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    import yaml, tempfile, os  # noqa: E401
    cfg = _config_from_dict(ck["config"])
    hierarchy = CellHierarchy.from_dict(ck["hierarchy"])
    model = GeoLocModel(cfg.model, [lc.num_classes for lc in hierarchy.levels], pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, cfg, hierarchy, ck.get("extra", {})


def _config_from_dict(d: dict) -> Config:
    from .config import _from_dict
    return _from_dict(Config, d)
