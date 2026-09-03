"""Training loop for the proxy-classification model (+ ground<->aerial contrastive term)."""
from __future__ import annotations

import csv
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .aerial import AerialPatchStore
from .config import Config
from .data import GroundDataset
from .database import build_database, embed_images
from .evaluate import errors_m, recall_at
from .geo import CellHierarchy, cell_edge_m, latlon_to_unit
from .localize import localize
from .losses import ground_aerial_infonce, hierarchical_loss
from .model import GeoLocModel, save_checkpoint

log = logging.getLogger(__name__)


def _lr_lambda(total_steps: int, warmup_steps: int):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return f


@torch.no_grad()
def quick_validate(model: GeoLocModel, val_ds, hierarchy: CellHierarchy, cfg: Config, device) -> dict:
    db = build_database(model, hierarchy, cfg, aerial_store=None, device=device, progress=False)
    q = embed_images(model, val_ds, cfg.train.batch_size, cfg.train.num_workers, device, progress=False)
    gt = np.stack([val_ds.lat, val_ds.lon], 1)
    res = localize(q, db, cfg.retrieval, alpha=0.0, image_index=None, refine=True)
    return recall_at(errors_m(res.latlon, gt), cfg.retrieval.eval_thresholds_m)


def train(cfg: Config, train_df: pd.DataFrame, val_df: pd.DataFrame | None, hierarchy: CellHierarchy,
          aerial_store: AerialPatchStore | None, device, pretrained: bool = True) -> tuple[GeoLocModel, Path]:
    """Ground pipeline: street-level images (+ the aerial patch of their cell) -> fit()."""
    use_aerial = cfg.model.aerial_enabled and aerial_store is not None
    train_ds = GroundDataset(train_df, hierarchy, cfg.model.image_size, train=True,
                             aerial=aerial_store if use_aerial else None, db_level=cfg.cells.database_level,
                             augment=cfg.train.augment)
    val_ds = GroundDataset(val_df, hierarchy, cfg.model.image_size, train=False) if val_df is not None and len(val_df) else None
    return fit(cfg, train_ds, val_ds, hierarchy, device, pretrained=pretrained, use_aerial=use_aerial)


def fit(cfg: Config, train_ds: Dataset, val_ds: Dataset | None, hierarchy: CellHierarchy, device,
        pretrained: bool = True, use_aerial: bool = False) -> tuple[GeoLocModel, Path]:
    """Training loop. `train_ds` yields {"image", "labels", "latlon"} (+ {"aerial", "cell"} when
    `use_aerial`); `val_ds` must expose `.lat` / `.lon` arrays for the per-epoch retrieval validation."""
    torch.manual_seed(cfg.train.seed)
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers,
                    drop_last=True, pin_memory=(str(device).startswith("cuda")), persistent_workers=cfg.train.num_workers > 0)

    model = GeoLocModel(cfg.model, [lc.num_classes for lc in hierarchy.levels], pretrained=pretrained).to(device)
    opt = torch.optim.AdamW(model.param_groups(cfg.train.lr, cfg.train.backbone_lr_mult, cfg.train.weight_decay))
    steps_per_epoch = len(dl) if cfg.train.max_steps_per_epoch is None else min(len(dl), cfg.train.max_steps_per_epoch)
    total = steps_per_epoch * cfg.train.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(total, int(cfg.train.warmup_epochs * steps_per_epoch)))
    use_amp = cfg.train.amp and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    centers_xyz = [torch.from_numpy(latlon_to_unit(lc.centers[:, 0], lc.centers[:, 1])).float().to(device)
                   for lc in hierarchy.levels]
    sigmas = [cfg.cells.label_smoothing_sigma_scale * cell_edge_m(lc.level) for lc in hierarchy.levels]

    log_path = out / "train_log.csv"
    best = (-1.0, -1.0, -float("inf"))
    step = 0
    with open(log_path, "w", newline="") as fh:
        writer = None
        for epoch in range(cfg.train.epochs):
            model.train()
            t0 = time.time()
            agg: dict[str, float] = {}
            n = 0
            for bi, batch in enumerate(dl):
                if bi >= steps_per_epoch:
                    break
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["labels"].to(device)
                gt = torch.from_numpy(latlon_to_unit(batch["latlon"][:, 0].numpy(), batch["latlon"][:, 1].numpy())).float().to(device)
                with torch.autocast("cuda", enabled=use_amp):
                    emb, logits, extra = model(images, return_extra=True)
                    loss, stats = hierarchical_loss(logits, labels, gt, centers_xyz, sigmas)
                    if "log_scale" in extra and "log_scale" in batch:  # overhead views: extent regression
                        l_s = F.smooth_l1_loss(extra["log_scale"].float(), batch["log_scale"].to(device).float())
                        loss = loss + cfg.overhead.scale_loss_weight * l_s
                        stats["loss_scale"] = float(l_s.detach())
                    if use_aerial and "aerial" in batch:
                        a_emb = model.encode_aerial(batch["aerial"].to(device, non_blocking=True))
                        l_a = ground_aerial_infonce(emb.float(), a_emb.float(), batch["cell"].to(device), cfg.model.aerial_temperature)
                        loss = loss + cfg.model.aerial_loss_weight * l_a
                        stats["loss_aerial"] = float(l_a.detach())
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                step += 1
                stats["loss"] = float(loss.detach())
                for k, v in stats.items():
                    agg[k] = agg.get(k, 0.0) + v
                n += 1
                if step % 50 == 0:
                    log.info("ep %d step %d loss %.3f %s", epoch, step, stats["loss"],
                             " ".join(f"{k}={v:.3f}" for k, v in stats.items() if k.startswith("acc")))
            row = {"epoch": epoch, "time_s": time.time() - t0, **{k: v / max(n, 1) for k, v in agg.items()}}
            if val_ds is not None and ((epoch + 1) % cfg.train.eval_every == 0 or epoch + 1 == cfg.train.epochs):
                val = quick_validate(model, val_ds, hierarchy, cfg, device)
                row.update({f"val_{k}": v for k, v in val.items()})
                # lexicographic: recall@100m, then @200m, then lower median error
                score = (val.get("R@100m", 0.0), val.get("R@200m", 0.0), -val.get("median_m", float("inf")))
                if score > best:
                    best = score
                    save_checkpoint(out / "best.pt", model, cfg, hierarchy, {"epoch": epoch, "val": val})
            log.info("epoch %d: %s", epoch, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row)
            fh.flush()
            save_checkpoint(out / "last.pt", model, cfg, hierarchy, {"epoch": epoch})
    if not (out / "best.pt").exists():
        save_checkpoint(out / "best.pt", model, cfg, hierarchy, {"epoch": cfg.train.epochs - 1})
    return model, out / "best.pt"
