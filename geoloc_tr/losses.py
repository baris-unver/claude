"""Losses: distance-smoothed hierarchical cross-entropy and ground<->aerial InfoNCE."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .geo import EARTH_RADIUS_M


def smoothed_targets(gt_xyz: torch.Tensor, centers_xyz: torch.Tensor, labels: torch.Tensor, sigma_m: float,
                     hard_frac: float = 0.5) -> torch.Tensor:
    """Soft targets over cells: mixture of the one-hot label and a Gaussian on cell-centre distance.

    gt_xyz: (B,3) unit vectors, centers_xyz: (C,3) unit vectors, labels: (B,) valid class indices.
    """
    chord = torch.cdist(gt_xyz, centers_xyz)  # (B,C), chord length on unit sphere ~ arc for small angles
    d_m = chord * EARTH_RADIUS_M
    w = torch.exp(-0.5 * (d_m / sigma_m) ** 2)
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
    onehot = F.one_hot(labels, centers_xyz.shape[0]).to(w.dtype)
    return hard_frac * onehot + (1.0 - hard_frac) * w


def hierarchical_loss(logits: list[torch.Tensor], labels: torch.Tensor, gt_xyz: torch.Tensor,
                      centers_xyz: list[torch.Tensor], sigmas_m: list[float], hard_frac: float = 0.5,
                      level_weights: list[float] | None = None) -> tuple[torch.Tensor, dict]:
    total = gt_xyz.new_zeros(())
    stats = {}
    level_weights = level_weights or [1.0] * len(logits)
    for li, (lg, cen, sig, wgt) in enumerate(zip(logits, centers_xyz, sigmas_m, level_weights)):
        lab = labels[:, li]
        valid = lab >= 0
        if valid.sum() == 0:
            continue
        lg_v = lg[valid].float()
        tgt = smoothed_targets(gt_xyz[valid], cen, lab[valid], sig, hard_frac)
        loss = -(tgt * F.log_softmax(lg_v, dim=1)).sum(1).mean()
        total = total + wgt * loss
        with torch.no_grad():
            stats[f"loss_l{li}"] = float(loss)
            stats[f"acc_l{li}"] = float((lg_v.argmax(1) == lab[valid]).float().mean())
    return total, stats


def ground_aerial_infonce(ground: torch.Tensor, aerial: torch.Tensor, cells: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric InfoNCE between ground embeddings and the aerial embedding of their cell.

    Samples from the same cell are not treated as negatives of each other.
    """
    sim = ground @ aerial.t() / temperature  # (B,B)
    same = cells[:, None] == cells[None, :]
    eye = torch.eye(len(cells), dtype=torch.bool, device=cells.device)
    sim = sim.masked_fill(same & ~eye, float("-inf"))
    target = torch.arange(len(cells), device=cells.device)
    return 0.5 * (F.cross_entropy(sim, target) + F.cross_entropy(sim.t(), target))
