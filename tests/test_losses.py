import numpy as np
import torch

from geoloc_tr.geo import latlon_to_unit
from geoloc_tr.losses import ground_aerial_infonce, hierarchical_loss, smoothed_targets


def test_smoothed_targets_peak_on_gt():
    centers = np.array([[39.90, 32.80], [39.90, 32.801], [39.95, 32.90]])
    cxyz = torch.from_numpy(latlon_to_unit(centers[:, 0], centers[:, 1])).float()
    gt = torch.from_numpy(latlon_to_unit([39.9001], [32.8002])).float()
    t = smoothed_targets(gt, cxyz, torch.tensor([0]), sigma_m=100.0, hard_frac=0.5)
    assert torch.allclose(t.sum(1), torch.ones(1), atol=1e-5)
    assert t[0, 0] > t[0, 1] > t[0, 2]
    assert t[0, 2] < 1e-3  # 6 km away


def test_hierarchical_loss_ignores_invalid():
    torch.manual_seed(0)
    logits = [torch.randn(4, 5, requires_grad=True), torch.randn(4, 7, requires_grad=True)]
    labels = torch.tensor([[0, 1], [1, -1], [-1, 2], [4, 6]])
    gt = torch.from_numpy(latlon_to_unit(np.full(4, 39.9), np.full(4, 32.8))).float()
    cen = [torch.from_numpy(latlon_to_unit(np.full(n, 39.9), 32.8 + np.arange(n) * 1e-3)).float() for n in (5, 7)]
    loss, stats = hierarchical_loss(logits, labels, gt, cen, [100.0, 50.0])
    loss.backward()
    assert torch.isfinite(loss) and "acc_l0" in stats
    assert logits[0].grad[2].abs().sum() == 0  # row with label -1 at level 0 gets no gradient there


def test_infonce_same_cell_not_negative():
    g = torch.nn.functional.normalize(torch.randn(6, 16), dim=1)
    a = torch.nn.functional.normalize(torch.randn(6, 16), dim=1)
    cells = torch.tensor([1, 1, 2, 3, 3, 3])
    loss = ground_aerial_infonce(g, a, cells, 0.1)
    assert torch.isfinite(loss)
    # if all pairs share one cell there are no negatives -> loss is exactly 0
    assert ground_aerial_infonce(g, a, torch.zeros(6, dtype=torch.long), 0.1).abs() < 1e-6
