from __future__ import annotations

import torch
import torch.nn.functional as F

from rcadnet.task_losses import (
    FrozenYOLOFeatureExtractor,
    HutchinsonJacobianPenalty,
    SpatiallyVaryingTDACLoss,
    TaskDrivenPerceptualLoss,
    TrainableDeepActiveContourLoss,
)


CascadedJacobianPenalty = HutchinsonJacobianPenalty


def train_step_task_driven(
    rmr_net: torch.nn.Module,
    frozen_detector: FrozenYOLOFeatureExtractor,
    optimizer: torch.optim.Optimizer,
    degraded_img: torch.Tensor,
    clean_img: torch.Tensor,
    metadata: torch.Tensor | None,
    *,
    tdp_layer_weights: list[float] | None = None,
    w_l1: float = 1.0,
    w_tdp: float = 0.5,
    w_jac: float = 0.1,
    w_ac: float = 0.2,
    tdac_loss: TrainableDeepActiveContourLoss | None = None,
    jacobian_loss: HutchinsonJacobianPenalty | None = None,
) -> dict[str, float]:
    """Single task-driven training step matching the manuscript equations.

    The main project uses ``train_rcadnet.py`` for full sweeps; this helper is
    provided for experiments and notebooks that want the exact compact block
    from Section III.H.
    """

    rmr_net.train()
    optimizer.zero_grad(set_to_none=True)

    result = rmr_net(degraded_img, metadata, return_aux=True)
    if not isinstance(result, dict):
        raise TypeError("RMR-Net must return a dict when return_aux=True")
    restored = result["restored"]

    loss_l1 = F.l1_loss(restored, clean_img)
    restored_features = frozen_detector(restored)
    with torch.no_grad():
        clean_features = [feature.detach() for feature in frozen_detector(clean_img)]

    if tdp_layer_weights is None:
        tdp_layer_weights = [1.0] * len(restored_features)
    loss_tdp = restored.new_tensor(0.0)
    for weight, restored_feat, clean_feat in zip(tdp_layer_weights, restored_features, clean_features):
        loss_tdp = loss_tdp + float(weight) * F.mse_loss(restored_feat, clean_feat)

    if jacobian_loss is None:
        jacobian_loss = HutchinsonJacobianPenalty(frozen_detector)
    loss_jac = jacobian_loss(restored)

    if tdac_loss is None:
        tdac_loss = TrainableDeepActiveContourLoss()
    loss_ac = tdac_loss(restored, result)

    total = w_l1 * loss_l1 + w_tdp * loss_tdp + w_jac * loss_jac + w_ac * loss_ac
    total.backward()
    optimizer.step()

    return {
        "loss": float(total.detach().cpu()),
        "loss_l1": float(loss_l1.detach().cpu()),
        "loss_tdp": float(loss_tdp.detach().cpu()),
        "loss_jacobian": float(loss_jac.detach().cpu()),
        "loss_tdac": float(loss_ac.detach().cpu()),
    }


__all__ = [
    "FrozenYOLOFeatureExtractor",
    "TaskDrivenPerceptualLoss",
    "CascadedJacobianPenalty",
    "HutchinsonJacobianPenalty",
    "SpatiallyVaryingTDACLoss",
    "TrainableDeepActiveContourLoss",
    "train_step_task_driven",
]
