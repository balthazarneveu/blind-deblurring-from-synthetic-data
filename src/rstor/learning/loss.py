import torch
from typing import Optional
from rstor.properties import LOSS_MSE, LOSS_VGG16, LOSS_MIXED_MSE_VGG16
from rstor.learning.perceptual import perceptual_loss

vgg_loss_instance = None


def compute_loss(
    predic: torch.Tensor,
    target: torch.Tensor,
    mode: Optional[str] = LOSS_MSE
) -> torch.Tensor:
    """
    Compute loss based on the predicted and true values.

    Args:
        predic (torch.Tensor): [N, C, H, W] predicted values
        target (torch.Tensor): [N, C, H, W] target values.
        mode (Optional[str], optional): mode of loss computation.

    Returns:
        torch.Tensor: The computed loss.
    """
    assert mode in [LOSS_MSE, LOSS_VGG16, LOSS_MIXED_MSE_VGG16], f"Mode {mode} not supported"
    if mode == LOSS_MSE:
        loss = torch.nn.functional.mse_loss(predic, target)
    elif mode == LOSS_VGG16:
        target.requires_grad = False
        loss = perceptual_loss(predic, target, device=predic.device, no_grad=False)
    elif mode == LOSS_MIXED_MSE_VGG16:
        # MSE ~ 2.10-3
        # VGG ~ 1.5
        mse_loss = torch.nn.functional.mse_loss(predic, target)
        vgg_loss = 1.E-3 * perceptual_loss(predic, target, device=predic.device, no_grad=False)
        alpha = 0.5
        # alpha = 0 -> pure MSE
        # alpha = 1 -> pure VGG
        loss = (1.-alpha)*mse_loss + alpha*vgg_loss
    return loss
