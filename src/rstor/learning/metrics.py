import torch
from rstor.properties import METRIC_PSNR, REDUCTION_AVERAGE, REDUCTION_SKIP


def compute_psnr(
    predic: torch.Tensor,
    target: torch.Tensor,
    clamp_mse=1e-10,
    reduction=REDUCTION_AVERAGE
) -> torch.Tensor:
    """
    Compute the average PSNR metric for a batch of predicted and true values.

    Args:
        predic (torch.Tensor): [N, C, H, W] predicted values.
        target (torch.Tensor): [N, C, H, W] target values.

    Returns:
        torch.Tensor: The average PSNR value for the batch.
    """
    mse_per_image = torch.mean((predic - target) ** 2, dim=(-3, -2, -1))
    mse_per_image = torch.clamp(mse_per_image, min=clamp_mse)
    psnr_per_image = 10 * torch.log10(1 / mse_per_image)
    if reduction == REDUCTION_AVERAGE:
        average_psnr = torch.mean(psnr_per_image)
    elif reduction == REDUCTION_SKIP:
        average_psnr = psnr_per_image
    else:
        raise ValueError(f"Unknown reduction {reduction}")
    return average_psnr


def compute_metrics(predic: torch.Tensor, target: torch.Tensor, reduction=REDUCTION_AVERAGE) -> dict:
    """
    Compute the metrics for a batch of predicted and true values.

    Args:
        predic (torch.Tensor): [N, C, H, W] predicted values.
        target (torch.Tensor): [N, C, H, W] target values.

    Returns:
        dict: computed metrics.
    """
    average_psnr = compute_psnr(predic, target, reduction=reduction)
    metrics = {METRIC_PSNR: average_psnr.item() if reduction != REDUCTION_SKIP else average_psnr}
    return metrics
