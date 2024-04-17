import torch
from rstor.properties import (
    METRIC_PSNR, METRIC_SSIM, METRIC_LPIPS, METRIC_PERCEPTUAL,
    REDUCTION_AVERAGE, REDUCTION_SKIP, REDUCTION_SUM
)
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from rstor.learning.perceptual import perceptual_loss
from typing import List, Optional
ALL_METRICS = [METRIC_PSNR, METRIC_SSIM, METRIC_LPIPS]

vgg_instance = None


def compute_psnr(
    predic: torch.Tensor,
    target: torch.Tensor,
    clamp_mse=1e-10,
    reduction: Optional[str] = REDUCTION_AVERAGE
) -> torch.Tensor:
    """
    Compute the average PSNR metric for a batch of predicted and true values.

    Args:
        predic (torch.Tensor): [N, C, H, W] predicted values.
        target (torch.Tensor): [N, C, H, W] target values.
        reduction (str): Reduction method. REDUCTION_AVERAGE/REDUCTION_SKIP/REDUCTION_SUM.

    Returns:
        torch.Tensor: The average PSNR value for the batch.
    """
    with torch.no_grad():
        mse_per_image = torch.mean((predic - target) ** 2, dim=(-3, -2, -1))
        mse_per_image = torch.clamp(mse_per_image, min=clamp_mse)
        psnr_per_image = 10 * torch.log10(1 / mse_per_image)
        if reduction == REDUCTION_AVERAGE:
            average_psnr = torch.mean(psnr_per_image)
        elif reduction == REDUCTION_SUM:
            average_psnr = torch.sum(psnr_per_image)
        elif reduction == REDUCTION_SKIP:
            average_psnr = psnr_per_image
        else:
            raise ValueError(f"Unknown reduction {reduction}")
    return average_psnr


def compute_ssim(
    predic: torch.Tensor,
    target: torch.Tensor,
    reduction: Optional[str] = REDUCTION_AVERAGE
) -> torch.Tensor:
    """
    Compute the average SSIM metric for a batch of predicted and true values.

    Args:
        predic (torch.Tensor): [N, C, H, W] predicted values.
        target (torch.Tensor): [N, C, H, W] target values.
        reduction (str): Reduction method. REDUCTION_AVERAGE/REDUCTION_SKIP.

    Returns:
        torch.Tensor: The average SSIM value for the batch.
    """
    with torch.no_grad():
        reduction_mode = {
            REDUCTION_SKIP: None,
            REDUCTION_AVERAGE: "elementwise_mean",
            REDUCTION_SUM: "sum"
        }[reduction]
        ssim = SSIM(data_range=1.0, reduction=reduction_mode).to(predic.device)
        assert predic.shape == target.shape, f"{predic.shape} != {target.shape}"
        assert predic.device == target.device, f"{predic.device} != {target.device}"
        ssim_value = ssim(predic, target)
    return ssim_value


def compute_lpips(
    predic: torch.Tensor,
    target: torch.Tensor,
    reduction: Optional[str] = REDUCTION_AVERAGE,
) -> torch.Tensor:
    """
    Compute the average LPIPS metric for a batch of predicted and true values.
    https://richzhang.github.io/PerceptualSimilarity/

    Args:
        predic (torch.Tensor): [N, C, H, W] predicted values.
        target (torch.Tensor): [N, C, H, W] target values.
        reduction (str): Reduction method. REDUCTION_AVERAGE/REDUCTION_SKIP.

    Returns:
        torch.Tensor: The average SSIM value for the batch.
    """
    reduction_mode = {
        REDUCTION_SKIP:  "sum",  # does not really matter
        REDUCTION_AVERAGE: "mean",
        REDUCTION_SUM: "sum"
    }[reduction]

    with torch.no_grad():
        lpip_metrics = LearnedPerceptualImagePatchSimilarity(
            reduction=reduction_mode,
            normalize=True  # If set to True will instead expect input to be in the [0,1] range.
        ).to(predic.device)
        assert predic.shape == target.shape, f"{predic.shape} != {target.shape}"
        assert predic.device == target.device, f"{predic.device} != {target.device}"
        if reduction == REDUCTION_SKIP:
            lpip_value = []
            for idx in range(predic.shape[0]):
                lpip_value.append(lpip_metrics(
                    predic[idx, ...].unsqueeze(0).clip(0, 1),
                    target[idx, ...].unsqueeze(0).clip(0, 1)
                ))
            lpip_value = torch.stack(lpip_value)
        elif reduction in [REDUCTION_SUM, REDUCTION_AVERAGE]:
            lpip_value = lpip_metrics(predic.clip(0, 1), target.clip(0, 1))
    return lpip_value


def compute_metrics(
        predic: torch.Tensor,
        target: torch.Tensor,
        reduction: Optional[str] = REDUCTION_AVERAGE,
        chosen_metrics: Optional[List[str]] = ALL_METRICS) -> dict:
    """
    Compute the metrics for a batch of predicted and true values.

    Args:
        predic (torch.Tensor): [N, C, H, W] predicted values.
        target (torch.Tensor): [N, C, H, W] target values.
        reduction (str): Reduction method. REDUCTION_AVERAGE/REDUCTION_SKIP/REDUCTION SUM.
        chosen_metrics (list): List of metrics to compute, default [METRIC_PSNR, METRIC_SSIM]

    Returns:
        dict: computed metrics.
    """
    metrics = {}
    if METRIC_PSNR in chosen_metrics:
        average_psnr = compute_psnr(predic, target, reduction=reduction)
        metrics[METRIC_PSNR] = average_psnr.item() if reduction != REDUCTION_SKIP else average_psnr
    if METRIC_SSIM in chosen_metrics:
        ssim_value = compute_ssim(predic, target, reduction=reduction)
        metrics[METRIC_SSIM] = ssim_value.item() if reduction != REDUCTION_SKIP else ssim_value
    if METRIC_LPIPS in chosen_metrics:
        lpip_value = compute_lpips(predic, target, reduction=reduction)
        metrics[METRIC_LPIPS] = lpip_value.item() if reduction != REDUCTION_SKIP else lpip_value
    if METRIC_PERCEPTUAL in chosen_metrics:
        with torch.no_grad():
            perceptual_l = perceptual_loss(predic, target, no_grad=True)
        metrics[METRIC_PERCEPTUAL] = perceptual_l.item() if reduction != REDUCTION_SKIP else perceptual_l
    return metrics
