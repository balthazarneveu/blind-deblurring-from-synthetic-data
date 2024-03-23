import torch
import numpy as np
from rstor.learning.metrics import compute_psnr, compute_ssim, compute_metrics, compute_lpips
from rstor.properties import REDUCTION_AVERAGE, REDUCTION_SKIP, REDUCTION_SUM, DEVICE
from rstor.properties import METRIC_PSNR, METRIC_SSIM, METRIC_LPIPS


def test_compute_psnr():

    # Test case 1: Identical values
    predic = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    assert torch.isinf(compute_psnr(predic, target, clamp_mse=0)), "Test case 1 failed"

    # Test case 2: Predic and target have different values
    predic = torch.tensor([[[[0., 0.], [0., 0.]]]])
    target = torch.tensor([[[[0.25, 0.25], [0.25, 0.25]]]])
    assert compute_psnr(predic, target).item() == (10. * torch.log10(torch.Tensor([4.**2]))).item()  # 12db

    print("All tests passed.")


def test_compute_ssim():
    x = torch.rand(8, 3, 256, 256)
    y = torch.rand(8, 3, 256, 256)
    ssim = compute_ssim(x, y, reduction=REDUCTION_AVERAGE)
    ssim_per_unit = compute_ssim(x, y, reduction=REDUCTION_SKIP)
    assert ssim_per_unit.shape == (8,), "SSIM Test case 1 failed"
    assert ssim_per_unit.mean() == ssim, "SSIM Test case 2 failed"


def test_compute_lpips():
    for i in range(2):
        x = torch.rand(8, 3, 256, 256).to(DEVICE)
        y = torch.rand(8, 3, 256, 256).to(DEVICE)
        lpips = compute_lpips(x, y, reduction=REDUCTION_AVERAGE)
        lpips_per_unit = compute_lpips(x, y, reduction=REDUCTION_SKIP)
        assert lpips_per_unit.shape == (8,), "LPIPS Test case 1 failed"
        assert torch.isclose(lpips_per_unit.mean(), lpips), "LPIPS Test case 2 failed"


def test_compute_metrics():
    x = torch.rand(8, 3, 256, 256)  # negative value ensures that we check clamping for LPIPS
    y = x.clone() + torch.randn(8, 3, 256, 256) * 0.01
    metrics = compute_metrics(x, y)
    print(metrics)
    metric_per_image = compute_metrics(x, y, reduction=REDUCTION_SKIP)

    metric_sum_reduction = compute_metrics(x, y, reduction=REDUCTION_SUM)
    assert metric_per_image[METRIC_PSNR].shape == (8,), "Metrics Test case 1 failed"
    assert metric_per_image[METRIC_SSIM].shape == (8,), "Metrics Test case 2 failed"
    assert metric_per_image[METRIC_LPIPS].shape == (8,), "Metrics Test case 3 failed"
    assert np.isclose(metric_per_image[METRIC_PSNR].mean().item(), metrics[METRIC_PSNR]), "Metrics Test case 4 failed"
    assert np.isclose(metric_per_image[METRIC_PSNR].sum().item(),
                      metric_sum_reduction[METRIC_PSNR]), "Metrics Test case 5 failed"
    assert np.isclose(metrics[METRIC_PSNR],
                      metric_sum_reduction[METRIC_PSNR]/8.), "Metrics Test case 6 failed"
