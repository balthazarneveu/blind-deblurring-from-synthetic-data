import torch
from rstor.learning.metrics import compute_psnr, compute_ssim, compute_metrics
from rstor.properties import REDUCTION_AVERAGE, REDUCTION_SKIP


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
    assert ssim_per_unit.shape == (8,), "Test case 1 failed"
    assert ssim_per_unit.mean() == ssim, "Test case 2 failed"


def test_compute_metrics():
    x = torch.randn(8, 3, 256, 256)
    y = x.clone() + torch.randn(8, 3, 256, 256) * 0.01
    metrics = compute_metrics(x, y)
    print(metrics)
    metric_per_image = compute_metrics(x, y, reduction=REDUCTION_SKIP)
    print(metric_per_image)
