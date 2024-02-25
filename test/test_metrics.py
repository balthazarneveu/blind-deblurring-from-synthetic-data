import torch
from rstor.learning.metrics import compute_psnr


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
