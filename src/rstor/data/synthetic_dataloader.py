
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Tuple
from rstor.data.degradation import DegradationBlurMat, DegradationBlurGauss, DegradationNoise
from rstor.synthetic_data.dead_leaves_cpu import cpu_dead_leaves_chart
from rstor.synthetic_data.dead_leaves_gpu import gpu_dead_leaves_chart
import cv2
from skimage.filters import gaussian
import random
import numpy as np

from rstor.utils import DEFAULT_TORCH_FLOAT_TYPE


class DeadLeavesDataset(Dataset):
    def __init__(
        self,
        size: Tuple[int, int] = (128, 128),
        length: int = 1000,
        frozen_seed: int = None,  # useful for validation set!
        blur_kernel_half_size: int = [0, 2],
        ds_factor: int = 5,
        noise_stddev: float = [0., 50.],
        use_gaussian_kernel=True,
        **config_dead_leaves
        # number_of_circles: int = -1,
        # background_color: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        # colored: Optional[bool] = False,
        # radius_mean: Optional[int] = -1,
        # radius_stddev: Optional[int] = -1,
    ):

        self.frozen_seed = frozen_seed
        self.ds_factor = ds_factor
        self.size = (size[0]*ds_factor, size[1]*ds_factor)
        self.length = length
        self.config_dead_leaves = config_dead_leaves
        self.blur_kernel_half_size = blur_kernel_half_size
        self.noise_stddev = noise_stddev

        self.use_gaussian_kernel = use_gaussian_kernel
        if use_gaussian_kernel:
            self.degradation_blur = DegradationBlurGauss(length,
                                                         blur_kernel_half_size,
                                                         frozen_seed)
        else:
            self.degradation_blur = DegradationBlurMat(length,
                                                       frozen_seed)
        self.degradation_noise = DegradationNoise(length,
                                                  noise_stddev,
                                                  frozen_seed)
        self.current_degradation = {}

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO there is a bug on this cpu version, the dead leaved dont appear ot be right
        seed = self.frozen_seed + idx if self.frozen_seed is not None else None
        chart = cpu_dead_leaves_chart(self.size, seed=seed, **self.config_dead_leaves)

        if self.ds_factor > 1:
            # print(f"Downsampling {chart.shape} with factor {self.ds_factor}...")
            sigma = 3/5
            chart = gaussian(
                chart, sigma=(sigma, sigma, 0), mode='nearest',
                cval=0, preserve_range=True, truncate=4.0)
            chart = chart[::self.ds_factor, ::self.ds_factor]

        th_chart = torch.from_numpy(chart).permute(2, 0, 1).unsqueeze(0)

        degraded_chart = self.degradation_blur(th_chart, idx)
        degraded_chart = self.degradation_noise(degraded_chart, idx)

        blur_deg_str = "blur_kernel_half_size" if self.use_gaussian_kernel else "blur_kernel_id"
        self.current_degradation[idx] = {
            blur_deg_str: self.degradation_blur.current_degradation[idx][blur_deg_str],
            "noise_stddev": self.degradation_noise.current_degradation[idx]["noise_stddev"]
        }

        degraded_chart = degraded_chart.squeeze(0)
        th_chart = th_chart.squeeze(0)

        return degraded_chart, th_chart


class DeadLeavesDatasetGPU(Dataset):
    def __init__(
        self,
        size: Tuple[int, int] = (128, 128),
        length: int = 1000,
        frozen_seed: int = None,  # useful for validation set!
        blur_kernel_half_size: int = [0, 2],
        ds_factor: int = 5,
        noise_stddev: float = [0., 50.],
        use_gaussian_kernel=True,
        **config_dead_leaves
        # number_of_circles: int = -1,
        # background_color: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        # colored: Optional[bool] = False,
        # radius_mean: Optional[int] = -1,
        # radius_stddev: Optional[int] = -1,
    ):
        self.frozen_seed = frozen_seed
        self.ds_factor = ds_factor
        self.size = (size[0]*ds_factor, size[1]*ds_factor)
        self.length = length
        self.config_dead_leaves = config_dead_leaves

        # downsample kernel
        sigma = 3/5
        k_size = 5  # This fits with sigma = 3/5, the cutoff value is 0.0038 (neglectable)
        x = (torch.arange(k_size) - 2).to('cuda')
        kernel = torch.stack(torch.meshgrid((x, x), indexing='ij'))
        kernel.requires_grad = False
        dist_sq = kernel[0]**2 + kernel[1]**2
        kernel = (-dist_sq.square()/(2*sigma**2)).exp()
        kernel = kernel / kernel.sum()
        self.downsample_kernel = kernel.repeat(3, 1, 1, 1)  # shape [3, 1, k_size, k_size]
        self.downsample_kernel.requires_grad = False
        self.use_gaussian_kernel = use_gaussian_kernel
        if use_gaussian_kernel:
            self.degradation_blur = DegradationBlurGauss(length,
                                                         blur_kernel_half_size,
                                                         frozen_seed)
        else:
            self.degradation_blur = DegradationBlurMat(length,
                                                       frozen_seed)

        self.degradation_noise = DegradationNoise(length,
                                                  noise_stddev,
                                                  frozen_seed)
        self.current_degradation = {}

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single deadleave chart and its degraded version.

        Args:
            idx (int): index of the item to retrieve

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: degraded chart, target chart
        """
        seed = self.frozen_seed + idx if self.frozen_seed is not None else None

        # Return numba device array
        numba_chart = gpu_dead_leaves_chart(self.size, seed=seed, **self.config_dead_leaves)
        th_chart = torch.as_tensor(numba_chart, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[
            None].permute(0, 3, 1, 2)  # [1, c, h, w]
        if self.ds_factor > 1:
            # Downsample using strided gaussian conv (sigma=3/5)
            th_chart = F.pad(th_chart,
                             pad=(2, 2, 0, 0),
                             mode="replicate")
            th_chart = F.conv2d(th_chart,
                                self.downsample_kernel,
                                padding='valid',
                                groups=3,
                                stride=self.ds_factor)

        degraded_chart = self.degradation_blur(th_chart, idx)
        degraded_chart = self.degradation_noise(degraded_chart, idx)

        blur_deg_str = "blur_kernel_half_size" if self.use_gaussian_kernel else "blur_kernel_id"
        self.current_degradation[idx] = {
            blur_deg_str: self.degradation_blur.current_degradation[idx][blur_deg_str],
            "noise_stddev": self.degradation_noise.current_degradation[idx]["noise_stddev"]
        }

        degraded_chart = degraded_chart.squeeze(0)
        th_chart = th_chart.squeeze(0)

        return degraded_chart, th_chart
