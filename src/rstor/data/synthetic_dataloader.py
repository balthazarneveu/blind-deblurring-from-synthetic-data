
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Tuple
from rstor.data.degradation import DegradationBlur, DegradationNoise
from rstor.synthetic_data.dead_leaves_cpu import cpu_dead_leaves_chart
from rstor.synthetic_data.dead_leaves_gpu import gpu_dead_leaves_chart
import cv2
from skimage.filters import gaussian
import random
import numpy as np

from numba import cuda

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
        if frozen_seed is not None:
            random.seed(self.frozen_seed)
            self.blur_kernel_half_size = [
                (
                    random.randint(self.blur_kernel_half_size[0], self.blur_kernel_half_size[1]),
                    random.randint(self.blur_kernel_half_size[0], self.blur_kernel_half_size[1])
                ) for _ in range(length)
            ]
            self.noise_stddev = [(self.noise_stddev[1] - self.noise_stddev[0]) *
                                 random.random() + self.noise_stddev[0] for _ in range(length)]
        self.current_degradation = {}

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seed = self.frozen_seed + idx if self.frozen_seed is not None else None
        chart = cpu_dead_leaves_chart(self.size, seed=seed, **self.config_dead_leaves)
        if self.ds_factor > 1:
            # print(f"Downsampling {chart.shape} with factor {self.ds_factor}...")
            sigma = 3/5
            chart = gaussian(
                chart, sigma=(sigma, sigma, 0), mode='nearest',
                cval=0, preserve_range=True, truncate=4.0)
            chart = chart[::self.ds_factor, ::self.ds_factor]
        if self.frozen_seed is not None:
            k_size_x, k_size_y = self.blur_kernel_half_size[idx]
            std_dev = self.noise_stddev[idx]
        else:
            k_size_x = random.randint(self.blur_kernel_half_size[0], self.blur_kernel_half_size[1])
            k_size_y = random.randint(self.blur_kernel_half_size[0], self.blur_kernel_half_size[1])
            std_dev = (self.noise_stddev[1] - self.noise_stddev[0]) * random.random() + self.noise_stddev[0]
        k_size_x = 2 * k_size_x + 1
        k_size_y = 2 * k_size_y + 1
        degraded_chart = cv2.GaussianBlur(chart, (k_size_x, k_size_y), 0)
        if std_dev > 0.:
            # print(f"Adding noise with std_dev={std_dev}...")
            degraded_chart += (std_dev/255.)*np.random.randn(*degraded_chart.shape)
        self.current_degradation[idx] = {
            "blur_kernel_half_size": (k_size_x, k_size_y),
            "noise_stddev": std_dev
        }

        def numpy_to_torch(ndarray):
            return torch.from_numpy(ndarray).permute(-1, 0, 1).float()
        return numpy_to_torch(degraded_chart), numpy_to_torch(chart)


class DeadLeavesDatasetGPU(Dataset):
    def __init__(
        self,
        size: Tuple[int, int] = (128, 128),
        length: int = 1000,
        frozen_seed: int = None,  # useful for validation set!
        blur_kernel_half_size: int = [0, 2],
        ds_factor: int = 5,
        noise_stddev: float = [0., 50.],
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
        dist_sq = kernel[0]**2 + kernel[1]**2
        kernel = (-dist_sq.square()/(2*sigma**2)).exp()
        kernel = kernel / kernel.sum()
        self.downsample_kernel = kernel.repeat(3, 1, 1, 1)  # shape [3, 1, k_size, k_size]

        self.degradation_blur = DegradationBlur(length,
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
        
        self.current_degradation[idx] = {
            "blur_kernel_id": self.degradation_blur.current_degradation[idx]["blur_kernel_id"],
            "noise_stddev": self.degradation_noise.current_degradation[idx]["noise_stddev"]
        }

        return degraded_chart.squeeze(0), th_chart.squeeze(0)
