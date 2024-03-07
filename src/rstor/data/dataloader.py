import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from rstor.synthetic_data.dead_leaves import dead_leaves_chart, gpu_dead_leaves_chart
from rstor.properties import DATALOADER, BATCH_SIZE, TRAIN, VALIDATION, LENGTH, CONFIG_DEAD_LEAVES, SIZE
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

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seed = self.frozen_seed + idx if self.frozen_seed is not None else None
        chart = dead_leaves_chart(self.size, seed=seed, **self.config_dead_leaves)
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

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seed = self.frozen_seed + idx if self.frozen_seed is not None else None

        # Return numba device array
        numba_chart = gpu_dead_leaves_chart(self.size, seed=seed, **self.config_dead_leaves)
        if self.ds_factor > 1:
            # print(f"Downsampling {chart.shape} with factor {self.ds_factor}...")

            # Downsample using strided gaussian conv (sigma=3/5)
            th_chart = torch.as_tensor(numba_chart, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[
                None].permute(0, 3, 1, 2)  # [1, c, h, w]
            th_chart = F.pad(th_chart,
                             pad=(2, 2, 0, 0),
                             mode="replicate")
            th_chart = F.conv2d(th_chart,
                                self.downsample_kernel,
                                padding='valid',
                                groups=3,
                                stride=self.ds_factor).squeeze(0)

            # Convert back to numba
            numba_chart = cuda.as_cuda_array(th_chart.permute(1, 2, 0))  # [h, w, c]

        # convert back to numpy (temporary for legacy)
        chart = numba_chart.copy_to_host()
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

        def numpy_to_torch(ndarray):
            return torch.from_numpy(ndarray).permute(-1, 0, 1).float()
        return numpy_to_torch(degraded_chart), numpy_to_torch(chart)


def get_data_loader(config, frozen_seed=42, **config_dead_leaves):
    dl_train = DeadLeavesDataset(config[DATALOADER][SIZE], config[DATALOADER][LENGTH][TRAIN],
                                 frozen_seed=None, **config[DATALOADER].get(CONFIG_DEAD_LEAVES, {}))
    dl_valid = DeadLeavesDataset(config[DATALOADER][SIZE], config[DATALOADER][LENGTH][VALIDATION],
                                 frozen_seed=frozen_seed, **config[DATALOADER].get(CONFIG_DEAD_LEAVES, {}))
    dl_dict = {
        TRAIN: DataLoader(
            dl_train,
            shuffle=True,
            batch_size=config[DATALOADER][BATCH_SIZE][TRAIN],
        ),
        VALIDATION: DataLoader(
            dl_valid,
            shuffle=False,
            batch_size=config[DATALOADER][BATCH_SIZE][VALIDATION]
        ),
        # TEST: DataLoader(dl_test, shuffle=False, batch_size=config[DATALOADER][BATCH_SIZE][TEST])
    }
    return dl_dict


if __name__ == "__main__":
    dead_leaves_dataset = DeadLeavesDataset(colored=True)
    dead_leaves_dataloader = DataLoader(dead_leaves_dataset, batch_size=4, shuffle=True)
    for i, (batch_inp, batch_target) in enumerate(dead_leaves_dataloader):
        print(batch_inp.shape, batch_target.shape)  # Should print [batch_size, size[0], size[1], 3] for each batch
        if i == 1:  # Just to break the loop after two batches for demonstration
            import matplotlib.pyplot as plt
            plt.subplot(1, 2, 1)
            plt.imshow(batch_inp.permute(0, 2, 3, 1).reshape(-1, 128, 3).numpy())
            plt.title("Degraded")
            plt.subplot(1, 2, 2)
            plt.imshow(batch_target.permute(0, 2, 3, 1).reshape(-1, 128, 3).numpy())
            plt.title("Target")
            plt.show()
            print(batch_target)
            break
