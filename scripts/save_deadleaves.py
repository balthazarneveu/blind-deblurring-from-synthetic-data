# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 15:38:28 2024

@author: jamyl
"""
import cv2
from pathlib import Path
from time import perf_counter
import matplotlib.pyplot as plt
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from numba import cuda
from tqdm import tqdm
import argparse
from rstor.synthetic_data.dead_leaves_gpu import gpu_dead_leaves_chart
from rstor.utils import DEFAULT_TORCH_FLOAT_TYPE
from rstor.properties import DATASET_PATH, DATASET_DL_RANDOMRGB_1024


class DeadLeavesDatasetGPU(Dataset):
    def __init__(
        self,
        size: Tuple[int, int] = (128, 128),
        length: int = 1000,
        frozen_seed: int = None,  # useful for validation set!
        ds_factor: int = 5,
        **config_dead_leaves
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
        if self.ds_factor > 1:
            # print(f"Downsampling {chart.shape} with factor {self.ds_factor}...")

            # Downsample using strided gaussian conv (sigma=3/5)
            th_chart = torch.as_tensor(numba_chart, dtype=DEFAULT_TORCH_FLOAT_TYPE,
                                       device="cuda").permute(2, 0, 1)[None]  # [b, c, h, w]
            th_chart = F.pad(th_chart,
                             pad=(2, 2, 0, 0),
                             mode="replicate")
            th_chart = F.conv2d(th_chart,
                                self.downsample_kernel,
                                padding='valid',
                                groups=3,
                                stride=self.ds_factor)

            # Convert back to numba
            numba_chart = cuda.as_cuda_array(th_chart.permute(0, 2, 3, 1))  # [b, h, w, c]

        # convert back to numpy (temporary for legacy)
        chart = numba_chart.copy_to_host()[0]

        return chart


def generate_random_rgb(path, imin=0):
    dataset = DeadLeavesDatasetGPU(
        size=(1_024, 1_024),
        length=1_000,
        frozen_seed=42,
        background_color=(0.2, 0.4, 0.6),
        colored=True,
        radius_min=5,
        radius_max=2_000,
        ds_factor=5)

    for i in tqdm(range(imin, dataset.length)):
        img = dataset[i]
        img = (img * 255).astype(np.uint8)
        out_path = path / "{:04d}.png".format(i)
        cv2.imwrite(out_path.as_posix(), img)


def bench():
    dataset = DeadLeavesDatasetGPU(
        size=(1_024, 1_024),
        length=1_000,
        frozen_seed=42,
        background_color=(0.2, 0.4, 0.6),
        colored=True,
        radius_min=5,
        radius_max=2_000,
        ds_factor=5)

    print("dataset initialised")
    t1 = perf_counter()
    chart = dataset[0]

    d = (perf_counter()-t1)
    print(f"generation done {d}")
    print(f"{d*1_000/60} min for 1_000")
    plt.imshow(chart)
    plt.show()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-o", "--output-dir", type=str, default=str(DATASET_PATH))
    argparser.add_argument("-n", "--name", type=str, default=DATASET_DL_RANDOMRGB_1024)
    args = argparser.parse_args()
    dataset_dir = args.output_dir
    name = args.name
    path = Path(dataset_dir)/name
    # print(path)
    path.mkdir(parents=True, exist_ok=True)
    if name == DATASET_DL_RANDOMRGB_1024:
        generate_random_rgb(path)
    else:
        raise NotImplementedError
