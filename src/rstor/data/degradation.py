# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 01:21:46 2024

@author: jamyl
"""
import torch
import torch.nn.functional as F
from rstor.properties import DATALOADER, BATCH_SIZE, TRAIN, VALIDATION, LENGTH, CONFIG_DEAD_LEAVES, SIZE, DATASET_BLUR_KERNEL_PATH, DEVICE
import random
from scipy.io import loadmat


class Degradation():
    def __init__(self,
                 length: int = 1000,
                 frozen_seed: int = None):
        self.frozen_seed = frozen_seed
        self.current_degradation = {}


class DegradationNoise(Degradation):
    def __init__(self,
                 length: int = 1000,
                 noise_stddev: float = [0., 50.],
                 frozen_seed: int = None):
        super().__init__(length, frozen_seed)
        self.noise_stddev = noise_stddev

        if frozen_seed is not None:
            random.seed(frozen_seed)
            self.noise_stddev = [(self.noise_stddev[1] - self.noise_stddev[0]) *
                                 random.random() + self.noise_stddev[0] for _ in range(length)]

    def __call__(self, x: torch.Tensor, idx: int):
        # expects x of shape [b, c, h, w]
        assert x.ndim == 4
        assert x.shape[1] in [1, 3]

        if self.frozen_seed is not None:
            std_dev = self.noise_stddev[idx]
        else:
            std_dev = (self.noise_stddev[1] - self.noise_stddev[0]) * random.random() + self.noise_stddev[0]

        if std_dev > 0.:
            # x += (std_dev/255.)*np.random.randn(*x.shape)
            x += (std_dev/255.)*torch.randn(*x.shape, device=x.device)
        self.current_degradation[idx] = {
            "noise_stddev": std_dev
        }
        return x


class DegradationBlur(Degradation):
    def __init__(self,
                 length: int = 1000,
                 frozen_seed: int = None):
        super().__init__(length, frozen_seed)

        kernels = loadmat(DATASET_BLUR_KERNEL_PATH)["kernels"].squeeze()
        # conversion to torch (the shape of the kernel is not constant)
        self.kernels = tuple([
            torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)
            for kernel in kernels])
        self.n_kernels = len(self.kernels)

        if frozen_seed is not None:
            random.seed(frozen_seed)
            self.kernel_ids = [random.randint(0, self.n_kernels-1) for _ in range(length)]

    def __call__(self, x: torch.Tensor, idx: int):
        # expects x of shape [b, c, h, w]
        assert x.ndim == 4
        assert x.shape[1] in [1, 3]
        device = x.device

        if self.frozen_seed is not None:
            kernel_id = self.kernel_ids[idx]
        else:
            kernel_id = random.randint(0, self.n_kernels-1)

        kernel = self.kernels[kernel_id].to(device).repeat(3, 1, 1, 1).float()  # repeat for grouped conv

        # We use padding = same to make
        # sure that the output size does not depend on the kernel.
        x = F.conv2d(x, kernel, padding="same", groups=3)

        self.current_degradation[idx] = {
            "blur_kernel_id": kernel_id
        }
        return x
