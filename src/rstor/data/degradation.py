# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 01:21:46 2024

@author: jamyl
"""
import torch
from rstor.properties import DATASET_BLUR_KERNEL_PATH
import random
from scipy.io import loadmat
import cv2


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
        # WARNING! INPLACE OPERATIONS!!!!!
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


class DegradationBlurMat(Degradation):
    def __init__(self,
                 length: int = 1000,
                 frozen_seed: int = None,
                 blur_index: int = None):
        super().__init__(length, frozen_seed)

        kernels = loadmat(DATASET_BLUR_KERNEL_PATH)["kernels"].squeeze()
        # conversion to torch (the shape of the kernel is not constant)
        self.kernels = tuple([
            torch.from_numpy(kernel/kernel.sum(keepdims=True)).unsqueeze(0).unsqueeze(0)
            for kernel in kernels] + [torch.ones((1, 1)).unsqueeze(0).unsqueeze(0)])
        self.n_kernels = len(self.kernels)

        if frozen_seed is not None:
            random.seed(frozen_seed)
            self.kernel_ids = [random.randint(0, self.n_kernels-1) for _ in range(length)]
        if blur_index is not None:
            self.frozen_seed = 42
            self.kernel_ids = [blur_index for _ in range(length)]

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
        _, _, kh, kw = kernel.shape
        # We use padding = same to make
        # sure that the output size does not depend on the kernel.

        # define nn.Conf layer to define both padding mode and padding value...
        conv_layer = torch.nn.Conv2d(in_channels=x.shape[1],
                                     out_channels=x.shape[1],
                                     kernel_size=(kh, kw),
                                     padding="same",
                                     padding_mode='replicate',
                                     groups=3,
                                     bias=False)

        # Set the predefined kernel as weights and freeze the parameters
        with torch.no_grad():
            conv_layer.weight = torch.nn.Parameter(kernel)
            conv_layer.weight.requires_grad = False
        # breakpoint()
        x = conv_layer(x)
        # Alternative Functional version with 0 padding :
        # x = F.conv2d(x, kernel, padding="same", groups=3)

        self.current_degradation[idx] = {
            "blur_kernel_id": kernel_id
        }
        return x


class DegradationBlurGauss(Degradation):
    def __init__(self,
                 length: int = 1000,
                 blur_kernel_half_size: int = [0, 2],
                 frozen_seed: int = None):
        super().__init__(length, frozen_seed)

        self.blur_kernel_half_size = blur_kernel_half_size
        # conversion to torch (the shape of the kernel is not constant)
        if frozen_seed is not None:
            random.seed(self.frozen_seed)
            self.blur_kernel_half_size = [
                (
                    random.randint(self.blur_kernel_half_size[0], self.blur_kernel_half_size[1]),
                    random.randint(self.blur_kernel_half_size[0], self.blur_kernel_half_size[1])
                ) for _ in range(length)
            ]

    def __call__(self, x: torch.Tensor, idx: int):
        # expects x of shape [b, c, h, w]
        assert x.ndim == 4
        assert x.shape[1] in [1, 3]
        device = x.device

        if self.frozen_seed is not None:
            k_size_x, k_size_y = self.blur_kernel_half_size[idx]
        else:
            k_size_x = random.randint(self.blur_kernel_half_size[0], self.blur_kernel_half_size[1])
            k_size_y = random.randint(self.blur_kernel_half_size[0], self.blur_kernel_half_size[1])

        k_size_x = 2 * k_size_x + 1
        k_size_y = 2 * k_size_y + 1

        x = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
        x = cv2.GaussianBlur(x, (k_size_x, k_size_y), 0)
        x = torch.from_numpy(x).to(device).permute(2, 0, 1).unsqueeze(0)

        self.current_degradation[idx] = {
            "blur_kernel_half_size": (k_size_x, k_size_y),
        }
        return x
