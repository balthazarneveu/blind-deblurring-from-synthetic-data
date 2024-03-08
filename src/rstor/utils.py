import numpy as np
import numba
import torch

THREADS_PER_BLOCK = 32  # 32 or 16
DEFAULT_NUMPY_FLOAT_TYPE = np.float32
DEFAULT_CUDA_FLOAT_TYPE = numba.float32
DEFAULT_TORCH_FLOAT_TYPE = torch.float32


DEFAULT_NUMPY_INT_TYPE = np.int32
DEFAULT_CUDA_INT_TYPE = numba.int32
