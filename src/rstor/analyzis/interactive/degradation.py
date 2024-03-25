import numpy as np
from interactive_pipe import interactive
from skimage.filters import gaussian
from rstor.properties import DATASET_BLUR_KERNEL_PATH
from scipy.io import loadmat
import cv2


@interactive(
    sigma=(3/5, [0., 2.])
)
def downsample(chart: np.ndarray, sigma=3/5, global_params={}):
    ds_factor = global_params.get("ds_factor", 5)
    if sigma > 0.:
        ds_chart = gaussian(chart, sigma=(sigma, sigma, 0), mode='nearest', cval=0, preserve_range=True, truncate=4.0)
    else:
        ds_chart = chart.copy()
    ds_chart = ds_chart[ds_factor//2::ds_factor, ds_factor//2::ds_factor]
    return ds_chart


@interactive(
    k_size_x=(0, [0, 10]),
    k_size_y=(0, [0, 10]),
)
def degrade_blur_gaussian(chart: np.ndarray, k_size_x: int = 1, k_size_y: int = 1):
    if k_size_x == 0 and k_size_y == 0:
        blurred = chart
    blurred = cv2.GaussianBlur(chart, (2*k_size_x+1, 2*k_size_y+1), 0)
    return blurred


@interactive(
    noise_stddev=(0., [0., 50.])
)
def degrade_noise(img: np.ndarray, noise_stddev=0., global_params={}):
    seed = global_params.get("seed", 42)
    np.random.seed(seed)
    if noise_stddev > 0.:
        noise = np.random.normal(0, noise_stddev/255., img.shape)
        img = img.copy()+noise
    return img


@interactive(
    ksize=(3, [1, 10])
)
def get_blur_kernel_box(ksize=3):
    return np.ones((ksize, ksize), dtype=np.float32) / (1.*ksize**2)


@interactive(
    blur_index=(-1, [-1, 1000])
)
def get_blur_kernel(blur_index: int = -1, global_params={}):
    if blur_index == -1:
        return None
    blur_mat = global_params.get("blur_mat", False)
    if blur_mat is False:
        blur_mat = loadmat(DATASET_BLUR_KERNEL_PATH)["kernels"].squeeze()
        global_params["blur_mat"] = blur_mat
    blur_k = blur_mat[blur_index]
    blur_k = blur_k/blur_k.sum()
    return blur_k


def degrade_blur(img: np.ndarray, blur_kernel: np.ndarray, global_params={}):
    if blur_kernel is None:
        return img
    img_blur = cv2.filter2D(img, -1, blur_kernel)
    return img_blur
