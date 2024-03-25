import numpy as np
from interactive_pipe import interactive
from skimage.filters import gaussian
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
        img += noise
    return img


@interactive(
    ksize=(3, [1, 10])
)
def get_blur_kernel(ksize=3):
    return np.ones((ksize, ksize), dtype=np.float32) / (1.*ksize**2)

# @interactive(blur_kernel_index)


def degrade_blur(img: np.ndarray, blur_kernel: np.ndarray, global_params={}):
    img = cv2.filter2D(img, -1, blur_kernel, img)
    return img
    # k_size = global_params.get("k_size", 1)
    # if k_size == 0:
    #     return img
    # blurred = cv2.GaussianBlur(img, (2*k_size+1, 2*k_size+1), 0)
    # return blurred
