import numpy as np
from interactive_pipe import interactive
from skimage.filters import gaussian
import cv2


@interactive(
    sigma=(3/5, [0., 2.])
)
def downsample(chart: np.ndarray, sigma=3/5):
    ds_factor = 5
    if sigma > 0.:
        ds_chart = gaussian(chart, sigma=(sigma, sigma, 0), mode='nearest', cval=0, preserve_range=True, truncate=4.0)
    else:
        ds_chart = chart.copy()
    ds_chart = ds_chart[ds_factor//2::ds_factor, ds_factor//2::ds_factor]
    return ds_chart


@interactive(
    k_size_x=(1, [0, 10]),
    k_size_y=(1, [0, 10]),
    noise_stddev=(0., [0., 50.])
)
def degrade(chart: np.ndarray, k_size_x: int = 1, k_size_y: int = 1, noise_stddev=0.):
    blurred = cv2.GaussianBlur(chart, (2*k_size_x+1, 2*k_size_y+1), 0)
    if noise_stddev > 0.:
        noise = np.random.normal(0, noise_stddev/255., blurred.shape)
        blurred += noise
    return blurred
