import numpy as np
from interactive_pipe import interactive
import cv2


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
