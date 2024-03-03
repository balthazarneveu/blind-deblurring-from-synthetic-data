import math
import random
from typing import Tuple, Optional
import numpy as np
import cv2
from numba import cuda

from rstor.utils import DEFAULT_NUMPY_FLOAT_TYPE, THREADS_PER_BLOCK

def dead_leaves_chart(size: Tuple[int, int] = (100, 100),
                      number_of_circles: int = -1,
                      background_color: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
                      colored: Optional[bool] = True,
                      radius_mean: Optional[int] = -1,
                      radius_stddev: Optional[int] = -1,
                      seed: int = None) -> np.ndarray:
    """
    Generation of a dead leaves chart by splatting circles on top of each other.

    Args:
        size (Tuple[int, int], optional): size of the generated chart. Defaults to (100, 100).
        number_of_circles (int, optional): number of circles to generate.
        If negative, it is computed based on the size. Defaults to -1.
        background_color (Optional[Tuple[float, float, float]], optional):
        background color of the chart. Defaults to gray (0.5, 0.5, 0.5).
        colored (Optional[bool], optional): Whether to generate colored circles. Defaults to True.
        radius_mean (Optional[int], optional): mean radius of the circles. Defaults to -1. (=> -2)
        radius_stddev (Optional[int], optional): standard deviation of the radius of the circles.
        If negative, it is calculated based on the size. Defaults to -1.
        seed (int, optional): seed for the random number generator. Defaults to None

    Returns:
        np.ndarray: generated dead leaves chart as a NumPy array.
    """
    if seed is not None:
        random.seed(seed)
    chart = np.multiply(background_color, np.ones((size[0], size[1], 3), dtype=np.float32))
    if number_of_circles < 0:
        number_of_circles = 4 * max(size)
    if radius_mean < 0.:
        radius_mean = 2.
    if radius_stddev < 0.:
        radius_stddev = min(size) / 60.
    for _ in range(number_of_circles):
        center_x, center_y = random.randint(0, size[1]), random.randint(0, size[0])
        radius = int(round(abs(random.gauss(radius_mean, radius_stddev))))
        if colored:
            color = tuple(sample_rgb_values(size=1, seed=random.randint(0, 1e10)).squeeze().astype(float))
        else:
            gray = random.uniform(0.25, 0.75)
            color = (gray, gray, gray)
        chart = cv2.circle(chart, (center_x, center_y), radius, color, -1)
    chart = chart.clip(0, 1)
    return chart


def sample_rgb_values(size: int, seed: int = None) -> np.ndarray:
    """
    Generate n random RGB values.

    Args:
        n (int): number of colors to sample
        seed (int, optional): Seed for the random number generator. Defaults to None.

    Returns:
        np.ndarray: Random RGB values as a numpy array.
    """
    random.seed(seed)
    random_samples = np.random.uniform(size=(size, 3))
    lab = (random_samples + np.array([0., -0.5, -0.5])[None]) * np.array([100., 127 * 2, 127 * 2])[None]
    rgb = cv2.cvtColor(lab[None, :].astype(np.float32), cv2.COLOR_Lab2RGB)
    return rgb.squeeze()



def gpu_dead_leaves_chart(size: Tuple[int, int] = (100, 100),
                          number_of_circles: int = -1,
                          background_color: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
                          colored: Optional[bool] = True,
                          radius_mean: Optional[int] = -1,
                          radius_stddev: Optional[int] = -1,
                          seed: int = None) -> np.ndarray:
    if seed is not None:
        random.seed(seed)
        
    
    if number_of_circles < 0:
        number_of_circles = 4 * max(size)
    if radius_mean < 0.:
        radius_mean = 2.
    if radius_stddev < 0.:
        radius_stddev = min(size) / 60.
    
    # Pick random circle centers and radia
    center_x = np.random.randint(0, size[1], size=number_of_circles)
    center_y = np.random.randint(0, size[0], size=number_of_circles)
    
    radius = np.abs(np.random.normal(loc=radius_mean, scale=radius_stddev, size=number_of_circles)).round().astype(int)
    
    # Pick random colors
    if colored:
        color = sample_rgb_values(number_of_circles, seed=random.randint(0, 1e10)).astype(float)
    else:
        color = np.random.uniform(0.25, 0.75, size=(number_of_circles, 1))
    
    # Generate on gpu
    chart = _generate_dead_leaves(
        size,
        centers=np.stack((center_x, center_y), axis=-1),
        radia=radius,
        colors=color,
        background=background_color
        )
    
    
    return chart



def _generate_dead_leaves(size, centers, radia, colors, background):
    assert centers.ndim == 2
    ny, nx = size
    nc = colors.shape[-1]
    
    # Init empty array on GPU    
    generation_ = cuda.device_array((ny, nx, nc), DEFAULT_NUMPY_FLOAT_TYPE)
    # Move useful array to GPU
    centers_ = cuda.to_device(centers)
    radia_ = cuda.to_device(radia)
    colors_ = cuda.to_device(colors)
    
    
    # Dispatch threads
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)
    blockspergrid_x = math.ceil(nx/threadsperblock[1])
    blockspergrid_y = math.ceil(ny/threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y, nc)
    
    cuda_dead_leaves_gen[blockspergrid, threadsperblock](
        generation_, 
        centers_,
        radia_,
        colors_,
        background)
    
    # cuda.synchronize()
    # print(f"-dead leaves gen (numba) {perf_counter() - t1}")
    generation = generation_.copy_to_host()
    
    # We systematically output rgb format
    if nc == 1:
        generation = np.repeat(generation, repeats=3, axis=-1)
    
    return generation


@cuda.jit(cache=False)
def cuda_dead_leaves_gen(generation, centers, radia, colors, background):
    idx, idy, c = cuda.grid(3)
    ny, nx, nc = generation.shape
    
    n_discs = centers.shape[0]
    
    # Out of bound threads
    if idx >= nx or idy >= ny:
        return
        
    # Init with background
    out = background[c]
    
    for disc_id in range(n_discs):
        dx = idx - centers[disc_id, 0]
        dy = idy - centers[disc_id, 1]
        dist_sq = dx*dx + dy*dy
        
        # Naive thread diverging version
        r = radia[disc_id]
        r_sq = r*r
        
        if dist_sq <= r_sq:
            out = colors[disc_id, c]
                
    # Copy back to global memory
    generation[idy, idx, c] = out