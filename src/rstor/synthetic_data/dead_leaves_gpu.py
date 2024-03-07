from rstor.utils import DEFAULT_NUMPY_FLOAT_TYPE, THREADS_PER_BLOCK
from typing import Tuple, Optional
from rstor.synthetic_data.color_sampler import sample_rgb_values
import numpy as np
from numba import cuda
import math


def gpu_dead_leaves_chart(size: Tuple[int, int] = (100, 100),
                          number_of_circles: int = -1,
                          background_color: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
                          colored: Optional[bool] = True,
                          radius_mean: Optional[int] = -1,
                          radius_stddev: Optional[int] = -1,
                          seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence(seed))

    if number_of_circles < 0:
        number_of_circles = 4 * max(size)
    if radius_mean < 0.:
        radius_mean = 2.
    if radius_stddev < 0.:
        radius_stddev = min(size) / 60.

    # Pick random circle centers and radia
    center_x = rng.integers(0, size[1], size=number_of_circles)
    center_y = rng.integers(0, size[0], size=number_of_circles)

    radius = np.abs(rng.normal(loc=radius_mean, scale=radius_stddev, size=number_of_circles)).round().astype(int)

    # Pick random colors
    if colored:
        color = sample_rgb_values(number_of_circles, seed=rng.integers(0, 1e10)).astype(float)
    else:
        color = rng.uniform(0.25, 0.75, size=(number_of_circles, 1))

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

    return generation_


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
