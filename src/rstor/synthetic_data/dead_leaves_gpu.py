from rstor.utils import DEFAULT_NUMPY_FLOAT_TYPE, THREADS_PER_BLOCK
from typing import Tuple, Optional
from rstor.synthetic_data.dead_leaves_cpu import define_dead_leaves_chart
import numpy as np
from numba import cuda
import math


def gpu_dead_leaves_chart(
    size: Tuple[int, int] = (100, 100),
        number_of_circles: int = -1,
        background_color: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        colored: Optional[bool] = True,
        radius_min: Optional[int] = -1,
        radius_max: Optional[int] = -1,
        radius_alpha: Optional[int] = 3,
        seed: int = None,
        reverse=True
) -> np.ndarray:
    center_x, center_y, radius, color = define_dead_leaves_chart(
        size,
        number_of_circles,
        colored,
        radius_min,
        radius_max,
        radius_alpha,
        seed
    )

    # Generate on gpu
    chart = _generate_dead_leaves(
        size,
        centers=np.stack((center_x, center_y), axis=-1),
        radia=radius,
        colors=color,
        background=background_color,
        reverse=reverse
    )

    return chart


def _generate_dead_leaves(size, centers, radia, colors, background, reverse):
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

    if reverse:
        cuda_dead_leaves_gen_reversed[blockspergrid, threadsperblock](
            generation_,
            centers_,
            radia_,
            colors_,
            background)
    else:
        cuda_dead_leaves_gen[blockspergrid, threadsperblock](
            generation_,
            centers_,
            radia_,
            colors_,
            background)

    return generation_


@cuda.jit(cache=False)
def cuda_dead_leaves_gen_reversed(generation, centers, radia, colors, background):
    idx, idy, c = cuda.grid(3)
    ny, nx, nc = generation.shape

    n_discs = centers.shape[0]

    # Out of bound threads
    if idx >= nx or idy >= ny:
        return

    for disc_id in range(n_discs):
        dx = idx - centers[disc_id, 0]
        dy = idy - centers[disc_id, 1]
        dist_sq = dx*dx + dy*dy

        # Naive thread diverging version
        r = radia[disc_id]
        r_sq = r*r

        if dist_sq <= r_sq:
            # Copy back to global memory
            generation[idy, idx, c] = colors[disc_id, c]
            return

    generation[idy, idx, c] = background[c]


@cuda.jit(cache=False)
def cuda_dead_leaves_gen(generation, centers, radia, colors, background):
    idx, idy, c = cuda.grid(3)
    ny, nx, nc = generation.shape

    n_discs = centers.shape[0]

    # Out of bound threads
    if idx >= nx or idy >= ny:
        return

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
