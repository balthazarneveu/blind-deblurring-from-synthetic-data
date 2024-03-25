from rstor.utils import DEFAULT_NUMPY_FLOAT_TYPE, THREADS_PER_BLOCK
from rstor.properties import SAMPLER_UNIFORM
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
        reverse=True,
        sampler=SAMPLER_UNIFORM,
        natural_image_list=None,
        circle_primitives: bool = True,
        anisotropy: float = 1.,
        angle: float = 0.
) -> np.ndarray:
    center_x, center_y, radius, color = define_dead_leaves_chart(
        size,
        number_of_circles,
        colored,
        radius_min,
        radius_max,
        radius_alpha,
        seed,
        sampler=sampler,
        natural_image_list=natural_image_list
    )

    # Generate on gpu
    chart = _generate_dead_leaves(
        size,
        centers=np.stack((center_x, center_y), axis=-1),
        radia=radius,
        colors=color,
        background=background_color,
        reverse=reverse,
        circle_primitives=circle_primitives,
        anisotropy=anisotropy,
        angle=angle
    )

    return chart


def _generate_dead_leaves(size, centers, radia, colors, background, reverse, circle_primitives: bool, anisotropy: float = 1., angle: float=0.):
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
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid_x = math.ceil(nx/threadsperblock[1])
    blockspergrid_y = math.ceil(ny/threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    if reverse:
        cuda_dead_leaves_gen_reversed[blockspergrid, threadsperblock](
            generation_,
            centers_,
            radia_,
            colors_,
            background,
            circle_primitives,
            anisotropy,
            np.deg2rad(angle)
        )
    else:
        cuda_dead_leaves_gen[blockspergrid, threadsperblock](
            generation_,
            centers_,
            radia_,
            colors_,
            background)

    return generation_


@cuda.jit(cache=False)
def cuda_dead_leaves_gen_reversed(generation, centers, radia, colors, background, circle_primitives: bool, anisotropy: float, angle: float):
    idx, idy = cuda.grid(2)
    ny, nx, nc = generation.shape

    n_discs = centers.shape[0]

    # Out of bound threads
    if idx >= nx or idy >= ny:
        return

    for disc_id in range(n_discs):
        dx_ = idx - centers[disc_id, 0]
        dy_ = idy - centers[disc_id, 1]
        dx = math.cos(angle)*dx_ + math.sin(angle)*dy_
        dy = -math.sin(angle)*dx_ + math.cos(angle)*dy_
        dx = dx * anisotropy
        dist_sq = dx*dx + dy*dy

        # Naive thread diverging version
        r = radia[disc_id]
        r_sq = r*r
        if circle_primitives:
            if dist_sq <= r_sq:
                # Copy back to global memory
                for c in range(nc):
                    generation[idy, idx, c] = colors[disc_id, c]
                return
        else:
            if (disc_id % 4) == 0 and dist_sq <= r_sq:
                # Copy back to global memory
                alpha = dist_sq/r_sq
                for c in range(nc):
                    generation[idy, idx, c] = colors[disc_id, c] * alpha + colors[disc_id, (c+1) % 3] * (1-alpha)
                return
            elif (disc_id % 4) == 1 and (abs(dx)+abs(dy)) <= r:
                # Copy back to global memory
                alpha = dist_sq/r_sq
                for c in range(nc):
                    generation[idy, idx, c] = colors[disc_id, c] * alpha + colors[disc_id, (c+1) % 3] * (1-alpha)
                return
            elif (disc_id % 4) == 2 and abs(dx) <= r and abs(dy) <= r:
                for c in range(nc):
                    generation[idy, idx, c] = colors[disc_id, c]
                return
            elif (disc_id % 200) == 3 and abs(dy) <= r//5:
                for c in range(nc):
                    generation[idy, idx, c] = colors[disc_id, c] * alpha + colors[disc_id, (c+1) % 3] * (1-alpha)
                return
            elif (disc_id % 200) == 4 and abs(dx) <= r//5:
                for c in range(nc):
                    generation[idy, idx, c] = colors[disc_id, c] * alpha + colors[disc_id, (c+1) % 3] * (1-alpha)
                return
    for c in range(nc):
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
