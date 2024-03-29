from typing import Tuple, Optional, List
from rstor.properties import SAMPLER_SATURATED, SAMPLER_NATURAL, SAMPLER_UNIFORM
from rstor.synthetic_data.color_sampler import sample_uniform_rgb, sample_saturated_color, sample_color_from_images
import numpy as np
from pathlib import Path


def define_dead_leaves_chart(
    size: Tuple[int, int] = (100, 100),
    number_of_circles: int = -1,
    colored: Optional[bool] = True,
    radius_min: Optional[int] = -1,
    radius_max: Optional[int] = -1,
    radius_alpha: Optional[int] = 3,
    seed: int = None,
    sampler=SAMPLER_UNIFORM,
    natural_image_list: Optional[List[Path]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Defines the geometric and color properties of the primitives in the dead leaves chart to later be sampled.

    Args:
        size (Tuple[int, int], optional): size of the generated chart. Defaults to (100, 100).
        number_of_circles (int, optional): number of circles to generate.
        If negative, it is computed based on the size. Defaults to -1.
        colored (Optional[bool], optional): Whether to generate colored circles. Defaults to True.
        radius_min (Optional[int], optional): minimum radius of the circles. Defaults to -1. (=> 1)
        radius_max (Optional[int], optional): maximum radius of the circles. Defaults to -1. (=> 2000)
        radius_alpha (Optional[int], optional): standard deviation of the radius of the circles.
        If negative, it is calculated based on the size. Defaults to -1.
        seed (int, optional): seed for the random number generator. Defaults to None

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: center_x, center_y, radius, color
    """
    rng = np.random.default_rng(np.random.SeedSequence(seed))

    if number_of_circles < 0:
        number_of_circles = 30 * max(size)
    if radius_min < 0.:
        radius_min = 1.
    if radius_max < 0.:
        radius_max = 2000.

    # Pick random circle centers and radii
    center_x = rng.integers(0, size[1], size=number_of_circles)
    center_y = rng.integers(0, size[0], size=number_of_circles)

    # Sample from a power law distribution for the p(radius=r) = (r.clip(radius_min, radius_max))^(-alpha)

    radius = rng.uniform(
        low=radius_max ** (1 - radius_alpha),
        high=radius_min ** (1 - radius_alpha),
        size=number_of_circles
    )
    # Using the change of variables formula for random variables.
    radius = radius ** (-1/(radius_alpha - 1))
    radius = radius.round().astype(int)

    # Pick random colors
    if colored:
        if sampler == SAMPLER_UNIFORM:
            color = sample_uniform_rgb(number_of_circles, seed=rng.integers(0, 1e10)).astype(float)
        elif sampler == SAMPLER_SATURATED:
            color = sample_saturated_color(number_of_circles, seed=rng.integers(0, 1e10)).astype(float)
        elif sampler == SAMPLER_NATURAL:
            assert natural_image_list is not None, "Please provide a list of images to sample colors from."
            color = sample_color_from_images(number_of_circles, seed=rng.integers(0, 1e10),
                                             path_to_images=natural_image_list).astype(float)
        else:
            raise NotImplementedError(f"Unknown color sampler {sampler}")
    else:
        color = rng.uniform(0.25, 0.75, size=(number_of_circles, 1))
    return center_x, center_y, radius, color
