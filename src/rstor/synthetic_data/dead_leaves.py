import numpy as np
import cv2
import random
from typing import Tuple, Optional


def dead_leaves_chart(size: Tuple[int, int] = (100, 100),
                      number_of_circles: int = -1,
                      background_color: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
                      colored: Optional[bool] = False,
                      radius_mean: Optional[int] = -1,
                      radius_stddev: Optional[int] = -1,
                      seed: int = 42) -> np.ndarray:
    """
    Generation of a dead leaves chart by splatting circles on top of each other.

    Args:
        size (Tuple[int, int], optional): size of the generated chart. Defaults to (100, 100).
        number_of_circles (int, optional): number of circles to generate.
        If negative, it is computed based on the size. Defaults to -1.
        background_color (Optional[Tuple[float, float, float]], optional):
        background color of the chart. Defaults to gray (0.5, 0.5, 0.5).
        colored (Optional[bool], optional): Whether to generate colored circles. Defaults to False.
        radius_mean (Optional[int], optional): mean radius of the circles. Defaults to -1. (=> -2)
        radius_stddev (Optional[int], optional): standard deviation of the radius of the circles.
        If negative, it is calculated based on the size. Defaults to -1.
        seed (int, optional): seed for the random number generator. Defaults to 42.

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
            color = tuple(sample_rgb_value(seed=random.randint(0, 1e10)).astype(float))
        else:
            gray = random.uniform(0.25, 0.75)
            color = (gray, gray, gray)
        chart = cv2.circle(chart, (center_x, center_y), radius, color, -1)
    chart = chart.clip(0, 1)
    return chart


def sample_rgb_value(seed: int = None) -> np.ndarray:
    """
    Generate a random RGB value.

    Args:
        seed (int, optional): Seed for the random number generator. Defaults to None.

    Returns:
        np.ndarray: Random RGB value as a numpy array.
    """
    random.seed(seed)
    random_samples = np.array([random.uniform(0., 1.), random.uniform(0., 1.), random.uniform(0., 1.)])
    lab = (random_samples + [0., -0.5, -0.5]) * [100., 127 * 2, 127 * 2]
    rgb = cv2.cvtColor(lab[None, None, :].astype(np.float32), cv2.COLOR_Lab2RGB)
    return rgb.squeeze()
