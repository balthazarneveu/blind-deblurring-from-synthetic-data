import numpy as np
import cv2


def sample_rgb_values(size: int, seed: int = None) -> np.ndarray:
    """
    Generate n random RGB values.

    Args:
        n (int): number of colors to sample
        seed (int, optional): Seed for the random number generator. Defaults to None.

    Returns:
        np.ndarray: Random RGB values as a numpy array.
    """
    # https://github.com/numpy/numpy/issues/17079
    # https://numpy.org/devdocs/reference/random/new-or-different.html#new-or-different
    rng = np.random.default_rng(np.random.SeedSequence(seed))

    random_samples = rng.uniform(size=(size, 3))
    lab = (random_samples + np.array([0., -0.5, -0.5])[None]) * np.array([100., 127 * 2, 127 * 2])[None]
    rgb = cv2.cvtColor(lab[None, :].astype(np.float32), cv2.COLOR_Lab2RGB)
    return rgb.squeeze()
