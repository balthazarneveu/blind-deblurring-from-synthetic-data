import numpy as np
import cv2

from pathlib import Path


def sample_uniform_rgb(size: int, seed: int = None) -> np.ndarray:
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
    rgb = random_samples
    
    ## Below old version with sturation
    # lab = (random_samples + np.array([0., -0.5, -0.5])[None]) * np.array([100., 127 * 2, 127 * 2])[None]
    # rgb = cv2.cvtColor(lab[None, :].astype(np.float32), cv2.COLOR_Lab2RGB)
    return rgb.squeeze()

def sample_saturated_color(size: int, seed: int = None) -> np.ndarray:
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

def sample_div2k_color(size: int, seed: int = None):
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    div2k_path = Path("__dataset/div2k/DIV2K_train_HR")

    png_paths = sorted([file for file in div2k_path.glob("*.png")])

    ## Randomly pick an image and load it
    img_id = rng.integers(0, len(png_paths))
    
    img = cv2.imread(png_paths[img_id].as_posix())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
    
    pixels = img.reshape(-1, 3)
    n_pixels = pixels.shape[0]
    
    # sample a pixel color for each disc
    pixel_ids = rng.integers(0, n_pixels, size)
    colors = pixels[pixel_ids, :]
    
    return colors