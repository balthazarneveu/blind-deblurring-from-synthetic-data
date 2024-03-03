from rstor.synthetic_data.interactive.interactive_dead_leaves import generate_deadleave
from rstor.analyzis.interactive.crop import crop_selector, crop
from rstor.analyzis.interactive.inference import infer
from rstor.analyzis.interactive.degradation import degrade, downsample
from rstor.analyzis.interactive.model_selection import model_selector
from rstor.analyzis.interactive.images import image_selector
from typing import Tuple, List
import numpy as np


def deadleave_inference_pipeline(models_dict: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    groundtruth = generate_deadleave()
    groundtruth = downsample(groundtruth)
    model = model_selector(models_dict)
    degraded_chart = degrade(groundtruth)
    restored_chart = infer(degraded_chart, model)
    crop_selector(restored_chart)
    groundtruth, degraded_chart, restored_chart = crop(groundtruth, degraded_chart, restored_chart)
    return groundtruth, degraded_chart, restored_chart


def natural_inference_pipeline(input_image_list: List[np.ndarray], models_dict: dict):
    model = model_selector(models_dict)
    img = image_selector(input_image_list)
    crop_selector(img)
    img = crop(img)
    restored = infer(img, model)
    return img, restored
