from rstor.synthetic_data.interactive.interactive_dead_leaves import generate_deadleave
from rstor.analyzis.interactive.crop import crop_selector, crop
from rstor.analyzis.interactive.inference import infer
from rstor.analyzis.interactive.degradation import degrade, downsample
from rstor.analyzis.interactive.model_selection import model_selector
from rstor.analyzis.interactive.images import image_selector
from rstor.analyzis.interactive.metrics import get_metrics, configure_metrics
from typing import Tuple, List
from functools import partial
import numpy as np


get_metrics_restored = partial(get_metrics, image_name="restored")
get_metrics_degraded = partial(get_metrics, image_name="degraded")


def deadleave_inference_pipeline(models_dict: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    groundtruth = generate_deadleave()
    groundtruth = downsample(groundtruth)
    model = model_selector(models_dict)
    degraded = degrade(groundtruth)
    restored = infer(degraded, model)
    crop_selector(restored)
    groundtruth, degraded, restored = crop(groundtruth, degraded, restored)
    configure_metrics()
    get_metrics_restored(restored, groundtruth)
    get_metrics_degraded(degraded, groundtruth)
    return groundtruth, degraded, restored


def natural_inference_pipeline(input_image_list: List[np.ndarray], models_dict: dict):
    model = model_selector(models_dict)
    img_clean = image_selector(input_image_list)
    crop_selector(img_clean)
    groundtruth = crop(img_clean)
    degraded = degrade(groundtruth)
    restored = infer(degraded, model)
    configure_metrics()
    get_metrics_restored(restored, groundtruth)
    get_metrics_degraded(degraded, groundtruth)
    return degraded, restored
