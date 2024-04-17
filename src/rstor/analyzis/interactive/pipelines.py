from rstor.synthetic_data.interactive.interactive_dead_leaves import generate_deadleave
from rstor.analyzis.interactive.crop import crop_selector, crop, rescale_thumbnail
from rstor.analyzis.interactive.inference import infer
from rstor.analyzis.interactive.degradation import degrade_noise, degrade_blur, downsample, degrade_blur_gaussian, get_blur_kernel
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
    degraded = degrade_blur_gaussian(groundtruth)
    degraded = degrade_noise(degraded)
    restored = infer(degraded, model)
    crop_selector(restored)
    groundtruth, degraded, restored = crop(groundtruth, degraded, restored)
    configure_metrics()
    get_metrics_restored(restored, groundtruth)
    get_metrics_degraded(degraded, groundtruth)
    return groundtruth, degraded, restored


CANVAS_DICT = {
    "demo": [["degraded", "restored"]],
    "landscape_light": [["degraded", "restored", "groundtruth"]],
    "landscape": [["degraded", "restored", "blur_kernel", "groundtruth"]],
    "full": [["degraded", "restored"], ["blur_kernel", "groundtruth"]]
}
CANVAS = list(CANVAS_DICT.keys())


def morph_canvas(canvas=CANVAS[0], global_params={}):
    global_params["__pipeline"].outputs = CANVAS_DICT[canvas]
    return None


def natural_inference_pipeline(input_image_list: List[np.ndarray], models_dict: dict):
    model = model_selector(models_dict)
    img_clean = image_selector(input_image_list)
    crop_selector(img_clean)
    groundtruth = crop(img_clean)
    blur_kernel = get_blur_kernel()
    degraded = degrade_blur(groundtruth, blur_kernel)
    degraded = degrade_noise(degraded)
    blur_kernel = rescale_thumbnail(blur_kernel)
    restored = infer(degraded, model)
    configure_metrics()
    get_metrics_restored(restored, groundtruth)
    get_metrics_degraded(degraded, groundtruth)
    morph_canvas()
    return [[degraded, restored], [blur_kernel, groundtruth]]
