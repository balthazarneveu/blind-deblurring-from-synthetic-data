from rstor.learning.metrics import compute_metrics, ALL_METRICS
import torch
import numpy as np
from rstor.properties import METRIC_PSNR, METRIC_SSIM, METRIC_PERCEPTUAL
from interactive_pipe import interactive, KeyboardControl
from typing import Optional


def plug_configure_metrics(key_shortcut: Optional[str] = None) -> None:
    interactive(
        advanced_metrics=KeyboardControl(False, keydown=key_shortcut) if key_shortcut is not None else (True,)
    )(configure_metrics)


def configure_metrics(advanced_metrics=False, global_params={}) -> None:
    chosen_metrics = ALL_METRICS if advanced_metrics else [METRIC_PSNR, METRIC_SSIM]
    chosen_metrics.append(METRIC_PERCEPTUAL)
    global_params["chosen_metrics"] = chosen_metrics


def get_metrics(prediction: torch.Tensor, target: torch.Tensor,
                image_name: str,  # use functools.partial to root where you want the title to appear
                global_params: dict = {}) -> None:
    if isinstance(prediction, np.ndarray):
        prediction_ = torch.from_numpy(prediction).permute(-1, 0, 1).float().unsqueeze(0)
    else:
        prediction_ = prediction
    if isinstance(target, np.ndarray):
        target_ = torch.from_numpy(target).permute(-1, 0, 1).float().unsqueeze(0)
    else:
        target_ = target
    chosen_metrics = global_params.get("chosen_metrics", [METRIC_PSNR])
    metrics = compute_metrics(prediction_, target_, chosen_metrics=chosen_metrics)
    global_params["metrics"] = metrics
    title = f"{image_name}: "
    title += " ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
    global_params["__output_styles"][image_name] = {"title": title, "image_name": image_name}
