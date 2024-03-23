from rstor.learning.metrics import compute_metrics
import torch
import numpy as np


def get_metrics(prediction: torch.Tensor, target: torch.Tensor, image_name, global_params: dict = {}):
    if isinstance(prediction, np.ndarray):
        prediction_ = torch.from_numpy(prediction).permute(-1, 0, 1).float().unsqueeze(0)
    else:
        prediction_ = prediction
    if isinstance(target, np.ndarray):
        target_ = torch.from_numpy(target).permute(-1, 0, 1).float().unsqueeze(0)
    else:
        target_ = target
    metrics = compute_metrics(prediction_, target_)
    global_params["metrics"] = metrics
    title = f"{image_name}: "
    title += " ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
    global_params["__output_styles"][image_name] = {"title": title, "image_name": image_name}