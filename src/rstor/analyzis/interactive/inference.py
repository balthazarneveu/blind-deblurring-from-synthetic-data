import torch
import numpy as np
from rstor.properties import DEVICE


def infer(degraded: np.ndarray, model: torch.nn.Module):
    degraded_tensor = torch.from_numpy(degraded).permute(-1, 0, 1).float().unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(degraded_tensor.to(DEVICE))
    output = output.squeeze().permute(1, 2, 0).cpu().numpy()
    return np.ascontiguousarray(output)
