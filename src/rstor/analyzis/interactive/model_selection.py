import torch
from interactive_pipe import KeyboardControl
from rstor.learning.experiments import get_training_content
from rstor.learning.experiments_definition import get_experiment_config
from rstor.properties import DEVICE, PRETTY_NAME
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple

from interactive_pipe import interactive
MODELS_PATH = Path("scripts")/"__output"


def model_selector(models_dict: dict, global_params={}, model_name="vanilla"):
    if isinstance(model_name, str):
        current_model = models_dict[model_name]
    elif isinstance(model_name, int):
        model_names = [name for name in models_dict.keys()]
        current_model = models_dict[model_names[model_name % len(model_names)]]
    else:
        raise ValueError(f"Model name {model_name} not understood")
    global_params["model_config"] = current_model["config"]
    return current_model["model"]


def get_model_from_exp(exp: int, model_storage: Path = MODELS_PATH, device=DEVICE) -> Tuple[torch.nn.Module, dict]:
    config = get_experiment_config(exp)
    model, _, _ = get_training_content(config, training_mode=False)
    model_path = torch.load(model_storage/f"{exp:04d}"/"best_model.pt")
    assert model_path is not None, f"Model {exp} not found"
    model.load_state_dict(model_path)
    model = model.to(device)
    return model, config


def get_default_models(
    exp_list: List[int] = [1000, 1001],
    model_storage: Path = MODELS_PATH,
    keyboard_control: bool = False,
    interactive_flag: bool = True
) -> dict:
    model_dict = {}
    assert model_storage.exists(), f"Model storage {model_storage} does not exist"
    for exp in tqdm(exp_list, desc="Loading models"):
        model, config = get_model_from_exp(exp, model_storage=model_storage)
        name = config.get(PRETTY_NAME, f"{exp:04d}")
        model_dict[name] = {
            "model": model,
            "config": config
        }
    exp_names = [name for name in model_dict.keys()]
    if interactive_flag:
        if keyboard_control:
            model_control = KeyboardControl(0, [0, len(exp_names)-1], keydown="pagedown", keyup="pageup", modulo=True)
        else:
            model_control = (exp_names[0], exp_names)
        interactive(model_name=model_control)(model_selector)  # Create the model dialog
    return model_dict
