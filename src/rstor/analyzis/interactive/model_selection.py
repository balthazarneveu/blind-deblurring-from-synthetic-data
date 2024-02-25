import torch
from rstor.learning.experiments import get_training_content
from rstor.learning.experiments_definition import get_experiment_config
from rstor.properties import DEVICE, PRETTY_NAME
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple

from interactive_pipe import interactive
MODELS_PATH = Path("scripts")/"__output"

# @TODO: link simple names to experiment definition
# @TODO: Find a proper way to define the models dialog list from CLI


def model_selector(models_dict: dict, global_params={}, model_name="vanilla"):
    current_model = models_dict[model_name]
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
    model_storage: Path = MODELS_PATH
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
    interactive(model_name=(exp_names[0], exp_names))(model_selector)  # Create the model dialog
    return model_dict
