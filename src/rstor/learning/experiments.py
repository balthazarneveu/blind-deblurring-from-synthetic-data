from rstor.properties import DEVICE, OPTIMIZER, PARAMS
from rstor.architecture.selector import load_architecture
from rstor.data.dataloader import get_data_loader
from typing import Tuple
import torch


def get_training_content(
        config: dict,
        training_mode: bool = False,
        device=DEVICE) -> Tuple[torch.nn.Module, torch.optim.Optimizer, dict]:
    model = load_architecture(config)
    optimizer, dl_dict = None, None
    if training_mode:
        optimizer = torch.optim.Adam(model.parameters(), **config[OPTIMIZER][PARAMS])
        dl_dict = get_data_loader(config)
    return model, optimizer, dl_dict


if __name__ == "__main__":
    from rstor.learning.experiments_definition import default_experiment
    config = default_experiment(1)
    model, optimizer, dl_dict = get_training_content(config, training_mode=True)
    print(config)
