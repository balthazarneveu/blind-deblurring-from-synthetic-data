from rstor.properties import MODEL, NAME, N_PARAMS, ARCHITECTURE
from rstor.architecture.stacked_convolutions import StackedConvolutions
import torch


def load_architecture(config: dict) -> torch.nn.Module:
    if config[MODEL][NAME] == StackedConvolutions.__name__:
        model = StackedConvolutions(**config[MODEL][ARCHITECTURE])
    else:
        raise ValueError(f"Unknown model {config[MODEL][NAME]}")
    config[MODEL][N_PARAMS] = model.count_parameters()
    config[MODEL]["receptive_field"] = model.receptive_field()
    return model
