from rstor.properties import MODEL, NAME, N_PARAMS, ARCHITECTURE
from rstor.architecture.stacked_convolutions import StackedConvolutions
from rstor.architecture.nafnet import NAFNet, UNet
import torch


def load_architecture(config: dict) -> torch.nn.Module:
    conf_model = config[MODEL][ARCHITECTURE]
    if config[MODEL][NAME] == StackedConvolutions.__name__:
        model = StackedConvolutions(**conf_model)
    elif config[MODEL][NAME] == NAFNet.__name__:
        model = NAFNet(**conf_model)
    elif config[MODEL][NAME] == UNet.__name__:
        model = UNet(**conf_model)
    else:
        raise ValueError(f"Unknown model {config[MODEL][NAME]}")
    config[MODEL][N_PARAMS] = model.count_parameters()
    # config[MODEL]["receptive_field"] = model.receptive_field() # way too slow
    return model
