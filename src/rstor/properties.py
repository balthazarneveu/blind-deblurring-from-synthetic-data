import torch
from pathlib import Path
import logging
RELU = "ReLU"
LEAKY_RELU = "LeakyReLU"
SIMPLE_GATE = "simple_gate"
LOSS = "loss"
LOSS_MSE = "MSE"
LOSS_VGG16 = "VGG16"
LOSS_MIXED_MSE_VGG16 = "MIXED_MSE_VGG16"
METRIC_PERCEPTUAL = LOSS_VGG16
METRIC_PSNR = "PSNR"
METRIC_SSIM = "SSIM"
METRIC_LPIPS = "LPIPS"
SELECTED_METRICS = "selected_metrics"
DATALOADER = "data_loader"
BATCH_SIZE = "batch_size"
SIZE = "size"
TRAIN, VALIDATION, TEST = "train", "validation", "test"
LENGTH = "length"
ID = "id"
NAME = "name"
PRETTY_NAME = "pretty_name"
NB_EPOCHS = "nb_epochs"
ARCHITECTURE = "architecture"
MODEL = "model"
NAME = "name"
N_PARAMS = "n_params"
OPTIMIZER = "optimizer"
LR = "lr"
PARAMS = "parameters"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE != "cuda":
    logging.warning("CUDA not available, using CPU!!!")
SCHEDULER_CONFIGURATION = "scheduler_configuration"
SCHEDULER = "scheduler"
REDUCELRONPLATEAU = "ReduceLROnPlateau"
ARCHITECTURE = "architecture"
CONFIG_DEAD_LEAVES = "config_dead_leaves"
CONFIG_DEGRADATION = "config_degradation"
REDUCTION_SUM = "reduction_sum"
REDUCTION_AVERAGE = "reduction_average"
REDUCTION_SKIP = "reduction_skip"
TRACES_TARGET = "target"
TRACES_DEGRADED = "degraded"
TRACES_RESTORED = "restored"
TRACES_METRICS = "metrics"
TRACES_ALL = "all"

DEGRADATION_BLUR_NONE = "none"
DEGRADATION_BLUR_MAT = "mat"
DEGRADATION_BLUR_GAUSS = "gauss"


SAMPLER_SATURATED = "saturated"
SAMPLER_UNIFORM = "uniform"
SAMPLER_NATURAL = "natural"
SAMPLER_DIV2K = "div2k"

DATASET_FOLDER = "__dataset"
DATASET_PATH = Path(__file__).parent.parent.parent/DATASET_FOLDER
DATASET_DL_RANDOMRGB_1024 = "deadleaves_randomrgb_1024"
DATASET_DL_DIV2K_1024 = "deadleaves_div2k_1024"
DATASET_DL_DIV2K_512 = "deadleaves_div2k_512"
DATASET_DL_EXTRAPRIMITIVES_DIV2K_512 = "deadleaves_primitives_div2k_512"
DATASET_SYNTH_LIST = [DATASET_DL_DIV2K_512, DATASET_DL_DIV2K_1024,
                      DATASET_DL_RANDOMRGB_1024, DATASET_DL_EXTRAPRIMITIVES_DIV2K_512]
DATASET_BLUR_KERNEL_PATH = DATASET_PATH / "kernels" / "custom_blur_centered.mat"
AUGMENTATION_FLIP = "flip"
AUGMENTATION_ROTATE = "rotate"


DATASET_DIV2K = "div2k"
