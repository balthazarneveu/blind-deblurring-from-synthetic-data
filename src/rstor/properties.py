import torch
RELU = "ReLU"
LEAKY_RELU = "LeakyReLU"
SIMPLE_GATE = "simple_gate"
LOSS = "loss"
LOSS_MSE = "MSE"
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
SCHEDULER_CONFIGURATION = "scheduler_configuration"
SCHEDULER = "scheduler"
REDUCELRONPLATEAU = "ReduceLROnPlateau"
ARCHITECTURE = "architecture"
CONFIG_DEAD_LEAVES = "config_dead_leaves"
REDUCTION_SUM = "reduction_sum"
REDUCTION_AVERAGE = "reduction_average"
REDUCTION_SKIP = "reduction_skip"
TRACES_TARGET = "target"
TRACES_DEGRADED = "degraded"
TRACES_RESTORED = "restored"
TRACES_METRICS = "metrics"
TRACES_ALL = "all"

