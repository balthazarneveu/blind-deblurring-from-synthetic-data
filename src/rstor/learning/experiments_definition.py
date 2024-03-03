from rstor.properties import (NB_EPOCHS, DATALOADER, BATCH_SIZE, SIZE, LENGTH,
                              TRAIN, VALIDATION, SCHEDULER, REDUCELRONPLATEAU,
                              MODEL, ARCHITECTURE, ID, NAME, SCHEDULER_CONFIGURATION, OPTIMIZER, PARAMS, LR,
                              LOSS, LOSS_MSE, CONFIG_DEAD_LEAVES,
                              PRETTY_NAME)


def default_experiment_vanilla(exp: int, b: int = 32, n: int = 50, bias: bool = True, length=5000) -> dict:
    config = {
        ID: exp,
        NAME: f"{exp:04d}",
        NB_EPOCHS: n
    }
    config[DATALOADER] = {
        BATCH_SIZE: {
            TRAIN: b,
            VALIDATION: b
        },
        SIZE: (128, 128),  # (width, height)
        LENGTH: {
            TRAIN: length,
            VALIDATION: 800
        }
    }
    config[OPTIMIZER] = {
        NAME: "Adam",
        PARAMS: {
            LR: 1e-3
        }
    }
    config[MODEL] = {
        ARCHITECTURE: dict(
            num_layers=8,
            k_size=3,
            h_dim=16,
            bias=bias
        ),
        NAME: "StackedConvolutions"
    }
    config[SCHEDULER] = REDUCELRONPLATEAU
    config[SCHEDULER_CONFIGURATION] = {
        "factor": 0.8,
        "patience": 5
    }
    config[LOSS] = LOSS_MSE
    return config


def get_experiment_config(exp: int) -> dict:
    if exp == -1:
        config = default_experiment_vanilla(exp, length=10, n=2)
    elif exp == 1000:
        config = default_experiment_vanilla(exp, n=60)
        config[PRETTY_NAME] = "Vanilla small blur"
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(blur_kernel_half_size=[0, 2], ds_factor=1, noise_stddev=[0., 0.])
    elif exp == 1001:
        config = default_experiment_vanilla(exp, n=60)
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(blur_kernel_half_size=[0, 6], ds_factor=1, noise_stddev=[0., 0.])
        config[PRETTY_NAME] = "Vanilla large blur 0 - 6"
    elif exp == 1002:
        config = default_experiment_vanilla(exp, n=6)  # Less epochs because of the large downsample factor
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(blur_kernel_half_size=[0, 2], ds_factor=5, noise_stddev=[0., 0.])
        config[PRETTY_NAME] = "Vanilla small blur - ds=5"
    elif exp == 1003:
        config = default_experiment_vanilla(exp, n=6)  # Less epochs because of the large downsample factor
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(blur_kernel_half_size=[0, 2], ds_factor=5, noise_stddev=[0., 50.])
        config[PRETTY_NAME] = "Vanilla small blur - ds=5 - noisy 0-50"
    elif exp == 1004:
        config = default_experiment_vanilla(exp, n=60)
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(blur_kernel_half_size=[0, 0], ds_factor=1, noise_stddev=[0., 50.])
        config[PRETTY_NAME] = "Vanilla denoise only - ds=5 - noisy 0-50"
    elif exp == 1005:
        config = default_experiment_vanilla(exp, bias=False, n=60)
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(blur_kernel_half_size=[0, 0], ds_factor=1, noise_stddev=[0., 50.])
        config[PRETTY_NAME] = "Vanilla denoise only - ds=5 - noisy 0-50 - bias free"
    elif exp == 1006:
        config = default_experiment_vanilla(exp, n=60)
        config[PRETTY_NAME] = "Vanilla small blur - noisy 0-50"
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(blur_kernel_half_size=[0, 2], ds_factor=1, noise_stddev=[0., 50.])
    elif exp == 1007:
        config = default_experiment_vanilla(exp, n=60)
        config[PRETTY_NAME] = "Vanilla large blur 0 - 6 - noisy 0-50"
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(blur_kernel_half_size=[0, 6], ds_factor=1, noise_stddev=[0., 50.])
    else:
        raise ValueError(f"Experiment {exp} not found")
    return config
