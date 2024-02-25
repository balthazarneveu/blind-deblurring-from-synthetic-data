from rstor.properties import (NB_EPOCHS, DATALOADER, BATCH_SIZE, SIZE, LENGTH,
                              TRAIN, VALIDATION, SCHEDULER, REDUCELRONPLATEAU,
                              MODEL, ARCHITECTURE, ID, NAME, SCHEDULER_CONFIGURATION, OPTIMIZER, PARAMS, LR,
                              LOSS, LOSS_MSE, CONFIG_DEAD_LEAVES)


def default_experiment(exp: int, b: int = 32, n: int = 50, length=5000) -> dict:
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
        config = default_experiment(exp, length=10, n=2)
    elif exp == 1000:
        config = default_experiment(exp)
    elif exp == 1001:
        config = default_experiment(exp)
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(blur_kernel_half_size=[0, 10])
    else:
        raise ValueError(f"Experiment {exp} not found")
    return config
