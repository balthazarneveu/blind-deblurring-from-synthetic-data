from rstor.properties import (NB_EPOCHS, DATALOADER, BATCH_SIZE, SIZE, LENGTH,
                              TRAIN, VALIDATION, SCHEDULER, REDUCELRONPLATEAU,
                              MODEL, ARCHITECTURE, ID, NAME, SCHEDULER_CONFIGURATION, OPTIMIZER, PARAMS, LR,
                              LOSS, LOSS_MSE, CONFIG_DEAD_LEAVES,
                              PRETTY_NAME)


def model_configurations(config, model_preset="StackedConvolutions", bias: bool = True) -> dict:
    if model_preset == "StackedConvolutions":
        config[MODEL] = {
            ARCHITECTURE: dict(
                num_layers=8,
                k_size=3,
                h_dim=16,
                bias=bias
            ),
            NAME: "StackedConvolutions"
        }
    elif model_preset == "NAFNet" or model_preset == "UNet":
        # https://github.com/megvii-research/NAFNet/blob/main/options/test/GoPro/NAFNet-width64.yml
        config[MODEL] = {
            ARCHITECTURE: dict(
                width=64,
                enc_blk_nums=[1, 1, 1, 28],
                middle_blk_num=1,
                dec_blk_nums=[1, 1, 1, 1],
            ),
            NAME: model_preset
        }
    else:
        raise ValueError(f"Unknown model preset {model_preset}")


def presets_experiments(
    exp: int,
    b: int = 32,
    n: int = 50,
    bias: bool = True,
    length: int = 5000,
    model_preset: str = "StackedConvolutions"
) -> dict:
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
    model_configurations(config, model_preset=model_preset, bias=bias)
    config[SCHEDULER] = REDUCELRONPLATEAU
    config[SCHEDULER_CONFIGURATION] = {
        "factor": 0.8,
        "patience": 5
    }
    config[LOSS] = LOSS_MSE
    return config


def get_experiment_config(exp: int) -> dict:
    if exp == -1:
        config = presets_experiments(exp, length=10, n=2)
    elif exp == 1000:
        config = presets_experiments(exp, n=60)
        config[PRETTY_NAME] = "Vanilla small blur"
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(blur_kernel_half_size=[0, 2], ds_factor=1, noise_stddev=[0., 0.])
    elif exp == 1001:
        config = presets_experiments(exp, n=60)
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(blur_kernel_half_size=[0, 6], ds_factor=1, noise_stddev=[0., 0.])
        config[PRETTY_NAME] = "Vanilla large blur 0 - 6"
    elif exp == 1002:
        config = presets_experiments(exp, n=6)  # Less epochs because of the large downsample factor
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(blur_kernel_half_size=[0, 2], ds_factor=5, noise_stddev=[0., 0.])
        config[PRETTY_NAME] = "Vanilla small blur - ds=5"
    elif exp == 1003:
        config = presets_experiments(exp, n=6)  # Less epochs because of the large downsample factor
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(blur_kernel_half_size=[0, 2], ds_factor=5, noise_stddev=[0., 50.])
        config[PRETTY_NAME] = "Vanilla small blur - ds=5 - noisy 0-50"
    elif exp == 1004:
        config = presets_experiments(exp, n=60)
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(blur_kernel_half_size=[0, 0], ds_factor=1, noise_stddev=[0., 50.])
        config[PRETTY_NAME] = "Vanilla denoise only - ds=5 - noisy 0-50"
    elif exp == 1005:
        config = presets_experiments(exp, bias=False, n=60)
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(blur_kernel_half_size=[0, 0], ds_factor=1, noise_stddev=[0., 50.])
        config[PRETTY_NAME] = "Vanilla denoise only - ds=5 - noisy 0-50 - bias free"
    elif exp == 1006:
        config = presets_experiments(exp, n=60)
        config[PRETTY_NAME] = "Vanilla small blur - noisy 0-50"
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(blur_kernel_half_size=[0, 2], ds_factor=1, noise_stddev=[0., 50.])
    elif exp == 1007:
        config = presets_experiments(exp, n=60)
        config[PRETTY_NAME] = "Vanilla large blur 0 - 6 - noisy 0-50"
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(blur_kernel_half_size=[0, 6], ds_factor=1, noise_stddev=[0., 50.])
    elif exp == 2000:
        config = presets_experiments(exp, n=60,  b=16, model_preset="NAFNet")
        config[PRETTY_NAME] = "NAFNet denoise 0-50"
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(
            blur_kernel_half_size=[0, 0],
            ds_factor=1,
            noise_stddev=[0., 50.]
        )
    elif exp == 2001:
        config = presets_experiments(exp, n=60,  b=16, model_preset="UNet")
        config[PRETTY_NAME] = "UNEt denoise 0-50"
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(
            blur_kernel_half_size=[0, 0],
            ds_factor=1,
            noise_stddev=[0., 50.]
        )
    else:
        raise ValueError(f"Experiment {exp} not found")
    return config
