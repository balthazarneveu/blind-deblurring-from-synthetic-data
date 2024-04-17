from rstor.properties import (NB_EPOCHS, DATALOADER, BATCH_SIZE, SIZE, LENGTH,
                              TRAIN, VALIDATION, SCHEDULER, REDUCELRONPLATEAU,
                              MODEL, ARCHITECTURE, ID, NAME, SCHEDULER_CONFIGURATION, OPTIMIZER, PARAMS, LR,
                              LOSS, LOSS_MSE, CONFIG_DEAD_LEAVES,
                              SELECTED_METRICS, METRIC_PSNR, METRIC_SSIM, METRIC_LPIPS,
                              DATASET_DL_DIV2K_512, DATASET_DIV2K,
                              CONFIG_DEGRADATION,
                              PRETTY_NAME,
                              DEGRADATION_BLUR_NONE, DEGRADATION_BLUR_MAT, DEGRADATION_BLUR_GAUSS,
                              AUGMENTATION_FLIP, AUGMENTATION_ROTATE,
                              DATASET_DL_EXTRAPRIMITIVES_DIV2K_512)


from typing import Tuple


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
    data_size: Tuple[int, int] = (128, 128),
    model_preset: str = "StackedConvolutions",
    lpips: bool = False
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
        SIZE: data_size,  # (width, height)
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
    config[SELECTED_METRICS] = [METRIC_PSNR, METRIC_SSIM]
    if lpips:
        config[SELECTED_METRICS].append(METRIC_LPIPS)
    return config


def get_experiment_config(exp: int) -> dict:
    if exp == -1:
        config = presets_experiments(exp, length=10, n=2)
    elif exp == -2:
        config = presets_experiments(exp, length=10, n=2, lpips=True)
    elif exp == -3:
        config = presets_experiments(exp, n=20)
        config[DATALOADER]["gpu_gen"] = True
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(
            blur_kernel_half_size=[0, 0],
            ds_factor=1,
            noise_stddev=[0., 50.]
        )
        config[PRETTY_NAME] = "Vanilla denoise only - ds=1 - noisy 0-50"
    elif exp == -4:
        config = presets_experiments(exp, b=4, n=20)
        config[DATALOADER][NAME] = DATASET_DL_DIV2K_512
        config[DATALOADER][CONFIG_DEGRADATION] = dict(
            noise_stddev=[0., 50.]
        )
        config[PRETTY_NAME] = "Vanilla exp from disk - noisy 0-50"
    # ---------------------------------
    # Pure DL DENOISING trainings
    # ---------------------------------
    elif exp == 3001:  # ENABLE GRADIENT CLIPPING
        config = presets_experiments(exp, n=30,  b=8, model_preset="NAFNet")
        config[PRETTY_NAME] = "NAFNet41.4M denoise - DL_DIV2K_512 0-50 256x256"
        config[DATALOADER][NAME] = DATASET_DL_DIV2K_512
        config[DATALOADER][CONFIG_DEGRADATION] = dict(
            noise_stddev=[0., 50.]
        )
        config[DATALOADER][SIZE] = (256, 256)
    elif exp == 3020:
        config = presets_experiments(exp, b=32, n=50)
        config[DATALOADER][NAME] = DATASET_DL_DIV2K_512
        config[DATALOADER][CONFIG_DEGRADATION] = dict(
            noise_stddev=[0., 50.]
        )
        config[PRETTY_NAME] = "Vanilla denoise DL  0-50 - noisy 0-50"
    # ---------------------------------
    # Pure DIV2K DENOISING trainings
    # ---------------------------------
    elif exp == 3101:
        config = presets_experiments(exp, n=30,  b=8, model_preset="NAFNet")
        config[PRETTY_NAME] = "NAFNet41.4M denoise - DIV2K_512 0-50 256x256"
        config[DATALOADER][NAME] = DATASET_DIV2K
        config[DATALOADER][CONFIG_DEGRADATION] = dict(
            noise_stddev=[0., 50.]
        )
        config[DATALOADER][SIZE] = (256, 256)
        # # 3.4M parameters
        # config[MODEL][ARCHITECTURE] = dict(
        #     width=64,
        #     enc_blk_nums=[1, 1, 2],
        #     middle_blk_num=1,
        #     dec_blk_nums=[1, 1, 1],
        # )
    else:
        raise ValueError(f"Experiment {exp} not found")
    return config
