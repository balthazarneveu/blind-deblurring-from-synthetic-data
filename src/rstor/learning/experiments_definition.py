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
        config[PRETTY_NAME] = "Vanilla denoise only - ds=1 - noisy 0-50"
    elif exp == 1005:
        config = presets_experiments(exp, bias=False, n=60)
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(blur_kernel_half_size=[0, 0], ds_factor=1, noise_stddev=[0., 50.])
        config[PRETTY_NAME] = "Vanilla denoise only - ds=1 - noisy 0-50 - bias free"
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
    elif exp == 2002:
        config = presets_experiments(exp, n=20,  b=8, data_size=(256, 256), model_preset="NAFNet")
        config[PRETTY_NAME] = "NAFNet denoise  0-50 gpu dl 256x256"
        config[DATALOADER]["gpu_gen"] = True
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(
            blur_kernel_half_size=[0, 0],
            ds_factor=1,
            noise_stddev=[0., 50.]
        )
    elif exp == 2003:
        config = presets_experiments(exp, n=20,  b=8, data_size=(128, 128), model_preset="NAFNet")
        config[PRETTY_NAME] = "NAFNet denoise  0-50 gpu dl - 128x128"
        config[DATALOADER]["gpu_gen"] = True
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(
            blur_kernel_half_size=[0, 0],
            ds_factor=1,
            noise_stddev=[0., 50.]
        )
    elif exp == 2004:
        config = presets_experiments(exp, n=20,  b=16, data_size=(128, 128), model_preset="NAFNet")
        config[PRETTY_NAME] = "NAFNet Light denoise  0-50 gpu dl - 128x128"
        config[DATALOADER]["gpu_gen"] = True
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(
            blur_kernel_half_size=[0, 0],
            ds_factor=1,
            noise_stddev=[0., 50.]
        )
        config[MODEL][ARCHITECTURE] = dict(
            width=64,
            enc_blk_nums=[1, 1, 1, 2],
            middle_blk_num=1,
            dec_blk_nums=[1, 1, 1, 1],
        )
    elif exp == 2005:
        config = presets_experiments(exp, n=20,  b=16, data_size=(128, 128), model_preset="NAFNet")
        config[PRETTY_NAME] = "NAFNet TresLight denoise  0-50 gpu dl - 128x128"
        config[DATALOADER]["gpu_gen"] = True
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(
            blur_kernel_half_size=[0, 0],
            ds_factor=1,
            noise_stddev=[0., 50.]
        )
        config[MODEL][ARCHITECTURE] = dict(
            width=64,
            enc_blk_nums=[1, 1, 2],
            middle_blk_num=1,
            dec_blk_nums=[1, 1, 1],
        )
    elif exp == 2006:
        config = presets_experiments(exp, n=20,  b=16, data_size=(128, 128), model_preset="NAFNet")
        config[PRETTY_NAME] = "NAFNet TresLight denoise  0-50 ds=5 gpu dl - 128x128"
        config[DATALOADER]["gpu_gen"] = True
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(
            blur_kernel_half_size=[0, 0],
            ds_factor=5,
            noise_stddev=[0., 50.]
        )
        config[MODEL][ARCHITECTURE] = dict(
            width=64,
            enc_blk_nums=[1, 1, 2],
            middle_blk_num=1,
            dec_blk_nums=[1, 1, 1],
        )
    elif exp == 2007:
        config = presets_experiments(exp, n=20,  b=16, data_size=(128, 128), model_preset="NAFNet")
        config[PRETTY_NAME] = "NAFNet denoise  0-50 gpu dl -ds=5 128x128"
        config[DATALOADER]["gpu_gen"] = True
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(
            blur_kernel_half_size=[0, 0],
            ds_factor=5,
            noise_stddev=[0., 50.]
        )
    elif exp == 1008:
        config = presets_experiments(exp, n=20)
        config[DATALOADER]["gpu_gen"] = True
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(
            blur_kernel_half_size=[0, 0],
            ds_factor=5,
            noise_stddev=[0., 50.]
        )
        config[PRETTY_NAME] = "Vanilla denoise only - ds=5 - noisy 0-50"
    # ---------------------------------
    # Pure DL DENOISING trainings!
    # ---------------------------------
    elif exp == 3000:
        config = presets_experiments(exp, n=30,  b=4, model_preset="NAFNet")
        config[PRETTY_NAME] = "NAFNet denoise - DL_DIV2K_512 0-50"
        config[DATALOADER][NAME] = DATASET_DL_DIV2K_512
        config[DATALOADER][CONFIG_DEGRADATION] = dict(
            noise_stddev=[0., 50.]
        )
        config[DATALOADER][SIZE] = (256, 256)
    elif exp == 3001:  # ENABLE GRADIENT CLIPPING
        config = presets_experiments(exp, n=30,  b=8, model_preset="NAFNet")
        config[PRETTY_NAME] = "NAFNet41.4M denoise - DL_DIV2K_512 0-50 256x256"
        config[DATALOADER][NAME] = DATASET_DL_DIV2K_512
        config[DATALOADER][CONFIG_DEGRADATION] = dict(
            noise_stddev=[0., 50.]
        )
        config[DATALOADER][SIZE] = (256, 256)
    elif exp == 3002:  # ENABLE GRADIENT CLIPPING
        config = presets_experiments(exp, n=30,  b=16, model_preset="NAFNet")
        config[PRETTY_NAME] = "NAFNet41.4M denoise - DL_DIV2K_512 0-50 128x128"
        config[DATALOADER][NAME] = DATASET_DL_DIV2K_512
        config[DATALOADER][CONFIG_DEGRADATION] = dict(
            noise_stddev=[0., 50.]
        )
        config[DATALOADER][SIZE] = (128, 128)
    elif exp == 3010 or exp == 3011:  # exp 3011 = REDO with Gradient clipping
        config = presets_experiments(exp, n=50,  b=4, model_preset="NAFNet")
        config[PRETTY_NAME] = "NAFNet3.4M Light denoise - DL_DIV2K_512 0-50 256x256"
        config[DATALOADER][NAME] = DATASET_DL_DIV2K_512
        config[DATALOADER][CONFIG_DEGRADATION] = dict(
            noise_stddev=[0., 50.]
        )
        config[MODEL][ARCHITECTURE] = dict(
            width=64,
            enc_blk_nums=[1, 1, 2],
            middle_blk_num=1,
            dec_blk_nums=[1, 1, 1],
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
    # Pure DIV2K DENOISING trainings!
    # ---------------------------------
    elif exp == 3120:
        config = presets_experiments(exp, b=32, n=50)
        config[DATALOADER][NAME] = DATASET_DIV2K
        config[DATALOADER][CONFIG_DEGRADATION] = dict(
            noise_stddev=[0., 50.]
        )
        config[PRETTY_NAME] = "Vanilla DIV2K_512 0-50 - noisy 0-50"
    elif exp == 3111:
        config = presets_experiments(exp, n=50,  b=4, model_preset="NAFNet")
        config[PRETTY_NAME] = "NAFNet3.4M Light denoise - DIV2K_512 0-50 256x256"
        config[DATALOADER][NAME] = DATASET_DIV2K
        config[DATALOADER][CONFIG_DEGRADATION] = dict(noise_stddev=[0., 50.])
        config[MODEL][ARCHITECTURE] = dict(
            width=64,
            enc_blk_nums=[1, 1, 2],
            middle_blk_num=1,
            dec_blk_nums=[1, 1, 1],
        )
        config[DATALOADER][SIZE] = (256, 256)
    elif exp == 3101:
        config = presets_experiments(exp, n=30,  b=8, model_preset="NAFNet")
        config[PRETTY_NAME] = "NAFNet41.4M denoise - DIV2K_512 0-50 256x256"
        config[DATALOADER][NAME] = DATASET_DIV2K
        config[DATALOADER][CONFIG_DEGRADATION] = dict(
            noise_stddev=[0., 50.]
        )
        config[DATALOADER][SIZE] = (256, 256)
    # ---------------------------------
    # Pure EXTRA PRIMITIVES
    # ---------------------------------
    elif exp == 3030:
        config = presets_experiments(exp, b=128, n=50)
        config[DATALOADER][NAME] = DATASET_DL_EXTRAPRIMITIVES_DIV2K_512
        config[DATALOADER][CONFIG_DEGRADATION] = dict(
            noise_stddev=[0., 50.]
        )
        config[PRETTY_NAME] = "Vanilla DL_PRIMITIVES_512 0-50 - noisy 0-50"
        # config[DATALOADER][SIZE] = (256, 256)
    elif exp == 3040:
        config = presets_experiments(exp, n=50,  b=8, model_preset="NAFNet")
        config[PRETTY_NAME] = "NAFNet3.4M Light denoise - DL_PRIMITIVES_512 0-50 256x256"
        config[DATALOADER][NAME] = DATASET_DL_EXTRAPRIMITIVES_DIV2K_512
        config[DATALOADER][CONFIG_DEGRADATION] = dict(noise_stddev=[0., 50.])
        config[MODEL][ARCHITECTURE] = dict(
            width=64,
            enc_blk_nums=[1, 1, 2],
            middle_blk_num=1,
            dec_blk_nums=[1, 1, 1],
        )
        config[DATALOADER][SIZE] = (256, 256)
    elif exp == 3050:
        config = presets_experiments(exp, n=30,  b=8, model_preset="NAFNet")
        config[PRETTY_NAME] = "NAFNet41.4M denoise - DL_PRIMITIVES_512 0-50 256x256"
        config[DATALOADER][NAME] = DATASET_DL_EXTRAPRIMITIVES_DIV2K_512
        config[DATALOADER][CONFIG_DEGRADATION] = dict(noise_stddev=[0., 50.])
        config[DATALOADER][SIZE] = (256, 256)
    # ---------------------------------
    # DEBLURRING
    # ---------------------------------
    elif exp == 5000:
        config = presets_experiments(exp, n=30,  b=8, model_preset="NAFNet")
        config[PRETTY_NAME] = "NAFNet deblur - DL_DIV2K_512 256x256"
        config[DATALOADER][NAME] = DATASET_DL_DIV2K_512
        config[DATALOADER][CONFIG_DEGRADATION] = dict(
            noise_stddev=[0., 0.],
            degradation_blur=DEGRADATION_BLUR_MAT,  # Using .mat kernels
            augmentation_list=[AUGMENTATION_FLIP]
        )
        config[DATALOADER][SIZE] = (256, 256)
    elif exp == 5001:
        config = presets_experiments(exp, n=30,  b=8, model_preset="NAFNet")
        config[PRETTY_NAME] = "NAFNet deblur - DIV2K_512 256x256"
        config[DATALOADER][NAME] = DATASET_DIV2K
        config[DATALOADER][CONFIG_DEGRADATION] = dict(
            noise_stddev=[0., 0.],
            degradation_blur=DEGRADATION_BLUR_MAT,  # Using .mat kernels
            augmentation_list=[AUGMENTATION_FLIP]
        )
        config[DATALOADER][SIZE] = (256, 256)
    elif exp == 5002:
        config = presets_experiments(exp, n=30,  b=8, model_preset="NAFNet")
        config[PRETTY_NAME] = "NAFNet deblur - DL_DIV2K_512 256x256"
        config[DATALOADER][NAME] = DATASET_DL_DIV2K_512
        config[DATALOADER][CONFIG_DEGRADATION] = dict(
            noise_stddev=[0., 0.],
            degradation_blur=DEGRADATION_BLUR_MAT,  # Using .mat kernels
            augmentation_list=[AUGMENTATION_FLIP]
        )
        config[DATALOADER][SIZE] = (256, 256)
    elif exp == 5003:
        config = presets_experiments(exp, n=30,  b=8, model_preset="NAFNet")
        config[PRETTY_NAME] = "NAFNet deblur - DIV2K_512 256x256"
        config[DATALOADER][NAME] = DATASET_DIV2K
        config[DATALOADER][CONFIG_DEGRADATION] = dict(
            noise_stddev=[0., 0.],
            degradation_blur=DEGRADATION_BLUR_MAT,  # Using .mat kernels
            augmentation_list=[AUGMENTATION_FLIP]
        )
        config[DATALOADER][SIZE] = (256, 256)
    elif exp == 5004:
        config = presets_experiments(exp, n=30,  b=8, model_preset="NAFNet")
        config[PRETTY_NAME] = "NAFNet deblur - DIV2K_512 256x256"
        config[DATALOADER][NAME] = DATASET_DIV2K
        config[DATALOADER][CONFIG_DEGRADATION] = dict(
            noise_stddev=[0., 0.],
            degradation_blur=DEGRADATION_BLUR_MAT,  # Using .mat kernels
            augmentation_list=[AUGMENTATION_FLIP]
        )
        config[DATALOADER][SIZE] = (256, 256)
    elif exp == 5005:
        config = presets_experiments(exp, n=30,  b=8, model_preset="UNet")
        config[PRETTY_NAME] = "UNet deblur - DL_512 256x256"
        config[DATALOADER][NAME] = DATASET_DL_DIV2K_512
        config[DATALOADER][CONFIG_DEGRADATION] = dict(
            noise_stddev=[0., 0.],
            degradation_blur=DEGRADATION_BLUR_MAT,  # Using .mat kernels
            augmentation_list=[AUGMENTATION_FLIP]
        )
        config[DATALOADER][SIZE] = (256, 256)
    # elif exp == 6000:  # -> FAILED, no kernels normalization!
    #     config = presets_experiments(exp, b=32, n=50)
    #     config[DATALOADER][NAME] = DATASET_DL_DIV2K_512
    #     config[DATALOADER][CONFIG_DEGRADATION] = dict(
    #         noise_stddev=[0., 50.],
    #         degradation_blur=DEGRADATION_BLUR_MAT,  # Deblur = Using .mat kernels
    #         augmentation_list=[AUGMENTATION_FLIP]
    #     )
    #     config[PRETTY_NAME] = "Vanilla deblur DL_DIV2K_512"
    elif exp == 6002:
        config = presets_experiments(exp, b=128, n=50)
        config[DATALOADER][NAME] = DATASET_DL_DIV2K_512
        config[DATALOADER][CONFIG_DEGRADATION] = dict(
            noise_stddev=[0., 50.],
            degradation_blur=DEGRADATION_BLUR_MAT,  # Deblur = Using .mat kernels
            augmentation_list=[AUGMENTATION_FLIP]
        )
        config[PRETTY_NAME] = "Vanilla deblur DL_DIV2K_512"
    # elif exp == 6001:  # -> FAILED, no kernels normalization!
    #     config = presets_experiments(exp, b=32, n=50)
    #     config[DATALOADER][NAME] = DATASET_DIV2K
    #     config[DATALOADER][CONFIG_DEGRADATION] = dict(
    #         noise_stddev=[0., 50.],
    #         degradation_blur=DEGRADATION_BLUR_MAT,  # Deblur = Using .mat kernels
    #         augmentation_list=[AUGMENTATION_FLIP]
    #     )
    #     config[PRETTY_NAME] = "Vanilla delbur DIV2K_512"
    elif exp == 6003:
        config = presets_experiments(exp, b=128, n=50)
        config[DATALOADER][NAME] = DATASET_DIV2K
        config[DATALOADER][CONFIG_DEGRADATION] = dict(
            noise_stddev=[0., 50.],
            degradation_blur=DEGRADATION_BLUR_MAT,  # Deblur = Using .mat kernels
            augmentation_list=[AUGMENTATION_FLIP]
        )
        config[PRETTY_NAME] = "Vanilla delbur DIV2K_512"

    elif exp == 7000:
        config = presets_experiments(exp, b=16, n=30, model_preset="NAFNet")
        config[MODEL][ARCHITECTURE] = dict(
            width=64,
            enc_blk_nums=[1, 1, 2],
            middle_blk_num=1,
            dec_blk_nums=[1, 1, 1],
        )
        config[DATALOADER][NAME] = DATASET_DL_DIV2K_512
        config[DATALOADER][CONFIG_DEGRADATION] = dict(
            noise_stddev=[0., 50.],
            degradation_blur=DEGRADATION_BLUR_MAT,  # Deblur = Using .mat kernels
            augmentation_list=[AUGMENTATION_FLIP]
        )
        config[PRETTY_NAME] = "NafNet Light deblur DL"
        config[DATALOADER][SIZE] = (256, 256)
    elif exp == 7001:
        config = presets_experiments(exp, b=16, n=50, model_preset="NAFNet")
        config[DATALOADER][NAME] = DATASET_DIV2K
        config[MODEL][ARCHITECTURE] = dict(
            width=64,
            enc_blk_nums=[1, 1, 2],
            middle_blk_num=1,
            dec_blk_nums=[1, 1, 1],
        )
        config[DATALOADER][CONFIG_DEGRADATION] = dict(
            noise_stddev=[0., 50.],
            degradation_blur=DEGRADATION_BLUR_MAT,  # Deblur = Using .mat kernels
            augmentation_list=[AUGMENTATION_FLIP]
        )
        config[PRETTY_NAME] = "NafNet Light deblur DIV2K"
        config[DATALOADER][SIZE] = (256, 256)
    elif exp == 7002:
        config = presets_experiments(exp, n=20,  b=8, model_preset="UNet")
        config[PRETTY_NAME] = "UNET deblur - DIV2K"
        config[DATALOADER][NAME] = DATASET_DIV2K
        config[DATALOADER][CONFIG_DEGRADATION] = dict(
            noise_stddev=[0., 0.],
            degradation_blur=DEGRADATION_BLUR_MAT,  # Using .mat kernels
            augmentation_list=[AUGMENTATION_FLIP]
        )
        config[DATALOADER][SIZE] = (256, 256)
    else:
        raise ValueError(f"Experiment {exp} not found")
    return config
