from torch.utils.data import DataLoader
from rstor.data.synthetic_dataloader import DeadLeavesDataset, DeadLeavesDatasetGPU
from rstor.data.stored_images_dataloader import RestorationDataset
from rstor.properties import (
    DATALOADER, BATCH_SIZE, TRAIN, VALIDATION, LENGTH, CONFIG_DEAD_LEAVES, SIZE, NAME, CONFIG_DEGRADATION,
    DATASET_SYNTH_LIST, DATASET_DIV2K,
    DATASET_PATH
)
from typing import Optional
from random import seed, shuffle


def get_data_loader_synthetic(config, frozen_seed=42):
    # print(config[DATALOADER].get(CONFIG_DEAD_LEAVES, {}))
    if config[DATALOADER].get("gpu_gen", False):
        print("Using GPU dead leaves generator")
        ds = DeadLeavesDatasetGPU
    else:
        ds = DeadLeavesDataset
    dl_train = ds(config[DATALOADER][SIZE], config[DATALOADER][LENGTH][TRAIN],
                  frozen_seed=None, **config[DATALOADER].get(CONFIG_DEAD_LEAVES, {}))
    dl_valid = ds(config[DATALOADER][SIZE], config[DATALOADER][LENGTH][VALIDATION],
                  frozen_seed=frozen_seed, **config[DATALOADER].get(CONFIG_DEAD_LEAVES, {}))
    dl_dict = create_dataloaders(config, dl_train, dl_valid)
    return dl_dict


def create_dataloaders(config, dl_train, dl_valid) -> dict:
    dl_dict = {
        TRAIN: DataLoader(
            dl_train,
            shuffle=True,
            batch_size=config[DATALOADER][BATCH_SIZE][TRAIN],
        ),
        VALIDATION: DataLoader(
            dl_valid,
            shuffle=False,
            batch_size=config[DATALOADER][BATCH_SIZE][VALIDATION]
        ),
        # TEST: DataLoader(dl_test, shuffle=False, batch_size=config[DATALOADER][BATCH_SIZE][TEST])
    }
    return dl_dict


def get_data_loader_from_disk(config, frozen_seed: Optional[int] = 42) -> dict:
    ds = RestorationDataset
    dataset_name = config[DATALOADER][NAME]  # NAME shall be here!
    if dataset_name == DATASET_DIV2K:
        dataset_root = DATASET_PATH/DATASET_DIV2K
        train_root = dataset_root/"DIV2K_train_HR"/"DIV2K_train_HR"
        valid_root = dataset_root/"DIV2K_valid_HR"/"DIV2K_valid_HR"
        train_files = sorted(list(train_root.glob("*.png")))
        train_files = 5*train_files  # Just to get 4000 elements...
        valid_files = sorted(list(valid_root.glob("*.png")))
    elif dataset_name in DATASET_SYNTH_LIST:
        dataset_root = DATASET_PATH/dataset_name
        all_files = sorted(list(dataset_root.glob("*.png")))
        seed(frozen_seed)
        shuffle(all_files)  # Easy way to perform cross validation if neeeded
        cut_index = int(0.9*len(all_files))
        train_files = all_files[:cut_index]
        valid_files = all_files[cut_index:]
    dl_train = ds(
        train_files,
        size=config[DATALOADER][SIZE],
        frozen_seed=None,
        # length=config[DATALOADER][LENGTH][TRAIN],
        **config[DATALOADER].get(CONFIG_DEGRADATION, {})
    )
    dl_valid = ds(
        valid_files,
        size=config[DATALOADER][SIZE],
        frozen_seed=frozen_seed,
        **config[DATALOADER].get(CONFIG_DEGRADATION, {})
    )
    dl_dict = create_dataloaders(config, dl_train, dl_valid)
    return dl_dict


def get_data_loader(config, frozen_seed=42):
    dataset_name = config[DATALOADER].get(NAME, False)
    if dataset_name:
        return get_data_loader_from_disk(config, frozen_seed)
    else:
        return get_data_loader_synthetic(config, frozen_seed)


if __name__ == "__main__":
    # Example of usage synthetic dataset
    for dataset_name in [DATASET_DIV2K, None, DATASET_DL_DIV2K_512, DATASET_DL_DIV2K_1024]:
        if dataset_name is None:
            dead_leaves_dataset = DeadLeavesDatasetGPU(colored=True)
            dl = DataLoader(dead_leaves_dataset, batch_size=4, shuffle=True)
        else:
            # Example of usage stored images dataset
            config = {
                DATALOADER: {
                    NAME: dataset_name,
                    SIZE: (128, 128),
                    BATCH_SIZE: {
                        TRAIN: 4,
                        VALIDATION: 4
                    },
                }
            }
            dl_dict = get_data_loader(config)
            dl = dl_dict[TRAIN]
            # dl = dl_dict[VALIDATION]
        for i, (batch_inp, batch_target) in enumerate(dl):
            print(batch_inp.shape, batch_target.shape)  # Should print [batch_size, size[0], size[1], 3] for each batch
            if i == 1:  # Just to break the loop after two batches for demonstration
                import matplotlib.pyplot as plt
                plt.subplot(1, 2, 1)
                plt.imshow(batch_inp.permute(0, 2, 3, 1).reshape(-1, batch_inp.shape[-1], 3).cpu().numpy())
                plt.title("Degraded")
                plt.subplot(1, 2, 2)
                plt.imshow(batch_target.permute(0, 2, 3, 1).reshape(-1, batch_inp.shape[-1], 3).cpu().numpy())
                plt.title("Target")
                plt.show()
                # print(batch_target)
                break
