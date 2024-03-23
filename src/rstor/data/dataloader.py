from torch.utils.data import DataLoader
from rstor.properties import DATALOADER, BATCH_SIZE, TRAIN, VALIDATION, LENGTH, CONFIG_DEAD_LEAVES, SIZE
from rstor.data.synthetic_dataloader import DeadLeavesDataset, DeadLeavesDatasetGPU


def get_data_loader(config, frozen_seed=42):
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


if __name__ == "__main__":
    dead_leaves_dataset = DeadLeavesDataset(colored=True)
    dead_leaves_dataloader = DataLoader(dead_leaves_dataset, batch_size=4, shuffle=True)
    for i, (batch_inp, batch_target) in enumerate(dead_leaves_dataloader):
        print(batch_inp.shape, batch_target.shape)  # Should print [batch_size, size[0], size[1], 3] for each batch
        if i == 1:  # Just to break the loop after two batches for demonstration
            import matplotlib.pyplot as plt
            plt.subplot(1, 2, 1)
            plt.imshow(batch_inp.permute(0, 2, 3, 1).reshape(-1, 128, 3).numpy())
            plt.title("Degraded")
            plt.subplot(1, 2, 2)
            plt.imshow(batch_target.permute(0, 2, 3, 1).reshape(-1, 128, 3).numpy())
            plt.title("Target")
            plt.show()
            print(batch_target)
            break
