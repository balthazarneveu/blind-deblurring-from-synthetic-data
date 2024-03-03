import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from rstor.synthetic_data.dead_leaves import dead_leaves_chart
from rstor.properties import DATALOADER, BATCH_SIZE, TRAIN, VALIDATION, LENGTH, CONFIG_DEAD_LEAVES, SIZE
import cv2
import random


class DeadLeavesDataset(Dataset):
    def __init__(
        self,
        size: Tuple[int, int] = (128, 128),
        length: int = 1000,
        frozen_seed: int = None,  # useful for validation set!
        blur_kernel_half_size: int = [0, 2],
        **config_dead_leaves
        # number_of_circles: int = -1,
        # background_color: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        # colored: Optional[bool] = False,
        # radius_mean: Optional[int] = -1,
        # radius_stddev: Optional[int] = -1,
    ):
        self.frozen_seed = frozen_seed

        self.size = size
        self.length = length
        self.config_dead_leaves = config_dead_leaves
        self.blur_kernel_half_size = blur_kernel_half_size
        if frozen_seed is not None:
            random.seed(self.frozen_seed)
            self.blur_kernel_half_size = [
                (self.blur_kernel_half_size[0], self.blur_kernel_half_size[1]) for _ in range(length)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seed = self.frozen_seed + idx if self.frozen_seed is not None else None
        chart = dead_leaves_chart(self.size, seed=seed, **self.config_dead_leaves)
        if self.frozen_seed is not None:
            k_size_x, k_size_y = self.blur_kernel_half_size[idx]
        else:
            k_size_x = random.randint(self.blur_kernel_half_size[0], self.blur_kernel_half_size[1])
            k_size_y = random.randint(self.blur_kernel_half_size[0], self.blur_kernel_half_size[1])
        k_size_x = 2 * k_size_x + 1
        k_size_y = 2 * k_size_y + 1
        degraded_chart = cv2.GaussianBlur(chart, (k_size_x, k_size_y), 0)

        def numpy_to_torch(ndarray):
            return torch.from_numpy(ndarray).permute(-1, 0, 1).float()
        return numpy_to_torch(degraded_chart), numpy_to_torch(chart)


def get_data_loader(config, frozen_seed=42, **config_dead_leaves):
    dl_train = DeadLeavesDataset(config[DATALOADER][SIZE], config[DATALOADER][LENGTH][TRAIN],
                                 frozen_seed=None, **config[DATALOADER].get(CONFIG_DEAD_LEAVES, {}))
    dl_valid = DeadLeavesDataset(config[DATALOADER][SIZE], config[DATALOADER][LENGTH][VALIDATION],
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
