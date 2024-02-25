import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from rstor.synthetic_data.dead_leaves import dead_leaves_chart


class DeadLeavesDataset(Dataset):
    def __init__(
        self,
        size: Tuple[int, int] = (128, 128),
        length: int = 1000,
        frozen_seed: int = None,  # useful for validation set!
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

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        seed = self.frozen_seed + idx if self.frozen_seed is not None else None
        chart = dead_leaves_chart(self.size, seed=seed, **self.config_dead_leaves)
        return torch.from_numpy(chart).permute(-1, 0, 1).float()


if __name__ == "__main__":
    dead_leaves_dataset = DeadLeavesDataset()
    dead_leaves_dataloader = DataLoader(dead_leaves_dataset, batch_size=4, shuffle=True)
    for i, batch in enumerate(dead_leaves_dataloader):
        print(batch.shape)  # Should print [batch_size, size[0], size[1], 3] for each batch
        if i == 1:  # Just to break the loop after two batches for demonstration
            break
