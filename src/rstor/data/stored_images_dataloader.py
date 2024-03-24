import torch
from torch.utils.data import DataLoader, Dataset
from rstor.data.augmentation import augment_flip
from rstor.properties import DEVICE, AUGMENTATION_FLIP
from rstor.properties import DATALOADER, BATCH_SIZE, TRAIN, VALIDATION, LENGTH, CONFIG_DEAD_LEAVES, SIZE
from typing import Tuple, Optional, Union
from torchvision.transforms import RandomCrop
from pathlib import Path
from tqdm import tqdm
from time import time
from torchvision.io import read_image
IMAGES_FOLDER = "images"


def load_image(path):
    return read_image(str(path))


class RestorationDataset(Dataset):
    def __init__(
        self,
        images_path: Path,
        device: str = DEVICE,
        preloaded: bool = False,
        augmentation_list: Optional[list] = [],
        freeze=True,
        **_extra_kwargs
    ):
        self.preloaded = preloaded
        self.augmentation_list = augmentation_list
        self.device = device
        self.freeze = freeze
        if not isinstance(images_path, list):
            self.path_list = sorted(list(images_path.glob("*.png")))
        else:
            self.path_list = images_path
        self.n_samples = len(self.path_list)
        # If we can preload everything in memory, we can do it
        if preloaded:
            self.data_list = [load_image(pth) for pth in tqdm(self.path_list)]
        else:
            self.data_list = self.path_list
        self.cropper = RandomCrop(size=(512, 512))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Access a specific image from dataset and augment

        Args:
            index (int): access index

        Returns:
            torch.Tensor: image tensor [C, H, W]
        """
        if self.preloaded:
            img_data = self.data_list[index]
        else:
            img_data = load_image(self.data_list[index])
        img_data = img_data.to(self.device)
        if AUGMENTATION_FLIP in self.augmentation_list:
            img_data = augment_flip(img_data)
        img_data = self.cropper(img_data)
        img_data = img_data.float()/255.
        degraded_img = img_data/2.
        return degraded_img, img_data

    def __len__(self):
        return self.n_samples


if __name__ == "__main__":
    dataset_restoration = RestorationDataset(
        Path("__dataset/div2k/DIV2K_train_HR/DIV2K_train_HR/"),
        preloaded=True,
    )
    dataloader = DataLoader(
        dataset_restoration,
        batch_size=16,
        shuffle=True
    )
    start = time()
    total = 0
    for batch in tqdm(dataloader):
        # print(batch.shape)
        torch.cuda.synchronize()
        total += batch.shape[0]
    end = time()
    print(f"Time elapsed: {(end-start)/total*1000.:.2f}ms/image")
