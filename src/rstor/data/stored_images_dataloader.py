import torch
from torch.utils.data import DataLoader, Dataset
from rstor.data.augmentation import augment_flip
from rstor.data.degradation import DegradationBlurMat, DegradationBlurGauss, DegradationNoise
from rstor.properties import (
    DEVICE, AUGMENTATION_FLIP, AUGMENTATION_ROTATE, DEGRADATION_BLUR_NONE,
    DEGRADATION_BLUR_MAT, DEGRADATION_BLUR_GAUSS
)
from typing import Tuple, Optional, Union
from torchvision import transforms
# from torchvision.transforms import RandomCrop
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
        size: Tuple[int, int] = (128, 128),
        device: str = DEVICE,
        preloaded: bool = False,
        augmentation_list: Optional[list] = [],
        frozen_seed: int = None,  # useful for validation set!
        blur_kernel_half_size: int = [0, 2],
        noise_stddev: float = [0., 50.],
        degradation_blur=DEGRADATION_BLUR_NONE,
        length=None,
        blur_index=None,
        **_extra_kwargs
    ):
        self.preloaded = preloaded
        self.augmentation_list = augmentation_list
        self.device = device
        self.frozen_seed = frozen_seed
        if not isinstance(images_path, list):
            self.path_list = sorted(list(images_path.glob("*.png")))
        else:
            self.path_list = images_path

        self.length = len(self.path_list) if length is None else length
        self.n_samples = len(self.path_list) if length is None else length
        # If we can preload everything in memory, we can do it
        if preloaded:
            self.data_list = [load_image(pth) for pth in tqdm(self.path_list)]
        else:
            self.data_list = self.path_list

        # if AUGMENTATION_FLIP in self.augmentation_list:
        #     img_data = augment_flip(img_data)
        # img_data = self.cropper(img_data)
        self.transforms = []

        if self.frozen_seed is None:
            if AUGMENTATION_FLIP in self.augmentation_list:
                self.transforms.append(transforms.RandomHorizontalFlip(p=0.5))
                self.transforms.append(transforms.RandomVerticalFlip(p=0.5))
            if AUGMENTATION_ROTATE in self.augmentation_list:
                self.transforms.append(transforms.RandomRotation(degrees=180))

        crop = transforms.RandomCrop(size) if frozen_seed is None else transforms.CenterCrop(size)
        self.transforms.append(crop)
        self.transforms = transforms.Compose(self.transforms)

        # self.cropper = RandomCrop(size=size)

        self.degradation_blur_type = degradation_blur
        if degradation_blur == DEGRADATION_BLUR_GAUSS:
            self.degradation_blur = DegradationBlurGauss(self.length,
                                                         blur_kernel_half_size,
                                                         frozen_seed)
            self.blur_deg_str = "blur_kernel_half_size"
        elif degradation_blur == DEGRADATION_BLUR_MAT:
            self.degradation_blur = DegradationBlurMat(self.length,
                                                       frozen_seed,
                                                       blur_index)
            self.blur_deg_str = "blur_kernel_id"
        elif degradation_blur == DEGRADATION_BLUR_NONE:
            pass
        else:
            raise ValueError(f"Unknown degradation blur {degradation_blur}")

        self.degradation_noise = DegradationNoise(self.length,
                                                  noise_stddev,
                                                  frozen_seed)
        self.current_degradation = {}

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

        # if AUGMENTATION_FLIP in self.augmentation_list:
        #     img_data = augment_flip(img_data)
        # img_data = self.cropper(img_data)

        img_data = self.transforms(img_data)
        img_data = img_data.float()/255.
        degraded_img = img_data.clone().unsqueeze(0)

        self.current_degradation[index] = {}
        if self.degradation_blur_type != DEGRADATION_BLUR_NONE:
            degraded_img = self.degradation_blur(degraded_img, index)
            self.current_degradation[index][self.blur_deg_str] = self.degradation_blur.current_degradation[index][self.blur_deg_str]

        degraded_img = self.degradation_noise(degraded_img, index)
        self.current_degradation[index]["noise_stddev"] = self.degradation_noise.current_degradation[
            index]["noise_stddev"]

        degraded_img = degraded_img.squeeze(0)
        self.current_degradation[index] = {
            "noise_stddev": self.degradation_noise.current_degradation[index]["noise_stddev"]
        }
        try:
            self.current_degradation[index][self.blur_deg_str] = self.degradation_blur.current_degradation[
                index][self.blur_deg_str]
        except:  # noqa: E722
            pass

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
