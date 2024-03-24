import torch
from rstor.data.stored_images_dataloader import RestorationDataset
from numba import cuda
from rstor.properties import DATASET_PATH


def test_dataloader_stored():
    if not cuda.is_available():
        print("cuda unavailable, exiting")
        return

    # Test case 1: Default parameters
    dataset = RestorationDataset(noise_stddev=(0, 0),
                                 images_path=DATASET_PATH/"sample")
    assert len(dataset) == 2
    assert dataset.frozen_seed is None

    # Test case 2: Custom parameters
    dataset = RestorationDataset(images_path=DATASET_PATH/"sample",
                                 frozen_seed=42,
                                 noise_stddev=(0, 0))
    assert len(dataset) == 2
    assert dataset.frozen_seed == 42

    # Test case 3: Check item retrieval
    item, item_tgt = dataset[0]
    assert isinstance(item, torch.Tensor)
    assert item.shape == item_tgt.shape

    # Test case 4: Repeatable results with frozen seed
    dataset1 = RestorationDataset(images_path=DATASET_PATH/"sample", 
                                  frozen_seed=42, noise_stddev=(0, 0))
    dataset2 = RestorationDataset(images_path=DATASET_PATH/"sample",
                                  frozen_seed=42, noise_stddev=(0, 0))
    item1, item_tgt1 = dataset1[0]
    item2, item_tgt2 = dataset2[0]
    
    assert torch.all(torch.eq(item1, item2))



    # Test case 5: Visualize
    # dataset = RestorationDataset(frozen_seed=42, noise_stddev=(0, 0))
    # item, item_tgt = dataset[0]
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(item.permute(1, 2, 0).detach().cpu())
    # plt.show()
    # print("done")

