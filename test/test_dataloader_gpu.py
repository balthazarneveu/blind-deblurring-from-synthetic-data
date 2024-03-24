import torch
from rstor.data.synthetic_dataloader import DeadLeavesDatasetGPU
import numba


def test_dead_leaves_dataset_gpu():
    if not numba.cuda.is_available():
        return

    # Test case 1: Default parameters
    dataset = DeadLeavesDatasetGPU(noise_stddev=(0, 0), ds_factor=1)
    assert len(dataset) == 1000
    assert dataset.size == (128, 128)
    assert dataset.frozen_seed is None
    assert dataset.config_dead_leaves == {}

    # Test case 2: Custom parameters
    dataset = DeadLeavesDatasetGPU(size=(256, 256), length=500, frozen_seed=42, number_of_circles=5,
                                   background_color=(0.2, 0.4, 0.6), colored=True, radius_min=1, radius_alpha=3,
                                   noise_stddev=(0, 0), ds_factor=1)
    assert len(dataset) == 500
    assert dataset.size == (256, 256)
    assert dataset.frozen_seed == 42
    assert dataset.config_dead_leaves == {
        'number_of_circles': 5,
        'background_color': (0.2, 0.4, 0.6),
        'colored': True,
        'radius_min': 1,
        'radius_alpha': 3
    }

    # Test case 3: Check item retrieval
    item, item_tgt = dataset[0]
    assert isinstance(item, torch.Tensor)
    assert item.shape == (3, 256, 256)

    # Test case 4: Repeatable results with frozen seed
    dataset1 = DeadLeavesDatasetGPU(frozen_seed=42, noise_stddev=(0, 0), number_of_circles=256)
    dataset2 = DeadLeavesDatasetGPU(frozen_seed=42, noise_stddev=(0, 0), number_of_circles=256)
    item1, item_tgt1 = dataset1[0]
    item2, item_tgt2 = dataset2[0]
    assert torch.all(torch.eq(item1, item2))



    # Test case 5: Visualize
    # dataset = DeadLeavesDatasetGPU(size=(256, 256), length=500, frozen_seed=44, number_of_circles=10_000,
    #                                 background_color=(0.2, 0.4, 0.6), colored=True, radius_min=1, radius_alpha=3,
    #                                 noise_stddev=(0, 0), ds_factor=1)
    # item, item_tgt = dataset[0]
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(item.permute(1, 2, 0).detach().cpu())
    # plt.show()
    # print("done")

