import torch
from rstor.data.dataloader import DeadLeavesDatasetGPU
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
                                   background_color=(0.2, 0.4, 0.6), colored=True, radius_mean=10, radius_stddev=3,
                                   noise_stddev=(0, 0), ds_factor=1)
    assert len(dataset) == 500
    assert dataset.size == (256, 256)
    assert dataset.frozen_seed == 42
    assert dataset.config_dead_leaves == {
        'number_of_circles': 5,
        'background_color': (0.2, 0.4, 0.6),
        'colored': True,
        'radius_mean': 10,
        'radius_stddev': 3
    }

    # Test case 3: Check item retrieval
    item, item_tgt = dataset[0]
    assert isinstance(item, torch.Tensor)
    assert item.shape == (3, 256, 256)

    # Test case 4: Repeatable results with frozen seed
    dataset1 = DeadLeavesDatasetGPU(frozen_seed=42, noise_stddev=(0, 0))
    dataset2 = DeadLeavesDatasetGPU(frozen_seed=42, noise_stddev=(0, 0))
    item1, item_tgt1 = dataset1[0]
    item2, item_tgt2 = dataset2[0]
    assert torch.all(torch.eq(item1, item2))
