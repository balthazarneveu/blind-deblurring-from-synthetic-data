import torch
from rstor.data.dataloader import DeadLeavesDataset


def test_dead_leaves_dataset():
    # Test case 1: Default parameters
    dataset = DeadLeavesDataset()
    assert len(dataset) == 1000
    assert dataset.size == (128, 128)
    assert dataset.frozen_seed is None
    assert dataset.config_dead_leaves == {}

    # Test case 2: Custom parameters
    dataset = DeadLeavesDataset(size=(256, 256), length=500, frozen_seed=42, number_of_circles=5,
                                background_color=(0.2, 0.4, 0.6), colored=True, radius_mean=10, radius_stddev=3)
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
    dataset1 = DeadLeavesDataset(frozen_seed=42)
    dataset2 = DeadLeavesDataset(frozen_seed=42)
    item1, item_tgt1 = dataset1[0]
    item2, item_tgt2 = dataset2[0]
    assert torch.all(torch.eq(item1, item2))