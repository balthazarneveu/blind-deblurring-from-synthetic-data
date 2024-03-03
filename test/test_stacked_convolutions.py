import torch
from rstor.architecture.stacked_convolutions import StackedConvolutions
from rstor.properties import RELU


def test_stacked_convolutions():
    # Test case 1: Default parameters
    model = StackedConvolutions()
    assert isinstance(model, torch.nn.Module)

    # Test case 2: Number of layers is not even
    try:
        model = StackedConvolutions(num_layers=7)
        assert False, "Expected AssertionError"
    except AssertionError:
        pass

    # Test case 3: Custom parameters
    n, c, h, w = 1, 3, 64, 64
    model = StackedConvolutions(ch_in=c, ch_out=2, h_dim=32, num_layers=4, k_size=5, activation=RELU, bias=False)
    assert isinstance(model, torch.nn.Module)

    # Test case 4: Forward pass
    input_tensor = torch.randn(n, c, h, w)
    output_tensor = model(input_tensor)
    assert model.receptive_field() == (25, 25)
    assert output_tensor.shape == (1, 2, h, w)
