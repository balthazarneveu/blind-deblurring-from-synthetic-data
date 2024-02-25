import torch
from rstor.properties import LEAKY_RELU, RELU


def get_non_linearity(activation: str):
    if activation == LEAKY_RELU:
        non_linearity = torch.nn.LeakyReLU()
    elif activation == RELU:
        non_linearity = torch.nn.ReLU()
    elif activation is None:
        non_linearity = torch.nn.Identity()
    else:
        raise ValueError(f"Unknown activation {activation}")
    return non_linearity


class BaseModel(torch.nn.Module):
    """Base class for all restoration models with additional useful methods"""

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def receptive_field(self) -> int:
        """Compute the receptive field of the model

        Returns:
            int: receptive field
        """
        input_tensor = torch.rand(1, 3, 128, 128, requires_grad=True)
        out = self.forward(input_tensor)
        grad = torch.zeros_like(out)
        grad[..., out.shape[-2]//2, out.shape[-1]//2] = torch.nan  # set NaN gradient at the middle of the output
        out.backward(gradient=grad)
        self.zero_grad()
        receptive_field_mask = input_tensor.grad.isnan()[0, 0]
        receptive_field_indexes = torch.where(receptive_field_mask)
        # Count NaN in the input
        receptive_x = 1+receptive_field_indexes[-1].max() - receptive_field_indexes[-1].min()  # Horizontal x
        receptive_y = 1+receptive_field_indexes[-2].max() - receptive_field_indexes[-2].min()  # Vertical y
        return receptive_x, receptive_y
