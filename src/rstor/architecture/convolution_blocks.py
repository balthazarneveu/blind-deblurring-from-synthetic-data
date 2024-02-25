import torch
from rstor.properties import LEAKY_RELU
from rstor.architecture.base import get_non_linearity


class BaseConvolutionBlock(torch.nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        k_size: int,
        activation=LEAKY_RELU,
        bias: bool = True
    ) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(ch_in, ch_out, k_size, padding=k_size//2, bias=bias)
        self.non_linearity = get_non_linearity(activation)
        self.conv_non_lin = torch.nn.Sequential(self.conv, self.non_linearity)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        return self.conv_non_lin(x_in)


class ResConvolutionBlock(torch.nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        k_size: int,
        activation=LEAKY_RELU,
        bias: bool = True,
        residual: bool = True
    ) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(ch_in, ch_out, k_size, padding=k_size//2, bias=bias)
        self.non_linearity = get_non_linearity(activation)
        self.conv2 = torch.nn.Conv2d(ch_out, ch_out, k_size, padding=k_size//2, bias=bias)
        self.residual = residual

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x_in)
        y = self.non_linearity(y)
        y = self.conv2(y)
        if self.residual:
            y = x_in + y
        y = self.non_linearity(y)
        return y
