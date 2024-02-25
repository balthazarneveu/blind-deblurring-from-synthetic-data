from rstor.architecture.base import BaseModel
from rstor.architecture.convolution_blocks import BaseConvolutionBlock, ResConvolutionBlock
from rstor.properties import LEAKY_RELU
import torch


class StackedConvolutions(BaseModel):
    def __init__(self,
                 ch_in: int = 3,
                 ch_out: int = 3,
                 h_dim: int = 64,
                 num_layers: int = 8,
                 k_size: int = 3,
                 activation: str = LEAKY_RELU,
                 bias: bool = True,
                 ) -> None:
        super().__init__()
        assert num_layers % 2 == 0, "Number of layers should be even"
        self.conv_in_modality = BaseConvolutionBlock(
            ch_in, h_dim, k_size, activation=activation, bias=bias)
        conv_list = []
        for _i in range(num_layers-2):
            conv_list.append(ResConvolutionBlock(
                h_dim, h_dim, k_size, activation=activation, bias=bias, residual=True))
        self.conv_out_modality = BaseConvolutionBlock(
            h_dim, ch_out, k_size, activation=None, bias=bias)
        self.conv_stack = torch.nn.Sequential(self.conv_in_modality, *conv_list, self.conv_out_modality)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        return self.conv_stack(x_in)
