"""
NAFNet: Non linear activation free neural network
Architecture adapted from Simple Baselines for Image Restoration
https://github.com/megvii-research/NAFNet/tree/main
"""
from torch import nn
import torch.nn.functional as F
import torch
from rstor.architecture.base import BaseModel, get_non_linearity
from typing import Optional, List
from rstor.properties import RELU, SIMPLE_GATE


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class NAFBlock(nn.Module):
    def __init__(
        self,
        c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.,
        activation: Optional[str] = SIMPLE_GATE,
        layer_norm_flag: Optional[bool] = True,
        channel_attention_flag: Optional[bool] = True,
    ):
        super().__init__()
        self.layer_norm_flag = layer_norm_flag
        self.channel_attention_flag = channel_attention_flag
        dw_channel = c * DW_Expand
        half_dw_channel = dw_channel // 2
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1,
                               padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel if activation == SIMPLE_GATE else half_dw_channel,
            kernel_size=3,
            padding=1, stride=1,
            groups=dw_channel if activation == SIMPLE_GATE else half_dw_channel,
            bias=True
        )
        # To grand the same amount of parameters between Simple Gate and ReLU versions...
        # Conv2 has to reduce the number of channels to half but... using grouped convolution
        # w -> w/2 ... not really a depthwise convolution but rather by channels of 2!
        self.conv3 = nn.Conv2d(in_channels=half_dw_channel, out_channels=c,
                               kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Simplified Channel Attention
        if self.channel_attention_flag:
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=half_dw_channel, out_channels=half_dw_channel, kernel_size=1,
                          padding=0, stride=1,
                          groups=1, bias=True),
            )

        # SimpleGate
        self.sg = get_non_linearity(activation)
        ffn_channel = FFN_Expand
        half_ffn_channel = ffn_channel // 2 if activation == SIMPLE_GATE else ffn_channel
        self.conv4 = nn.Conv2d(
            in_channels=c,
            out_channels=ffn_channel if activation == SIMPLE_GATE else half_ffn_channel,
            kernel_size=1,
            padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=half_ffn_channel, out_channels=c,
                               kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        if self.layer_norm_flag:
            self.norm1 = LayerNorm2d(c)
            self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        if self.layer_norm_flag:
            x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        if self.channel_attention_flag:
            x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y) if self.layer_norm_flag else y)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(BaseModel):
    def __init__(
            self,
            img_channel: Optional[int] = 3,
            width: Optional[int] = 16,
            middle_blk_num: Optional[int] = 1,
            enc_blk_nums: List[int] = [],
            dec_blk_nums: List[int] = [],
            activation: Optional[bool] = SIMPLE_GATE,
            layer_norm_flag: Optional[bool] = True,
            channel_attention_flag: Optional[bool] = True,
    ) -> None:
        super().__init__()

        self.intro = nn.Conv2d(
            in_channels=img_channel,
            out_channels=width,
            kernel_size=3,
            padding=1, stride=1,
            groups=1,
            bias=True
        )
        config_block = {
            "activation": activation,
            "layer_norm_flag": layer_norm_flag,
            "channel_attention_flag": channel_attention_flag
        }
        self.ending = nn.Conv2d(
            in_channels=width, out_channels=img_channel, kernel_size=3,
            padding=1, stride=1, groups=1,
            bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, **config_block) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan, **config_block) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, **config_block) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        B, C, H, W = inp.shape
        inp = self.sanitize_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def sanitize_image_size(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class UNet(NAFNet):
    def __init__(
            self,
            activation: Optional[bool] = RELU,
            layer_norm_flag: Optional[bool] = False,
            channel_attention_flag: Optional[bool] = False,
            **kwargs):
        super().__init__(
            activation=activation,
            layer_norm_flag=layer_norm_flag,
            channel_attention_flag=channel_attention_flag, **kwargs)


if __name__ == '__main__':
    tiny_recetive_field = True
    if tiny_recetive_field:
        enc_blks = [1, 1, 2]
        middle_blk_num = 1
        dec_blks = [1, 1, 1]
        width = 16
        # Receptive field is 208x208
    else:
        enc_blks = [1, 1, 1, 28]
        middle_blk_num = 1
        dec_blks = [1, 1, 1, 1]
        width = 2
        # Receptive field is 544x544
    device = "cpu"

    for model_name in ["NAFNet", "UNet"]:
        if model_name == "NAFNet":
            model = NAFNet(
                img_channel=3,
                width=width,
                middle_blk_num=middle_blk_num,
                enc_blk_nums=enc_blks,
                dec_blk_nums=dec_blks,
                activation=SIMPLE_GATE,
                layer_norm_flag=False,
                channel_attention_flag=False
            )
        if model_name == "UNet":
            model = UNet(
                img_channel=3,
                width=width,
                middle_blk_num=middle_blk_num,
                enc_blk_nums=enc_blks,
                dec_blk_nums=dec_blks
            )
        model.to(device)
        with torch.no_grad():
            x = torch.randn(1, 3, 256, 256).to(device)
            y = model(x)

            # print(y.shape)
            # print(y)
            # print(model)
            print(f"{model.count_parameters()/1E3:.2f}k parameters")
        print(model.receptive_field(size=256 if tiny_recetive_field else 1024, device=device))
