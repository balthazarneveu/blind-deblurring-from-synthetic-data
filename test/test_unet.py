import torch
from rstor.architecture.nafnet import UNet
from rstor.properties import LEAKY_RELU


def test_unet():
    enc_blks = [1, 2]
    middle_blk_num = 2
    dec_blks = [2, 1]

    model = UNet(
        img_channel=3,
        width=2,
        activation=LEAKY_RELU,
        # We need leaky relu ...
        # otherwise it seems like ReLU may block propagation of NaN (with zeros!)
        # NaN and ReLu do not work correctly for receptive field estimation technique
        middle_blk_num=middle_blk_num,
        enc_blk_nums=enc_blks,
        dec_blk_nums=dec_blks,
    )
    rx, ry = model.receptive_field(channels=3)
    assert rx == ry
    assert rx == 44, "Receptive field should be {rx} x {ry}"
    x = torch.rand(2, 3, 128, 128)
    y = model(x)
    assert y.shape == (2, 3, 128, 128)
