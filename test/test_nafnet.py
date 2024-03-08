import torch
from rstor.architecture.nafnet import NAFNet


def test_nafnet():
    enc_blks = [1, 1]
    middle_blk_num = 1
    dec_blks = [1, 2]

    model = NAFNet(
        img_channel=3,
        width=2,
        middle_blk_num=middle_blk_num,
        enc_blk_nums=enc_blks,
        dec_blk_nums=dec_blks,
    )
    x = torch.rand(2, 3, 128, 128)
    y = model(x)
    assert y.shape == (2, 3, 128, 128)
