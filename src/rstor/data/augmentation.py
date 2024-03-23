import torch
from typing import Tuple, Optional


def augment_flip(
    img: torch.Tensor,
    flip: Optional[Tuple[bool, bool]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Roll pixels horizontally to avoid negative index

    Args:
        img (torch.Tensor): [N, 3, H, W] image tensor
        lab (torch.Tensor): [N, 3, H, W] label tensor
        flip (Optional[bool], optional): forced flip_h, flip_v value. Defaults to None.
        If not provided, a random flip_h, flip_v values are used
    Returns:
        torch.Tensor, torch.Tensor: flipped image, labels

    """
    if flip is None:
        flip = torch.randint(0, 2, (2,))
    flipped_img = img
    if flip[0] > 0:
        flipped_img = torch.flip(flipped_img, (-1,))
    if flip[1] > 0:
        flipped_img = torch.flip(flipped_img, (-2,))
    return flipped_img
