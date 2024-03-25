import cv2
import numpy as np
from interactive_pipe import interactive


def get_color_channel_offset(image):
    # size is defined in power of 2
    if len(image.shape) == 2:
        offset = 0
    elif len(image.shape) == 3:
        channel_guesser_max_size = 4
        if image.shape[0] <= channel_guesser_max_size:  # channel first C,H,W
            offset = 0
        elif image.shape[-1] <= channel_guesser_max_size:  # channel last or numpy H,W,C
            offset = 1
    else:
        raise NameError(f"Not supported shape {image.shape}")
    return offset


def crop_selector(image, center_x=0.5, center_y=0.5, size=9., global_params={}):
    offset = get_color_channel_offset(image)
    crop_size_pixels = int(2.**(size)/2.)
    h, w = image.shape[-2-offset], image.shape[-1-offset]
    ar = w/h
    half_crop_h, half_crop_w = crop_size_pixels, int(ar*crop_size_pixels)

    def round(val):
        return int(np.round(val))
    center_x_int = round(half_crop_w + center_x*(w-2*half_crop_w))
    center_y_int = round(half_crop_h + center_y*(h-2*half_crop_h))
    start_x = max(0, center_x_int-half_crop_w)
    start_y = max(0, center_y_int-half_crop_h)
    end_x = min(start_x+2*half_crop_w, w-1)
    end_y = min(start_y+2*half_crop_h, h-1)
    start_x = max(0, end_x-2*half_crop_w)
    start_y = max(0, end_y-2*half_crop_h)
    MAX_ALLOWED_SIZE = 512
    w_resize = int(min(MAX_ALLOWED_SIZE, w))
    h_resize = int(w_resize/w*h)
    h_resize = int(min(MAX_ALLOWED_SIZE, h_resize))
    w_resize = int(h_resize/h*w)
    global_params["crop"] = (start_x, start_y, end_x, end_y)
    global_params["resize"] = (w_resize, h_resize)
    return


def plug_crop_selector(num_pad: bool = False):
    interactive(
        center_x=(0.5, [0., 1.], "cx", ["4" if num_pad else "left", "6" if num_pad else "right"]),
        center_y=(0.5, [0., 1.], "cy", ["8" if num_pad else "up", "2" if num_pad else "down"]),
        size=(9., [6., 13., 0.3], "crop size", ["+", "-"])
    )(crop_selector)


def crop(*images, global_params={}):
    images_resized = []
    for image in images:
        offset = get_color_channel_offset(image)
        start_x, start_y, end_x, end_y = global_params["crop"]
        w_resize, h_resize = global_params["resize"]
        if offset == 0:
            crop = image[..., start_y:end_y, start_x:end_x]
        if offset == 1:
            crop = image[..., start_y:end_y, start_x:end_x, :]
        image_resized = cv2.resize(crop, (w_resize, h_resize), interpolation=cv2.INTER_NEAREST)
        images_resized.append(image_resized)
    return tuple(images_resized)


def rescale_thumbnail(image, global_params={}):
    resize_dim = max(global_params.get("resize", (512, 512)))
    return cv2.resize(image, (resize_dim, resize_dim), interpolation=cv2.INTER_NEAREST)
