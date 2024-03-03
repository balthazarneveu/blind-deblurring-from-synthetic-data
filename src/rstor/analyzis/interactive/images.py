from interactive_pipe.data_objects.image import Image
from typing import List


def image_selector(image_list: List[dict], image_index: int = 0) -> dict:
    current_image = image_list[image_index % len(image_list)]
    img = current_image.get("buffer", None)
    if img is None:
        img = Image.from_file(current_image["path"]).data
    return img
