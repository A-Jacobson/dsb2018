import numpy as np
from PIL import Image


def combine_masks(masks):
    """
    :param masks: list of PIL images
    :return: combined PIL masks
    """
    combined_mask = np.zeros(np.array(masks[0]).shape, dtype='uint8')
    for mask in masks:
        combined_mask[np.array(mask) > 0] = 255
    return Image.fromarray(combined_mask)