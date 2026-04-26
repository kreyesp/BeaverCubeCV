import cv2
import numpy as np
import random
from config import SCALE_FACTOR, CAMERA_W, CAMERA_H


def rescale_to_iss_resolution(image: np.ndarray, scale: float = SCALE_FACTOR, is_label: bool = False) -> np.ndarray:
    """
    Upscale image to simulate ISS GSD vs Sentinel-2 GSD.
    Args:
        image:    numpy array (H, W) or (H, W, C)
        scale:    altitude ratio (default: 786/400 ≈ 1.965)
        is_label: if True, uses nearest-neighbour to preserve class values
    Returns:
        Upscaled array (~1006x1006 for 512x512 input)
    """
    original_h, original_w = image.shape[:2]
    new_h = int(original_h * scale)
    new_w = int(original_w * scale)
    interp = cv2.INTER_NEAREST if is_label else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def random_crop(image: np.ndarray, mask: np.ndarray, crop_h: int = CAMERA_H, crop_w: int = CAMERA_W):
    """
    Random crop to camera resolution.
    Args:
        image:  numpy array (H, W, C)
        mask:   numpy array (H, W)
        crop_h: target height (default: 480)
        crop_w: target width  (default: 752)
    Returns:
        Cropped image and mask
    """
    h, w = image.shape[:2]
    top  = random.randint(0, h - crop_h)
    left = random.randint(0, w - crop_w)
    image = image[top:top + crop_h, left:left + crop_w]
    mask  = mask[top:top + crop_h, left:left + crop_w]
    return image, mask