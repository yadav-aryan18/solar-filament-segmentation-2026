import numpy as np
from pycocotools import mask as coco_mask

def polygon_to_mask(segmentation, shape=(2048, 2048)):
    """
    Converts a COCO polygon segmentation to a binary mask.

    Args:
        segmentation (list): List of polygons [[x1, y1, x2, y2, ...]].
        shape (tuple): Target mask shape.

    Returns:
        mask (np.ndarray): Binary mask (0 or 1).
    """
    # pycocotools.mask.decode expects a list of RLEs, but we have polygons.
    # For polygons, the standard way is to use coco.annToMask.
    # Since we don't want to instantiate the whole COCO API, we use the underlying
    # mask.decode for RLE or simple cv2.fillPoly for polygons.

    import cv2
    mask = np.zeros(shape, dtype=np.uint8)
    for poly in segmentation:
        # COCO polygons are [x0, y0, x1, y1, ...]
        poly_np = np.array(poly).reshape((-1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [poly_np], 1)
    return mask

def mask_to_rle(mask):
    """
    Converts a binary mask to a COCO RLE string (counts only).

    Args:
        mask (np.ndarray): Binary mask (0 or 1).

    Returns:
        rle_string (str): The RLE encoded counts string.
    """
    # mask must be uint8
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # pycocotools.mask.encode returns a dictionary {'counts': ..., 'size': ...}
    rle = coco_mask.encode(np.asfortranarray(mask))
    return rle['counts']

def rle_to_mask(rle_string, shape=(2048, 2048)):
    """
    Converts a COCO RLE string back to a binary mask.

    Args:
        rle_string (str): RLE encoded counts string.
        shape (tuple): Target mask shape.

    Returns:
        mask (np.ndarray): Binary mask (0 or 1).
    """
    rle = {'counts': rle_string, 'size': shape}
    return coco_mask.decode(rle)
