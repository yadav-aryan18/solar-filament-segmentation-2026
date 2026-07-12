import cv2
import numpy as np

def get_disk_mask(image):
    """
    Separates the solar disk from the black space background using Otsu thresholding
    and circle fitting.
    """
    # 1. Otsu Thresholding to separate foreground (sun) from background (space)
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Find the largest contour to identify the solar disk
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(image)

    largest_contour = max(contours, key=cv2.contourArea)

    # 3. Fit a circle to the largest contour for a smooth mask
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)

    mask = np.zeros_like(image)
    cv2.circle(mask, (int(x), int(y)), int(radius), 255, -1)
    return mask

def apply_clahe(image):
    """
    Applies Contrast Limited Adaptive Histogram Equalization to amplify
    the subtle contrast between filaments and the solar disk.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

def preprocess_image(image):
    """
    Full preprocessing pipeline: Disk Masking -> Space Zeroing -> CLAHE.

    Args:
        image (np.ndarray): Grayscale image of size (2048, 2048).

    Returns:
        enhanced_image (np.ndarray): CLAHE enhanced image.
        disk_mask (np.ndarray): Binary mask of the solar disk.
    """
    # Get disk mask to isolate the sun from space
    disk_mask = get_disk_mask(image)

    # Mask out space (set all pixels outside the disk to 0)
    masked_image = cv2.bitwise_and(image, image, mask=disk_mask)

    # Apply CLAHE to amplify local contrast of filaments
    enhanced_image = apply_clahe(masked_image)

    return enhanced_image, disk_mask
