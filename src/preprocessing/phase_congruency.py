import cv2
import numpy as np


def destripe_pushbroom_raster(img_gray: np.ndarray) -> np.ndarray:
    """Suppresses vertical sensor line artifacts via median column profiling."""
    if img_gray is None:
        return None
    norm = img_gray.astype(np.float32)
    col_median = np.median(norm, axis=0, keepdims=True)
    global_median = np.median(norm)
    destriped = norm - (col_median - global_median)
    destriped = np.clip(destriped, 0, 255).astype(np.uint8)

    # 1D morphological closure to heal 1-pixel dead detector gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    return cv2.morphologyEx(destriped, cv2.MORPH_CLOSE, kernel)


def apply_clahe(img_gray: np.ndarray) -> np.ndarray:
    """Adaptive histogram equalization with bilateral edge preservation."""
    denoised = cv2.bilateralFilter(img_gray, d=5, sigmaColor=35, sigmaSpace=35)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(denoised)


def preprocess_lunar_frame(img_gray: np.ndarray) -> np.ndarray:
    """
    Destripes pushbroom lines -> Multi-frequency edge extraction.
    Isolates physical crater rims from changing shadow illumination.
    """
    if img_gray is None:
        return None
    clean_img = destripe_pushbroom_raster(img_gray)
    enhanced = apply_clahe(clean_img)

    # Multi-directional Sobel gradients to isolate crater crest energy
    grad_x = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    norm_mag = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Blend structural edge contours with enhanced regolith frame
    return cv2.addWeighted(enhanced, 0.60, norm_mag, 0.40, 0)