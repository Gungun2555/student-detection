"""
Face Alignment using 5-point landmarks from RetinaFace.
Produces 112x112 aligned face crops (ArcFace standard).
"""

import numpy as np
import cv2
from insightface.utils.face_align import norm_crop

# ArcFace standard output size
ALIGN_SIZE = 112


def align_face(image: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """
    Align a face using 5-point landmarks.

    Args:
        image: BGR numpy array (full image)
        kps: 5x2 landmark array from RetinaFace

    Returns:
        112x112 aligned BGR face crop
    """
    aligned = norm_crop(image, landmark=kps, image_size=ALIGN_SIZE)
    return aligned


def crop_face(image: np.ndarray, bbox: list, padding: float = 0.1) -> np.ndarray:
    """
    Fallback: crop face by bbox with optional padding (no alignment).
    Used when landmarks are unavailable.
    """
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    crop = image[y1:y2, x1:x2]
    return cv2.resize(crop, (ALIGN_SIZE, ALIGN_SIZE))
