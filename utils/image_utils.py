"""
Image utility helpers.
"""

import numpy as np
import cv2
from fastapi import UploadFile
import logging

logger = logging.getLogger(__name__)


async def decode_upload(upload: UploadFile) -> np.ndarray | None:
    """Decode an uploaded file to a BGR numpy array."""
    try:
        data = await upload.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Failed to decode upload {upload.filename}: {e}")
        return None


def read_image(path: str) -> np.ndarray | None:
    """Read image from disk."""
    img = cv2.imread(path)
    if img is None:
        logger.error(f"Cannot read image: {path}")
    return img


def resize_if_large(image: np.ndarray, max_dim: int = 1920) -> np.ndarray:
    """Downscale large images to speed up detection without losing faces."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
