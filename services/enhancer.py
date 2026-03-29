"""
Face Enhancement using GFPGAN.
Applied only when face resolution is low or blur is detected.

Patches torchvision.transforms.functional_tensor missing in newer torchvision.
"""

import numpy as np
import cv2
import logging
import os
import sys

logger = logging.getLogger(__name__)

MIN_FACE_SIZE_FOR_ENHANCEMENT = 80  # pixels
BLUR_VARIANCE_THRESHOLD = 100.0  # Laplacian variance

_enhancer_instance = None
_gfpgan_available = None  # None = not yet checked


def _patch_torchvision():
    """
    Patch missing torchvision.transforms.functional_tensor for newer torchvision.
    In torchvision >= 0.16 this module was removed; GFPGAN still imports it.
    """
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
    except ModuleNotFoundError:
        try:
            import torchvision.transforms.functional as F
            import types

            mod = types.ModuleType("torchvision.transforms.functional_tensor")
            # Copy over the attributes GFPGAN actually uses
            for attr in dir(F):
                setattr(mod, attr, getattr(F, attr))
            sys.modules["torchvision.transforms.functional_tensor"] = mod
        except Exception:
            pass  # will surface as GFPGAN unavailable below


def _is_blurry(face_img: np.ndarray) -> bool:
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_VARIANCE_THRESHOLD


def _needs_enhancement(face_img: np.ndarray) -> bool:
    h, w = face_img.shape[:2]
    if w < MIN_FACE_SIZE_FOR_ENHANCEMENT or h < MIN_FACE_SIZE_FOR_ENHANCEMENT:
        return True
    return _is_blurry(face_img)


def get_enhancer():
    """Singleton: load GFPGAN once (with torchvision patch applied first)."""
    global _enhancer_instance, _gfpgan_available

    if _gfpgan_available is False:
        return None
    if _enhancer_instance is not None:
        return _enhancer_instance

    _patch_torchvision()

    try:
        from gfpgan import GFPGANer

        model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "GFPGANv1.4.pth"
        )
        model_path = os.path.abspath(model_path)

        if not os.path.exists(model_path):
            logger.warning(
                "GFPGANv1.4.pth not found in models/. Enhancement disabled. "
                "Download from: https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
            )
            _gfpgan_available = False
            return None

        _enhancer_instance = GFPGANer(
            model_path=model_path,
            upscale=2,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
        _gfpgan_available = True
        logger.info("GFPGAN enhancer loaded.")
    except Exception as e:
        logger.warning(f"GFPGAN unavailable: {e}. Enhancement disabled.")
        _gfpgan_available = False

    return _enhancer_instance


def enhance_face(face_img: np.ndarray) -> np.ndarray:
    """
    Enhance face if quality is low. Returns original if enhancement
    is not needed or unavailable.
    """
    if not _needs_enhancement(face_img):
        return face_img

    enhancer = get_enhancer()
    if enhancer is None:
        return face_img

    try:
        _, _, enhanced_list = enhancer.enhance(
            face_img,
            has_aligned=True,
            only_center_face=True,
            paste_back=False,
        )
        if enhanced_list:
            result = enhanced_list[0]
            return cv2.resize(result, (face_img.shape[1], face_img.shape[0]))
    except Exception as e:
        logger.warning(f"Enhancement failed: {e}")

    return face_img
