"""
Face Detection using RetinaFace via InsightFace.
Returns bounding boxes + 5-point landmarks for alignment.
"""
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo
import logging

logger = logging.getLogger(__name__)

_detector_instance = None


def get_detector(ctx_id: int = 0, det_size: tuple = (640, 640)):
    """Singleton: load detector once."""
    global _detector_instance
    if _detector_instance is None:
        logger.info("Loading RetinaFace detector...")
        app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=ctx_id, det_size=det_size)
        _detector_instance = app
        logger.info("Detector loaded.")
    return _detector_instance


def detect_faces(image: np.ndarray, min_face_size: int = 20) -> list[dict]:
    """
    Detect all faces in an image.

    Args:
        image: BGR numpy array
        min_face_size: minimum face bounding box dimension to keep

    Returns:
        List of dicts with keys: bbox, kps, det_score
    """
    detector = get_detector()
    faces = detector.get(image)

    results = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        w, h = x2 - x1, y2 - y1
        if w < min_face_size or h < min_face_size:
            continue
        results.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "kps": face.kps,           # 5-point landmarks (np array)
            "det_score": float(face.det_score),
        })

    logger.debug(f"Detected {len(results)} faces.")
    return results
