"""
Face Matching with adaptive thresholds and full debug output.
"""

import numpy as np
import cv2
import logging
from database.faiss_db import get_database, SEARCH_K

logger = logging.getLogger(__name__)

# Thresholds — tuned for ArcFace cosine similarity on real-world group photos
THRESHOLD_HIGH = 0.55  # sharp, well-lit, frontal
THRESHOLD_MEDIUM = 0.42  # slight blur / angle
THRESHOLD_LOW = 0.32  # heavy blur / small face / occlusion


def estimate_quality(face_img: np.ndarray) -> str:
    """Return 'high', 'medium', or 'low' based on blur + resolution."""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    h, w = face_img.shape[:2]
    area = h * w
    if blur > 300 and area >= 112 * 112:
        return "high"
    if blur > 80 or area >= 64 * 64:
        return "medium"
    return "low"


def get_threshold(quality: str) -> float:
    return {
        "high": THRESHOLD_HIGH,
        "medium": THRESHOLD_MEDIUM,
        "low": THRESHOLD_LOW,
    }.get(quality, THRESHOLD_MEDIUM)


def match_face(embedding: np.ndarray, face_img: np.ndarray = None) -> dict:
    """
    Match embedding against FAISS database.

    Returns dict with:
      student_id, confidence, quality, threshold_used, top_matches, emb_norm
    """
    db = get_database()
    quality = estimate_quality(face_img) if face_img is not None else "medium"
    threshold = get_threshold(quality)

    # Retrieve large k so voting has enough per-student hits
    hits = db.search(embedding, k=SEARCH_K)
    result = db.vote(hits, threshold=threshold)

    emb_norm = float(np.linalg.norm(embedding))

    # Debug log
    logger.debug(
        f"  → {result['student_id']:25s} conf={result['confidence']:.3f} "
        f"quality={quality} thr={threshold:.2f} norm={emb_norm:.3f}"
    )
    if result["top_matches"]:
        logger.debug(
            "     top5: "
            + "  ".join(f"{m['id']}={m['score']:.3f}" for m in result["top_matches"])
        )

    return {
        **result,
        "quality": quality,
        "threshold_used": threshold,
        "emb_norm": emb_norm,
    }
