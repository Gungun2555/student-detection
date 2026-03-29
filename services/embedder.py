"""
Face Embedding using ArcFace (buffalo_l).

Uses InsightFace's FaceAnalysis.get() end-to-end for enrollment (passport photos)
so the full InsightFace preprocessing pipeline is applied correctly.

For recognition of already-aligned 112x112 crops (from group photos), uses the
direct ONNX path with correct manual preprocessing.
"""

import numpy as np
import cv2
import logging
import os

logger = logging.getLogger(__name__)

_onnx_session = None  # for aligned 112x112 crops (recognition path)
_face_app = None  # for full-image enrollment path


# ── ONNX session (recognition path) ──────────────────────────────────────────


def _get_onnx_session():
    global _onnx_session
    if _onnx_session is None:
        import onnxruntime as ort

        model_path = os.path.expanduser(
            "~/.insightface/models/buffalo_l/w600k_r50.onnx"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ArcFace model not found: {model_path}\n"
                "Run enroll_students.py once to trigger the InsightFace download."
            )
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _onnx_session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        logger.info("ArcFace ONNX session loaded.")
    return _onnx_session


def _preprocess_aligned(face_112: np.ndarray) -> np.ndarray:
    """
    Correct ArcFace preprocessing for a 112x112 aligned BGR face:
      BGR → RGB → float32 → normalize [-1,1] → CHW → batch
    """
    img = cv2.cvtColor(face_112, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = (img - 127.5) / 127.5
    img = img.transpose(2, 0, 1)
    return np.expand_dims(img, 0)  # (1,3,112,112)


def extract_embedding(aligned_face: np.ndarray) -> np.ndarray:
    """
    Extract L2-normalised 512-d embedding from an aligned 112x112 BGR face.
    This is the main path used during recognition.
    """
    # Ensure 112x112
    if aligned_face.shape[:2] != (112, 112):
        aligned_face = cv2.resize(aligned_face, (112, 112))

    session = _get_onnx_session()
    blob = _preprocess_aligned(aligned_face)
    input_name = session.get_inputs()[0].name
    emb = session.run(None, {input_name: blob})[0].flatten()
    return _l2norm(emb)


def extract_embedding_from_full_image(image: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """
    Extract embedding by letting InsightFace handle alignment internally.
    Used during enrollment for maximum accuracy on clean passport photos.

    Args:
        image: full BGR image
        kps:   5-point landmarks from RetinaFace

    Returns:
        512-d L2-normalised embedding
    """
    from insightface.utils.face_align import norm_crop

    aligned = norm_crop(image, landmark=kps, image_size=112)
    return extract_embedding(aligned)


def _l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)


def extract_embeddings_batch(aligned_faces: list) -> np.ndarray:
    if not aligned_faces:
        return np.empty((0, 512), dtype=np.float32)
    return np.stack([extract_embedding(f) for f in aligned_faces], axis=0)
