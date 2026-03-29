import numpy as np
import cv2
import logging
import os
import base64
from dataclasses import dataclass, field, asdict
from services.detector import detect_faces
from services.aligner import align_face, crop_face
from services.enhancer import enhance_face
from services.embedder import extract_embedding
from services.matcher import match_face

logger = logging.getLogger(__name__)


@dataclass
class FaceResult:
    bbox: list
    name: str
    confidence: float
    quality: str
    det_score: float
    top_matches: list = field(default_factory=list)
    emb_norm: float = 0.0
    embedding: list = None
    cropped_face_b64: str = None


def recognize_group_photo(image, return_embeddings=False, return_crops=False, debug_crop_dir=None, photo_name=""):
    faces = detect_faces(image)
    if not faces:
        logger.info("No faces detected.")
        return []
    if debug_crop_dir:
        os.makedirs(debug_crop_dir, exist_ok=True)
    results = []
    seen_ids = {}
    for i, face_data in enumerate(faces):
        bbox = face_data["bbox"]
        kps = face_data["kps"]
        det_score = face_data["det_score"]
        try:
            aligned = align_face(image, kps)
        except Exception as e:
            logger.warning("Alignment failed face %d: %s", i, e)
            aligned = crop_face(image, bbox)
        enhanced = enhance_face(aligned)
        embedding = extract_embedding(enhanced)
        emb_norm = float(np.linalg.norm(embedding))
        match = match_face(embedding, enhanced)
        name = match["student_id"]
        confidence = match["confidence"]
        quality = match["quality"]
        top_matches = match.get("top_matches", [])
        logger.info("  Face %02d | det=%.2f | quality=%s | norm=%.3f | -> %s (%.3f)", i+1, det_score, quality, emb_norm, name, confidence)
        if top_matches:
            logger.info("           top5: %s", "  ".join("%s=%.3f" % (m["id"], m["score"]) for m in top_matches))
        if name != "Unknown":
            seen_ids[name] = seen_ids.get(name, 0) + 1
            if seen_ids[name] > 1:
                name = "%s (#%d)" % (name, seen_ids[name])
        if debug_crop_dir:
            prefix = "%s_" % photo_name if photo_name else ""
            cv2.imwrite(os.path.join(debug_crop_dir, "%sface%02d_%s.jpg" % (prefix, i+1, name)), enhanced)
        result = FaceResult(bbox=bbox, name=name, confidence=round(confidence,4), quality=quality, det_score=round(det_score,4), top_matches=top_matches, emb_norm=round(emb_norm,4))
        if return_embeddings:
            result.embedding = embedding.tolist()
        if return_crops:
            _, buf = cv2.imencode(".jpg", enhanced)
            result.cropped_face_b64 = base64.b64encode(buf).decode("utf-8")
        results.append(asdict(result))
    identified = sum(1 for r in results if r["name"] != "Unknown")
    logger.info("  => %d faces | %d identified | %d unknown", len(results), identified, len(results)-identified)
    return results


def annotate_image(image, results):
    out = image.copy()
    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        name = r["name"]
        conf = r["confidence"]
        color = (0, 200, 0) if name != "Unknown" else (0, 0, 220)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = "%s %.2f" % (name, conf)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        label_y = y1 - 6 if y1 > th + 6 else y2 + th + 6
        cv2.rectangle(out, (x1, label_y-th-2), (x1+tw+2, label_y+2), color, -1)
        cv2.putText(out, label, (x1+1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    return out
