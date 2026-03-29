"""
Enrollment Pipeline — high-accuracy multi-embedding enrollment.

Strategy:
  For each student passport photo:
    1. Detect face + get landmarks
    2. Generate augmented variants on the ORIGINAL image (before alignment)
       so alignment is re-applied fresh to each variant → clean embeddings
    3. Align each variant → extract embedding → store
  Target: 20-30 embeddings per student
"""

import numpy as np
import cv2
import logging
import re
from pathlib import Path

from services.detector import detect_faces
from services.aligner import align_face
from services.enhancer import enhance_face
from services.embedder import extract_embedding
from database.faiss_db import get_database

logger = logging.getLogger(__name__)


# ── Augmentation on original image (before alignment) ────────────────────────


def _augment_image(img: np.ndarray) -> list[np.ndarray]:
    """
    Generate augmented variants of the full image.
    Augmentation is applied BEFORE alignment so each variant gets
    a fresh, clean alignment pass — avoiding distorted embeddings.
    """
    variants = []
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    # 1. Brightness: darker and brighter
    for alpha in [0.65, 0.80, 1.20, 1.40]:
        v = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        variants.append(v)

    # 2. Contrast adjustment (CLAHE on L channel)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l2 = clahe.apply(l)
    variants.append(cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR))

    # 3. Slight rotations (small angles keep face well-aligned)
    for angle in [-8, -4, 4, 8]:
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        variants.append(cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR))

    # 4. Gaussian blur (simulate motion/defocus blur in group photos)
    for ksize in [3, 5]:
        variants.append(cv2.GaussianBlur(img, (ksize, ksize), 0))

    # 5. Gaussian noise
    noise = np.random.normal(0, 8, img.shape).astype(np.float32)
    variants.append(np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8))

    # 6. Horizontal flip (some group photos may have mirrored angles)
    variants.append(cv2.flip(img, 1))

    # 7. Slight scale crop (simulate different distances)
    for scale in [0.90, 1.10]:
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        if scale < 1.0:
            # pad back to original size
            pad_x = (w - new_w) // 2
            pad_y = (h - new_h) // 2
            canvas = img.copy()
            canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
            variants.append(canvas)
        else:
            # centre-crop back to original size
            sx = (new_w - w) // 2
            sy = (new_h - h) // 2
            variants.append(resized[sy : sy + h, sx : sx + w])

    return variants


# ── Student ID parsing ────────────────────────────────────────────────────────


def parse_student_id(filename: str) -> str:
    """
    '1-Chandan B L - Photo.jpg'  → 'Chandan B L'
    '17 - Roshan - Photo.jpg'    → 'Roshan'
    '45 - BHOOMIKA.jpg'          → 'BHOOMIKA'
    """
    stem = Path(filename).stem
    stem = re.sub(r"[-\s]*[Pp][Hh][Oo][Tt][Oo][Tt]?\s*$", "", stem).strip()
    stem = re.sub(r"^\d+\s*[-–]\s*", "", stem).strip()
    return stem if stem else Path(filename).stem


# ── Core enrollment ───────────────────────────────────────────────────────────


def enroll_student(
    student_id: str,
    images: list[np.ndarray],
    augment: bool = True,
) -> dict:
    db = get_database()
    stored = 0
    failed = 0

    for img in images:
        faces = detect_faces(img)
        if not faces:
            logger.warning(f"[{student_id}] No face detected.")
            failed += 1
            continue

        face = max(faces, key=lambda f: f["det_score"])
        kps = face["kps"]

        # --- Original image embedding ---
        aligned = align_face(img, kps)
        enhanced = enhance_face(aligned)
        emb = extract_embedding(enhanced)
        db.add_embedding(student_id, emb)
        stored += 1

        if not augment:
            continue

        # --- Augmented variants ---
        for aug_img in _augment_image(img):
            aug_faces = detect_faces(aug_img)
            if not aug_faces:
                # Fall back: reuse original landmarks on augmented image
                aug_faces = [face]

            aug_face = max(aug_faces, key=lambda f: f["det_score"])
            try:
                aug_aligned = align_face(aug_img, aug_face["kps"])
            except Exception:
                aug_aligned = align_face(aug_img, kps)

            aug_enhanced = enhance_face(aug_aligned)
            aug_emb = extract_embedding(aug_enhanced)
            db.add_embedding(student_id, aug_emb)
            stored += 1

    db.save()
    logger.info(f"[{student_id}] enrolled: {stored} embeddings, {failed} failed.")
    return {
        "student_id": student_id,
        "embeddings_stored": stored,
        "failed_images": failed,
    }


def enroll_from_directory(students_dir: str, augment: bool = True) -> dict:
    students_dir = Path(students_dir)
    results = []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for file in sorted(students_dir.iterdir()):
        if file.suffix.lower() not in exts:
            continue
        student_id = parse_student_id(file.name)
        img = cv2.imread(str(file))
        if img is None:
            logger.error(f"Cannot read: {file}")
            continue
        result = enroll_student(student_id, [img], augment=augment)
        results.append(result)
        logger.info(f"  ✓ {student_id}: {result['embeddings_stored']} embeddings")

    total_students = len(results)
    total_embeddings = sum(r["embeddings_stored"] for r in results)
    logger.info(
        f"Enrollment done: {total_students} students, {total_embeddings} embeddings."
    )
    return {
        "total_students": total_students,
        "total_embeddings": total_embeddings,
        "details": results,
    }
