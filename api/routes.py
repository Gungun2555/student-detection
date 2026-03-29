"""
FastAPI routes for enrollment and recognition.
"""

import cv2
import numpy as np
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

from services.enrollment import enroll_student
from services.recognition import recognize_group_photo
from database.faiss_db import get_database
from utils.image_utils import decode_upload

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Enrollment ─────────────────────────────────────────────────────────────────


@router.post("/enroll")
async def enroll(
    student_id: str = Form(...),
    images: list[UploadFile] = File(...),
    augment: bool = Form(True),
):
    """
    Enroll a student with one or more passport-size photos.
    """
    imgs = []
    for upload in images:
        img = await decode_upload(upload)
        if img is None:
            raise HTTPException(400, f"Cannot decode image: {upload.filename}")
        imgs.append(img)

    result = enroll_student(student_id, imgs, augment=augment)
    return JSONResponse(content=result)


# ── Recognition ────────────────────────────────────────────────────────────────


@router.post("/recognize")
async def recognize(
    image: UploadFile = File(...),
    return_embeddings: bool = Form(False),
    return_crops: bool = Form(False),
):
    """
    Recognize all faces in a group photo.
    Returns list of {bbox, name, confidence, quality, det_score}.
    """
    img = await decode_upload(image)
    if img is None:
        raise HTTPException(400, "Cannot decode image.")

    results = recognize_group_photo(
        img,
        return_embeddings=return_embeddings,
        return_crops=return_crops,
    )
    return JSONResponse(content={"faces": results, "total": len(results)})


@router.post("/batch-recognize")
async def batch_recognize(
    images: list[UploadFile] = File(...),
    return_embeddings: bool = Form(False),
):
    """
    Recognize faces in multiple group photos.
    Returns results per image.
    """
    batch_results = []
    for upload in images:
        img = await decode_upload(upload)
        if img is None:
            batch_results.append(
                {"filename": upload.filename, "error": "decode failed"}
            )
            continue

        faces = recognize_group_photo(img, return_embeddings=return_embeddings)
        batch_results.append(
            {
                "filename": upload.filename,
                "faces": faces,
                "total": len(faces),
            }
        )

    return JSONResponse(content={"results": batch_results})


# ── Database info ──────────────────────────────────────────────────────────────


@router.get("/db/stats")
def db_stats():
    db = get_database()
    return {
        "students": db.get_student_count(),
        "total_embeddings": db.get_embedding_count(),
        "student_list": db.list_students(),
    }
