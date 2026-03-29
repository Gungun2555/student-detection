"""
Entry point: FastAPI application.
"""

import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

from utils.logger import setup_logging
from api.routes import router
from database.faiss_db import get_database

setup_logging("INFO")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load database on startup
    get_database()
    yield


app = FastAPI(
    title="Student Face Recognition API",
    version="1.0.0",
    description="Production-grade face recognition using RetinaFace + ArcFace + FAISS",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
