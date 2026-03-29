"""
FAISS vector database for student face embeddings.

Key design:
  - IndexFlatIP on L2-normalised vectors = cosine similarity
  - Search k=50 to get enough hits per student for robust voting
  - Per-student aggregation: weighted average (top-3 scores per student)
"""

import numpy as np
import faiss
import pickle
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "faiss_index.bin")
META_PATH = os.path.join(os.path.dirname(__file__), "metadata.pkl")
EMBEDDING_DIM = 512

# How many FAISS hits to retrieve before voting
SEARCH_K = 50


class FaceDatabase:
    def __init__(self):
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.id_map: list[str] = []
        self._counts: dict[str, int] = {}

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_embedding(self, student_id: str, embedding: np.ndarray):
        emb = embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(emb)
        self.index.add(emb)
        self.id_map.append(student_id)
        self._counts[student_id] = self._counts.get(student_id, 0) + 1

    def add_embeddings_batch(self, student_id: str, embeddings: np.ndarray):
        for emb in embeddings:
            self.add_embedding(student_id, emb)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query_embedding: np.ndarray, k: int = SEARCH_K) -> list[dict]:
        """Return top-k hits as list of {student_id, similarity, rank}."""
        if self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        q = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q)
        sims, idxs = self.index.search(q, k)
        return [
            {"student_id": self.id_map[idx], "similarity": float(sim), "rank": i}
            for i, (sim, idx) in enumerate(zip(sims[0], idxs[0]))
            if idx >= 0
        ]

    def vote(self, search_results: list[dict], threshold: float = 0.40) -> dict:
        """
        Aggregate top-k hits per student.

        Strategy:
          1. Group hits by student_id
          2. For each student take the mean of their TOP-3 similarity scores
             (top-3 is more robust than all hits which include noisy augmented ones)
          3. Pick student with highest aggregated score
          4. Return Unknown if below threshold

        Also returns top_matches list for debugging.
        """
        if not search_results:
            return {"student_id": "Unknown", "confidence": 0.0, "top_matches": []}

        per_student: dict[str, list[float]] = {}
        for r in search_results:
            per_student.setdefault(r["student_id"], []).append(r["similarity"])

        # Mean of top-3 scores per student
        agg = {
            sid: float(np.mean(sorted(scores, reverse=True)[:3]))
            for sid, scores in per_student.items()
        }

        # Top-5 students for debug output
        top_matches = sorted(
            [{"id": sid, "score": round(s, 4)} for sid, s in agg.items()],
            key=lambda x: x["score"],
            reverse=True,
        )[:5]

        best_id = top_matches[0]["id"]
        best_score = top_matches[0]["score"]

        if best_score < threshold:
            return {
                "student_id": "Unknown",
                "confidence": best_score,
                "top_matches": top_matches,
            }

        return {
            "student_id": best_id,
            "confidence": best_score,
            "top_matches": top_matches,
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, db_path: str = DB_PATH, meta_path: str = META_PATH):
        Path(os.path.dirname(db_path)).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, db_path)
        with open(meta_path, "wb") as f:
            pickle.dump({"id_map": self.id_map, "counts": self._counts}, f)
        logger.info(
            f"DB saved: {self.index.ntotal} embeddings, {len(self._counts)} students."
        )

    def load(self, db_path: str = DB_PATH, meta_path: str = META_PATH):
        if not os.path.exists(db_path) or not os.path.exists(meta_path):
            logger.warning("No existing database found. Starting fresh.")
            return
        self.index = faiss.read_index(db_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.id_map = meta["id_map"]
        self._counts = meta["counts"]
        logger.info(
            f"DB loaded: {self.index.ntotal} embeddings, {len(self._counts)} students."
        )

    # ── Info ──────────────────────────────────────────────────────────────────

    def get_student_count(self) -> int:
        return len(self._counts)

    def get_embedding_count(self) -> int:
        return self.index.ntotal

    def list_students(self) -> list[str]:
        return list(self._counts.keys())


_db_instance = None


def get_database() -> FaceDatabase:
    global _db_instance
    if _db_instance is None:
        _db_instance = FaceDatabase()
        _db_instance.load()
    return _db_instance
