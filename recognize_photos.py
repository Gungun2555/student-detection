"""
Standalone script: run recognition on all group photos in data/.

Usage:
    python recognize_photos.py
    python recognize_photos.py --save-json
    python recognize_photos.py --debug-crops        # saves face chips to output/debug_crops/
    python recognize_photos.py --photos-dir ../data --output-dir ./output
"""

import sys
import os
import argparse
import json
import logging
import cv2
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from utils.logger import setup_logging
from utils.image_utils import read_image, resize_if_large
from services.recognition import recognize_group_photo, annotate_image
from database.faiss_db import get_database

setup_logging("INFO")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--photos-dir", default=os.path.join(os.path.dirname(__file__), "..", "data")
    )
    parser.add_argument(
        "--output-dir", default=os.path.join(os.path.dirname(__file__), "output")
    )
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument(
        "--debug-crops",
        action="store_true",
        help="Save individual face chips to output/debug_crops/ for inspection.",
    )
    parser.add_argument("--return-embeddings", action="store_true")
    args = parser.parse_args()

    photos_dir = Path(os.path.abspath(args.photos_dir))
    output_dir = Path(os.path.abspath(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_crop_dir = str(output_dir / "debug_crops") if args.debug_crops else None

    db = get_database()
    if db.get_student_count() == 0:
        logger.error("Database is empty. Run enroll_students.py first.")
        sys.exit(1)

    logger.info(
        f"DB: {db.get_student_count()} students | {db.get_embedding_count()} embeddings"
    )

    photo_files = sorted(
        [
            f
            for f in photos_dir.iterdir()
            if f.is_file()
            and f.name.upper().startswith("PHOTO")
            and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )

    if not photo_files:
        logger.error(f"No group photos found in {photos_dir}")
        sys.exit(1)

    logger.info(f"Processing {len(photo_files)} group photos...\n")
    all_results = {}

    for photo_path in photo_files:
        logger.info(f"{'='*60}")
        logger.info(f"Photo: {photo_path.name}")
        img = read_image(str(photo_path))
        if img is None:
            continue

        img = resize_if_large(img, max_dim=1920)

        results = recognize_group_photo(
            img,
            return_embeddings=args.return_embeddings,
            return_crops=False,
            debug_crop_dir=debug_crop_dir,
            photo_name=photo_path.stem,
        )

        identified = [r for r in results if r["name"] != "Unknown"]
        unknown = [r for r in results if r["name"] == "Unknown"]

        logger.info(
            f"  SUMMARY: {len(results)} faces | "
            f"{len(identified)} identified | {len(unknown)} unknown"
        )
        for r in identified:
            logger.info(
                f"    [OK] {r['name']:30s} conf={r['confidence']:.3f}  "
                f"quality={r['quality']}  norm={r['emb_norm']:.3f}"
            )
        for r in unknown:
            logger.info(
                f"    [??] Unknown  conf={r['confidence']:.3f}  "
                f"quality={r['quality']}  norm={r['emb_norm']:.3f}  "
                f"top={r['top_matches'][:2] if r['top_matches'] else []}"
            )

        annotated = annotate_image(img, results)
        out_path = output_dir / f"annotated_{photo_path.name}"
        cv2.imwrite(str(out_path), annotated)

        all_results[photo_path.name] = results

    if args.save_json:
        json_path = output_dir / "recognition_results.json"
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"JSON saved: {json_path}")

    print(f"\nDone. Annotated images -> {output_dir}")
    if debug_crop_dir:
        print(f"Face chips          -> {debug_crop_dir}")


if __name__ == "__main__":
    main()
