"""
Standalone script: bulk enroll all students from data/students/ directory.
Run this ONCE before starting the API.

Usage:
    python enroll_students.py
    python enroll_students.py --no-augment
    python enroll_students.py --students-dir ../data/students
"""

import sys
import os
import argparse
import logging

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from utils.logger import setup_logging
from services.enrollment import enroll_from_directory

setup_logging("INFO")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Enroll students into face database.")
    parser.add_argument(
        "--students-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "students"),
        help="Path to directory containing student photos.",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable augmentation (faster but less robust).",
    )
    args = parser.parse_args()

    students_dir = os.path.abspath(args.students_dir)
    if not os.path.isdir(students_dir):
        logger.error(f"Students directory not found: {students_dir}")
        sys.exit(1)

    logger.info(f"Starting enrollment from: {students_dir}")
    logger.info(f"Augmentation: {'disabled' if args.no_augment else 'enabled'}")

    summary = enroll_from_directory(
        students_dir=students_dir,
        augment=not args.no_augment,
    )

    print("\n" + "=" * 50)
    print(f"Enrollment Complete")
    print(f"  Students enrolled : {summary['total_students']}")
    print(f"  Total embeddings  : {summary['total_embeddings']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
