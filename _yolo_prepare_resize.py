import os
import shutil
import cv2
import numpy as np
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from config import (
    RESIZE_SOURCE_ROOT as SOURCE_ROOT,
    RESIZE_DEST_ROOT as DEST_ROOT,
    RESIZE_TARGET_SIZE as TARGET_SIZE,
    VALID_IMAGE_EXTENSIONS,
    RESIZE_MAX_WORKERS as MAX_WORKERS,
)

EXTENSIONS = tuple(VALID_IMAGE_EXTENSIONS)

# ==========================================
# LOGGING
# ==========================================
os.makedirs("logs", exist_ok=True)
log_filename = os.path.join("logs", f"resize_{time.strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

def process_file(file_info):
    """Resizes image or copies label using OpenCV."""
    src_path, dest_path, is_image = file_info

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if os.path.exists(dest_path):
        return None

    if not is_image:
        try:
            shutil.copy2(src_path, dest_path)
            return None
        except Exception as e:
            return f"ERROR copying label {src_path}: {e}"

    try:
        img = cv2.imread(src_path)

        if img is None:
            return f"SKIPPED (Corrupt/Unreadable): {os.path.basename(src_path)}"

        h, w = img.shape[:2]
        longest_side = max(h, w)

        if longest_side > TARGET_SIZE:
            scale = TARGET_SIZE / float(longest_side)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(dest_path, resized_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            msg = f"Resized: {os.path.basename(src_path)} ({w}x{h} -> {new_w}x{new_h})"
            del resized_img
            del img
            return msg
        else:
            shutil.copy2(src_path, dest_path)
            del img
            return f"Copied (Small enough): {os.path.basename(src_path)}"

    except Exception as e:
        return f"ERROR processing {src_path}: {e}"

def main():
    if os.path.exists(DEST_ROOT):
        logger.warning("Destination folder '%s' already exists.", DEST_ROOT)

    tasks = []
    logger.info("Scanning '%s' for files...", SOURCE_ROOT)

    for root, dirs, files in os.walk(SOURCE_ROOT):
        if DEST_ROOT in root:
            continue

        for file in files:
            src_path = os.path.join(root, file)
            rel_path = os.path.relpath(src_path, SOURCE_ROOT)
            dest_path = os.path.join(DEST_ROOT, rel_path)

            if file.lower().endswith(EXTENSIONS):
                tasks.append((src_path, dest_path, True))
            elif file.lower().endswith('.txt') and 'labels' in root:
                tasks.append((src_path, dest_path, False))

    logger.info("Found %d files. Processing with OpenCV...", len(tasks))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for result in executor.map(process_file, tasks):
            if result and "ERROR" in result:
                logger.error(result)

    logger.info("Done!")

if __name__ == "__main__":
    main()
