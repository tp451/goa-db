"""
Step 0 (OPTIONAL): batched upscale→resize pipeline.

Side-by-side OCR comparison (eval_ocr.py) showed Real-ESRGAN upscaling does
not improve OCR accuracy on this corpus, so the main pipeline now reads from
images_original/ by default. Run this script only if you want to experiment
with the upscaled images, then set USE_UPSCALED = True in config.py.

Processes images in batches of PIPELINE_BATCH_SIZE so that images_esrgan/
never holds more than one batch at a time, saving disk space.

Resumable: re-running skips images that already exist in images_resize/.
"""

import os
import sys
import logging
import time
import gc

from config import (
    UPSCALE_INPUT_ROOT,
    UPSCALE_OUTPUT_ROOT,
    RESIZE_DEST_ROOT,
    VALID_IMAGE_EXTENSIONS,
    PIPELINE_BATCH_SIZE,
)

# ==========================================
# LOGGING
# ==========================================
os.makedirs("logs", exist_ok=True)
log_filename = os.path.join("logs", f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def find_pending_images():
    """Return list of relative paths from UPSCALE_INPUT_ROOT that have no
    corresponding file in RESIZE_DEST_ROOT yet."""
    pending = []
    for root, _, files in os.walk(UPSCALE_INPUT_ROOT):
        for f in files:
            if os.path.splitext(f)[1].lower() in VALID_IMAGE_EXTENSIONS:
                src = os.path.join(root, f)
                rel = os.path.relpath(src, UPSCALE_INPUT_ROOT)
                dest = os.path.join(RESIZE_DEST_ROOT, rel)
                if not os.path.exists(dest):
                    pending.append(rel)
    return sorted(pending)


def upscale_batch(batch_rel_paths, upsampler):
    """Upscale a batch of images using script 1's core functions."""
    import torch
    from importlib import import_module
    script1 = import_module("_upscale_batch")

    for rel in batch_rel_paths:
        src = os.path.join(UPSCALE_INPUT_ROOT, rel)
        dst = os.path.join(UPSCALE_OUTPUT_ROOT, rel)

        if os.path.exists(dst):
            logger.info("  Upscale SKIP (exists): %s", rel)
            continue

        logger.info("  Upscaling: %s", rel)
        try:
            img = script1.sanitize_and_read(src)
            if img is None:
                logger.warning("  Could not read: %s", rel)
                continue

            with torch.no_grad():
                output = script1.process_image(img, upsampler)

            if script1.APPLY_CONTRAST:
                output = script1.apply_clahe(output)

            os.makedirs(os.path.dirname(dst), exist_ok=True)
            import cv2
            cv2.imwrite(dst, output, [int(cv2.IMWRITE_JPEG_QUALITY), script1.JPEG_QUALITY])
            del output, img

        except Exception as e:
            logger.error("  Upscale failed for %s: %s", rel, e)
            gc.collect()


def resize_batch(batch_rel_paths):
    """Resize a batch of images using script 2's core function."""
    from importlib import import_module
    script2 = import_module("_yolo_prepare_resize")

    tasks = []
    for rel in batch_rel_paths:
        src = os.path.join(UPSCALE_OUTPUT_ROOT, rel)
        dst = os.path.join(RESIZE_DEST_ROOT, rel)
        tasks.append((src, dst, True))

    from concurrent.futures import ThreadPoolExecutor
    from config import RESIZE_MAX_WORKERS

    with ThreadPoolExecutor(max_workers=RESIZE_MAX_WORKERS) as executor:
        for result in executor.map(script2.process_file, tasks):
            if result and "ERROR" in result:
                logger.error("  %s", result)
            elif result:
                logger.info("  %s", result)


def delete_upscaled_batch(batch_rel_paths):
    """Delete upscaled files from images_esrgan/ after confirming resize exists."""
    deleted = 0
    for rel in batch_rel_paths:
        resized = os.path.join(RESIZE_DEST_ROOT, rel)
        upscaled = os.path.join(UPSCALE_OUTPUT_ROOT, rel)

        if os.path.exists(resized) and os.path.exists(upscaled):
            os.remove(upscaled)
            deleted += 1

            # Clean up empty parent directories
            parent = os.path.dirname(upscaled)
            while parent != UPSCALE_OUTPUT_ROOT:
                try:
                    os.rmdir(parent)  # only removes if empty
                    parent = os.path.dirname(parent)
                except OSError:
                    break

    logger.info("  Deleted %d upscaled files.", deleted)


def main():
    pending = find_pending_images()
    total = len(pending)

    if total == 0:
        logger.info("0 pending images. Nothing to do.")
        return

    batch_size = PIPELINE_BATCH_SIZE if PIPELINE_BATCH_SIZE > 0 else total
    num_batches = (total + batch_size - 1) // batch_size

    logger.info("%d pending images, batch size %d, %d batches.", total, batch_size, num_batches)

    # Set up upsampler once (expensive)
    import torch
    from importlib import import_module
    script1 = import_module("_upscale_batch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    upsampler = script1.setup_upsampler(device)

    processed = 0
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        batch = pending[start:start + batch_size]

        logger.info("=== Batch %d/%d (%d images) ===", batch_idx + 1, num_batches, len(batch))

        # 1. Upscale
        upscale_batch(batch, upsampler)

        # 2. Resize
        resize_batch(batch)

        # 3. Delete upscaled copies
        delete_upscaled_batch(batch)

        processed += len(batch)
        remaining = total - processed
        logger.info("Batch %d/%d done, %d images processed, %d remaining.",
                     batch_idx + 1, num_batches, processed, remaining)

    logger.info("Pipeline complete. %d images processed.", processed)


if __name__ == "__main__":
    main()
