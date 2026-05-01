"""
Unified YOLO training script — replaces 3_yolo_run_segmenting.py and 5_yolo_run_detection.py.

Usage:
    python train_yolo.py --task seg       # segmentation training
    python train_yolo.py --task detect    # detection training
"""
import os
import sys
import gc
import argparse
import logging
import time
import shutil
from pathlib import Path

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import torch
from ultralytics import YOLO, YOLOE
from ultralytics.models.yolo.yoloe.train_seg import YOLOEPESegTrainer
import matplotlib.pyplot as plt
import cv2

from config import (
    SEG_YAML_PATH, SEG_PROJECT_NAME, SEG_MODEL_NAME,
    DET_YAML_PATH, DET_PROJECT_NAME, DET_MODEL_NAME,
    TRAIN_EPOCHS, TRAIN_PATIENCE, TRAIN_BATCH_SIZE, TRAIN_DEVICE,
    TRAIN_SEG_IMGSZ, TRAIN_DET_IMGSZ, TRAIN_TEST_IMAGE_DIR,
    VALID_IMAGE_EXTENSIONS,
)

# ==========================================
# LOGGING
# ==========================================
os.makedirs("logs", exist_ok=True)
log_filename = os.path.join("logs", f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def get_task_config(task):
    if task == "seg":
        return {
            "model_name": SEG_MODEL_NAME,
            "yaml_path": SEG_YAML_PATH,
            "project_name": SEG_PROJECT_NAME,
            "img_size": TRAIN_SEG_IMGSZ,
            "is_seg": True,
        }
    elif task == "detect":
        return {
            "model_name": DET_MODEL_NAME,
            "yaml_path": DET_YAML_PATH,
            "project_name": DET_PROJECT_NAME,
            "img_size": TRAIN_DET_IMGSZ,
            "is_seg": False,
        }
    else:
        raise ValueError(f"Unknown task: {task}. Use 'seg' or 'detect'.")


_DETECT_SWAP_MARKER = ".detect_swap_active"

# Each directory pair: (active_name, det_source, seg_stash)
# Swap-in:  active_name → seg_stash,  det_source → active_name
# Swap-out: active_name → det_source, seg_stash  → active_name
_DIR_TRIPLES = [
    ("images", "images_headers", "images_rows"),
    ("labels", "labels_headers", "labels_rows"),
]


def _swap_dirs_for_detect(swap_in):
    """Swap images/labels dirs so YOLO's 'images'→'labels' resolution works for detect.

    YOLO resolves label paths by replacing '/images/' with '/labels/' in the
    image path.  Segmentation data lives in images/ + labels/ (works as-is).
    Detection data lives in images_headers/ + labels_headers/ (breaks).

    When swap_in=True  : seg dirs stashed to *_rows, detect dirs become images/labels.
    When swap_in=False : undo the above.

    Uses a marker file (.detect_swap_active) to survive interrupted runs.
    """
    already_swapped = os.path.isfile(_DETECT_SWAP_MARKER)

    if swap_in:
        if already_swapped:
            logger.info("Dirs already swapped (previous run interrupted), skipping swap-in")
            return
        for active, det_src, seg_stash in _DIR_TRIPLES:
            if os.path.isdir(active) and not os.path.isdir(seg_stash):
                os.rename(active, seg_stash)
            if os.path.isdir(det_src):
                os.rename(det_src, active)
        Path(_DETECT_SWAP_MARKER).touch()
        logger.info("Swapped dirs: *_headers → images/labels, seg stashed to *_rows")
    else:
        if not already_swapped:
            logger.info("Dirs already restored, skipping swap-out")
            return
        for active, det_src, seg_stash in _DIR_TRIPLES:
            if os.path.isdir(active) and not os.path.isdir(det_src):
                os.rename(active, det_src)
            if os.path.isdir(seg_stash) and not os.path.isdir(active):
                os.rename(seg_stash, active)
        Path(_DETECT_SWAP_MARKER).unlink(missing_ok=True)
        logger.info("Restored dirs: images/labels → *_headers, *_rows → images/labels")


def _remove_yolo_caches():
    """Delete YOLO .cache files under images/ so labels are re-scanned."""
    for cache in Path("images").rglob("*.cache"):
        cache.unlink()
        logger.info("Removed stale cache: %s", cache)


def main():
    parser = argparse.ArgumentParser(description="Unified YOLO training script")
    parser.add_argument("--task", required=True, choices=["seg", "detect"],
                        help="Training task: 'seg' for segmentation, 'detect' for detection")
    args = parser.parse_args()

    cfg = get_task_config(args.task)
    task_label = "SEGMENTATION" if cfg["is_seg"] else "DETECTION"
    is_detect = args.task == "detect"

    if is_detect:
        _swap_dirs_for_detect(swap_in=True)
        _remove_yolo_caches()

    try:
        _run_training(cfg, task_label)
    finally:
        if is_detect:
            _swap_dirs_for_detect(swap_in=False)


def _run_training(cfg, task_label):
    # ==========================================
    # TRAIN
    # ==========================================
    logger.info("Starting %s training (%s)", task_label, cfg["model_name"])

    train_kwargs = dict(
        data=cfg["yaml_path"],
        epochs=TRAIN_EPOCHS,
        patience=TRAIN_PATIENCE,
        batch=TRAIN_BATCH_SIZE,
        imgsz=cfg["img_size"],
        device=TRAIN_DEVICE,
        project=cfg["project_name"],
        name=".",
        exist_ok=True,
        save=True,
        plots=True,
    )

    if cfg["is_seg"]:
        # YOLOe segmentation models require the YOLOE class and PE-Seg trainer
        model = YOLOE(cfg["model_name"])
        train_kwargs["trainer"] = YOLOEPESegTrainer
    else:
        model = YOLO(cfg["model_name"])

    model.train(**train_kwargs)

    # Free training model before loading best weights
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ==========================================
    # VALIDATE (Split=Test)
    # ==========================================
    task_dir = "segment" if cfg["is_seg"] else "detect"
    best_weight_path = os.path.join("runs", task_dir, cfg["project_name"], "weights", "best.pt")

    if not os.path.exists(best_weight_path):
        logger.error("Could not find weights at %s. Check if training finished.", best_weight_path)
        return

    best_model = YOLO(best_weight_path)

    logger.info("Running validation on TEST split...")
    metrics = best_model.val(
        data=cfg["yaml_path"],
        split="test",
        imgsz=cfg["img_size"],
        batch=1,
        save_json=True,
    )

    logger.info("RESULTS FOR PUBLICATION (%s):", task_label)
    logger.info("Box mAP50:    %.4f", metrics.box.map50)
    logger.info("Box mAP50-95: %.4f", metrics.box.map)
    if cfg["is_seg"]:
        logger.info("Mask mAP50-95: %.4f", metrics.seg.map)

    # ==========================================
    # HIGH-RES VISUALIZATION
    # ==========================================
    test_images = sorted(
        p for p in Path(TRAIN_TEST_IMAGE_DIR).iterdir()
        if p.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ) if os.path.isdir(TRAIN_TEST_IMAGE_DIR) else []

    if test_images:
        test_image_path = str(test_images[0])
        logger.info("Predicting on %s ...", test_image_path)
        predict_kwargs = dict(imgsz=cfg["img_size"], conf=0.4, iou=0.6)
        if cfg["is_seg"]:
            predict_kwargs["retina_masks"] = True

        results = best_model.predict(test_image_path, **predict_kwargs)

        for result in results:
            plot_kwargs = dict(line_width=2, font_size=15)

            im_array = result.plot(**plot_kwargs)
            im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(15, 15))
            plt.imshow(im_rgb)
            plt.axis("off")

            suffix = "seg" if cfg["is_seg"] else "det"
            save_path = os.path.join("runs", task_dir, cfg["project_name"], f"publication_figure_{suffix}.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Saved figure to %s", save_path)
            plt.close()
    else:
        logger.warning("No test images found in %s", TRAIN_TEST_IMAGE_DIR)


if __name__ == "__main__":
    main()
