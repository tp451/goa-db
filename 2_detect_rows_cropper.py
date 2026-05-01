import os
import cv2
import json
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import torch
import gc
import logging
import time

from config import (
    CROPPER_MODEL_PATH as MODEL_PATH,
    CROPPER_INPUT_DIR as INPUT_DIR,
    CROPPER_OUTPUT_ROOT as OUTPUT_ROOT,
    CROPPER_CONF_THRESHOLD as CONF_THRESHOLD,
    CROPPER_IMGSZ as IMGSZ,
    CROPPER_RESIZE_PERCENTAGE as RESIZE_PERCENTAGE,
    CROPPER_JPG_QUALITY as JPG_QUALITY,
    VALID_IMAGE_EXTENSIONS,
)

Image.MAX_IMAGE_PIXELS = None

EXTENSIONS = VALID_IMAGE_EXTENSIONS

# ==========================================
# LOGGING
# ==========================================
os.makedirs("logs", exist_ok=True)
log_filename = os.path.join("logs", f"cropper_{time.strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def sort_rows_top_to_bottom(rows):
    """Sort row detections top-to-bottom by vertical center."""
    return sorted(rows, key=lambda d: (d['box_xyxy'][1] + d['box_xyxy'][3]) / 2)


def validate_image(img_path):
    """Validate that an image can be loaded and has reasonable dimensions (B3)."""
    img = cv2.imread(str(img_path))
    if img is None:
        logger.error("Could not load %s", img_path.name)
        return None
    h, w = img.shape[:2]
    if h < 10 or w < 10:
        logger.warning("Image too small (%dx%d): %s", w, h, img_path.name)
        return None
    return img


def process_single_image(model, img_path, output_root):
    image_name = img_path.stem
    save_dir = output_root / image_name

    if save_dir.exists() and any(save_dir.iterdir()):
        logger.info("Skipping %s (already done)", image_name)
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Processing: %s", img_path.name)

    # 1. Load & validate
    original_img = validate_image(img_path)
    if original_img is None:
        return

    # 2. Downsize by percentage
    h, w = original_img.shape[:2]
    work_img = original_img

    if RESIZE_PERCENTAGE < 100:
        scale = RESIZE_PERCENTAGE / 100.0
        new_w = int(w * scale)
        new_h = int(h * scale)
        work_img = cv2.resize(original_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info("  Downsized %d%%: %dx%d -> %dx%d", RESIZE_PERCENTAGE, w, h, new_w, new_h)
        del original_img
    else:
        logger.info("  Processing at original size: %dx%d", w, h)

    img_h, img_w = work_img.shape[:2]

    # 3. Inference
    try:
        with torch.no_grad():
            results = model.predict(
                source=work_img, imgsz=IMGSZ, conf=CONF_THRESHOLD,
                retina_masks=True, verbose=False, device=0,
            )
    except Exception as e:
        logger.error("  Prediction failed: %s", e)
        return

    # 4. Separate fullpage and row detections
    fullpages = []
    rows = []

    for result in results:
        if result.masks is None:
            continue

        polygons = result.masks.xy
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        class_names = result.names

        for i, polygon in enumerate(polygons):
            if len(polygon) == 0:
                continue

            box = boxes[i]
            class_name = class_names[class_ids[i]]

            item = {
                'polygon': polygon,
                'box_xyxy': box,
                'class_name': class_name,
            }

            if class_name == "fullpage":
                fullpages.append(item)
            elif class_name == "row":
                rows.append(item)

    # 5. Assign rows to pages using fullpage detections
    if len(fullpages) >= 2:
        # Sort fullpages right-to-left (Japanese reading order)
        fullpages.sort(key=lambda d: (d['box_xyxy'][0] + d['box_xyxy'][2]) / 2, reverse=True)
        page_centers = [(d['box_xyxy'][0] + d['box_xyxy'][2]) / 2 for d in fullpages[:2]]
        right_page, left_page = [], []
        for row in rows:
            row_cx = (row['box_xyxy'][0] + row['box_xyxy'][2]) / 2
            if abs(row_cx - page_centers[0]) <= abs(row_cx - page_centers[1]):
                right_page.append(row)
            else:
                left_page.append(row)
        logger.info("  Two pages detected via fullpage segments")
    elif len(fullpages) == 1:
        right_page = rows
        left_page = []
        logger.info("  Single page detected via fullpage segment")
    else:
        # Fallback: aspect-ratio heuristic
        aspect_ratio = img_w / img_h
        if aspect_ratio > 1.4:
            page_split_x = img_w // 2
            logger.info("  Two-page spread detected via fallback heuristic (aspect ratio %.2f)", aspect_ratio)
            right_page, left_page = [], []
            for row in rows:
                row_cx = (row['box_xyxy'][0] + row['box_xyxy'][2]) / 2
                if row_cx > page_split_x:
                    right_page.append(row)
                else:
                    left_page.append(row)
        else:
            right_page = rows
            left_page = []
            logger.info("  Single page detected via fallback heuristic (aspect ratio %.2f)", aspect_ratio)

    final_sorted_list = sort_rows_top_to_bottom(right_page) + sort_rows_top_to_bottom(left_page)

    # 5b. Save row metadata for step 4 (header-to-row mapping)
    row_metadata = {
        "source_image": img_path.name,
        "resize_scale": RESIZE_PERCENTAGE / 100.0,
        "rows": [],
    }
    for i, det in enumerate(final_sorted_list):
        poly_int = det['polygon'].astype(np.int32)
        x, y, w_b, h_b = cv2.boundingRect(poly_int)
        x, y = max(0, x), max(0, y)
        w_b, h_b = min(w_b, img_w - x), min(h_b, img_h - y)
        row_metadata["rows"].append({
            "index": i,
            "filename": f"{i:03d}_{det['class_name']}.jpg",
            "bbox_xyxy": [x, y, x + w_b, y + h_b],
        })
    with open(save_dir / "row_metadata.json", 'w') as f:
        json.dump(row_metadata, f, indent=2)

    # 6. Generate Visual Index
    vis_img = work_img.copy()
    font_scale = max(1, img_w // 1000)
    thickness = max(1, img_w // 800)

    for i, det in enumerate(final_sorted_list):
        poly_int = det['polygon'].astype(np.int32)
        x, y, w_box, h_box = cv2.boundingRect(poly_int)
        cv2.polylines(vis_img, [poly_int], isClosed=True, color=(0, 255, 0), thickness=thickness)
        cx, cy = int(x + w_box / 2), int(y + h_box / 2)
        text = str(i)
        cv2.putText(vis_img, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness + 2)
        cv2.putText(vis_img, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

    vis_path = save_dir / "00_visual_index.jpg"
    cv2.imwrite(str(vis_path), vis_img, [cv2.IMWRITE_JPEG_QUALITY, 70])

    # 7. Save JPG Crops (White Background)
    count = 0
    for i, det in enumerate(final_sorted_list):
        polygon = det['polygon']
        class_name = det['class_name']

        poly_int = polygon.astype(np.int32)
        x, y, w, h = cv2.boundingRect(poly_int)
        x, y = max(0, x), max(0, y)
        w, h = min(w, img_w - x), min(h, img_h - y)

        crop_img = work_img[y:y + h, x:x + w]
        if crop_img.size == 0:
            continue

        mask = np.zeros(crop_img.shape[:2], dtype=np.uint8)
        poly_relative = poly_int - [x, y]
        cv2.fillPoly(mask, [poly_relative], 255)

        fg = cv2.bitwise_and(crop_img, crop_img, mask=mask)
        bg = np.ones_like(crop_img, dtype=np.uint8) * 255
        bg = cv2.bitwise_and(bg, bg, mask=cv2.bitwise_not(mask))
        final_jpg = cv2.add(fg, bg)

        filename = save_dir / f"{i:03d}_{class_name}.jpg"
        cv2.imwrite(str(filename), final_jpg, [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])
        count += 1

    logger.info("  Saved %d JPG crops.", count)

    del results, work_img, vis_img
    gc.collect()
    torch.cuda.empty_cache()


def main():
    logger.info("Loading model: %s ...", MODEL_PATH)
    model = YOLO(MODEL_PATH)

    if not INPUT_DIR.exists():
        logger.error("Input directory does not exist: %s", INPUT_DIR)
        return

    images = [p for p in INPUT_DIR.glob('*') if p.suffix.lower() in EXTENSIONS]
    logger.info("Found %d images.", len(images))

    for img_path in images:
        process_single_image(model, img_path, OUTPUT_ROOT)

if __name__ == "__main__":
    main()
