import io
import json
import base64
import cv2
import numpy as np
import os
import time
import logging
import requests
from ultralytics import YOLO

from config import (
    OCR_MODEL_PATH as MODEL_PATH,
    OCR_IMAGES_ROOT as IMAGES_ROOT,
    OCR_OUTPUT_ROOT as OUTPUT_ROOT,
    OCR_SOURCE_IMAGE_DIR,
    VALID_IMAGE_EXTENSIONS as VALID_EXTENSIONS,
    OCR_SHIFT_OFFSET as SHIFT_OFFSET,
    OCR_CONF_THRESHOLD as CONF_THRESHOLD,
    OCR_MAX_RETRIES as MAX_RETRIES,
    OCR_RETRY_BASE_DELAY as RETRY_BASE_DELAY,
    OCR_BATCH_SIZE as BATCH_SIZE,
    CROPPER_RESIZE_PERCENTAGE as RESIZE_PERCENTAGE,
    OCR_BACKEND,
    OCR_OLLAMA_BASE_URL,
    OCR_OLLAMA_MODEL,
    OCR_OLLAMA_MAX_TOKENS,
)

# ==========================================
# 2. LOGGING SETUP
# ==========================================
log_dir = os.path.join(OUTPUT_ROOT, "logs")
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

error_log_path = os.path.join(log_dir, f"errors_{time.strftime('%Y%m%d_%H%M%S')}.log")
error_logger = logging.getLogger("errors")
error_handler = logging.FileHandler(error_log_path, encoding="utf-8")
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
error_logger.addHandler(error_handler)

# ==========================================
# 3. SETUP
# ==========================================
if OCR_BACKEND == "google":
    from google.cloud import vision
    try:
        client = vision.ImageAnnotatorClient()
        logger.info("Google Cloud Client active.")
    except Exception as e:
        logger.warning("Google Cloud Client failed: %s", e)
        client = None
else:
    vision = None
    client = None
    OLLAMA_CHAT_URL = f"{OCR_OLLAMA_BASE_URL}/api/chat"
    logger.info("Using Ollama backend: %s @ %s", OCR_OLLAMA_MODEL, OCR_OLLAMA_BASE_URL)

logger.info("Loading YOLO Model: %s", MODEL_PATH)
model = YOLO(MODEL_PATH)

# ==========================================
# 4. HELPER FUNCTIONS — Google Vision
# ==========================================
def perform_ocr_on_bytes(image_content, hints=None):
    """OCR a single image with retry logic and exponential backoff."""
    if client is None:
        return None
    image = vision.Image(content=image_content)
    context = vision.ImageContext(language_hints=hints) if hints else None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.document_text_detection(image=image, image_context=context)
            return response
        except Exception as e:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            if attempt < MAX_RETRIES:
                logger.warning("OCR attempt %d/%d failed: %s — retrying in %.1fs",
                               attempt, MAX_RETRIES, e, delay)
                time.sleep(delay)
            else:
                logger.error("OCR failed after %d attempts: %s", MAX_RETRIES, e)
                error_logger.error("OCR permanently failed: %s", e)
                return None


def batch_ocr(image_contents_list, hints=None):
    """OCR multiple images in a single batch API call (up to 16).
    Returns a list of responses in the same order as inputs."""
    if client is None:
        return [None] * len(image_contents_list)

    context = vision.ImageContext(language_hints=hints) if hints else None
    reqs = []
    for content in image_contents_list:
        image = vision.Image(content=content)
        req = vision.AnnotateImageRequest(
            image=image,
            features=[vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)],
        )
        if context:
            req.image_context = context
        reqs.append(req)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            batch_response = client.batch_annotate_images(requests=reqs)
            return batch_response.responses
        except Exception as e:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            if attempt < MAX_RETRIES:
                logger.warning("Batch OCR attempt %d/%d failed: %s — retrying in %.1fs",
                               attempt, MAX_RETRIES, e, delay)
                time.sleep(delay)
            else:
                logger.error("Batch OCR failed after %d attempts: %s", MAX_RETRIES, e)
                error_logger.error("Batch OCR permanently failed: %s", e)
                return [None] * len(image_contents_list)


def get_symbol_text(symbol):
    text = symbol.text
    break_type = symbol.property.detected_break.type
    if break_type == vision.TextAnnotation.DetectedBreak.BreakType.SPACE:
        text += ' '
    elif break_type in [vision.TextAnnotation.DetectedBreak.BreakType.EOL_SURE_SPACE,
                        vision.TextAnnotation.DetectedBreak.BreakType.LINE_BREAK]:
        text += '\n'
    return text


def symbol_bbox(symbol):
    """Return (x1, y1, x2, y2) for a symbol's bounding box."""
    verts = symbol.bounding_box.vertices
    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_overlap_area(a, b):
    """Compute overlap area between two (x1, y1, x2, y2) bboxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def symbol_area(bbox):
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


# ==========================================
# 4b. HELPER FUNCTIONS — Ollama
# ==========================================
def ollama_ocr_image(image_bytes):
    """OCR an image crop using a local Ollama VLM.
    Returns the recognized text as a string."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": OCR_OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a character-level OCR engine. Your ONLY job is to identify each printed character by its shape. "
                    "This is vertical Japanese text (tategaki): read each column top-to-bottom, columns right-to-left. "
                    "Output one column per line.\n"
                    "CRITICAL RULES:\n"
                    "- Recognise each character ONLY by its visual shape. NEVER substitute a character based on what would make grammatical or semantic sense.\n"
                    "- Keep original character forms exactly (眞 not 真, 區 not 区, 國 not 国, 實 not 実, 學 not 学, 會 not 会).\n"
                    "- Do NOT expand abbreviations (keep 明 not 明治, 大 not 大正).\n"
                    "- Do NOT add, remove, or reorder characters. Do NOT guess characters that are unclear — skip them.\n"
                    "- No commentary, no translation, no headers, no formatting."
                ),
            },
            {
                "role": "user",
                "content": "Read all the text in this image.",
                "images": [b64],
            },
        ],
        "stream": False,
        "think": False,
        "options": {"num_predict": OCR_OLLAMA_MAX_TOKENS, "temperature": 0, "top_k": 1},
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except Exception as e:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            if attempt < MAX_RETRIES:
                logger.warning("Ollama OCR attempt %d/%d failed: %s — retrying in %.1fs",
                               attempt, MAX_RETRIES, e, delay)
                time.sleep(delay)
            else:
                logger.error("Ollama OCR failed after %d attempts: %s", MAX_RETRIES, e)
                error_logger.error("Ollama OCR permanently failed: %s", e)
                return ""


# ==========================================
# 4c. SHARED HELPERS
# ==========================================
def validate_image(image_path):
    """Check that an image file can be loaded and has valid dimensions."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if h < 10 or w < 10:
        logger.warning("Image too small (%dx%d): %s", w, h, image_path)
        return None
    return img


# ==========================================
# 5. FULL-PAGE HEADER DETECTION
# ==========================================
def detect_headers_fullpage(work_img):
    """Run YOLO header detection on a full-page image.
    Returns list of [x1, y1, x2, y2] in work_img coordinates."""
    results = model.predict(work_img, conf=CONF_THRESHOLD, verbose=False)
    headers = []
    boxes = results[0].boxes.xyxy.cpu().numpy()
    h_img, w_img = work_img.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        headers.append([x1, y1, x2, y2])
    return headers


def map_headers_to_rows(headers, rows_meta):
    """Map detected headers to rows by checking that the header's center
    (both x and y) falls within the row's bounding box.
    Returns dict: row_index -> list of header bboxes."""
    row_headers = {i: [] for i in range(len(rows_meta))}
    for hbox in headers:
        h_cx = (hbox[0] + hbox[2]) / 2
        h_cy = (hbox[1] + hbox[3]) / 2
        for ri, row in enumerate(rows_meta):
            rx1, ry1, rx2, ry2 = row["bbox_xyxy"]
            if rx1 <= h_cx <= rx2 and ry1 <= h_cy <= ry2:
                row_headers[ri].append(hbox)
                break
    return row_headers


# ==========================================
# 6. ROW PROCESSING
# ==========================================
def build_segments(local_headers, width):
    """Build the segment list from local (row-relative) headers.
    Shared by both OCR backends."""
    segments = []

    def get_shifted_edge(bbox_val):
        return min(int(bbox_val + SHIFT_OFFSET), width)

    if local_headers:
        first_header_right_x = get_shifted_edge(local_headers[0]['bbox'][2])
        if first_header_right_x < width - 5:
            segments.append({
                "type": "orphan",
                "header_ocr": "Previous_Page_Content",
                "crop_index": -1,
                "bbox_header": [],
                "x_range": (first_header_right_x, width),
                "text_content": "",
            })

    if not local_headers:
        segments.append({
            "type": "full_row",
            "header_ocr": "None",
            "crop_index": 0,
            "bbox_header": [],
            "x_range": (0, width),
            "text_content": "",
        })
    else:
        for i, header in enumerate(local_headers):
            x_right = get_shifted_edge(header['bbox'][2])
            if i < len(local_headers) - 1:
                x_left = get_shifted_edge(local_headers[i + 1]['bbox'][2])
            else:
                x_left = 0
            segments.append({
                "type": "standard",
                "header_info": header,
                "crop_index": header['crop_index'],
                "header_ocr": "",
                "bbox_header": header['bbox'],
                "x_range": (x_left, x_right),
                "text_content": "",
            })

    return segments


def assign_symbols_from_vision(segments, ocr_response, height):
    """Assign OCR symbols from a Google Vision response to segments
    based on bounding-box overlap."""
    header_bboxes = []
    for seg_idx, seg in enumerate(segments):
        if seg["type"] == "standard" and seg.get("bbox_header"):
            hb = seg["bbox_header"]
            header_bboxes.append((hb[0], hb[1], hb[2], hb[3], seg_idx))

    full_text = ocr_response.full_text_annotation
    if full_text:
        for page in full_text.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        for symbol in word.symbols:
                            sb = symbol_bbox(symbol)
                            s_area = symbol_area(sb)
                            sym_text = get_symbol_text(symbol)

                            best_header_overlap = 0
                            best_header_idx = -1
                            for hx1, hy1, hx2, hy2, seg_idx in header_bboxes:
                                overlap = bbox_overlap_area(sb, (hx1, hy1, hx2, hy2))
                                if overlap > best_header_overlap:
                                    best_header_overlap = overlap
                                    best_header_idx = seg_idx

                            if s_area > 0 and best_header_overlap / s_area > 0.5:
                                segments[best_header_idx]['header_ocr'] += sym_text
                                continue

                            cx = (sb[0] + sb[2]) / 2
                            assigned = False
                            for seg in segments:
                                x_min, x_max = seg['x_range']
                                seg_bbox = (x_min, 0, x_max, height)
                                overlap = bbox_overlap_area(sb, seg_bbox)
                                if s_area > 0 and overlap / s_area > 0.5:
                                    seg['text_content'] += sym_text
                                    assigned = True
                                    break

                            if not assigned:
                                for seg in segments:
                                    x_min, x_max = seg['x_range']
                                    if x_min <= cx < x_max:
                                        seg['text_content'] += sym_text
                                        break

    for seg in segments:
        if seg["type"] == "standard":
            seg["header_ocr"] = seg["header_ocr"].strip()


def ocr_segments_with_ollama(segments, original_img):
    """OCR each segment by cropping the relevant region from the row image
    and sending it to Ollama. For standard segments, also crops the header
    region separately."""
    height = original_img.shape[0]

    for seg in segments:
        x_min, x_max = seg['x_range']
        x_min_i, x_max_i = int(x_min), int(x_max)
        if x_max_i <= x_min_i or (x_max_i - x_min_i) < 30:
            seg["text_content"] = ""
            continue

        # OCR the header region for standard segments
        if seg["type"] == "standard" and seg.get("bbox_header"):
            hb = seg["bbox_header"]
            hx1, hy1, hx2, hy2 = int(hb[0]), int(hb[1]), int(hb[2]), int(hb[3])
            hx1, hy1 = max(0, hx1), max(0, hy1)
            hx2, hy2 = min(original_img.shape[1], hx2), min(height, hy2)
            if hx2 > hx1 and hy2 > hy1:
                header_crop = original_img[hy1:hy2, hx1:hx2]
                _, header_buf = cv2.imencode('.jpg', header_crop)
                seg["header_ocr"] = ollama_ocr_image(header_buf.tobytes())
                logger.info("    Header OCR [seg %d]: %s",
                            seg['crop_index'], seg["header_ocr"][:40].replace('\n', ''))

        # OCR the body (full x_range column)
        body_crop = original_img[:, x_min_i:x_max_i]
        _, body_buf = cv2.imencode('.jpg', body_crop)
        seg["text_content"] = ollama_ocr_image(body_buf.tobytes())


def write_output(segments, original_img, image_path, page_dir_name):
    """Generate JSON output and debug visualization image."""
    height, width = original_img.shape[:2]
    output_json = []
    debug_img = original_img.copy()
    overlay = debug_img.copy()

    for seg in segments:
        x_min, x_max = seg['x_range']
        entry = {
            "type": seg.get("type", "standard"),
            "segment_index": seg['crop_index'],
            "header_ocr": seg['header_ocr'],
            "segment_x_range": [x_min, x_max],
            "body_text": seg['text_content'].strip(),
            "source_image": os.path.basename(image_path),
        }
        if seg.get("bbox_header"):
            entry["bbox_header"] = seg["bbox_header"]
        output_json.append(entry)

        color = (180, 180, 180) if seg.get("type") == "orphan" else np.random.randint(100, 255, 3).tolist()
        cv2.rectangle(overlay, (int(x_min), 0), (int(x_max), height), color, -1)
        cv2.line(debug_img, (int(x_min), 0), (int(x_min), height), (0, 0, 255), 2)

        if seg.get("type") == "standard":
            hx1, hy1, hx2, hy2 = map(int, seg['bbox_header'])
            cv2.rectangle(debug_img, (hx1, hy1), (hx2, hy2), (0, 0, 0), 2)
            cv2.putText(debug_img, str(seg['crop_index']), (int(x_min) + 10, int(hy2) + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.addWeighted(overlay, 0.25, debug_img, 0.75, 0, debug_img)

    output_dir = os.path.join(OUTPUT_ROOT, page_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(output_dir, base_name + "_segmented_output.json")
    img_path = os.path.join(output_dir, base_name + "_debug_segmentation.jpg")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)
    cv2.imwrite(img_path, debug_img)
    logger.info("  Saved: %s", os.path.basename(json_path))


def process_row(image_path, ocr_response, headers_fullpage, row_bbox, page_dir_name):
    """Process a single row image with pre-detected headers and OCR response."""
    original_img = validate_image(image_path)
    if original_img is None:
        logger.error("Could not read row image: %s", image_path)
        return

    height, width = original_img.shape[:2]
    rx1, ry1 = row_bbox[0], row_bbox[1]

    # Translate fullpage headers to row-local coordinates, sort right-to-left
    local_headers = []
    for hbox in sorted(headers_fullpage, key=lambda b: b[0], reverse=True):
        local_headers.append({
            'bbox': [hbox[0] - rx1, hbox[1] - ry1, hbox[2] - rx1, hbox[3] - ry1],
            'crop_index': len(local_headers),
        })

    segments = build_segments(local_headers, width)

    if OCR_BACKEND == "google":
        if ocr_response is None:
            logger.error("OCR response is None for: %s", image_path)
            error_logger.error("OCR returned None: %s", image_path)
            return
        assign_symbols_from_vision(segments, ocr_response, height)
    else:
        ocr_segments_with_ollama(segments, original_img)

    write_output(segments, original_img, image_path, page_dir_name)


# ==========================================
# 7. PAGE PROCESSING
# ==========================================
def process_page(page_dir_name):
    """Process all rows in a page directory by detecting headers on
    the full-page image and mapping them to individual row crops."""
    page_dir = os.path.join(IMAGES_ROOT, page_dir_name)

    # Load row metadata (saved by step 2)
    metadata_path = os.path.join(page_dir, "row_metadata.json")
    if not os.path.exists(metadata_path):
        logger.warning("No row_metadata.json for %s, skipping", page_dir_name)
        return

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Filter to row entries only (exclude fullpage crops)
    rows_meta = [r for r in metadata["rows"] if r["filename"].endswith("_row.jpg")]
    if not rows_meta:
        logger.info("No rows in %s, skipping", page_dir_name)
        return

    # Find pending rows (not yet processed)
    pending = []
    for ri, row in enumerate(rows_meta):
        base_name = os.path.splitext(row["filename"])[0]
        json_check = os.path.join(OUTPUT_ROOT, page_dir_name, base_name + "_segmented_output.json")
        if os.path.exists(json_check):
            continue
        row_path = os.path.join(page_dir, row["filename"])
        if not os.path.exists(row_path):
            logger.warning("Row image not found: %s", row_path)
            continue
        pending.append((ri, row, row_path))

    if not pending:
        logger.info("Skipping %s (all rows done)", page_dir_name)
        return

    logger.info("Processing page %s (%d pending rows)", page_dir_name, len(pending))

    # Load full-page image and downsize to match step 2 coordinate space
    source_img_path = os.path.join(OCR_SOURCE_IMAGE_DIR, metadata["source_image"])
    full_img = cv2.imread(source_img_path)
    if full_img is None:
        logger.error("Could not load source image: %s", source_img_path)
        return

    if RESIZE_PERCENTAGE < 100:
        scale = RESIZE_PERCENTAGE / 100.0
        h, w = full_img.shape[:2]
        work_img = cv2.resize(full_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        del full_img
    else:
        work_img = full_img

    # Detect headers on full page
    headers = detect_headers_fullpage(work_img)
    logger.info("  Detected %d headers on full page", len(headers))
    del work_img

    # Map headers to rows by y-coordinate overlap
    row_headers = map_headers_to_rows(headers, rows_meta)

    if OCR_BACKEND == "google":
        # Encode pending row images for batch OCR
        encoded_rows = []
        for ri, row, row_path in pending:
            img = cv2.imread(row_path)
            if img is None:
                logger.error("Could not read row image: %s", row_path)
                continue
            success, encoded = cv2.imencode('.jpg', img)
            if success:
                encoded_rows.append((ri, row, row_path, encoded.tobytes()))

        if not encoded_rows:
            return

        # Process in OCR batches
        for batch_start in range(0, len(encoded_rows), BATCH_SIZE):
            batch = encoded_rows[batch_start:batch_start + BATCH_SIZE]
            contents = [e[3] for e in batch]

            logger.info("  Sending batch of %d row images to Google Vision...", len(batch))
            responses = batch_ocr(contents, hints=["ja"])

            for (ri, row, row_path, _), ocr_resp in zip(batch, responses):
                process_row(row_path, ocr_resp, row_headers.get(ri, []),
                            row["bbox_xyxy"], page_dir_name)
    else:
        # Ollama: process each row individually (no batching)
        for ri, row, row_path in pending:
            logger.info("  OCR row %s via Ollama...", row["filename"])
            process_row(row_path, None, row_headers.get(ri, []),
                        row["bbox_xyxy"], page_dir_name)


# ==========================================
# 8. MAIN LOOP
# ==========================================
def main():
    logger.info("Scanning '%s'... (backend: %s)", IMAGES_ROOT, OCR_BACKEND)

    page_dirs = sorted([
        d for d in os.listdir(IMAGES_ROOT)
        if os.path.isdir(os.path.join(IMAGES_ROOT, d))
    ])

    logger.info("Found %d page directories.", len(page_dirs))

    for page_dir_name in page_dirs:
        process_page(page_dir_name)

    logger.info("Done.")


if __name__ == "__main__":
    main()
