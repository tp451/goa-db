"""
Shared configuration for the archival Japanese OCR pipeline.
All paths, thresholds, and model settings in one place.
"""
from pathlib import Path

# ==========================================
# DIRECTORY PATHS
# ==========================================
# Set True to feed the cropper + OCR from the upscaled/resized output of
# 0_optional_upscale.py (images_resize/). Default False uses the raw scans
# in images_original/ — eval_ocr.py showed upscaling does not improve OCR
# accuracy on this corpus, so the upscale step is now optional.
USE_UPSCALED = False
_DOWNSTREAM_IMAGE_DIR = "images_resize" if USE_UPSCALED else "images_original"

# Step 0 (optional): Upscale + resize
UPSCALE_INPUT_ROOT = "images_original"
UPSCALE_OUTPUT_ROOT = "images_esrgan"

# Step 0 (resize phase)
RESIZE_SOURCE_ROOT = "images_esrgan"
RESIZE_DEST_ROOT = "images_resize"

# Steps 1/3 (train_yolo.py): Training
SEG_YAML_PATH = "dataset_rows.yaml"
SEG_PROJECT_NAME = "yolo_rows"
SEG_MODEL_NAME = "yoloe-26x-seg.pt"

DET_YAML_PATH = "dataset_headers.yaml"
DET_PROJECT_NAME = "yolo_headers"
DET_MODEL_NAME = "yolo26x.pt"

# Step 2: Row cropping
CROPPER_MODEL_PATH = "runs/segment/yolo_rows/weights/best.pt"
CROPPER_INPUT_DIR = Path(_DOWNSTREAM_IMAGE_DIR)
CROPPER_OUTPUT_ROOT = Path("detected_rows")

# Step 4: Segmentation & OCR
OCR_MODEL_PATH = "runs/detect/yolo_headers/weights/best.pt"
OCR_IMAGES_ROOT = "detected_rows"
OCR_OUTPUT_ROOT = "ocr_results"
OCR_SOURCE_IMAGE_DIR = _DOWNSTREAM_IMAGE_DIR

# ==========================================
# IMAGE SETTINGS
# ==========================================
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Step 0 (upscale)
UPSCALE_FACTOR = 4
UPSCALE_JPEG_QUALITY = 90
UPSCALE_APPLY_CONTRAST = True
UPSCALE_USE_CUSTOM_MODEL = True
UPSCALE_CUSTOM_MODEL_PATH = "RealESRGAN_x4plus.pth"
UPSCALE_TILE_SIZE = 256
UPSCALE_TILE_PAD = 10
UPSCALE_MAX_OUTPUT_DIMENSION = 24000

# Step 0 (resize)
RESIZE_TARGET_SIZE = 4096
RESIZE_MAX_WORKERS = 4

# Step 2
CROPPER_CONF_THRESHOLD = 0.25
CROPPER_IMGSZ = 1280
CROPPER_RESIZE_PERCENTAGE = 100
CROPPER_JPG_QUALITY = 95
CROPPER_COLUMN_TOLERANCE_RATIO = 0.02  # multiplied by image width

# Step 4
OCR_SHIFT_OFFSET = 8
OCR_CONF_THRESHOLD = 0.5
OCR_MAX_RETRIES = 3
OCR_RETRY_BASE_DELAY = 3.0
OCR_BATCH_SIZE = 16

# OCR backend: "google" or "ollama"
OCR_BACKEND = "ollama"

# Ollama OCR settings (used when OCR_BACKEND = "ollama")
OCR_OLLAMA_BASE_URL = "http://localhost:11434"
OCR_OLLAMA_MODEL = "qwen3.5:9b"
OCR_OLLAMA_MAX_TOKENS = 2000

# ==========================================
# TRAINING SETTINGS
# ==========================================
TRAIN_EPOCHS = 300
TRAIN_PATIENCE = 50
TRAIN_BATCH_SIZE = 2
TRAIN_SEG_IMGSZ = 1280
TRAIN_DET_IMGSZ = 1280
TRAIN_DEVICE = 0
TRAIN_TEST_IMAGE_DIR = "images/test"

# ==========================================
# BIOGRAPHY EXTRACTION SETTINGS
# ==========================================
# Step 5: Biography extraction (local LLM)
BIO_INPUT_ROOT = "ocr_results"          # = OCR_OUTPUT_ROOT from step 4
BIO_OUTPUT_FILE = "biographies_extracted.jsonl"
VOLUME_ID = "1927"
BIO_OUTPUT_PATTERN = "biographies_extracted_{volume}.jsonl"
BIO_MODEL_NAME = "qwen3.5:9b"
BIO_OLLAMA_BASE_URL = "http://localhost:11434/v1"
BIO_MAX_TOKENS = 8000
DEDUP_SIMILARITY_THRESHOLD = 0.85  # skip consecutive rows with ≥85% identical OCR text

# ==========================================
# STRUCTURING SETTINGS
# ==========================================
# Step 6: Structure biographies into relational tables
STRUCT_INPUT_FILE = "biographies_extracted.jsonl"
STRUCT_OUTPUT_DIR = "structured"
STRUCT_OLLAMA_BATCH_SIZE = 50  # names per romanization request

# Validation year ranges
STRUCT_BIRTHYEAR_MIN = 1820
STRUCT_BIRTHYEAR_MAX = 1940
STRUCT_YEAR_MIN = 1850        # career start, graduation
STRUCT_YEAR_MAX = 1943
STRUCT_FAMILY_AGE_GAP = 15    # min years between parent/child birth
STRUCT_SPOUSE_AGE_GAP = 15    # max years between spouses

# Gender identification
STRUCT_GENDER_MODEL = "tarudesu/gendec-with-distilmbert"
STRUCT_GENDER_THRESHOLD = 0.97

# Geolocation
STRUCT_MCGD_PATH = "MCGD_Data2025_08_06.csv"
STRUCT_MCGD_CODES = {"A", "C", "P", "PP"}
STRUCT_GEONAMES_FILES = ["JP.txt", "TW.txt", "CN.txt", "KR.txt", "KP.txt"]
STRUCT_GEONAMES_CLASSES = {"A", "P"}
STRUCT_GEOJSON_DIR = "geojson"
STRUCT_GEOJSON_FILES = [
    ("japan_hijmans.geojson", "NAME_1"),
    ("china_1928.geojson", "province"),
    ("korea_imperial.geojson", "province"),
    ("taiwan_1946.geojson", "COUNTYENG"),
]

# Organization / job title translation
STRUCT_ORG_TRANSLATION_MAX_LEN = 50

# Organization location assignment (item 8)
STRUCT_ORG_LOCATION_MAX_SPREAD_KM = 200  # max km spread to assign org location

# Step 7: Cross-volume disambiguation
DISAMBIG_OUTPUT_DIR = "disambiguated"
DISAMBIG_ORG_FUZZY_THRESHOLD = 0.90  # Jaro-Winkler similarity threshold for org fuzzy matching
DISAMBIG_LLM_MAX_TOKENS = 50           # org verification (no thinking)
DISAMBIG_ORG_HIERARCHY_LLM_MAX_TOKENS = 50  # org hierarchy verification
DISAMBIG_ORG_HIERARCHY_SKIP_PARENTS = {
    # Geographic: traditional/archaic/historical forms missing from GeoNames
    "朝鮮", "臺灣", "台灣", "横濱", "九州", "奉天",
    "滿洲", "満洲",
    # Country names in Japanese kanji
    "米国", "米國", "獨逸", "英國", "英国", "佛國", "佛国",
    # Generic sector/descriptor terms (not actual organizations)
    "鐵道", "鉄道", "臨時", "農商務",
}

# ==========================================
# BATCHED UPSCALE→RESIZE PIPELINE
# ==========================================
PIPELINE_BATCH_SIZE = 20  # images per batch (0 = process all at once)

# ==========================================
# OCCUPATION & INDUSTRY CLASSIFICATION
# ==========================================
# Step 8: OccCANINE + ISIC classification
HISCO_CONFIDENCE_THRESHOLD = 0.22  # OccCANINE default best-F1 for English
ISIC_LLM_MAX_TOKENS = 50            # letter + label, e.g. "C - Manufacturing"
