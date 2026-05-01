import os
import sys
import types
import traceback
import gc
import cv2
import numpy as np
import torch
import re
import time
import logging
from torch.nn import functional as F_torch
from PIL import Image

from config import (
    UPSCALE_INPUT_ROOT as INPUT_ROOT,
    UPSCALE_OUTPUT_ROOT as OUTPUT_ROOT,
    VALID_IMAGE_EXTENSIONS,
    UPSCALE_FACTOR as SCALE,
    UPSCALE_JPEG_QUALITY as JPEG_QUALITY,
    UPSCALE_APPLY_CONTRAST as APPLY_CONTRAST,
    UPSCALE_USE_CUSTOM_MODEL as USE_CUSTOM_MODEL,
    UPSCALE_CUSTOM_MODEL_PATH as CUSTOM_MODEL_PATH,
    UPSCALE_TILE_SIZE as TILE_SIZE,
    UPSCALE_TILE_PAD as TILE_PAD,
    UPSCALE_MAX_OUTPUT_DIMENSION as MAX_OUTPUT_DIMENSION,
)

VALID_EXTS = VALID_IMAGE_EXTENSIONS

# ==========================================
# LOGGING
# ==========================================
os.makedirs("logs", exist_ok=True)
log_filename = os.path.join("logs", f"upscale_{time.strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ==========================================
# MONKEY PATCH (D1: fragile — pin torchvision in requirements.txt instead)
# ==========================================
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    import torchvision.transforms.functional as F
    dummy = types.ModuleType("torchvision.transforms.functional_tensor")
    dummy.rgb_to_grayscale = F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = dummy

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# ==========================================
# KEY TRANSLATOR
# ==========================================
def translate_keys(state_dict):
    new_dict = {}
    logger.info("  Translating keys...")

    for k, v in state_dict.items():
        new_k = k
        if "model.0." in k: new_k = k.replace("model.0.", "conv_first.")
        elif "model.3." in k: new_k = k.replace("model.3.", "conv_up1.")
        elif "model.6." in k: new_k = k.replace("model.6.", "conv_up2.")
        elif "model.8." in k: new_k = k.replace("model.8.", "conv_hr.")
        elif "model.10." in k: new_k = k.replace("model.10.", "conv_last.")
        elif "model.1." in k:
            if "model.1.23" in k and "weight" in k and "conv" not in k and "RDB" not in k:
                 new_k = k.replace("model.1.23.", "conv_body.")

            sub_pattern = r"model\.1\.sub\.(\d+)\."
            if re.search(sub_pattern, k):
                new_k = re.sub(sub_pattern, r"body.\1.", k)
            else:
                std_pattern = r"model\.1\.(\d+)\."
                new_k = re.sub(std_pattern, r"body.\1.", k)

            if "body.23." in new_k: new_k = new_k.replace("body.23.", "conv_body.")

        new_k = new_k.replace("RDB", "rdb")
        new_k = re.sub(r"\.conv(\d+)\.0\.", r".conv\1.", new_k)
        new_dict[new_k] = v

    return new_dict

# ==========================================
# SAFE UPSAMPLER
# ==========================================
class SafeRealESRGANer(RealESRGANer):
    def __init__(self, scale, model, tile=0, tile_pad=10, pre_pad=10, half=True, device=None):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.half = half
        self.device = device
        self.model = model

        self.model.to(self.device)
        self.model.eval()
        if self.half and self.device.type == 'cuda':
            self.model = self.model.half()

def setup_upsampler(device):
    if USE_CUSTOM_MODEL:
        logger.info("Initializing Custom Model: %s", CUSTOM_MODEL_PATH)
        if not os.path.exists(CUSTOM_MODEL_PATH):
            raise FileNotFoundError(f"Missing file: {CUSTOM_MODEL_PATH}")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        path = CUSTOM_MODEL_PATH
    else:
        logger.info("Initializing Official Anime_6B...")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
        path = "RealESRGAN_x4plus_anime_6B.pth"
        if not os.path.exists(path):
            torch.hub.download_url_to_file(url, path)

    logger.info("  Loading weights from %s...", path)
    checkpoint = torch.load(path, map_location=device)

    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    elif 'params_ema' in checkpoint:
        state_dict = checkpoint['params_ema']
    else:
        state_dict = checkpoint

    if USE_CUSTOM_MODEL:
        state_dict = translate_keys(state_dict)

    try:
        model.load_state_dict(state_dict, strict=True)
        logger.info("  Weights loaded successfully (strict=True).")
    except RuntimeError as e:
        logger.error("Key mismatch detected!")
        logger.error("File keys: %s", list(state_dict.keys())[:5])
        logger.error("Expected keys: %s", list(model.state_dict().keys())[:5])
        raise e

    return SafeRealESRGANer(
        scale=SCALE, model=model, tile=TILE_SIZE, tile_pad=TILE_PAD,
        pre_pad=0, half=False, device=device,
    )

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def sanitize_and_read(path):
    try:
        with Image.open(path) as pil_img:
            pil_img = pil_img.convert('RGB')
            img = np.array(pil_img)
            img = img[:, :, ::-1].copy()
            return img
    except Exception:
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def process_image(img, upsampler):
    h, w = img.shape[:2]
    target_h, target_w = h * SCALE, w * SCALE

    if target_h > MAX_OUTPUT_DIMENSION or target_w > MAX_OUTPUT_DIMENSION:
        ratio = min(MAX_OUTPUT_DIMENSION / target_h, MAX_OUTPUT_DIMENSION / target_w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        logger.warning("Huge image. Resizing input to %dx%d.", new_w, new_h)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    output, _ = upsampler.enhance(img, outscale=SCALE)

    final_h, final_w = output.shape[0] // 2, output.shape[1] // 2
    logger.info("  Resizing from %dx%d to %dx%d for storage.",
                output.shape[1], output.shape[0], final_w, final_h)
    output = cv2.resize(output, (final_w, final_h), interpolation=cv2.INTER_LANCZOS4)
    return output

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")

    try:
        upsampler = setup_upsampler(device)
    except Exception as e:
        logger.critical("FATAL: %s", e)
        return

    tasks = []
    for root, _, files in os.walk(INPUT_ROOT):
        for f in files:
            if os.path.splitext(f)[1].lower() in VALID_EXTS:
                src = os.path.join(root, f)
                rel = os.path.relpath(src, INPUT_ROOT)
                dst = os.path.join(OUTPUT_ROOT, rel)
                tasks.append((src, dst))

    logger.info("Found %d images.", len(tasks))

    for i, (src, dst) in enumerate(tasks, 1):
        if os.path.exists(dst):
            logger.info("[%d/%d] SKIP: %s", i, len(tasks), dst)
            continue

        logger.info("[%d/%d] Processing: %s", i, len(tasks), src)

        try:
            img = sanitize_and_read(src)
            if img is None:
                continue

            with torch.no_grad():
                output = process_image(img, upsampler)

            if APPLY_CONTRAST:
                output = apply_clahe(output)

            os.makedirs(os.path.dirname(dst), exist_ok=True)
            cv2.imwrite(dst, output, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

        except Exception as e:
            logger.error("  Failed: %s", e)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    main()
