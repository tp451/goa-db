"""
OCR Evaluation: compare 4 OCR configurations + manual gold standard.

Usage:
    python eval_ocr.py setup [--n 8] [--seed 42]
    python eval_ocr.py run   [--backend ollama|google|all]
    python eval_ocr.py compare

Configurations compared:
    1. Ollama (qwen3.5) on rows cropped from images_resize/
    2. Ollama (qwen3.5) on rows cropped from images_original/
    3. Google Cloud Vision on rows cropped from images_resize/
    4. Google Cloud Vision on rows cropped from images_original/
    5. Gold standard — manually proofread (user fills in gold.json)
"""
import argparse
import base64
import csv
import json
import os
import random
import shutil
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import requests

from config import (
    OCR_OLLAMA_BASE_URL,
    OCR_OLLAMA_MODEL,
    OCR_OLLAMA_MAX_TOKENS,
    OCR_MAX_RETRIES,
    OCR_RETRY_BASE_DELAY,
)

EVAL_DIR = Path("ocr_eval")
DETECTED_ROWS = Path("detected_rows")
OCR_RESULTS = Path("ocr_results")
IMAGES_RESIZE = Path("images_resize")
IMAGES_ORIGINAL = Path("images_original")

OLLAMA_CHAT_URL = f"{OCR_OLLAMA_BASE_URL}/api/chat"

OLLAMA_SYSTEM_PROMPT = (
    "You are a character-level OCR engine. Your ONLY job is to identify each printed character by its shape. "
    "This is vertical Japanese text (tategaki): read each column top-to-bottom, columns right-to-left. "
    "Output one column per line.\n"
    "CRITICAL RULES:\n"
    "- Recognise each character ONLY by its visual shape. NEVER substitute a character based on what would make grammatical or semantic sense.\n"
    "- Keep original character forms exactly (眞 not 真, 區 not 区, 國 not 国, 實 not 実, 學 not 学, 會 not 会).\n"
    "- Do NOT expand abbreviations (keep 明 not 明治, 大 not 大正).\n"
    "- Do NOT add, remove, or reorder characters. Do NOT guess characters that are unclear — skip them.\n"
    "- No commentary, no translation, no headers, no formatting."
)

VARIANTS = [
    "ollama_resize", "ollama_original",
    "ollama_vanilla_resize", "ollama_vanilla_original",
    "google_resize", "google_original",
]


# ──────────────────────────────────────────────
# Utility functions (from compare_ocr.py / compute_paper_stats.py)
# ──────────────────────────────────────────────

def cjk_only(text: str) -> str:
    """Keep only CJK Unified Ideographs and Kana characters."""
    return "".join(
        c for c in text
        if "\u4E00" <= c <= "\u9FFF"
        or "\u3400" <= c <= "\u4DBF"
        or "\u3040" <= c <= "\u309F"
        or "\u30A0" <= c <= "\u30FF"
    )


def levenshtein(s1, s2):
    """Standard Levenshtein edit distance."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for c1 in s1:
        curr = [prev[0] + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(curr[j] + 1, prev[j + 1] + 1,
                            prev[j] + (0 if c1 == c2 else 1)))
        prev = curr
    return prev[-1]


def compute_cer(pred, gold):
    """Character Error Rate = edit_distance / len(gold)."""
    if not gold:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, gold) / len(gold)


def compute_prf(pred, gold):
    """Character-level Precision, Recall, F1 via matching characters."""
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    if not gold or not pred:
        return 0.0, 0.0, 0.0
    matching = sum(b.size for b in SequenceMatcher(None, gold, pred).get_matching_blocks())
    precision = matching / len(pred)
    recall = matching / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


# ──────────────────────────────────────────────
# setup
# ──────────────────────────────────────────────

def cmd_setup(args):
    """Select sample rows, copy/crop images, create gold template."""
    n = args.n
    seed = args.seed

    # Collect all rows with their OCR text length for stratified sampling
    rows = []
    for page_dir in sorted(DETECTED_ROWS.iterdir()):
        meta_path = page_dir / "row_metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        page_name = page_dir.name
        source_image = meta["source_image"]
        resize_scale = meta["resize_scale"]

        for row in meta["rows"]:
            # Get body text length from existing OCR output
            body_len = 0
            seg_path = OCR_RESULTS / page_name / f"{row['filename'].replace('.jpg', '')}_segmented_output.json"
            if seg_path.exists():
                with open(seg_path) as f:
                    segs = json.load(f)
                body_len = sum(len(s.get("body_text", "")) for s in segs)

            rows.append({
                "page": page_name,
                "row_filename": row["filename"],
                "source_image": source_image,
                "bbox_xyxy": row["bbox_xyxy"],
                "resize_scale": resize_scale,
                "body_len": body_len,
            })

    if not rows:
        print("ERROR: No rows found in detected_rows/")
        return

    # Stratified sampling: sort by body_len, pick evenly from terciles
    rows.sort(key=lambda r: r["body_len"])
    rng = random.Random(seed)
    tercile_size = len(rows) // 3
    selected = []
    per_tercile = max(1, n // 3)
    for t in range(3):
        start = t * tercile_size
        end = (t + 1) * tercile_size if t < 2 else len(rows)
        pool = rows[start:end]
        pick = min(per_tercile, len(pool))
        selected.extend(rng.sample(pool, pick))
    # Fill remaining slots
    remaining = [r for r in rows if r not in selected]
    while len(selected) < n and remaining:
        selected.append(rng.choice(remaining))
        remaining.remove(selected[-1])

    # Create output dirs
    for sub in ["resize", "original", "results"]:
        (EVAL_DIR / sub).mkdir(parents=True, exist_ok=True)

    samples = []
    gold_entries = []

    for row_info in selected:
        page = row_info["page"]
        row_fn = row_info["row_filename"]
        row_id = f"{page}_{row_fn.replace('.jpg', '')}"
        bbox = row_info["bbox_xyxy"]
        scale = row_info["resize_scale"]

        # 1. Copy existing crop from detected_rows/
        src = DETECTED_ROWS / page / row_fn
        dst_resize = EVAL_DIR / "resize" / f"{row_id}.jpg"
        if src.exists():
            shutil.copyfile(src, dst_resize)
        else:
            print(f"WARNING: {src} not found, skipping")
            continue

        # 2. Re-crop from images_original/
        orig_path = IMAGES_ORIGINAL / row_info["source_image"]
        resize_path = IMAGES_RESIZE / row_info["source_image"]
        dst_original = EVAL_DIR / "original" / f"{row_id}.jpg"

        if orig_path.exists() and resize_path.exists():
            img_resize = cv2.imread(str(resize_path))
            img_orig = cv2.imread(str(orig_path))
            if img_resize is not None and img_orig is not None:
                # bbox is in work_img coords (resize_img × resize_scale)
                work_h = img_resize.shape[0] * scale
                work_w = img_resize.shape[1] * scale
                orig_h, orig_w = img_orig.shape[:2]
                sx = orig_w / work_w
                sy = orig_h / work_h
                x1 = max(0, int(bbox[0] * sx))
                y1 = max(0, int(bbox[1] * sy))
                x2 = min(orig_w, int(bbox[2] * sx))
                y2 = min(orig_h, int(bbox[3] * sy))
                crop = img_orig[y1:y2, x1:x2]
                cv2.imwrite(str(dst_original), crop)
            else:
                print(f"WARNING: Could not read images for {page}")
        else:
            print(f"WARNING: Original/resize image not found for {page}")

        samples.append(row_info)
        gold_entries.append({
            "id": row_id,
            "page": page,
            "row": row_fn,
            "resize_image": f"resize/{row_id}.jpg",
            "original_image": f"original/{row_id}.jpg",
            "gold_text": "",
        })

    # Save samples metadata
    with open(EVAL_DIR / "samples.json", "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    # Save gold template
    with open(EVAL_DIR / "gold.json", "w") as f:
        json.dump(gold_entries, f, indent=2, ensure_ascii=False)

    print(f"Setup complete: {len(samples)} rows in {EVAL_DIR}/")
    print(f"  resize/    — row crops from images_resize")
    print(f"  original/  — row crops from images_original")
    print(f"  gold.json  — fill in 'gold_text' for each entry")


# ──────────────────────────────────────────────
# run
# ──────────────────────────────────────────────

def ocr_ollama(image_bytes):
    """OCR via Ollama VLM."""
    import time
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": OCR_OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": OLLAMA_SYSTEM_PROMPT},
            {"role": "user", "content": "Read all the text in this image.", "images": [b64]},
        ],
        "stream": False,
        "think": False,
        "options": {"num_predict": OCR_OLLAMA_MAX_TOKENS, "temperature": 0, "top_k": 1},
    }
    for attempt in range(1, OCR_MAX_RETRIES + 1):
        try:
            resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except Exception as e:
            delay = OCR_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            if attempt < OCR_MAX_RETRIES:
                print(f"  Ollama attempt {attempt}/{OCR_MAX_RETRIES} failed: {e}, retrying in {delay}s")
                time.sleep(delay)
            else:
                print(f"  Ollama failed after {OCR_MAX_RETRIES} attempts: {e}")
                return ""


def ocr_ollama_vanilla(image_bytes):
    """OCR via Ollama VLM — zero-shot, no system prompt."""
    import time
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": OCR_OLLAMA_MODEL,
        "messages": [
            {"role": "user", "content": "Read all the text in this image.", "images": [b64]},
        ],
        "stream": False,
        "think": False,
        "options": {"num_predict": OCR_OLLAMA_MAX_TOKENS, "temperature": 0, "top_k": 1},
    }
    for attempt in range(1, OCR_MAX_RETRIES + 1):
        try:
            resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except Exception as e:
            delay = OCR_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            if attempt < OCR_MAX_RETRIES:
                print(f"  Ollama attempt {attempt}/{OCR_MAX_RETRIES} failed: {e}, retrying in {delay}s")
                time.sleep(delay)
            else:
                print(f"  Ollama failed after {OCR_MAX_RETRIES} attempts: {e}")
                return ""


def ocr_google(image_bytes):
    """OCR via Google Cloud Vision."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.document_text_detection(image=image)
    if response.full_text_annotation:
        return response.full_text_annotation.text.strip()
    return ""


def cmd_run(args):
    """Run OCR on all sample images."""
    backend = args.backend
    gold_path = EVAL_DIR / "gold.json"
    if not gold_path.exists():
        print("ERROR: Run 'setup' first")
        return

    with open(gold_path) as f:
        entries = json.load(f)

    backends = []
    if backend in ("ollama", "all"):
        backends.append(("ollama", ocr_ollama))
    if backend in ("ollama_vanilla", "all"):
        backends.append(("ollama_vanilla", ocr_ollama_vanilla))
    if backend in ("google", "all"):
        backends.append(("google", ocr_google))

    for bname, ocr_fn in backends:
        for source in ("resize", "original"):
            variant = f"{bname}_{source}"
            print(f"\nRunning {variant}...")
            results = {}
            for entry in entries:
                row_id = entry["id"]
                img_path = EVAL_DIR / source / f"{row_id}.jpg"
                if not img_path.exists():
                    print(f"  SKIP {row_id}: image not found")
                    continue
                print(f"  {row_id}...", end=" ", flush=True)
                with open(img_path, "rb") as f:
                    image_bytes = f.read()
                text = ocr_fn(image_bytes)
                results[row_id] = text
                cjk_len = len(cjk_only(text))
                print(f"{cjk_len} CJK chars")

            out_path = EVAL_DIR / "results" / f"{variant}.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────
# compare
# ──────────────────────────────────────────────

def cmd_compare(args):
    """Compare all OCR variants against gold standard."""
    gold_path = EVAL_DIR / "gold.json"
    if not gold_path.exists():
        print("ERROR: Run 'setup' first")
        return

    with open(gold_path) as f:
        entries = json.load(f)

    # Filter to entries with gold text
    gold = {e["id"]: e["gold_text"] for e in entries if e["gold_text"]}
    if not gold:
        print("ERROR: No gold_text filled in. Edit gold.json first.")
        return

    # Load result files
    variant_results = {}
    for v in VARIANTS:
        path = EVAL_DIR / "results" / f"{v}.json"
        if path.exists():
            with open(path) as f:
                variant_results[v] = json.load(f)
        else:
            print(f"NOTE: {path} not found, skipping {v}")

    if not variant_results:
        print("ERROR: No result files found. Run 'run' first.")
        return

    # Compute metrics
    row_ids = sorted(gold.keys())
    detail_rows = []
    variant_cers = {v: [] for v in variant_results}
    variant_sims = {v: [] for v in variant_results}
    variant_precs = {v: [] for v in variant_results}
    variant_recs = {v: [] for v in variant_results}
    variant_f1s = {v: [] for v in variant_results}

    for row_id in row_ids:
        gold_cjk = cjk_only(gold[row_id])
        if not gold_cjk:
            continue
        row_detail = {"id": row_id, "gold_cjk_len": len(gold_cjk)}
        for v, results in variant_results.items():
            pred_text = results.get(row_id, "")
            pred_cjk = cjk_only(pred_text)
            cer = compute_cer(pred_cjk, gold_cjk)
            sim = SequenceMatcher(None, gold_cjk, pred_cjk).ratio()
            prec, rec, f1 = compute_prf(pred_cjk, gold_cjk)
            variant_cers[v].append(cer)
            variant_sims[v].append(sim)
            variant_precs[v].append(prec)
            variant_recs[v].append(rec)
            variant_f1s[v].append(f1)
            row_detail[f"{v}_cer"] = cer
            row_detail[f"{v}_sim"] = sim
            row_detail[f"{v}_prec"] = prec
            row_detail[f"{v}_rec"] = rec
            row_detail[f"{v}_f1"] = f1
            row_detail[f"{v}_cjk_len"] = len(pred_cjk)
        detail_rows.append(row_detail)

    # Summary table
    lines = []
    sep = "=" * 90
    lines.append(sep)
    lines.append(f"OCR Evaluation — Character-Level Metrics (n = {len(detail_rows)} rows)")
    lines.append(sep)
    lines.append(f"{'Configuration':<25} {'CER':>8} {'P':>8} {'R':>8} {'F1':>8} {'n':>6}")
    lines.append("-" * 90)
    for v in variant_results:
        cers = variant_cers[v]
        if cers:
            mean_cer = sum(cers) / len(cers)
            mean_prec = sum(variant_precs[v]) / len(cers)
            mean_rec = sum(variant_recs[v]) / len(cers)
            mean_f1 = sum(variant_f1s[v]) / len(cers)
            lines.append(f"{v:<25} {mean_cer:>7.1%} {mean_prec:>7.1%} {mean_rec:>7.1%} {mean_f1:>7.1%} {len(cers):>6}")
    lines.append(sep)
    lines.append("CER = character error rate; P = precision; R = recall; F1 = harmonic mean of P and R.")

    # Per-row detail
    lines.append("")
    lines.append("Per-row detail (CER / F1):")
    header = f"{'Row':<30} {'n':>5}"
    for v in variant_results:
        label = v.replace("_", " ")
        header += f" {label:>22}"
    lines.append(header)
    lines.append("-" * len(header))
    for rd in detail_rows:
        line = f"{rd['id']:<30} {rd['gold_cjk_len']:>5}"
        for v in variant_results:
            cer = rd.get(f"{v}_cer", float("nan"))
            f1 = rd.get(f"{v}_f1", float("nan"))
            line += f" {cer:>9.1%} / {f1:<9.1%}"
        lines.append(line)

    output = "\n".join(lines)
    print(output)

    # Save text
    with open(EVAL_DIR / "comparison.txt", "w") as f:
        f.write(output + "\n")

    # Save CSV
    csv_path = EVAL_DIR / "comparison.csv"
    if detail_rows:
        fieldnames = list(detail_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(detail_rows)

    print(f"\nSaved: {EVAL_DIR}/comparison.txt, {csv_path}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OCR evaluation: 4 backends + gold standard")
    sub = parser.add_subparsers(dest="command")

    p_setup = sub.add_parser("setup", help="Select samples, copy/crop images, create gold template")
    p_setup.add_argument("--n", type=int, default=8, help="Number of sample rows (default: 8)")
    p_setup.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    p_run = sub.add_parser("run", help="Run OCR on sample images")
    p_run.add_argument("--backend", choices=["ollama", "ollama_vanilla", "google", "all"], default="all",
                        help="Which backend to run (default: all)")

    sub.add_parser("compare", help="Compare OCR results vs gold standard")

    args = parser.parse_args()
    if args.command == "setup":
        cmd_setup(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
