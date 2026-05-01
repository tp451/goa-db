# Archival Japanese OCR Pipeline

Extracts and OCRs text from scanned 1940s Japanese biographical directories. The pipeline trains YOLO models to find text rows and headers, crops rows, segments pages by header position, OCRs each segment with a local vision LLM (Ollama/Qwen), then uses local LLMs for biography extraction, name romanization, translation, gender classification, organisation verification, and HISCO/ISIC coding. Cross-volume disambiguation merges duplicate persons and organisations. Google Cloud Vision is supported as an optional alternative OCR backend, and a Real-ESRGAN upscaling preprocess (step 0) is also optional — side-by-side OCR comparison showed upscaling does not improve accuracy on this corpus.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (required for YOLO training, local LLM inference, and the optional step 0 upscaling)
- [Ollama](https://ollama.ai/) with a vision-capable model pulled — used end-to-end for OCR (step 4), biography extraction (step 5), and name romanization / translation / gender / org verification / HISCO–ISIC classification (steps 6–8)
- *(Optional alternative OCR backend)* Google Cloud Vision API credentials — only needed if you set `OCR_BACKEND = "google"` in `config.py` instead of the default Ollama backend
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) + PyTorch (for Japanese gender classification in step 6)
- [Shapely](https://shapely.readthedocs.io/) (for spatial join in step 6)
- [namedivider](https://github.com/rskmoi/namedivider-python) (`BasicNameDivider`) — for Japanese name splitting in step 6
- [pykakasi](https://github.com/miurahr/pykakasi) — fast first-pass Hepburn romanization of Japanese names in steps 6 and 7 (Ollama is the fallback for non-Japanese names and pykakasi failures)
- GeoNames gazetteer files (`JP.txt`, `TW.txt`, `CN.txt`, `KR.txt`, `KP.txt`) — download from [GeoNames](https://download.geonames.org/export/dump/) (for geolocation in step 6)
- MCGD gazetteer (`MCGD_Data2025_08_06.csv`) — for Chinese historical place matching in step 6
- Historical GeoJSON boundary files (`japan_hijmans.geojson`, `china_1928.geojson`, `korea_imperial.geojson`, `taiwan_1946.geojson`) — for spatial province/country assignment in step 6

## Installation

```bash
pip install -r requirements.txt
# Then start Ollama and pull the vision/LLM model configured in config.py.
# Only required if you opt into the Google Vision OCR backend:
# export GOOGLE_APPLICATION_CREDENTIALS="path/to/your-credentials.json"
```

## Pipeline Overview

```
Step 0:  0_optional_upscale.py         OPTIONAL: batched Real-ESRGAN upscale + resize.
                                       Skip unless you specifically want to experiment with
                                       upscaled inputs; toggle USE_UPSCALED in config.py.
Step 1:  1_train_seg.py                Train row segmentation model (wrapper → train_yolo.py --task seg)
Step 2:  2_detect_rows_cropper.py      Crop rows using segmentation model
Step 3:  3_train_detect.py             Train header detection model (wrapper → train_yolo.py --task detect)
Step 4:  4_segment_recognise.py        Segment by headers + OCR
Step 5:  5_extract_biographies.py      Extract structured biographies via LLM (per-volume)
Step 6:  6_structure_biographies.py    Structure into relational tables + gender/geo/translation
Step 7:  7_disambiguate.py             Cross-volume entity disambiguation
Step 8:  8_classify_occupations.py     Occupation & industry classification (HISCO/ISIC) via LLM
```

### Detailed flow

```mermaid
flowchart TD
    classDef script fill:#4a90d9,color:#fff,stroke:#2c5f8a
    classDef optional fill:#9aa5b1,color:#fff,stroke:#5a6877,stroke-dasharray:5 5
    classDef data fill:#f5f5f5,stroke:#999,color:#333
    classDef model fill:#f0ad4e,color:#fff,stroke:#c68e3c
    classDef service fill:#5cb85c,color:#fff,stroke:#3d8b3d

    %% Data stores
    raw[(images_original/)]:::data
    upscaled[(images_resize/<br/>only if step 0 was run)]:::data
    rows_out[(detected_rows/)]:::data
    ocr_out[(ocr_results/)]:::data
    bios_out[(biographies_extracted_*.jsonl)]:::data
    struct_out[(structured/<br/>11 JSONL tables)]:::data
    disambig_out[(disambiguated/<br/>11 JSONL tables)]:::data

    %% Models
    seg_model([seg model<br/>yolo_rows/]):::model
    det_model([detect model<br/>yolo_headers/]):::model

    %% External services
    ocr_svc{{Ollama vision LLM<br/>(default; Google Cloud Vision optional)}}:::service
    ollama_bio{{Ollama LLM}}:::service
    ollama_struct{{Ollama LLM}}:::service
    hf{{HuggingFace model}}:::service
    geo{{GeoNames + MCGD<br/>+ GeoJSON}}:::service
    namediv{{namedivider}}:::service
    pykakasi{{pykakasi}}:::service
    shapely{{Shapely}}:::service

    %% Step 0 (optional)
    raw -.-> s0[0_optional_upscale.py<br/>Step 0 OPTIONAL: upscale + resize]:::optional
    s0 -.-> upscaled

    %% Step 1
    raw --> s1[1_train_seg.py<br/>Step 1: train segmentation]:::script
    s1 --> seg_model

    %% Step 2
    raw --> s2[2_detect_rows_cropper.py<br/>Step 2: crop rows]:::script
    seg_model --> s2
    s2 --> rows_out

    %% Step 3
    s3[3_train_detect.py<br/>Step 3: train detection]:::script
    s3 --> det_model

    %% Step 4
    raw -->|full-page images| s4[4_segment_recognise.py<br/>Step 4: segment + OCR]:::script
    rows_out -->|row crops + metadata| s4
    det_model --> s4
    ocr_svc -.-> s4
    s4 --> ocr_out

    %% Step 5
    ocr_out --> s5[5_extract_biographies.py<br/>Step 5: extract biographies]:::script
    ollama_bio -.-> s5
    s5 --> bios_out

    %% Step 6
    bios_out --> s6[6_structure_biographies.py<br/>Step 6: structure + enrich]:::script
    pykakasi -.->|romanize ja names| s6
    ollama_struct -.->|romanize non-ja + translate<br/>+ gender (non-ja)| s6
    hf -.->|gender classify (ja)| s6
    geo -.->|geolocation| s6
    shapely -.->|spatial join| s6
    namediv -.->|name splitting| s6
    s6 --> struct_out

    %% Step 7
    struct_out --> s7[7_disambiguate.py<br/>Step 7: disambiguate]:::script
    pykakasi -.->|re-romanize merged ja names| s7
    ollama_struct -.->|org verification<br/>+ hierarchy| s7
    geo -.->|place name filter| s7
    s7 --> disambig_out

    %% Step 8
    disambig_out --> s8[8_classify_occupations.py<br/>Step 8: classify occupations]:::script
    ollama_bio -.->|HISCO + ISIC| s8
```

> The dashed step 0 path is optional. With `USE_UPSCALED = False` (default in `config.py`), steps 2 and 4 read directly from `images_original/`. Setting `USE_UPSCALED = True` makes them read from `images_resize/` instead — only meaningful after step 0 has been run.

### Why training happens at two points

The pipeline trains **two different YOLO models** for two different tasks:

1. **Segmentation model** (step 1, `--task seg`) finds **text rows** in full page scans. This model is needed by step 2 to crop the rows.
2. **Detection model** (step 3, `--task detect`) finds **section headers** in full-page images. This model is needed by step 4 to segment rows by header position before OCR.

Training step 1 must happen before step 2 (which needs its output model). Training step 3 must happen before step 4 (which needs its output model). If you already have trained model weights, you can skip the training steps.

Both YOLO models are trained on the original (non-upscaled) images in `images/train/` — running step 0 is not a prerequisite for training.

---

## Running Each Step

### Step 0 (OPTIONAL): Upscale and resize

```bash
python 0_optional_upscale.py
```

**Optional.** `eval_ocr.py` showed Real-ESRGAN upscaling does **not** improve OCR accuracy on this corpus, so the main pipeline now reads `images_original/` by default. Run this script only if you want to experiment with upscaled inputs; afterwards set `USE_UPSCALED = True` in `config.py` to make steps 2 and 4 read from `images_resize/`.

Reads raw scans from `images_original/`, upscales with Real-ESRGAN, resizes to fit within `RESIZE_TARGET_SIZE` (default: 4096px), and writes to `images_resize/`. Processes images in batches of `PIPELINE_BATCH_SIZE` (default: 20) to limit disk usage — upscaled intermediates in `images_esrgan/` are deleted after each batch is resized. Resumable: skips images that already exist in `images_resize/`.

The underlying scripts (`_upscale_batch.py` and `_yolo_prepare_resize.py`) still exist as importable modules and can be run individually if needed.

### Step 1: Train segmentation model

```bash
python 1_train_seg.py
```

Trains a YOLOe segmentation model (`yoloe-26x-seg.pt`) using `dataset_rows.yaml`. The script automatically uses the `YOLOE` class and `YOLOEPESegTrainer` required by YOLOe models. Outputs weights to `yolo_rows/yolo_rows/weights/best.pt`. Runs validation on the test split and saves a publication-quality figure.

### Step 2: Crop rows

```bash
python 2_detect_rows_cropper.py
```

Uses the segmentation model from step 1 to detect text rows, then crops each row with polygon masking (white background). Reads from `CROPPER_INPUT_DIR` (default `images_original/`; becomes `images_resize/` if `USE_UPSCALED = True`), outputs to `detected_rows/<image_name>/`. Generates a visual index image (`00_visual_index.jpg`) and a `row_metadata.json` per source image.

The metadata file records each row's bounding box in the downsized coordinate space so that step 4 can map full-page header detections back to individual row crops.

### Step 3: Train detection model

```bash
python 3_train_detect.py
```

Trains a YOLO26 detection model (`yolo26x.pt`) using `dataset_headers.yaml`. Outputs weights to `yolo_headers/yolo_headers/weights/best.pt`.

### Step 4: Segment and OCR

```bash
python 4_segment_recognise.py
```

For each page directory in `detected_rows/`:
1. Loads the corresponding full-page image from `OCR_SOURCE_IMAGE_DIR` (same default as the cropper — `images_original/`, or `images_resize/` if `USE_UPSCALED = True`) and downsizes it to match step 2's coordinate space (`CROPPER_RESIZE_PERCENTAGE`).
2. Runs header detection on the **full page** (the model was trained on full pages and fails on small row crops).
3. Maps each detected header to its row using bounding-box overlap from `row_metadata.json`.
4. For each row: translates header coordinates to row-local space, defines column segments based on header positions.
5. OCRs each **cropped row image** with the configured backend — by default, a local Ollama/Qwen vision model. Google Cloud Vision (batched up to 16 images) is available as an optional alternative; set `OCR_BACKEND` in `config.py` to `"ollama"` (default) or `"google"`.
6. Assigns each OCR symbol to either a header or body segment using bounding-box overlap.
7. Writes JSON + debug image to `ocr_results/`.

### Step 5: Extract biographies

```bash
python 5_extract_biographies.py
python 5_extract_biographies.py --volume 1927
python 5_extract_biographies.py --reprocess          # re-run post-processing without re-calling LLM
python 5_extract_biographies.py --reprocess --volume 1927
```

Reads OCR results from `ocr_results/`, stitches orphan segments back onto their preceding entries, and sends each biography to a local Ollama model for structured extraction. The `--volume` flag (default: `VOLUME_ID` from config) sets the volume identifier, and output is written to a per-volume file: `biographies_extracted_{volume}.jsonl`.

The script is **resumable** — on restart it skips entries that were already successfully extracted and retries any previous errors. Post-processing includes:
- **Era date conversion** — deterministic conversion of Japanese/Chinese/Korean era dates to Western years, overriding LLM arithmetic (supports Meiji, Taisho, Showa, Manchukuo, Republic of China, Qing, Korean Empire eras)
- **Schema normalization** — fixes type violations (phone_number lists → strings, hobbies → arrays, etc.)
- **Family name cleaning** — nulls compound descriptors (e.g. "高木金之助三女") and normalizes relation strings to single kinship terms
- **Validation check** — flags entries with suspiciously few career records given the source text

Use `--reprocess` to re-run all post-processing on existing records without making new LLM calls (useful after updating the post-processing logic).

Requires Ollama running locally (default: `http://localhost:11434`). Configure the model name via `BIO_MODEL_NAME` in `config.py`.

### Step 6: Structure biographies

```bash
python 6_structure_biographies.py
```

Auto-discovers all per-volume files matching `biographies_extracted_*.jsonl` (via `load_all_volume_records()`), tags each record with its volume ID, and builds normalised relational tables in `structured/`. Falls back to `STRUCT_INPUT_FILE` if no per-volume files exist. Processing steps:

1. **Build tables** — Splits flat JSON records into `person_core`, `person_career`, `person_education`, `person_hobbies`, `person_ranks`, `person_religions`, `person_family_members`, `person_family_education`, and `person_family_career` tables. Person IDs include the volume prefix (e.g., `P1927_0`). Cleans names, validates years, strips kinship noise from family member names. Assigns `relation_id` per family member and validates unique relations (one father/mother) and sibling birth order.
2. **Domain detection** — Classifies each person as `ja` (Japanese), `zh` (Chinese), `ko` (Korean), or `other` (katakana-only foreigners) based on origin place and name patterns.
3. **Geolocation** — Loads GeoNames (`JP.txt`, `TW.txt`, `CN.txt`, `KR.txt`, `KP.txt`) and MCGD gazetteers into CJK reverse indexes. Matches place strings using progressively shorter prefixes, disambiguates by admin level and origin province overlap. Also geolocates family member places and career places. Coordinates live only in `locations.jsonl`; all other tables reference locations by `location_id`.
4. **Spatial join** — Assigns `province` and `country` to each location via point-in-polygon join against historical GeoJSON boundary files (Japan, China 1928, Korea Imperial, Taiwan 1946) using Shapely. Overrides text-based domain detection with the authoritative spatial result when origin place geolocates inside a different country's boundaries.
5. **Admin hierarchy** — Parses CJK place names into hierarchical admin levels (`admin1`/`admin2`/`admin3`) with normalised and English-translated variants.
6. **Name splitting** — Uses `namedivider` (`BasicNameDivider`) to split Japanese names into family and given components; uses compound surname lookup for Chinese names. Falls back to character-based splitting if the divider fails.
7. **Gender identification** — Japanese subjects classified using [gendec-with-distilmbert](https://huggingface.co/tarudesu/gendec-with-distilmbert) (threshold 0.97); non-Japanese subjects classified via Qwen/Ollama. Name-based classification is a weak prior only — biographical signals override: having a wife → male, having a husband → female, military ranks → male, attending a women's school → female. Family member gender inferred from the relation field.
8. **Name romanization** — Batches all unique names to Ollama for Hepburn (Japanese), Pinyin without tone marks (Chinese), Revised Romanization (Korean), or original Latin spelling (katakana foreigners).
9. **Location translation** — Translates location names and admin-level names to English via Ollama.
10. **Organisation deduplication** — Assigns `organization_id` to career and education records across both subject and family member tables.
11. **Plausibility checks** — Applies corrections and logs all editorial changes to `editorial_changes.jsonl`: removes male subjects' girls'-school education records, nulls conflicting family ordinals (長男+一男), resolves wife/husband conflicts using gender, validates education/career years against birthyear (min age 14/16), and validates dates against volume publication year.
12. **Org/job title translation** — Batches unique organisation names and job titles (including family members') to Ollama for English translation. Cleans output (ASCII only, max 50 chars).
13. **Org location assignment** — Assigns `location_id` to organisations using geographic clustering of career records (most common location), with a name-prefix fallback. Skips generic government bodies and organisations whose career locations span more than `STRUCT_ORG_LOCATION_MAX_SPREAD_KM`.

Requires Ollama running locally, the GeoNames/MCGD data files, and the GeoJSON boundary files in the working directory.

### Step 7: Cross-volume disambiguation

```bash
python 7_disambiguate.py
```

Reads the relational tables from `structured/` (all 11 tables) and produces a deduplicated dataset in `disambiguated/`. Processing steps:

1. **Person disambiguation** — Groups persons by (name, birthyear) across volumes. When the same person appears in multiple volumes, the earliest volume's record is selected as canonical and all later person_id references are remapped. Scalar fields are merged with later volumes taking precedence. Origin-place conflicts are flagged for review. After remapping, deduplicates identical entries and removes subset records (entries that are strict subsets of a more specific record for the same person + organisation).
2. **Organisation fuzzy matching** — Computes Jaro-Winkler similarity between organisation names, finding candidate pairs above `DISAMBIG_ORG_FUZZY_THRESHOLD` (default: 0.85). Each candidate pair is then **verified by LLM** (considers character variants, OCR errors, abbreviations; rejects similar but distinct entities). Verified merges are automatically applied — merged org IDs are remapped across all career/education tables and duplicate entries removed. Results are checkpointed so interrupted runs can resume.
3. **Manual org overrides** — Applies manual org ID remappings from `org_overrides.jsonl` if present (resolves transitive chains).
4. **Organisation hierarchy detection** — Detects parent-child relationships among organisations by prefix matching (org A's name is a prefix of org B's), filters out geographic prefixes using GeoNames data, and verifies remaining candidates via LLM. Assigns `parent_organization_id` to child organisations.
5. **Output** — Writes the same 11 table files as step 6 (with canonical IDs) plus `entity_mappings.jsonl` tracking all person merges and confirmed organisation matches.

### Step 8: Classify occupations

```bash
python 8_classify_occupations.py
```

Classifies career entries and organisations using Qwen/Ollama in two parts:

- **Part A: HISCO** — Assigns a 2-digit HISCO minor group code (ISCO-68 based) and a 1-digit major group to each career record based on its `job_title_en`. Unique titles are classified individually and cached to `_cache_hisco.json`.
- **Part B: ISIC** — Assigns an ISIC Rev 4 section code (A–U) and label to each organisation based on its name. Cached to `_cache_isic.json`.

Reads `person_career.jsonl` and `organizations.jsonl` from `disambiguated/`, writes enriched records back in place. Resumable via caches (skips already-classified titles/orgs on restart).

Requires Ollama running locally with `BIO_MODEL_NAME`.

---

## OCR Backend Comparison

Step 4 defaults to a local Ollama vision LLM. Google Cloud Vision is supported only as an optional alternative; switch via `OCR_BACKEND` in `config.py`:

| Backend | Setting | Requires |
|---|---|---|
| Ollama (Qwen 3.5 vision) — **default** | `"ollama"` | Ollama running locally with the model pulled |
| Google Cloud Vision (optional) | `"google"` | `GOOGLE_APPLICATION_CREDENTIALS` env var |

To compare output quality between backends, run step 4 once per backend (renaming `ocr_results/` between runs) and diff the JSON outputs.

---

## Configuration

All settings live in `config.py`. Key groups:

| Prefix | Controls |
|---|---|
| `USE_UPSCALED` | Whether the cropper (step 2) and OCR (step 4) read from `images_resize/` (True) or `images_original/` (False, default) |
| `UPSCALE_*` | Step 0 — upscaling model, quality, tile size |
| `RESIZE_*` | Step 0 — target size, thread count |
| `SEG_*` / `DET_*` | Steps 1/3 — model names, YAML paths, project names |
| `TRAIN_*` | Steps 1/3 — epochs, patience, batch size, image size |
| `CROPPER_*` | Step 2 — confidence, resize %, column tolerance |
| `OCR_*` | Step 4 — confidence, retry settings, batch size, source image dir |
| `VOLUME_ID` | Default volume identifier (used by step 5's `--volume` flag) |
| `BIO_*` | Step 5 — Ollama URL, model name, max tokens, `BIO_OUTPUT_PATTERN` for per-volume filenames |
| `STRUCT_*` | Step 6 — output dir, batch size, year validation, gender model/threshold, geolocation paths, GeoJSON settings, org location spread, translation settings |
| `DISAMBIG_*` | Step 7 — output dir, org fuzzy-match threshold, LLM max tokens for org verification/hierarchy, hierarchy skip-parents list |
| `ISIC_*` | Step 8 — LLM max tokens for HISCO/ISIC classification |
| `PIPELINE_BATCH_SIZE` | `0_optional_upscale.py` — images per upscale+resize batch (0 = all at once) |

To change any path, threshold, or model setting, edit `config.py` once. All scripts import from it.

---

## Output Structure

### Step 2 output: `detected_rows/`

```
detected_rows/
  <image_name>/
    row_metadata.json            # row bounding boxes for step 4
    00_visual_index.jpg          # numbered overlay of all detected rows
    000_row.jpg                  # first cropped row
    001_row.jpg                  # second cropped row
    ...
```

### Step 4 output: `ocr_results/`

```
ocr_results/
  <image_name>/
    <crop_name>_segmented_output.json    # OCR results per segment
    <crop_name>_debug_segmentation.jpg   # visual debug overlay
  logs/
    run_YYYYMMDD_HHMMSS.log              # full run log
    errors_YYYYMMDD_HHMMSS.log           # errors only
```

### Step 5 output: `biographies_extracted_{volume}.jsonl`

One JSON object per line, per volume. Each record contains:

```json
{
  "entry_index": 0,
  "source_image": "003_row.jpg",
  "validation": "Passed",
  "extraction": {
    "name": "...",
    "rank": "...",
    "place": "...",
    "birth_year": 1880,
    "career": [...],
    "family_member": [...]
  }
}
```

### Step 6 output: `structured/`

```
structured/
  person_core.jsonl              # one record per person (name, name_family, name_given, *_latin, domain, gender, birthyear, location_id, origin_location_id, volume)
  person_career.jsonl            # career entries (job_title, job_title_en, organization_id, location_id, start_year, current)
  person_education.jsonl         # education entries (organization_id, major_of_study, year_graduated)
  person_hobbies.jsonl           # hobbies
  person_ranks.jsonl             # ranks/titles
  person_religions.jsonl         # religions
  person_family_members.jsonl    # family members (relation, relation_id, name, name_latin, gender, birth_year, location_id)
  person_family_education.jsonl  # family member education (person_id, relation_id, organization_id, major_of_study, year_graduated)
  person_family_career.jsonl     # family member career (person_id, relation_id, job_title, job_title_en, organization_id, location_id, start_year)
  organizations.jsonl            # deduplicated orgs (name, name_en, location_id)
  locations.jsonl                # deduplicated locations (lat, lon, geonameid, mcgd_locid, province, country, admin1/2/3, *_en)
  editorial_changes.jsonl        # log of plausibility corrections applied
  logs/
    run_YYYYMMDD_HHMMSS.log      # full run log
```

Coordinates are stored **only** in `locations.jsonl`. All other tables reference locations via `location_id`, avoiding coordinate duplication.

### Step 7 output: `disambiguated/`

```
disambiguated/
  person_core.jsonl              # canonical persons (cross-volume duplicates merged, volume becomes a list)
  person_career.jsonl            # career entries with remapped person_ids and org_ids (hisco_code, hisco_major added by step 8)
  person_education.jsonl         # education entries with remapped person_ids and org_ids
  person_hobbies.jsonl           # hobbies with remapped person_ids
  person_ranks.jsonl             # ranks with remapped person_ids
  person_religions.jsonl         # religions with remapped person_ids
  person_family_members.jsonl    # family members with remapped person_ids
  person_family_education.jsonl  # family member education with remapped person_ids and org_ids
  person_family_career.jsonl     # family member career with remapped person_ids and org_ids
  organizations.jsonl            # orgs with parent_organization_id (isic_section, isic_label added by step 8)
  locations.jsonl                # locations
  entity_mappings.jsonl          # person merges + LLM-confirmed org matches
```

---

## File Inventory

| File | Purpose |
|---|---|
| `config.py` | All paths, thresholds, and model settings — the single source of truth (includes `USE_UPSCALED` toggle) |
| `0_optional_upscale.py` | OPTIONAL batched upscale+resize pipeline (combines upscale and resize, saves disk space). Skip unless experimenting with upscaled inputs |
| `_upscale_batch.py` | Upscale archival photos with Real-ESRGAN (used as module by `0_optional_upscale.py`) |
| `_yolo_prepare_resize.py` | Resize images + copy labels for YOLO training (used as module by `0_optional_upscale.py`) |
| `1_train_seg.py` | Train row segmentation model (wrapper around `train_yolo.py --task seg`) |
| `2_detect_rows_cropper.py` | Crop text rows from page images using the segmentation model |
| `3_train_detect.py` | Train header detection model (wrapper around `train_yolo.py --task detect`) |
| `4_segment_recognise.py` | Detect headers, segment pages, OCR via local Ollama vision LLM (default) or Google Cloud Vision (optional) |
| `5_extract_biographies.py` | Extract structured biographies from OCR results via Ollama (per-volume) |
| `6_structure_biographies.py` | Structure biographies into relational tables with gender, geolocation, and translation |
| `7_disambiguate.py` | Cross-volume entity disambiguation (person dedup + LLM-verified org matching + org hierarchy) |
| `8_classify_occupations.py` | Occupation & industry classification via LLM (HISCO minor groups + ISIC sections) |
| `train_yolo.py` | Train YOLO models (`--task seg` or `--task detect`) — used by wrapper scripts |
| `eval_ocr.py` | OCR evaluation harness — compares Ollama/Google Vision on `images_resize/` vs `images_original/` against a manual gold standard |
| `compute_paper_stats.py` | Descriptive stats + gold-standard sample export, scoring, and SLM-vs-Gemini extraction comparison. See note below for optional deps. |
| `geojson/` | Historical boundary files (Japan, China 1928, Korea Imperial, Taiwan 1946) used by step 6 spatial join |
| `website/` | R Shiny web interface (`app.R`) and pre-processed dataset (`data/`, `app_data.rds`) for browsing the published database |
| `requirements.txt` | Pinned Python dependencies (includes optional deps for `compute_paper_stats.py`) |

### Optional dependencies for `compute_paper_stats.py`

These are listed in `requirements.txt` but only needed if you run the paper-stats pipeline:

- **Always** (for `--export-gold` / `--score-gold`): `pandas`
- **`--bertscore`** (semantic-similarity scoring of extracted bios): `bert_score`, `fugashi`, `unidic-lite` (the last two power MeCab tokenisation for the default `cl-tohoku/bert-base-japanese-v3` BERT model)
- **`--compare-hisco`** (OccCANINE comparison): not on PyPI — install via `uv pip install "git+https://github.com/christianvedels/OccCANINE.git"`. The package is named `histocc` internally.
- **`--run-cloud --cloud-provider google`** (Gemini extraction comparison): `google-generativeai`, plus `GOOGLE_API_KEY` or `GEMINI_API_KEY` env var.
