"""
Step 6: Structure extracted biographies into relational JSONL tables.

Reads biographies_extracted.jsonl (from step 5), normalises flat JSON records
into relational tables (person_core, person_career, person_education, etc.),
romanises names via Ollama, and deduplicates organisations.
"""
import json
import re
import os
import logging
import time
import requests
from namedivider import BasicNameDivider

import csv
from collections import Counter, defaultdict
from transformers import pipeline as hf_pipeline

from math import radians, sin, cos, sqrt, atan2
import pykakasi

from config import (
    STRUCT_INPUT_FILE, STRUCT_OUTPUT_DIR, BIO_OUTPUT_PATTERN,
    STRUCT_BIRTHYEAR_MIN, STRUCT_BIRTHYEAR_MAX,
    STRUCT_YEAR_MIN, STRUCT_YEAR_MAX,
    STRUCT_FAMILY_AGE_GAP, STRUCT_SPOUSE_AGE_GAP,
    BIO_OLLAMA_BASE_URL, BIO_MODEL_NAME, BIO_MAX_TOKENS,
    STRUCT_GENDER_MODEL, STRUCT_GENDER_THRESHOLD,
    STRUCT_MCGD_PATH, STRUCT_MCGD_CODES,
    STRUCT_GEONAMES_FILES, STRUCT_GEONAMES_CLASSES,
    STRUCT_GEOJSON_DIR, STRUCT_GEOJSON_FILES,
    STRUCT_ORG_TRANSLATION_MAX_LEN,
    STRUCT_ORG_LOCATION_MAX_SPREAD_KM,
)

# ==========================================
# LOGGING
# ==========================================
os.makedirs(STRUCT_OUTPUT_DIR, exist_ok=True)
log_dir = os.path.join(STRUCT_OUTPUT_DIR, "logs")
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
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

OLLAMA_CHAT_URL = BIO_OLLAMA_BASE_URL.replace("/v1", "").rstrip("/") + "/api/chat"

# ==========================================
# CONSTANTS
# ==========================================
SUFFIX_TITLES = {"長", "員", "官", "監", "佐"}

# Chinese province / region indicators for domain detection
ZH_REGION_PATTERNS = re.compile(
    # Modern provinces (traditional + simplified/Japanese character variants)
    r"(?:河北|山西|遼寧|吉林|黑龍江|黒龍江|江蘇|浙江|安徽|福建|江西|山東|河南|"
    r"湖北|湖南|廣東|広東|廣西|広西|海南|四川|貴州|雲南|陝西|陕西|甘肅|青海|"
    r"新疆|西藏|內蒙古|寧夏|"
    # Historical provinces / regions
    r"直隸|直隷|直轄|奉天|熱河|察哈爾|綏遠|江南|蒙古|"
    # Major cities
    r"北京|天津|上海|重慶|南京|廣州|広州|蘇州|杭州|武漢|"
    # Taiwan / Manchuria (all character variants)
    r"臺灣|台湾|滿洲|満洲|滿州|満州)"
    # Also match any of these followed by 省
    r"|(?:(?:河北|山西|遼寧|吉林|黑龍江|黒龍江|江蘇|浙江|安徽|福建|江西|山東|"
    r"河南|湖北|湖南|廣東|広東|廣西|広西|四川|貴州|雲南|陝西|陕西|甘肅|青海|"
    r"直隸|直隷|奉天|熱河|察哈爾|綏遠)省)"
)

# Korean region patterns for domain detection
KO_REGION_PATTERNS = re.compile(
    r"(?:朝鮮|韓國|韓国|慶尚|全羅|忠清|平安|黃海|黄海|江原|咸鏡|咸鏡|"
    r"京畿|京城|仁川|釜山|平壤|大邱|光州|大田)"
)

# Han/Kana character ranges for name filtering
CJK_PATTERN = re.compile(r"[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]")
HAN_PATTERN = re.compile(r"[\u4E00-\u9FFF]")
NON_CJK_KANA = re.compile(r"[^\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]")
KATAKANA_ONLY = re.compile(r"^[\u30A0-\u30FFー]+$")

# Unambiguous Chinese single-character surnames (百家姓 minus those that commonly
# start Japanese surnames: 田林山中森石松木井川上大小長高本原内北南西東前後金)
COMMON_ZH_SURNAMES = {
    "王", "李", "張", "劉", "陳", "楊", "黃", "趙", "周", "吳",
    "徐", "孫", "馬", "朱", "胡", "郭", "何", "羅", "梁", "宋",
    "鄭", "謝", "韓", "唐", "馮", "于", "董", "蕭", "程", "曹",
    "袁", "鄧", "許", "傅", "沈", "曾", "彭", "呂", "蘇", "盧",
    "蔣", "蔡", "賈", "丁", "魏", "薛", "葉", "閻", "余", "潘",
    "杜", "戴", "夏", "鍾", "汪", "邱", "任", "姜", "范", "方",
    "陸", "鄒", "熊", "孟", "秦", "白", "江", "段", "雷", "侯",
    "龍", "史", "陶", "賀", "顧", "毛", "郝", "龔", "邵", "萬",
    "覃", "武", "錢", "嚴", "尹", "廖", "譚", "魯", "喬", "莊",
    "翁", "鄂", "岳", "齊", "沙", "柳", "奚", "紀", "賴", "甄",
}

name_divider = BasicNameDivider()


# ==========================================
# A. LOAD & FILTER
# ==========================================
def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("validation") == "Passed" and rec.get("extraction"):
                records.append(rec)
    logger.info(f"Loaded {len(records)} valid records from {path}")
    return records


def load_all_volume_records():
    """Discover all biographies_extracted_*.jsonl, load and tag with volume."""
    import glob as globmod
    pattern = BIO_OUTPUT_PATTERN.replace("{volume}", "*")
    files = sorted(globmod.glob(pattern))
    if not files:
        # Fallback to old single file
        if os.path.exists(STRUCT_INPUT_FILE):
            files = [STRUCT_INPUT_FILE]
        else:
            logger.error("No input files found")
            return []
    all_records = []
    for filepath in files:
        basename = os.path.basename(filepath)
        match = re.search(r'biographies_extracted_(\w+)\.jsonl', basename)
        volume = match.group(1) if match else "unknown"
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("validation") == "Passed" and rec.get("extraction"):
                    rec["volume"] = volume
                    all_records.append(rec)
        logger.info(f"Loaded {sum(1 for r in all_records if r.get('volume') == volume)} valid records from {filepath} (volume={volume})")
    logger.info(f"Total: {len(all_records)} valid records from {len(files)} file(s)")
    return all_records


# ==========================================
# B. NAME CLEANING
# ==========================================
def clean_name(name):
    """Strip parenthetical content, Latin chars, whitespace. Return None if invalid."""
    if not name:
        return None
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"（.*?）", "", name)
    name = re.sub(r"[A-Za-z\s]+", "", name)
    # Keep only Han/Hiragana/Katakana (strip ・ middle dot)
    name = "".join(CJK_PATTERN.findall(name)).replace("・", "")
    if len(name) <= 1:
        return None
    return name


# Multi-char kinship terms safe to strip from names (longer terms first for greedy match)
KINSHIP_NOISE = [
    "婿養子", "娘婿",
    "養父", "養母", "養子", "養女",
    "叔父", "叔母", "伯父", "伯母",
    "義父", "義母", "義兄", "義弟", "義姉", "義妹",
    "長男", "二男", "三男", "四男", "五男", "六男", "七男",
    "長女", "二女", "三女", "四女", "五女", "六女", "七女",
]


def strip_kinship_noise(name):
    """Remove kinship terms that leaked into family member names from OCR/extraction.

    Finds the earliest 2+ char kinship term in the name and truncates from there.
    Only truncates if the portion before the term is >= 2 characters (a plausible name).
    Iterates in case truncation reveals a new match.
    Single-char terms (父, 母, 子, etc.) are NOT stripped — they appear in real names.
    """
    if not name or len(name) < 4:
        return name
    changed = True
    while changed:
        changed = False
        earliest_pos = len(name)
        for term in KINSHIP_NOISE:
            pos = name.find(term, 2)
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos
        if earliest_pos < len(name):
            name = name[:earliest_pos]
            changed = True
    return name


COMPOUND_CHINESE_SURNAMES = {
    "歐陽", "司馬", "上官", "諸葛", "公孫", "令狐", "夏侯", "東方",
    "尉遲", "慕容", "端木", "皇甫", "獨孤", "長孫", "司徒", "軒轅",
}

# Era date pattern: e.g. "明二・三" (Meiji 2, March) — not a valid org/institution name
ERA_DATE_RE = re.compile(r'^[明大昭][一二三四五六七八九十〇]+[・年]')


def split_name(name, domain=None):
    """Split full name into family/given using namedivider (or heuristic for Chinese)."""
    if not name or len(name) <= 1:
        return None, None
    if len(name) == 2:
        return name[0], name[1]
    # For Chinese names, use heuristic: most surnames are 1 char, compound surnames are 2 chars
    if domain == "zh":
        if name[:2] in COMPOUND_CHINESE_SURNAMES:
            return name[:2], name[2:] if len(name) > 2 else None
        return name[0], name[1:]
    try:
        result = name_divider.divide_name(name)
        return result.family, result.given
    except Exception:
        return name[0], name[1:]


def validate_year(year, min_y, max_y):
    """Return year if valid, else None."""
    if year is None:
        return None
    try:
        year = int(year)
    except (TypeError, ValueError):
        return None
    if min_y <= year <= max_y:
        return year
    return None


def clean_phone(phone):
    """Strip non-digits, null if not 3-5 digits."""
    if not phone:
        return None
    digits = re.sub(r"\D", "", str(phone))
    if 3 <= len(digits) <= 5:
        return digits
    return None


def clean_rank(rank):
    """Null if length < 2."""
    if not rank or len(str(rank)) < 2:
        return None
    return str(rank)


# ==========================================
# C. DOMAIN DETECTION
# ==========================================
def detect_domain(origin_place, name=None):
    """Determine domain: 'zh', 'ko', 'ja', or 'other' based on origin_place and name."""
    if origin_place and ZH_REGION_PATTERNS.search(origin_place):
        return "zh"
    if origin_place and KO_REGION_PATTERNS.search(origin_place):
        return "ko"
    if name:
        stripped = re.sub(r"\s+", "", name)
        # Katakana-only full name → likely non-CJK foreigner (e.g., Einstein)
        if stripped and KATAKANA_ONLY.match(stripped):
            return "other"
        # Short all-han name (2-3 chars) with unambiguous Chinese surname
        han_chars = HAN_PATTERN.findall(stripped)
        if len(han_chars) == len(stripped) and 2 <= len(stripped) <= 3:
            if stripped[0] in COMMON_ZH_SURNAMES:
                return "zh"
    return "ja"


# ==========================================
# D. CAREER CLEANING
# ==========================================
def extract_place_from_org(org):
    """Extract place prefix up to 州 or 地方 from org name."""
    m = re.match(r"^(.+?州)", org)
    if m:
        return m.group(1)
    m = re.match(r"^(.+?地方)", org)
    if m:
        return m.group(1)
    return None


def fix_suffix_title(job_title, organization):
    """Prepend last CJK char from org to single-char titles in SUFFIX_TITLES."""
    if job_title not in SUFFIX_TITLES or not organization:
        return job_title
    han_chars = HAN_PATTERN.findall(organization)
    if han_chars:
        return han_chars[-1] + job_title
    return job_title


def clean_org(org):
    """Strip whitespace, null if length < 2 or > 15."""
    if not org:
        return None
    org = re.sub(r"\s+", "", str(org))
    if len(org) < 2 or len(org) > 15:
        return None
    return org


# ==========================================
# E. FAMILY BIRTH YEAR CROSS-VALIDATION
# ==========================================
def validate_family_birthyear(family_birth_year, subject_birth_year, relation):
    """Cross-validate family member birth year against subject."""
    if family_birth_year is None or subject_birth_year is None or not relation:
        return family_birth_year

    # Parents must be born STRUCT_FAMILY_AGE_GAP+ years before subject
    if re.search(r"[父母]", relation):
        if family_birth_year > (subject_birth_year - STRUCT_FAMILY_AGE_GAP):
            return None

    # Children must be born STRUCT_FAMILY_AGE_GAP+ years after subject
    if re.search(r"[女男子]", relation):
        if family_birth_year < (subject_birth_year + STRUCT_FAMILY_AGE_GAP):
            return None

    # Spouse within STRUCT_SPOUSE_AGE_GAP years
    if re.search(r"妻", relation):
        if (family_birth_year > (subject_birth_year + STRUCT_SPOUSE_AGE_GAP) or
                family_birth_year < (subject_birth_year - STRUCT_SPOUSE_AGE_GAP)):
            return None

    return family_birth_year


# ==========================================
# F. GENDER IDENTIFICATION
# ==========================================
def _classify_gender_qwen(names):
    """Classify gender for non-Japanese names via Qwen/Ollama. Returns list of 'm'/'f'/'x'."""
    results = []
    cache_path = os.path.join(STRUCT_OUTPUT_DIR, "_cache_gender_qwen.json")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

    todo = [(i, n) for i, n in enumerate(names) if n not in cache]
    logger.info(f"Qwen gender: {len(names)} names, {len(names) - len(todo)} cached, {len(todo)} remaining")

    for batch_idx, (i, name) in enumerate(todo, 1):
        prompt = (
            "You are classifying the gender of a historical East Asian person (early 20th century) "
            "based on their Chinese/Korean name. Many of these individuals held public positions, "
            "so the vast majority are male. Only classify as female if the name is clearly and "
            "unambiguously a female name.\n"
            f"Name: {name}\n"
            "Reply with exactly one character: M or F or X (unknown)."
        )
        try:
            resp = requests.post(OLLAMA_CHAT_URL, json={
                "model": BIO_MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You classify gender from East Asian names. Reply with exactly one character: M, F, or X."},
                    {"role": "user", "content": prompt},
                ],
                "think": False,
                "stream": False,
                "options": {"temperature": 0.0, "top_k": 1, "num_predict": 4},
            })
            resp.raise_for_status()
            answer = resp.json()["message"]["content"].strip().upper()
            if answer.startswith("M"):
                cache[name] = "m"
            elif answer.startswith("F"):
                cache[name] = "f"
            else:
                cache[name] = "x"
        except Exception as e:
            logger.error(f"  Qwen gender {batch_idx}/{len(todo)} failed: {e}")
            cache[name] = "x"

        if batch_idx % 100 == 0 or batch_idx == len(todo):
            logger.info(f"  Qwen gender: {batch_idx}/{len(todo)} classified")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)

    return [cache.get(n, "x") for n in names]


def classify_gender(person_core, person_family, person_education, person_career, person_ranks):
    """Add gender field to person_core and person_family using biographical signals.

    Strategy: name-based ML is a weak prior only. Biographical indicators override:
      - Has 妻 (wife) in family   → male
      - Has 夫 (husband) in family → female
      - Has military ranks         → male
      - Attended women's school (女 in institution) → female
      - Name-only female is never trusted alone → unknown
    """
    # --- Build biographical signal indices ---
    # Family relations per person
    person_relations = defaultdict(list)
    for fm in person_family:
        if fm.get("relation"):
            person_relations[fm["person_id"]].append(fm["relation"])

    # Persons with military ranks
    persons_with_ranks = {r["person_id"] for r in person_ranks}

    # Persons who attended women's schools (女 in institution name)
    persons_womens_school = set()
    for edu in person_education:
        inst = edu.get("institution") or ""
        if "女" in inst:
            persons_womens_school.add(edu["person_id"])

    # Persons who worked at women's orgs (weaker signal — they could be male staff)
    persons_womens_work = set()
    for job in person_career:
        org = job.get("organization") or ""
        if "女" in org:
            persons_womens_work.add(job["person_id"])

    # --- Name-based classification (weak prior) ---
    ja_indices = [i for i, c in enumerate(person_core) if c.get("domain") == "ja"]
    non_ja_indices = [i for i, c in enumerate(person_core) if c.get("domain") != "ja"]

    logger.info(f"Classifying gender for {len(person_core)} subjects "
                f"({len(ja_indices)} ja via gendec, {len(non_ja_indices)} non-ja via Qwen)")

    # Japanese: gendec (weak prior only)
    name_prior = {}  # person_id → 'm'/'f'/'x' from name model
    if ja_indices:
        ja_names = [person_core[i]["name"] for i in ja_indices]
        classifier = hf_pipeline("text-classification", model=STRUCT_GENDER_MODEL)
        predictions = classifier(ja_names, batch_size=64)
        for idx, pred in zip(ja_indices, predictions):
            label = pred["label"]
            score = pred["score"]
            pid = person_core[idx]["person_id"]
            if label == "LABEL_0" and score > STRUCT_GENDER_THRESHOLD and len(person_core[idx]["name"]) > 2:
                name_prior[pid] = "m"
            elif label == "LABEL_1" and score > STRUCT_GENDER_THRESHOLD and len(person_core[idx]["name"]) > 2:
                name_prior[pid] = "f"
            else:
                name_prior[pid] = "x"

    # Non-Japanese: Qwen (weak prior only)
    if non_ja_indices:
        non_ja_names = [person_core[i]["name"] for i in non_ja_indices]
        qwen_genders = _classify_gender_qwen(non_ja_names)
        for idx, gender in zip(non_ja_indices, qwen_genders):
            name_prior[person_core[idx]["person_id"]] = gender

    # --- Resolve final gender using biographical signals ---
    bio_override_counts = {"male_wife": 0, "female_husband": 0, "male_military": 0,
                           "female_school": 0, "name_female_demoted": 0}
    for core in person_core:
        pid = core["person_id"]
        relations = person_relations.get(pid, [])
        has_wife = any("妻" in r for r in relations)
        has_husband = any("夫" in r for r in relations)
        has_ranks = pid in persons_with_ranks
        attended_womens_school = pid in persons_womens_school
        prior = name_prior.get(pid, "x")

        # Strong biographical signals (override name)
        if has_wife and not has_husband:
            core["gender"] = "m"
            if prior == "f":
                bio_override_counts["male_wife"] += 1
        elif has_husband and not has_wife:
            core["gender"] = "f"
            if prior == "m":
                bio_override_counts["female_husband"] += 1
        elif has_ranks:
            core["gender"] = "m"
            if prior == "f":
                bio_override_counts["male_military"] += 1
        elif attended_womens_school:
            # Attended women's school is a strong female indicator
            core["gender"] = "f"
            bio_override_counts["female_school"] += 1
        elif prior == "m":
            # Name says male — trust it (male is the overwhelming majority)
            core["gender"] = "m"
        else:
            # Name says female or unknown, but no biographical confirmation → unknown
            # Name-only female is NOT trusted in this dataset
            if prior == "f":
                bio_override_counts["name_female_demoted"] += 1
            core["gender"] = "x"

    logger.info(f"Gender bio overrides: {bio_override_counts}")

    # Family member gender from relation field
    male_markers = re.compile(r"父|兄|弟|祖父|婿|男$")
    female_markers = re.compile(r"母|妻|姉|妹|祖母|女$")
    for fm in person_family:
        rel = fm.get("relation") or ""
        if male_markers.search(rel):
            fm["gender"] = "m"
        elif female_markers.search(rel):
            fm["gender"] = "f"
        else:
            fm["gender"] = "x"

    m_count = sum(1 for c in person_core if c["gender"] == "m")
    f_count = sum(1 for c in person_core if c["gender"] == "f")
    logger.info(f"Gender subjects: {m_count} male, {f_count} female, {len(person_core)-m_count-f_count} unknown")
    ja_f = sum(1 for i in ja_indices if person_core[i]["gender"] == "f")
    non_ja_f = sum(1 for i in non_ja_indices if person_core[i]["gender"] == "f")
    logger.info(f"  (ja female: {ja_f}, non-ja female: {non_ja_f})")
    fm_m = sum(1 for fm in person_family if fm["gender"] == "m")
    fm_f = sum(1 for fm in person_family if fm["gender"] == "f")
    logger.info(f"Gender family: {fm_m} male, {fm_f} female, {len(person_family)-fm_m-fm_f} unknown")


# ==========================================
# G. GEOLOCATION
# ==========================================
CJK_CHAR = re.compile(r"[\u4E00-\u9FFF]")
# CJK + katakana ケ(U+30B1)/ヶ(U+30F6) — these appear in Japanese place names
# as connective particles (龍ケ崎, 茅ヶ崎, 霞ヶ関, 関ヶ原, etc.) and must be
# preserved during prefix extraction so lookups hit the correct GeoNames entry
# instead of falling through to a shorter, wrong-country match.
_PLACE_CHAR = re.compile(r"[\u4E00-\u9FFF\u30B1\u30F6]")

FEATURE_CODE_PRIORITY = {"ADM1": 0, "ADM2": 1, "ADM3": 2, "PPLC": 3, "PPL": 4, "PPLA": 3, "PPLA2": 3}


def load_geonames_index():
    """Load GeoNames files into a reverse CJK index."""
    index = defaultdict(list)  # kanji_token → [(geonameid, lat, lon, feature_code, country_code)]
    total = 0
    for filename in STRUCT_GEONAMES_FILES:
        if not os.path.exists(filename):
            logger.warning(f"GeoNames file not found: {filename}")
            continue
        with open(filename, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 8:
                    continue
                feature_class = row[6]
                if feature_class not in STRUCT_GEONAMES_CLASSES:
                    continue
                geonameid = int(row[0])
                lat = float(row[4])
                lon = float(row[5])
                feature_code = row[7]
                country_code = filename.replace(".txt", "")  # JP, TW, etc.
                alternatenames = row[3]
                entry = (geonameid, lat, lon, feature_code, country_code)
                for token in alternatenames.split(","):
                    token = token.strip()
                    if token and CJK_CHAR.search(token):
                        index[token].append(entry)
                        total += 1
    # Override: GeoNames 9930571 (新京/Xinjing) has wrong coords (Guangxi)
    # Correct location: Hsinking, capital of Manchukuo = modern Changchun, Jilin
    GEONAMES_OVERRIDES = {
        9930571: (43.88, 125.32),  # 新京特別市 → Changchun
    }
    overridden = 0
    for token, entries in index.items():
        new_entries = []
        for gid, lat, lon, fcode, cc in entries:
            if gid in GEONAMES_OVERRIDES:
                lat, lon = GEONAMES_OVERRIDES[gid]
                overridden += 1
            new_entries.append((gid, lat, lon, fcode, cc))
        index[token] = new_entries
    if overridden:
        logger.info(f"GeoNames overrides: corrected {overridden} entries")

    # Synthetic entries for Manchukuo/Mengjiang historical place names not in GeoNames.
    # Without these, names like "興安西省" match only a Taiwanese township (wrong).
    # Coordinates are approximate capital / center of each historical province.
    # geonameid=0 signals a synthetic entry (no real GeoNames record).
    HISTORICAL_PLACES = {
        "興安西省": (0, 44.0, 118.5, "ADM1", "CN"),   # Kailu, Inner Mongolia
        "興安北省": (0, 49.2, 119.7, "ADM1", "CN"),   # Hailar
        "興安東省": (0, 48.0, 122.7, "ADM1", "CN"),   # Zhalantun
        "興安南省": (0, 46.1, 122.1, "ADM1", "CN"),   # Wangyemiao (Ulanhot)
        "興安總省": (0, 46.0, 121.0, "ADM1", "CN"),   # Xing'an center
        "厚和":     (0, 40.84, 111.75, "ADM2", "CN"), # Hohhot wartime name
        "厚和市":   (0, 40.84, 111.75, "ADM2", "CN"),
        "厚和特別市": (0, 40.84, 111.75, "ADM2", "CN"),
        "三江省":   (0, 46.8, 130.4, "ADM1", "CN"),   # Jiamusi, Heilongjiang
        "綏遠":     (0, 40.84, 111.75, "ADM1", "CN"), # Suiyuan = Hohhot area
        "綏遠省":   (0, 40.84, 111.75, "ADM1", "CN"),
        "察哈爾":   (0, 40.82, 114.88, "ADM1", "CN"), # Chahar = Zhangjiakou area
        "察哈爾省": (0, 40.82, 114.88, "ADM1", "CN"),
        "蒙疆":     (0, 40.84, 111.75, "ADM1", "CN"), # Mengjiang = Inner Mongolia
        # Jiandao / Gando (間島省) — GeoNames has 間島 as Mashima, Niigata
        "間島省":   (0, 42.88, 129.47, "ADM1", "CN"), # Yanji, Jilin
        # Andong (安東) — GeoNames has 安東 as Andō, Hiroshima
        "安東省":   (0, 40.12, 124.39, "ADM1", "CN"), # Dandong, Liaoning
        "安東縣":   (0, 40.12, 124.39, "ADM2", "CN"),
        "安東県":   (0, 40.12, 124.39, "ADM2", "CN"), # simplified kanji
        "安東郡":   (0, 40.12, 124.39, "ADM2", "CN"),
        "安東邑":   (0, 40.12, 124.39, "ADM3", "CN"),
        "安東地方": (0, 40.12, 124.39, "ADM2", "CN"),
        # Shanhaiguan — GeoNames matched to Taiwan
        "山海關":   (0, 40.00, 119.77, "ADM3", "CN"), # Hebei
    }
    added = 0
    for name, entry in HISTORICAL_PLACES.items():
        if name not in index:
            index[name] = [entry]
            added += 1
    if added:
        logger.info(f"GeoNames index: added {added} historical Manchukuo/Mengjiang entries")

    logger.info(f"GeoNames index: {len(index)} CJK tokens, {total} entries")
    return dict(index)


def load_mcgd_index():
    """Load MCGD CSV into a CJK name index."""
    index = defaultdict(list)  # CJK name → [(lat, lon, prov_zh, locid)]
    if not os.path.exists(STRUCT_MCGD_PATH):
        logger.warning(f"MCGD file not found: {STRUCT_MCGD_PATH}")
        return dict(index)
    with open(STRUCT_MCGD_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row.get("Code", "")
            if code not in STRUCT_MCGD_CODES:
                continue
            name = row.get("Name", "").strip()
            if not name or not CJK_CHAR.search(name):
                continue
            try:
                lat = float(row["LAT"])
                lon = float(row["LONG"])
            except (ValueError, KeyError):
                continue
            prov_zh = row.get("Prov_Zh", "")
            locid = row.get("LocID", "")
            index[name].append((lat, lon, prov_zh, locid))
    logger.info(f"MCGD index: {len(index)} CJK place names")
    return dict(index)


def extract_place_prefixes(place):
    """Generate progressively shorter prefixes from a place string."""
    if not place:
        return []
    # Keep CJK characters + ケ/ヶ (Japanese place-name particles)
    chars = _PLACE_CHAR.findall(place)
    if not chars:
        return []
    full = "".join(chars)
    prefixes = []
    for length in range(len(full), 1, -1):
        prefixes.append(full[:length])
    return prefixes


_JA_ADMIN_PATTERN = re.compile(
    r"郡.{1,10}(?:町|村)"  # 郡+町/村 (gun + machi/mura)
    r"|[區区]"             # 區/区 (Japanese ward — bare match, no 町 required)
    r"|丁目"               # chōme block numbering
    r"|番地"               # banchi lot numbering
    r"|番町"               # banchō
)


def match_place(place, origin_place, geonames_idx, mcgd_idx):
    """Match a place string to coordinates. Returns (lat, lon, geonameid, mcgd_locid, province) or None."""
    if not place:
        return None
    prefixes = extract_place_prefixes(place)
    origin_cjk = "".join(CJK_CHAR.findall(origin_place)) if origin_place else ""

    # Detect preferred country from origin_place
    preferred_cc = None
    if origin_cjk:
        if re.search(r'臺灣|台灣|台北', origin_cjk):
            preferred_cc = "TW"
        elif re.search(r'朝鮮|韓', origin_cjk):
            preferred_cc = "KR"
        elif re.search(r'中國|中国|上海|北京|南京|廣東', origin_cjk):
            preferred_cc = "CN"

    # Detect Japanese admin patterns in the place name itself
    # (郡+町/村, 區+町, 丁目, 番地 are uniquely Japanese admin structures)
    place_implies_jp = bool(_JA_ADMIN_PATTERN.search(place))

    # Try GeoNames first
    for prefix in prefixes:
        # Try original prefix, then 區→区 normalized (GeoNames uses modern 区)
        candidates = geonames_idx.get(prefix)
        if not candidates and "區" in prefix:
            candidates = geonames_idx.get(prefix.replace("區", "区"))
        if not candidates:
            continue

        # If place has Japanese admin patterns, filter to JP candidates only.
        # Skip this prefix if no JP candidates — try shorter prefixes instead
        # of matching to a wrong country.
        if place_implies_jp:
            jp_candidates = [c for c in candidates if c[4] == "JP"]
            if jp_candidates:
                candidates = jp_candidates
            else:
                continue  # no JP entry at this prefix — try shorter

        # Sort by feature code priority (lower = better)
        scored = []
        for gid, lat, lon, fcode, cc in candidates:
            prio = FEATURE_CODE_PRIORITY.get(fcode, 5)
            # Strong bonus if origin indicates a specific country
            if preferred_cc and cc == preferred_cc:
                prio -= 1.5
            elif origin_cjk and cc == "JP":
                prio -= 0.1  # slight default preference for JP
            scored.append((prio, gid, lat, lon, fcode, cc))
        scored.sort(key=lambda x: x[0])
        best = scored[0]
        return (best[2], best[3], best[1], None, None)

    # Try MCGD
    for prefix in prefixes:
        candidates = mcgd_idx.get(prefix)
        if not candidates:
            continue
        # Prefer candidate whose province overlaps with origin
        if origin_cjk and len(candidates) > 1:
            for lat, lon, prov, locid in candidates:
                if prov and prov in origin_cjk:
                    return (lat, lon, None, locid, None)
        # Take first
        lat, lon, prov, locid = candidates[0]
        return (lat, lon, None, locid, None)

    return None


def build_locations(person_core, person_career, geonames_idx, mcgd_idx,
                    person_family=None, person_family_career=None):
    """Match all place fields, build locations.jsonl, assign location_ids."""
    # Collect all (place, origin_place) pairs to match
    place_results = {}  # place_text → match result

    def _str(v):
        """Coerce a value to str (LLM sometimes returns lists/dicts)."""
        if isinstance(v, list):
            return "、".join(str(x) for x in v) if v else None
        if isinstance(v, dict):
            return "、".join(str(x) for x in v.values() if x) or None
        return v

    all_places = set()
    for core in person_core:
        core["place"] = _str(core.get("place"))
        core["origin_place"] = _str(core.get("origin_place"))
        if core.get("place"):
            all_places.add((core["place"], core.get("origin_place")))
        if core.get("origin_place"):
            all_places.add((core["origin_place"], core.get("origin_place")))
    for career in person_career:
        career["place_name"] = _str(career.get("place_name"))
        if career.get("place_name"):
            # Find the person's origin for context
            all_places.add((career["place_name"], None))
    # Family member places
    for fm in (person_family or []):
        fm["place"] = _str(fm.get("place"))
        if fm.get("place"):
            all_places.add((fm["place"], None))
    for fc in (person_family_career or []):
        fc["place_name"] = _str(fc.get("place_name"))
        if fc.get("place_name"):
            all_places.add((fc["place_name"], None))

    # Match each unique place
    logger.info(f"Matching {len(all_places)} unique places against GeoNames/MCGD...")
    for match_idx, (place, origin) in enumerate(all_places, 1):
        if match_idx % 5000 == 0:
            logger.info(f"  Place matching: {match_idx}/{len(all_places)}")
        if place not in place_results:
            place_results[place] = match_place(place, origin, geonames_idx, mcgd_idx)

    # Build deduplicated location table (round to 0.1° for grouping)
    loc_key_to_id = {}  # (rounded_lat, rounded_lon, name_cjk) → location_id
    locations = []  # list of location dicts
    loc_counter = 0

    def get_location_id(place_text):
        if not place_text or place_text not in place_results or place_results[place_text] is None:
            return None
        lat, lon, geonameid, mcgd_locid, province = place_results[place_text]
        name_cjk = "".join(_PLACE_CHAR.findall(place_text))
        key = (round(lat, 1), round(lon, 1), name_cjk)
        if key in loc_key_to_id:
            return loc_key_to_id[key]
        nonlocal loc_counter
        loc_counter += 1
        lid = f"L{loc_counter}"
        loc_key_to_id[key] = lid
        locations.append({
            "location_id": lid,
            "name": place_text,
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "geonameid": geonameid,
            "mcgd_locid": mcgd_locid,
            "province": province,
        })
        return lid

    # Assign to person_core
    for core in person_core:
        core["location_id"] = get_location_id(core.get("place"))
        core["origin_location_id"] = get_location_id(core.get("origin_place"))

    # Assign to person_career
    for career in person_career:
        career["location_id"] = get_location_id(career.get("place_name"))

    # Assign to family members
    for fm in (person_family or []):
        fm["location_id"] = get_location_id(fm.get("place"))
    for fc in (person_family_career or []):
        fc["location_id"] = get_location_id(fc.get("place_name"))

    logger.info(f"Geolocation: matched {sum(1 for v in place_results.values() if v)} / {len(place_results)} places, {len(locations)} unique locations")
    return locations


_GEOJSON_COUNTRY_MAP = {
    "japan_hijmans.geojson": "ja",
    "china_1928.geojson": "zh",
    "korea_imperial.geojson": "ko",
    "taiwan_1946.geojson": "zh",
}


def fill_provinces(locations):
    """Assign province and country to each location via spatial join with GeoJSON boundaries."""
    from shapely.geometry import shape, Point
    from shapely.prepared import prep

    polygons = []  # list of (prepared_geom, province_name, country_code)
    for filename, prop_key in STRUCT_GEOJSON_FILES:
        path = os.path.join(STRUCT_GEOJSON_DIR, filename)
        cc = _GEOJSON_COUNTRY_MAP.get(filename, "ja")
        with open(path) as f:
            geojson = json.load(f)
        for feature in geojson["features"]:
            geom = prep(shape(feature["geometry"]))
            name = feature["properties"][prop_key]
            polygons.append((geom, name, cc))

    filled = 0
    for loc in locations:
        lat, lon = loc.get("latitude"), loc.get("longitude")
        if lat is None or lon is None:
            continue
        pt = Point(lon, lat)
        for geom, prov_name, cc in polygons:
            if geom.contains(pt):
                loc["province"] = prov_name
                loc["country"] = cc
                filled += 1
                break
    logger.info(f"Spatial join: assigned province to {filled}/{len(locations)} locations")


def resolve_domain_from_spatial(person_core, locations_table, person_career=None):
    """Override domain using the spatial country of origin_location_id.

    The text-based detect_domain() is a preliminary guess. The spatial join
    against historical GeoJSON boundaries is authoritative: if origin_place
    geolocates inside china_1928.geojson, the person is Chinese, not Japanese.
    When domain changes, re-split the name accordingly.

    Second pass: for persons still "ja" with no origin_location_id, check if
    all career locations map to a single non-ja country.
    """
    loc_country = {loc["location_id"]: loc.get("country") for loc in locations_table}

    def _apply_override(core, new_domain):
        core["domain"] = new_domain
        name_family, name_given = split_name(core["name"], new_domain)
        core["name_family"] = name_family
        core["name_given"] = name_given

    # Pass 1: origin-based override
    updated_origin = 0
    for core in person_core:
        origin_lid = core.get("origin_location_id")
        if not origin_lid:
            continue
        spatial_country = loc_country.get(origin_lid)
        if not spatial_country or spatial_country == core.get("domain"):
            continue
        if core.get("domain") == "other":
            continue
        _apply_override(core, spatial_country)
        updated_origin += 1

    # Pass 2: career-location override for remaining "ja" without origin
    updated_career = 0
    if person_career:
        career_locs_by_pid = defaultdict(set)
        for c in person_career:
            lid = c.get("location_id")
            if lid and lid in loc_country:
                career_locs_by_pid[c["person_id"]].add(loc_country[lid])

        for core in person_core:
            if core.get("domain") != "ja" or core.get("origin_location_id"):
                continue
            countries = career_locs_by_pid.get(core["person_id"], set())
            countries.discard("ja")
            countries.discard(None)
            if len(countries) == 1:
                _apply_override(core, countries.pop())
                updated_career += 1

    logger.info(f"Spatial domain: updated {updated_origin} (origin-based) + "
                f"{updated_career} (career-based) persons")
    return updated_origin + updated_career


def fix_zh_onechar_surnames(person_core):
    """Fix zh-domain persons whose 1-char surname is not a recognized Chinese
    surname. Re-classify as ja, re-split with namedivider, then Ollama fallback.

    These are Japanese names that were wrongly assigned zh domain (e.g. by the
    career-location spatial override pointing to a mis-geolocated location).
    """
    # Pass 1: namedivider (fast, no LLM)
    fixed_nd = 0
    ollama_candidates = []
    for core in person_core:
        if core.get("domain") != "zh":
            continue
        nf = core.get("name_family") or ""
        if len(nf) != 1 or nf in COMMON_ZH_SURNAMES:
            continue
        name = core.get("name") or ""
        if len(name) < 2:
            continue
        try:
            result = name_divider.divide_name(name)
        except Exception:
            result = None
        if result and len(result.family) > 1:
            core["domain"] = "ja"
            core["name_family"] = result.family
            core["name_given"] = result.given
            fixed_nd += 1
        else:
            # namedivider couldn't help — queue for Ollama
            fl = core.get("name_family_latin") or ""
            if len(name) >= 3 and len(fl) <= 2:
                ollama_candidates.append(core)
    logger.info(f"zh 1-char surname fix: {fixed_nd} reclassified via namedivider")

    # Pass 2: Ollama fallback for remaining candidates
    if not ollama_candidates:
        return fixed_nd
    logger.info(f"zh 1-char surname fix: {len(ollama_candidates)} candidates for Ollama")
    fixed_llm = 0
    for i, core in enumerate(ollama_candidates, 1):
        name = core["name"]
        result = _ollama_split_romanize(name)
        if result is None:
            continue
        fam, giv, fl, gl = result
        if len(fam) <= 1:
            continue
        core["domain"] = "ja"
        core["name_family"] = fam
        core["name_given"] = giv
        core["name_family_latin"] = fl
        core["name_given_latin"] = gl
        fixed_llm += 1
        if i % 50 == 0:
            logger.info(f"  Ollama: {i}/{len(ollama_candidates)} processed, {fixed_llm} fixed")
    logger.info(f"zh 1-char surname fix: {fixed_llm} reclassified via Ollama")
    return fixed_nd + fixed_llm


def _ollama_split_romanize(name):
    """Ask Ollama to split a Japanese name and romanize it.

    Returns (family, given, family_latin, given_latin) or None on failure.
    """
    prompt = (
        "This is a Japanese person's full name written in kanji. "
        "Split it into family name (surname) and given name, then romanize "
        "using modified Hepburn. Return a JSON object with keys: "
        '"family", "given", "family_latin", "given_latin".\n'
        f"Name: {name}\n"
        "Output JSON only, no explanation."
    )
    try:
        resp = requests.post(OLLAMA_CHAT_URL, json={
            "model": BIO_MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a Japanese name analysis engine. Output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "think": False,
            "stream": False,
            "options": {"temperature": 0.0, "top_k": 1, "num_predict": BIO_MAX_TOKENS},
        }, timeout=30)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        obj_match = re.search(r"\{.*\}", content, re.DOTALL)
        if not obj_match:
            return None
        parsed = json.loads(obj_match.group(0))
        fam = parsed.get("family") or ""
        giv = parsed.get("given") or ""
        fl = parsed.get("family_latin") or ""
        gl = parsed.get("given_latin") or ""
        if fam + giv != name and fam + giv != name.replace(" ", ""):
            return None
        if not fl or not fl[0].isupper():
            return None
        return fam, giv, fl, gl
    except Exception as e:
        logger.debug(f"Ollama failed for {name}: {e}")
        return None


def parse_admin_hierarchy(locations):
    """Parse CJK place names into hierarchical admin levels."""
    ADMIN1_SUFFIXES = r'[都道府県縣省]'
    ADMIN_SUFFIX_CHAR = re.compile(r'[都道府県縣省市郡區区町村]$')
    STREET_PATTERN = re.compile(
        r'第\d+番|番地|丁目|通り|ストリート|No\.\s*\d|Street|号|條|路\d|巷'
    )

    def _parse(name):
        name = re.sub(r'[（(].*?[）)]', '', name)
        name = re.sub(r'[\d一二三四五六七八九十百千]+$', '', name)
        remaining = name
        levels = []

        # L1: prefecture/province (2+ chars before suffix to avoid splitting 京都)
        m = re.match(r'^(.{2,}?' + ADMIN1_SUFFIXES + r')(.*)', remaining)
        if m:
            levels.append(m.group(1))
            remaining = m.group(2)
            l2_pat = r'[市郡縣県]'  # 縣/県 = county after province
        else:
            l2_pat = r'[市郡]'

        # L2: city/county
        m = re.match(r'^(.+?' + l2_pat + r')(.*)', remaining)
        if m:
            levels.append(m.group(1))
            remaining = m.group(2)

        # L3: ward/district
        m = re.match(r'^(.+?[區区])(.*)', remaining)
        if m:
            levels.append(m.group(1))
            remaining = m.group(2)

        # L4: town/village
        m = re.match(r'^(.+?[町村])(.*)', remaining)
        if m:
            levels.append(m.group(1))
            remaining = m.group(2)

        if not levels:
            levels = [name]

        return levels

    parsed = 0
    for loc in locations:
        levels = _parse(loc['name'])
        # Sanity check: truncate levels at street-level detail
        for i, lvl in enumerate(levels):
            if STREET_PATTERN.search(lvl) or len("".join(CJK_CHAR.findall(lvl))) > 15:
                levels = levels[:i]  # keep only levels above the street-level one
                break
        loc['admin1'] = levels[0] if len(levels) > 0 else None
        loc['admin2'] = levels[1] if len(levels) > 1 else None
        loc['admin3'] = levels[2] if len(levels) > 2 else None
        # Normalized admin1: strip suffix for cross-variant queries
        loc['admin1_norm'] = ADMIN_SUFFIX_CHAR.sub('', levels[0]) if levels else None
        if len(levels) > 0:
            parsed += 1

    logger.info(f"Admin hierarchy: parsed {parsed}/{len(locations)} locations")


# ==========================================
# H. ORG / JOB TITLE TRANSLATION
# ==========================================
TRANSLATE_BATCH_SIZE = 10

_TRANSLATE_SYSTEM_PROMPTS = {
    "job title": (
        "You are a translation engine for historical East Asian job titles (1900s-1940s). "
        "For each numbered input, output ONLY the number and the English translation on one line. "
        "Translate the function of the title, not a phonetic transcription. "
        "Example:\n1. Secretary\n2. Section Chief"
    ),
    "organization": (
        "You are a translation engine for historical East Asian organization names (1900s-1940s). "
        "For each numbered input, output ONLY the number and the English translation on one line. "
        "Use established English spellings where they exist "
        "(e.g., 三菱 → Mitsubishi, ハーバード → Harvard). "
        "Translate generic components like 銀行 → Bank, 大学 → University, 株式会社 → Co. Ltd. "
        "Example:\n1. Mitsubishi Bank\n2. Tokyo Imperial University"
    ),
    "location": (
        "You are a translation engine for historical East Asian place names (1900s-1940s). "
        "For each numbered input, output ONLY the number and the English translation on one line. "
        "If the name contains katakana, identify the real-world place "
        "(e.g., ベルリン → Berlin, ロンドン → London). Use the established English name. "
        "Example:\n1. Tokyo\n2. Berlin"
    ),
    "administrative region": (
        "You are a translation engine for historical East Asian administrative region names (1900s-1940s). "
        "For each numbered input, output ONLY the number and the English translation on one line. "
        "If the name contains katakana, identify the real-world place "
        "(e.g., ベルリン → Berlin, ロンドン → London). Use the established English name. "
        "Example:\n1. Tokyo Prefecture\n2. Gyeonggi Province"
    ),
}
_TRANSLATE_SYSTEM_DEFAULT = (
    "You are a translation engine for historical East Asian names (1900s-1940s). "
    "For each numbered input, output ONLY the number and the English translation on one line. "
    "Example:\n1. Tokyo\n2. Berlin"
)


def translate_batch_ollama(items, item_type, batch_size=None):
    """Translate a list of CJK strings to English via Ollama. Returns dict name→english.

    Sends batches of items per LLM call for efficiency.
    """
    if not items:
        return {}
    unique = list(set(items))
    bs = batch_size or TRANSLATE_BATCH_SIZE

    # Load cache
    cache_path = os.path.join(STRUCT_OUTPUT_DIR, f"_cache_translate_{item_type.replace(' ', '_')}.json")
    results = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            results = json.load(f)

    todo = [name for name in unique if name not in results]
    logger.info(f"Translating {item_type}: {len(unique)} unique, {len(unique) - len(todo)} cached, {len(todo)} remaining")
    if not todo:
        return results

    system_prompt = _TRANSLATE_SYSTEM_PROMPTS.get(item_type, _TRANSLATE_SYSTEM_DEFAULT)
    num_batches = (len(todo) + bs - 1) // bs

    for batch_idx in range(num_batches):
        start = batch_idx * bs
        batch = todo[start:start + bs]

        # Build numbered prompt
        user_content = "\n".join(f"{i}. {name}" for i, name in enumerate(batch, 1))

        try:
            resp = requests.post(OLLAMA_CHAT_URL, json={
                "model": BIO_MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "think": False,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "top_k": 1,
                    "num_predict": len(batch) * 40,
                },
            })
            resp.raise_for_status()
            content = resp.json()["message"].get("content", "")

            # Parse numbered responses
            for line in content.splitlines():
                m = re.match(r"(\d+)\.\s*(.+)", line)
                if m:
                    idx = int(m.group(1)) - 1
                    if 0 <= idx < len(batch):
                        t = m.group(2).strip()
                        t = re.sub(r"[^\x20-\x7E]", "", t)  # keep ASCII printable
                        t = re.sub(r"\s+", " ", t).strip()
                        if t and len(t) <= STRUCT_ORG_TRANSLATION_MAX_LEN:
                            results[batch[idx]] = t.title() if not t[0].isupper() else t
                        else:
                            results[batch[idx]] = None

        except Exception as e:
            logger.error(f"  {item_type} batch {batch_idx + 1}/{num_batches} failed: {e}")

        # Mark any unparsed items in this batch as None
        for name in batch:
            if name not in results:
                results[name] = None

        # Save cache periodically
        done = min((batch_idx + 1) * bs, len(todo))
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
            logger.info(f"  {item_type}: {done}/{len(todo)} translated")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)

    return results


# ==========================================
# I. NAME ROMANIZATION
# ==========================================
# Tone mark stripping for Pinyin post-processing
TONE_MAP = str.maketrans(
    "āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜĀÁǍÀĒÉĚÈĪÍǏÌŌÓǑÒŪÚǓÙǕǗǙǛ",
    "aaaaeeeeiiiioooouuuuüüüüAAAAEEEEIIIIOOOOUUUUÜÜÜÜ",
)

# CJK + kana character set for sanitizing name parts (exclude middle dot ・ U+30FB)
_CJK_KANA_RE = re.compile(r"[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FA\u30FC-\u30FFー]")

_kakasi = pykakasi.kakasi()


def _sanitize_name_part(text):
    """Strip non-CJK/non-kana characters from a name part."""
    if not text:
        return text
    return "".join(_CJK_KANA_RE.findall(text)) or None


def _romanize_pykakasi(family, given):
    """Romanize Japanese name via pykakasi. Returns dict or None if suspicious."""
    result = {}
    for field, text in [("family", family), ("given", given)]:
        if not text:
            continue
        items = _kakasi.convert(text)
        hepburn = "".join(item["hepburn"] for item in items)
        # Suspicious: empty, passthrough, non-ASCII, or unreasonably long
        if not hepburn or hepburn == text or not hepburn.isascii() or len(hepburn) > 20:
            return None
        # Any chunk passed through unconverted (unknown kanji)
        if any(item["hepburn"] == item["orig"] for item in items):
            return None
        result[field] = hepburn.capitalize()
    return result if result else None


def _domain_instruction(domain, origin_place=None):
    """Return romanization prompt instructions for a given domain."""
    if domain == "zh":
        inst = (
            "This is a Chinese name. Use Pinyin romanization WITHOUT tone marks. "
            "Write the given name syllables joined together without spaces "
            "(e.g., 'Mingwei' not 'Ming Wei'). Capitalize the first letter of each part."
        )
    elif domain == "ko":
        inst = (
            "This is a Korean name. Use Revised Romanization of Korean. "
            "Capitalize the first letter of each part."
        )
    elif domain == "other":
        inst = (
            "This name is written in katakana and likely represents a non-Japanese, "
            "non-Chinese person (e.g., a Western name). Transliterate the katakana "
            "to the most likely original Latin-alphabet spelling. Consider English "
            "and German origins especially. For example, アインシュタイン → Einstein, "
            "シュミット → Schmidt. Return the original Western spelling if recognizable, "
            "otherwise a phonetic Latin transliteration."
        )
    else:  # "ja"
        inst = (
            "This is a Japanese name. Use modified Hepburn romanization with macrons "
            "(ō, ū). Capitalize the first letter of each part."
        )
    if origin_place:
        inst += (
            f"\nThe person's origin place is: {origin_place}. "
            "Consider this when choosing the romanization system."
        )
    return inst


def _romanize_ollama_single(family, given, domain, origin_place):
    """Romanize a single name via Ollama. Returns dict or None on failure."""
    entry = {}
    if family and given:
        entry["family"] = family
        entry["given"] = given
    elif family:
        entry["given"] = family
    elif given:
        entry["given"] = given

    prompt = (
        f"Romanize this name. {_domain_instruction(domain, origin_place)}\n"
        "Return a JSON object.\n"
        f"Input: {json.dumps(entry, ensure_ascii=False)}\n"
        'Output format: {"family": "Surname", "given": "Givenname"}\n'
        "If only a given name is provided, return only the given field.\n"
    )
    resp = requests.post(OLLAMA_CHAT_URL, json={
        "model": BIO_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a name romanization engine. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "think": False,
        "stream": False,
        "options": {"temperature": 0.0, "top_k": 1, "num_predict": BIO_MAX_TOKENS},
    })
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    obj_match = re.search(r"\{.*\}", content, re.DOTALL)
    if not obj_match:
        raise ValueError("No JSON object in response")
    parsed = json.loads(obj_match.group(0))
    if domain == "zh":
        for field in ("family", "given"):
            val = parsed.get(field)
            if val:
                parsed[field] = val.translate(TONE_MAP)
    return {"family": parsed.get("family"), "given": parsed.get("given")}


def _romanize_ollama_batch(batch, domain):
    """Romanize a batch of names via a single Ollama call.

    batch: list of (family, given, domain, origin_place) tuples (all same domain).
    Returns dict of (family, given) → {"family": ..., "given": ...} for successful items.
    """
    entries = []
    for idx, (family, given, _dom, _op) in enumerate(batch, 1):
        entry = {"id": idx}
        if family and given:
            entry["family"] = family
            entry["given"] = given
        elif family:
            entry["given"] = family
        elif given:
            entry["given"] = given
        entries.append(entry)

    prompt = (
        f"Romanize these names. {_domain_instruction(domain)}\n"
        "Return a JSON array of objects, one per input, preserving the id field.\n"
        f"Input: {json.dumps(entries, ensure_ascii=False)}\n"
        'Output format: [{"id": 1, "family": "Surname", "given": "Givenname"}, ...]\n'
        "If only a given name is provided, return only the given field for that entry.\n"
    )
    resp = requests.post(OLLAMA_CHAT_URL, json={
        "model": BIO_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a name romanization engine. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "think": False,
        "stream": False,
        "options": {"temperature": 0.0, "top_k": 1, "num_predict": BIO_MAX_TOKENS * len(batch)},
    })
    resp.raise_for_status()
    content = resp.json()["message"]["content"]

    # Parse JSON array from response
    arr_match = re.search(r"\[.*\]", content, re.DOTALL)
    if not arr_match:
        raise ValueError("No JSON array in batch response")
    parsed_arr = json.loads(arr_match.group(0))

    # Map results by id
    results = {}
    id_to_item = {idx: batch[idx - 1] for idx in range(1, len(batch) + 1)}
    for item in parsed_arr:
        item_id = item.get("id")
        if item_id not in id_to_item:
            continue
        family, given, _d, _o = id_to_item[item_id]
        parsed_result = {"family": item.get("family"), "given": item.get("given")}
        if domain == "zh":
            for field in ("family", "given"):
                val = parsed_result.get(field)
                if val:
                    parsed_result[field] = val.translate(TONE_MAP)
        results[(family, given)] = parsed_result
    return results


def romanize_names_batch(name_pairs, batch_size=None):
    """Romanize all unique (family, given, domain, origin_place) tuples.

    Strategy:
    1. Sanitize name parts (strip non-CJK, skip 1-char total names)
    2. Try pykakasi for ja-domain names (instant, no LLM)
    3. Batch-call Ollama for remaining names (non-ja + pykakasi failures)
    Returns dict keyed by (family, given) → {"family": latin, "given": latin}
    """
    if not name_pairs:
        return {}

    unique = list(set(name_pairs))

    # Load cache (keys stored as "family\tgiven")
    cache_path = os.path.join(STRUCT_OUTPUT_DIR, "_cache_romanize.json")
    cache_raw = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_raw = json.load(f)

    results = {}
    for key_str, val in cache_raw.items():
        parts = key_str.split("\t", 1)
        if len(parts) == 2:
            f_part = None if parts[0] == "None" else parts[0]
            g_part = None if parts[1] == "None" else parts[1]
            results[(f_part, g_part)] = val

    todo = [(f, g, d, op) for f, g, d, op in unique if (f, g) not in results]
    total = len(unique)
    logger.info(f"Romanizing names: {total} unique, {total - len(todo)} cached, {len(todo)} remaining")
    if not todo:
        return results

    # --- Sanitize and filter ---
    sanitized = []
    skipped_short = 0
    for family, given, domain, origin_place in todo:
        fam_clean = _sanitize_name_part(family)
        giv_clean = _sanitize_name_part(given)
        combined_len = len(fam_clean or "") + len(giv_clean or "")
        if combined_len <= 1:
            results[(family, given)] = {"family": None, "given": None}
            skipped_short += 1
            continue
        sanitized.append((fam_clean or family, giv_clean or given, domain, origin_place,
                          family, given))  # keep originals for cache key
    if skipped_short:
        logger.info(f"  Skipped {skipped_short} names (≤1 CJK char total)")

    # --- pykakasi for ja-domain names ---
    ollama_todo = []  # (family, given, domain, origin_place, orig_family, orig_given)
    pykakasi_ok = 0
    pykakasi_fail = 0
    for fam, giv, domain, origin_place, orig_f, orig_g in sanitized:
        if domain == "ja":
            result = _romanize_pykakasi(fam, giv)
            if result:
                results[(orig_f, orig_g)] = result
                pykakasi_ok += 1
            else:
                ollama_todo.append((fam, giv, domain, origin_place, orig_f, orig_g))
                pykakasi_fail += 1
        else:
            ollama_todo.append((fam, giv, domain, origin_place, orig_f, orig_g))

    logger.info(f"  pykakasi: {pykakasi_ok} ja names resolved, {pykakasi_fail} ja need LLM fallback, "
                f"{sum(1 for _, _, d, _, _, _ in ollama_todo if d != 'ja')} non-ja names")

    if not ollama_todo:
        cache_raw = {f"{f}\t{g}": v for (f, g), v in results.items()}
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_raw, f, ensure_ascii=False)
        return results

    # --- Batched Ollama calls, grouped by domain ---
    ollama_batch_size = batch_size or 15
    by_domain = defaultdict(list)
    for item in ollama_todo:
        by_domain[item[2]].append(item)

    processed = 0
    total_ollama = len(ollama_todo)
    for domain, items in by_domain.items():
        logger.info(f"  Ollama romanizing {len(items)} '{domain}' names in batches of {ollama_batch_size}")
        for batch_start in range(0, len(items), ollama_batch_size):
            batch = items[batch_start:batch_start + ollama_batch_size]
            batch_tuples = [(f, g, d, op) for f, g, d, op, _of, _og in batch]
            try:
                batch_results = _romanize_ollama_batch(batch_tuples, domain)
                # Store successful batch results
                for f, g, d, op, orig_f, orig_g in batch:
                    key = (f, g)
                    if key in batch_results:
                        results[(orig_f, orig_g)] = batch_results[key]
                    else:
                        # Item missing from batch response — try individually
                        try:
                            results[(orig_f, orig_g)] = _romanize_ollama_single(f, g, d, op)
                        except Exception:
                            results[(orig_f, orig_g)] = {"family": None, "given": None}
            except Exception as e:
                logger.warning(f"  Batch failed ({e}), falling back to individual calls")
                for f, g, d, op, orig_f, orig_g in batch:
                    try:
                        results[(orig_f, orig_g)] = _romanize_ollama_single(f, g, d, op)
                    except Exception as e2:
                        logger.error(f"  Name failed: {e2}")
                        results[(orig_f, orig_g)] = {"family": None, "given": None}

            processed += len(batch)
            if processed % 100 < ollama_batch_size or processed == total_ollama:
                logger.info(f"  Ollama names: {processed}/{total_ollama}")
                cache_raw = {f"{f}\t{g}": v for (f, g), v in results.items()}
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache_raw, f, ensure_ascii=False)

    return results


# ==========================================
# FALSE JOB FILTER
# ==========================================
_RELATION_EXACT = frozenset({
    "前名", "舊名", "旧名",
    "長男", "二男", "三男", "四男", "五男", "六男", "七男", "八男",
    "長女", "二女", "三女", "四女", "五女", "六女",
    "養子", "養女", "嫡男", "嫡子", "庶子",
    "入夫", "婿養子", "養嗣子", "嗣子", "養妹",
    "妻", "夫", "男", "女", "��", "兄", "弟", "姉", "妹",
    "孫",
})

_LEGITIMATE_JOBS = frozenset({
    "弟子", "徒弟", "大夫", "中大夫", "農夫", "人夫", "樵夫", "工夫",
    "園子", "菓子", "高女", "芸妓",
})

_WIFE_OF_RE = re.compile(r"^.{1,10}[の之]?妻$")

# Abbreviated Japanese calendar years misextracted as job titles
_ERA_YEAR_RE = re.compile(
    r"^(?:昭和?|大正?|明治?)[一二三四五六七八九十〇\d]{1,4}年?$"
)

# Regex for place fields that end with a family relation type (not a location)
_RELATION_SUFFIX_RE = re.compile(
    r"(?:長男|二男|三男|四男|五男|六男|七男|八男"
    r"|長女|二女|三女|四女|五女|六女"
    r"|養子|養女|嫡男|嫡子|庶子"
    r"|入夫|婿養子|養嗣子|嗣子|養妹|娘婿|後裔"
    r"|[の之](?:妻|夫|男|女|子|兄|弟|姉|妹|孫))$"
)

# Organization-type suffixes that should not appear at the end of place fields
_ORG_PLACE_SUFFIXES = ("領事館", "鐵道", "鉄道", "中學", "中学", "造船部")

# University abbreviation prefixes (sorted longest first) — these leak into
# place fields via OCR/segmentation errors (e.g. "東大法科卒" as a place).
_UNI_PREFIXES = ("東北大", "東大", "北大", "京大", "阪大", "帝大", "九大", "名大", "慶大", "早大")
# First chars after uni prefix that indicate a real place name (not academic)
_UNI_SAFE_PLACE_CHARS = frozenset("阪曽曾崎久門海野浦杉")

# Kyūjitai → Shinjitai mapping for hobby normalization
_KYUJITAI_TO_SHINJITAI = {
    "衞": "衛", "國": "国", "學": "学", "會": "会",
    "齋": "斎", "齊": "斉", "澤": "沢", "濱": "浜", "邊": "辺",
    "邉": "辺", "瀨": "瀬", "廣": "広", "鐵": "鉄",
    "嶋": "島", "嶌": "島", "藏": "蔵", "澁": "渋", "條": "条",
    "德": "徳", "惠": "恵", "譽": "誉", "櫻": "桜", "龍": "竜",
    "禮": "礼", "證": "証", "醫": "医", "縣": "県", "經": "経",
    "營": "営", "壽": "寿", "實": "実", "寶": "宝", "萬": "万",
    "與": "与", "樂": "楽", "關": "関", "鑛": "鉱",
    "驛": "駅", "體": "体", "應": "応", "黑": "黒", "淺": "浅",
    "藝": "芸", "圓": "円", "歲": "歳", "遞": "逓", "豐": "豊",
    "參": "参", "獻": "献", "雜": "雑", "錢": "銭", "兒": "児",
    "穗": "穂", "榮": "栄", "靜": "静", "嚴": "厳", "總": "総",
    "觀": "観", "權": "権", "顯": "顕",
    "齡": "齢", "辯": "弁", "辨": "弁",
    "賴": "頼", "顏": "顔", "號": "号", "聲": "声", "氣": "気",
    "緣": "縁", "譯": "訳", "讀": "読", "單": "単", "歷": "歴",
    "禪": "禅", "繩": "縄", "佛": "仏", "圖": "図", "勸": "勧",
    "勳": "勲", "輕": "軽", "郞": "郎",
    "滿": "満",
    # Hobby-specific: 謠→謡, 將→将
    "謠": "謡", "將": "将",
}
_KYUJITAI_TABLE = str.maketrans(_KYUJITAI_TO_SHINJITAI)

# Regex for org names that are actually family relation strings
_RELATION_ORG_RE = re.compile(
    r"(?:長男|二男|三男|四男|五男|六男|七男|八男"
    r"|長女|二女|三女|四女|五女|六女"
    r"|養子|養女|嫡男|嫡子|庶子"
    r"|入夫|婿養子|養嗣子|嗣子|養妹|娘婿|後裔)$"
)


def _dedup_hobby_variants(hobby_records):
    """Merge hobby records that are kyūjitai/shinjitai variants of the same word."""
    groups = defaultdict(list)
    for i, rec in enumerate(hobby_records):
        pid = rec["person_id"]
        norm = rec["hobby"].translate(_KYUJITAI_TABLE)
        groups[(pid, norm)].append(i)

    drop = set()
    merged = 0
    for (pid, norm), indices in groups.items():
        if len(indices) < 2:
            continue
        forms = {hobby_records[i]["hobby"] for i in indices}
        if len(forms) < 2:
            for idx in indices[1:]:
                drop.add(idx)
                merged += 1
            continue
        canonical_idx = indices[0]
        for idx in indices:
            if hobby_records[idx]["hobby"] == norm:
                canonical_idx = idx
                break
        canon = hobby_records[canonical_idx]
        cvols = canon.get("source_volume")
        if isinstance(cvols, str):
            cvols = [cvols]
        elif cvols is None:
            cvols = []
        cpages = canon.get("source_page")
        if isinstance(cpages, str):
            cpages = [cpages]
        elif cpages is None:
            cpages = []
        for idx in indices:
            if idx == canonical_idx:
                continue
            dup = hobby_records[idx]
            dvols = dup.get("source_volume")
            if isinstance(dvols, str):
                dvols = [dvols]
            elif dvols is None:
                dvols = []
            dpages = dup.get("source_page")
            if isinstance(dpages, str):
                dpages = [dpages]
            elif dpages is None:
                dpages = []
            cvols.extend(v for v in dvols if v not in cvols)
            cpages.extend(p for p in dpages if p not in cpages)
            drop.add(idx)
            merged += 1
        canon["hobby"] = norm
        if len(cvols) == 1:
            canon["source_volume"] = cvols[0]
        else:
            canon["source_volume"] = cvols
        if len(cpages) == 1:
            canon["source_page"] = cpages[0]
        else:
            canon["source_page"] = cpages

    if drop:
        hobby_records[:] = [r for i, r in enumerate(hobby_records) if i not in drop]
    return merged


def _null_relation_org_refs(org_records, edu_records, fe_records,
                            career_records, fc_records):
    """Null organization_id refs pointing to orgs whose names are family
    relation strings (長男, 二女, 養子, etc.) — not real institutions."""
    bad_oids = set()
    for org in org_records:
        if _RELATION_ORG_RE.search(org.get("name", "")):
            bad_oids.add(org["organization_id"])
    if not bad_oids:
        return 0
    n = 0
    for table in (edu_records, fe_records, career_records, fc_records):
        for rec in table:
            if rec.get("organization_id") in bad_oids:
                rec["organization_id"] = None
                n += 1
    org_records[:] = [o for o in org_records if o["organization_id"] not in bad_oids]
    n += len(bad_oids)
    return n


_COLONIAL_PREFIXES = ("大連", "新京", "京城", "臺北", "臺中", "臺南", "臺灣",
                      "撫順", "四平", "鞍山", "奉天", "哈爾")


def _fix_misgeocoded_ke_locations(loc_records):
    """Fix locations where ケ/ヶ stripping caused wrong-country geocoding,
    and 東大曽根/東大曾根 entries mapped to the wrong 東区 ward."""
    n = 0
    for loc in loc_records:
        name = loc.get("name", "")
        if loc.get("latitude") is None:
            continue
        if ("ケ" in name or "ヶ" in name) and loc.get("country") != "ja":
            cjk = "".join(c for c in name if "\u4e00" <= c <= "\u9fff")
            if not any(cjk.startswith(p) for p in _COLONIAL_PREFIXES):
                for field in ("latitude", "longitude", "geonameid",
                              "province", "country"):
                    loc[field] = None
                n += 1
                continue
        if ("東大曽根" in name or "東大曾根" in name):
            if loc.get("province") != "Aichi":
                loc["latitude"] = 35.17886
                loc["longitude"] = 136.92604
                loc["geonameid"] = 1862790
                loc["province"] = "Aichi"
                loc["country"] = "ja"
                n += 1
    return n


_ORG_NAME_TOKENS = ("株", "製作所", "工場", "會社", "会社", "銀行", "商店",
                    "合資", "合名", "事務所", "本社", "支社", "出張所")


def _null_orgname_location_refs(loc_records, org_records):
    """Null org location_id where the location's name contains org tokens."""
    loc_by_id = {loc["location_id"]: loc for loc in loc_records}
    n = 0
    for org in org_records:
        lid = org.get("location_id")
        if not lid:
            continue
        loc = loc_by_id.get(lid)
        if not loc:
            continue
        if any(tok in loc.get("name", "") for tok in _ORG_NAME_TOKENS):
            org["location_id"] = None
            n += 1
    return n


_FM_IDENTITY_FIELDS = ("name", "name_latin", "relation", "birth_year", "place", "location_id")


def _drop_phantom_family_members(fm_records, fe_records, fc_records):
    """Remove family member records where all identity fields are null.
    Also removes orphaned family_education and family_career records."""
    phantom_keys = set()
    drop_indices = set()
    for i, rec in enumerate(fm_records):
        if all(not rec.get(f) for f in _FM_IDENTITY_FIELDS):
            phantom_keys.add((rec["person_id"], rec["relation_id"]))
            drop_indices.add(i)
    if not drop_indices:
        return 0, 0, 0
    fm_records[:] = [r for i, r in enumerate(fm_records) if i not in drop_indices]
    fe_before = len(fe_records)
    fe_records[:] = [r for r in fe_records
                     if (r.get("person_id"), r.get("relation_id")) not in phantom_keys]
    fc_before = len(fc_records)
    fc_records[:] = [r for r in fc_records
                     if (r.get("person_id"), r.get("relation_id")) not in phantom_keys]
    return len(drop_indices), fe_before - len(fe_records), fc_before - len(fc_records)


# Era-year and deictic place cleanup
_ERA_NUM_RE = re.compile(
    r"^[明大昭天][一二三四五六七八九十〇\d]{0,3}$"
)
_STANDALONE_NUM_RE = re.compile(r"^[一二三四五六七八九十〇\d]{1,4}$")
_DEICTIC_PLACE_SET = frozenset({
    "本縣", "本県", "本府", "本市", "本郡",
    "同縣", "同県", "同校", "同府", "同市", "同町", "同村",
    "同郡", "同地", "同國", "同国", "同",
})
_SCHOOL_PLACE_SUFFIXES = (
    "高女卒", "高女", "女高師", "女學", "女学",
    "高商", "高工", "高校", "高等",
    "大學", "大学", "中學", "中学", "小學", "小学",
    "師範", "商業", "工業", "農業", "卒",
)
_ERA_SAFE_PLACES = frozenset({
    "明石", "明野", "昭和", "明治",
    "天津", "天理", "天草", "天城", "天王", "天竜", "天神",
    "天野", "天沼", "天長", "天水", "天山", "天河", "天田",
    "天門", "天香", "天本",
})


def _fix_garbled_and_deictic_places(core_records, fm_records,
                                     career_records, fc_records):
    """Null place values that are era-year fragments, deictic references,
    or standalone numbers."""
    n = 0

    def _should_null(place):
        if not place:
            return False
        if place in _DEICTIC_PLACE_SET:
            return True
        if _STANDALONE_NUM_RE.match(place):
            return True
        if _ERA_NUM_RE.match(place):
            return True
        if (len(place) == 2 and place[0] in "明昭天"
                and place not in _ERA_SAFE_PLACES):
            return True
        if place.endswith(_SCHOOL_PLACE_SUFFIXES):
            return True
        return False

    for rec in core_records:
        for field in ("place", "origin_place"):
            if _should_null(rec.get(field)):
                rec[field] = None
                loc_field = "location_id" if field == "place" else "origin_location_id"
                if rec.get(loc_field):
                    rec[loc_field] = None
                n += 1

    for rec in fm_records:
        if _should_null(rec.get("place")):
            rec["place"] = None
            if rec.get("location_id"):
                rec["location_id"] = None
            n += 1

    for table in (career_records, fc_records):
        for rec in table:
            if _should_null(rec.get("place_name")):
                rec["place_name"] = None
                if rec.get("location_id"):
                    rec["location_id"] = None
                n += 1

    return n


_CORPORATE_SUFFIXES = ("株式會社", "株式会社", "株式", "株")


def _merge_corporate_suffix_orgs(org_records, career_records, edu_records,
                                 fc_records, fe_records):
    """Merge orgs identical after stripping corporate suffixes and kyūjitai
    normalization.  Keeps the shorter (suffix-free) form as canonical."""
    norm_to_org = {}
    for org in org_records:
        norm = org["name"].translate(_KYUJITAI_TABLE)
        if norm not in norm_to_org:
            norm_to_org[norm] = org

    remap = {}
    for org in org_records:
        norm = org["name"].translate(_KYUJITAI_TABLE)
        for suffix in _CORPORATE_SUFFIXES:
            if norm.endswith(suffix) and len(norm) > len(suffix):
                base = norm[:-len(suffix)]
                if base in norm_to_org:
                    target = norm_to_org[base]
                    if target["organization_id"] != org["organization_id"]:
                        remap[org["organization_id"]] = target["organization_id"]
                break

    if not remap:
        return 0

    changed = True
    while changed:
        changed = False
        for mid in remap:
            cid = remap[mid]
            if cid in remap:
                remap[mid] = remap[cid]
                changed = True

    merged_ids = set(remap.keys())
    org_records[:] = [o for o in org_records if o["organization_id"] not in merged_ids]

    for table in (career_records, edu_records, fc_records, fe_records):
        for rec in table:
            if rec.get("organization_id") in remap:
                rec["organization_id"] = remap[rec["organization_id"]]

    return len(remap)


def _is_false_job(job_title):
    """Return True if job_title is a family relation string, not a real job."""
    if not job_title:
        return False
    if job_title in _LEGITIMATE_JOBS:
        return False
    if job_title in _RELATION_EXACT:
        return True
    if _WIFE_OF_RE.match(job_title):
        return True
    if _ERA_YEAR_RE.match(job_title):
        return True
    return False


# ==========================================
# G. MAIN PROCESSING
# ==========================================
def build_tables(records):
    person_core = []
    person_career = []
    person_education = []
    person_hobbies = []
    person_ranks = []
    person_religions = []
    person_political_parties = []
    person_family = []
    person_family_education = []
    person_family_career = []

    # Collect names for romanization
    name_pairs = []  # (family, given, domain, origin_place)

    for rec in records:
        ext = rec["extraction"]
        entry_index = rec["entry_index"]
        source_image = rec.get("source_image")
        source_page = rec.get("source_page")
        volume = rec.get("volume", "unknown")
        person_id = f"P{volume}_{entry_index}"

        # --- Name cleaning ---
        raw_name = ext.get("name")
        name = clean_name(raw_name)
        if not name:
            logger.debug(f"Skipping {entry_index}: invalid name '{raw_name}'")
            continue

        origin_place = ext.get("origin_place")
        if origin_place == "現地":
            origin_place = None
        domain = detect_domain(origin_place, name=name)

        name_family, name_given = split_name(name, domain)
        # For "other" domain (katakana-only foreigners), treat entire name as family name
        if domain == "other":
            name_family = name
            name_given = None

        birthyear = validate_year(ext.get("birth_year"), STRUCT_BIRTHYEAR_MIN, STRUCT_BIRTHYEAR_MAX)
        phone = clean_phone(ext.get("phone_number"))
        tax_amount = ext.get("tax_amount")
        if isinstance(tax_amount, (int, float)):
            tax_amount = str(tax_amount)
        rank = clean_rank(ext.get("rank"))

        # Core record (latin names filled later)
        core = {
            "person_id": person_id,
            "entry_index": entry_index,
            "source_image": source_image,
            "source_page": source_page,
            "volume": volume,
            "name": name,
            "name_family": name_family,
            "name_given": name_given,
            "name_family_latin": None,
            "name_given_latin": None,
            "domain": domain,
            "birthyear": birthyear,
            "phone": phone,
            "tax_amount": tax_amount,
            "place": None if ext.get("place") == "現地" else ext.get("place"),
            "origin_place": origin_place,
            "rank": rank,
        }
        person_core.append(core)

        # Convert 支店 (branch office) place fields to undated career entries
        if origin_place and origin_place.endswith("支店"):
            person_career.append({
                "person_id": person_id,
                "job_title": None,
                "organization": origin_place,
                "start_year": None,
                "place_name": None,
                "current": None,
                "source_volume": volume,
                "source_page": source_page,
            })
            core["origin_place"] = None
        place_val = core.get("place")
        if place_val and place_val.endswith("支店"):
            person_career.append({
                "person_id": person_id,
                "job_title": None,
                "organization": place_val,
                "start_year": None,
                "place_name": None,
                "current": None,
                "source_volume": volume,
                "source_page": source_page,
            })
            core["place"] = None

        # Null out place fields ending with family relation or org suffix,
        # or starting with a university abbreviation (東大法科卒 etc.)
        for _pf in ("origin_place", "place"):
            _pv = core.get(_pf)
            if not _pv:
                continue
            if _RELATION_SUFFIX_RE.search(_pv) or _pv.endswith(_ORG_PLACE_SUFFIXES):
                core[_pf] = None
                continue
            # University abbreviation leak: 東大+academic → not a place
            for _upfx in _UNI_PREFIXES:
                if _pv.startswith(_upfx):
                    _urem = _pv[len(_upfx):]
                    if not _urem or _urem[0] not in _UNI_SAFE_PLACE_CHARS:
                        core[_pf] = None
                    break

        if name_family and name_given:
            name_pairs.append((name_family, name_given, domain, origin_place))
        elif domain == "other" and name_family:
            # Katakana-only name: romanize as single family name
            name_pairs.append((name_family, None, domain, origin_place))

        # --- Career ---
        for career in ext.get("career", []) or []:
            if not isinstance(career, dict):
                continue
            job_title = career.get("job_title")
            if isinstance(job_title, (dict, list)):
                job_title = str(job_title) if job_title else None
            if _is_false_job(job_title):
                continue
            if job_title == "現職":
                job_title = None
            organization = career.get("organization")
            org_clean = clean_org(organization)
            place_name = career.get("place_name")
            if isinstance(place_name, list):
                place_name = place_name[0] if place_name else None
            if place_name == "現地":
                place_name = None

            # Fix single-char suffix titles
            if job_title:
                job_title = fix_suffix_title(job_title, organization or "")

            # Place fallback from org
            if not place_name and org_clean:
                place_name = extract_place_from_org(org_clean)

            start_year = validate_year(career.get("start_year"), STRUCT_YEAR_MIN, STRUCT_YEAR_MAX)
            current = career.get("current", False)

            person_career.append({
                "person_id": person_id,
                "job_title": job_title,
                "organization": org_clean,
                "start_year": start_year,
                "place_name": place_name,
                "current": current if current else None,
                "source_volume": volume,
                "source_page": source_page,
            })

        # --- Education ---
        for edu in ext.get("education", []) or []:
            if not isinstance(edu, dict):
                continue
            year_grad = validate_year(edu.get("year_graduated"), STRUCT_YEAR_MIN, STRUCT_YEAR_MAX)
            major = edu.get("major_of_study")
            if isinstance(major, (dict, list)):
                major = str(major) if major else None
            person_education.append({
                "person_id": person_id,
                "institution": edu.get("institution"),
                "major_of_study": major,
                "year_graduated": year_grad,
                "source_volume": volume,
                "source_page": source_page,
            })

        # --- Hobbies ---
        for hobby in ext.get("hobbies", []) or []:
            if isinstance(hobby, dict):
                hobby = hobby.get("name") or hobby.get("hobby") or hobby.get("item")
            if isinstance(hobby, list):
                hobby = "、".join(str(x) for x in hobby) if hobby else None
            if hobby and isinstance(hobby, str):
                person_hobbies.append({
                    "person_id": person_id,
                    "hobby": hobby,
                    "source_volume": volume,
                    "source_page": source_page,
                })

        # --- Ranks ---
        if ext.get("rank"):
            for part in str(ext["rank"]).split():
                cleaned = NON_CJK_KANA.sub("", part)
                if 2 <= len(cleaned) <= 6:
                    person_ranks.append({
                        "person_id": person_id,
                        "rank": cleaned,
                        "source_volume": volume,
                        "source_page": source_page,
                    })

        # --- Religion ---
        religion = ext.get("religion")
        if isinstance(religion, dict):
            religion = religion.get("name") or religion.get("religion") or next(
                (v for v in religion.values() if isinstance(v, str)), None)
        if isinstance(religion, list):
            religion = "、".join(str(x) for x in religion) if religion else None
        if religion and isinstance(religion, str):
            person_religions.append({
                "person_id": person_id,
                "religion": religion,
                "source_volume": volume,
                "source_page": source_page,
            })

        # --- Political party ---
        political_party = ext.get("political_party")
        if isinstance(political_party, dict):
            political_party = political_party.get("name") or political_party.get("party") or next(
                (v for v in political_party.values() if isinstance(v, str)), None)
        if isinstance(political_party, list):
            political_party = "、".join(str(x) for x in political_party) if political_party else None
        if political_party and isinstance(political_party, str):
            person_political_parties.append({
                "person_id": person_id,
                "political_party": political_party,
                "source_volume": volume,
                "source_page": source_page,
            })

        # --- Family members ---
        subject_birthyear = birthyear
        for fm in ext.get("family_member", []) or []:
            relation = fm.get("relation")
            if relation and len(relation) >= 7:
                relation = None

            fm_name = fm.get("name")
            fm_birth_year = validate_year(fm.get("birth_year"), STRUCT_BIRTHYEAR_MIN, STRUCT_BIRTHYEAR_MAX)
            fm_birth_year = validate_family_birthyear(fm_birth_year, subject_birthyear, relation)

            # Collect family member names for romanization
            fm_family_ctx = None
            if fm_name:
                fm_name_clean = clean_name(fm_name)
                if fm_name_clean:
                    fm_name_clean = strip_kinship_noise(fm_name_clean)
                    # Only add subject's family name as context when the member's
                    # name is short (1-2 chars = given name only, needs surname context).
                    # 3+ chars likely already contains its own family name.
                    fm_family_ctx = name_family if len(fm_name_clean) <= 2 else None
                    name_pairs.append((fm_family_ctx, fm_name_clean, domain, origin_place))

            fm_record = {
                "person_id": person_id,
                "relation_id": None,  # assigned below
                "_subject_name_family": fm_family_ctx,  # transient: used by apply_romanization
                "relation": relation,
                "name": fm_name,
                "name_latin": None,  # filled after romanization
                "birth_year": fm_birth_year,
                "place": None if fm.get("place") == "現地" else fm.get("place"),
                "source_volume": volume,
                "source_page": source_page,
            }

            fm_idx = len(person_family)  # index for linking edu/career records

            # Convert 支店 (branch office) family member place to career entry
            fm_place_val = fm_record.get("place")
            if fm_place_val and fm_place_val.endswith("支店"):
                person_family_career.append({
                    "person_id": person_id,
                    "_fm_idx": fm_idx,
                    "job_title": None,
                    "organization": fm_place_val,
                    "place_name": None,
                    "start_year": None,
                    "current": None,
                    "source_volume": volume,
                    "source_page": source_page,
                })
                fm_record["place"] = None

            # Null out family member place ending with family relation, org suffix,
            # or starting with university abbreviation
            _fm_pv = fm_record.get("place")
            if _fm_pv:
                if (_RELATION_SUFFIX_RE.search(_fm_pv)
                        or _fm_pv.endswith(_ORG_PLACE_SUFFIXES)):
                    fm_record["place"] = None
                else:
                    for _upfx in _UNI_PREFIXES:
                        if _fm_pv.startswith(_upfx):
                            _urem = _fm_pv[len(_upfx):]
                            if not _urem or _urem[0] not in _UNI_SAFE_PLACE_CHARS:
                                fm_record["place"] = None
                            break

            # Extract ALL education records (not just first)
            fm_edu_list = fm.get("education", []) or []
            if not isinstance(fm_edu_list, list):
                fm_edu_list = [fm_edu_list]
            for edu_entry in fm_edu_list:
                if not isinstance(edu_entry, dict):
                    continue
                fm_inst = edu_entry.get("institution")
                if fm_inst and ERA_DATE_RE.match(fm_inst):
                    fm_inst = None
                if not fm_inst:
                    continue
                fm_major = edu_entry.get("major_of_study")
                if isinstance(fm_major, (dict, list)):
                    fm_major = str(fm_major) if fm_major else None
                person_family_education.append({
                    "person_id": person_id,
                    "_fm_idx": fm_idx,
                    "institution": fm_inst,
                    "major_of_study": fm_major,
                    "year_graduated": validate_year(
                        edu_entry.get("year_graduated"), STRUCT_YEAR_MIN, STRUCT_YEAR_MAX
                    ),
                    "source_volume": volume,
                    "source_page": source_page,
                })

            # Extract ALL career records (not just first)
            fm_career_list = fm.get("career", []) or []
            if not isinstance(fm_career_list, list):
                fm_career_list = [fm_career_list]
            for career_entry in fm_career_list:
                if not isinstance(career_entry, dict):
                    continue
                fm_job = career_entry.get("job_title")
                if isinstance(fm_job, (dict, list)):
                    fm_job = str(fm_job) if fm_job else None
                if fm_job == "嫁" or _is_false_job(fm_job):
                    continue
                if fm_job == "現職":
                    fm_job = None
                fm_org = clean_org(career_entry.get("organization"))
                if fm_org and ERA_DATE_RE.match(fm_org):
                    fm_org = None
                if not fm_job and not fm_org:
                    continue
                fm_place = career_entry.get("place_name")
                if isinstance(fm_place, list):
                    fm_place = fm_place[0] if fm_place else None
                if fm_place == "現地":
                    fm_place = None
                person_family_career.append({
                    "person_id": person_id,
                    "_fm_idx": fm_idx,
                    "job_title": fm_job,
                    "organization": fm_org,
                    "place_name": fm_place,
                    "start_year": validate_year(
                        career_entry.get("start_year"), STRUCT_YEAR_MIN, STRUCT_YEAR_MAX
                    ),
                    "current": career_entry.get("current", False) or None,
                    "source_volume": volume,
                    "source_page": source_page,
                })

            person_family.append(fm_record)

    # Assign relation_ids per person
    person_fm_counts = {}
    for fm in person_family:
        pid = fm["person_id"]
        person_fm_counts[pid] = person_fm_counts.get(pid, 0) + 1
        fm["relation_id"] = f"R{person_fm_counts[pid]}"

    # Propagate relation_id to family education/career tables
    for rec in person_family_education:
        rec["relation_id"] = person_family[rec.pop("_fm_idx")]["relation_id"]
    for rec in person_family_career:
        rec["relation_id"] = person_family[rec.pop("_fm_idx")]["relation_id"]

    # Validate unique relations: a person can only have one father or one mother
    UNIQUE_RELATIONS = re.compile(r'^(父|母)$')
    person_fm_by_pid = defaultdict(list)
    for fm in person_family:
        person_fm_by_pid[fm["person_id"]].append(fm)

    nullified = 0
    for pid, fms in person_fm_by_pid.items():
        relation_groups = defaultdict(list)
        for fm in fms:
            rel = fm.get("relation")
            if rel and UNIQUE_RELATIONS.match(rel):
                relation_groups[rel].append(fm)
        for rel, members in relation_groups.items():
            if len(members) >= 2:
                for fm in members:
                    fm["relation"] = None
                    nullified += 1
    if nullified:
        logger.info(f"Nullified {nullified} duplicate unique-role family relations")

    # Validate sibling birth order: Nth child cannot be born before (N-1)th child
    ORDINAL_MAP = {"長": 1, "一": 1, "次": 2, "二": 2, "三": 3, "四": 4, "五": 5,
                   "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    CHILD_PATTERN = re.compile(r"[男女子]$")
    sibling_nullified = 0
    for pid, fms in person_fm_by_pid.items():
        # Collect children with ordinals, split by gender suffix (男/女)
        for gender_suffix in ("男", "女"):
            siblings = []
            for fm in fms:
                rel = fm.get("relation") or ""
                if not rel.endswith(gender_suffix):
                    continue
                if fm.get("birth_year") is None:
                    continue
                # Extract ordinal from relation
                ordinal = None
                for char in rel:
                    if char in ORDINAL_MAP:
                        ordinal = ORDINAL_MAP[char]
                        break
                if ordinal is not None:
                    siblings.append((ordinal, fm))
            if len(siblings) < 2:
                continue
            siblings.sort(key=lambda x: x[0])
            # Check monotonicity: later ordinals must have >= birth year
            for i in range(1, len(siblings)):
                prev_ord, prev_fm = siblings[i - 1]
                curr_ord, curr_fm = siblings[i]
                if curr_fm["birth_year"] < prev_fm["birth_year"]:
                    # Birth order violation — nullify all conflicting siblings' birth years
                    for _, fm in siblings:
                        if fm.get("birth_year") is not None:
                            fm["birth_year"] = None
                            sibling_nullified += 1
                    break  # already nullified all in this group
    if sibling_nullified:
        logger.info(f"Nullified {sibling_nullified} family birth years (sibling order violation)")

    return (person_core, person_career, person_education, person_hobbies, person_ranks,
            person_religions, person_political_parties,
            person_family, person_family_education, person_family_career, name_pairs)


def apply_romanization(person_core, person_family, roman_cache):
    """Apply romanization results back to core and family tables."""
    null_count = 0
    for core in person_core:
        key = (core["name_family"], core["name_given"])
        if key in roman_cache:
            core["name_family_latin"] = roman_cache[key].get("family")
            core["name_given_latin"] = roman_cache[key].get("given")
        if not core["name_family_latin"]:
            null_count += 1

    fm_null_count = 0
    for fm in person_family:
        if fm["name"]:
            fm_name_clean = clean_name(fm["name"])
            if fm_name_clean:
                fm_name_clean = strip_kinship_noise(fm_name_clean)
                subject_family = fm.get("_subject_name_family")
                key = (subject_family, fm_name_clean)
                if key in roman_cache:
                    fm["name_latin"] = roman_cache[key].get("given")
                elif (None, fm_name_clean) in roman_cache:
                    # Fallback to old-style key for backward compat
                    fm["name_latin"] = roman_cache[(None, fm_name_clean)].get("given")
        # Remove transient field before output
        fm.pop("_subject_name_family", None)
        if not fm.get("name_latin"):
            fm_null_count += 1

    logger.info(f"Romanization: {null_count} core names without latin, {fm_null_count} family names without latin")


def _pykakasi_romanize_single(text):
    """Romanize a single text string via pykakasi. Returns capitalized or None."""
    if not text:
        return None
    items = _kakasi.convert(text)
    hepburn = "".join(item["hepburn"] for item in items)
    if not hepburn or not hepburn.isascii() or hepburn == text:
        return None
    if any(item["hepburn"] == item["orig"] for item in items):
        return None
    return hepburn.capitalize()


# Prefecture suffixes — signal that the "given name" is actually a place name
_PREFECTURE_SUFFIX_RE = re.compile(r"[縣県]")


def _fix_family_name_split_and_length(family_records, known_families, core_family_by_pid):
    """Fix family member name/name_latin in multiple passes:

    1. If name starts with a known multi-char family name different from the
       core person's, split the romanization into "Family Given" with a space.
    2. If name starts with the same family as the core person, re-romanize
       with a space between family and given.
    3. If the effective given-name part is >6 chars, null BOTH name and name_latin.
    4. If the given part starts with a prefecture suffix (縣/県), null both.
    5. If name_latin (without spaces) is >15 Latin chars, null both.
    """
    n_split = 0
    n_nulled = 0

    for rec in family_records:
        name = rec.get("name") or ""
        if not name or len(name) < 2:
            continue

        pid = rec["person_id"]
        core_family = core_family_by_pid.get(pid, "")

        diff_family = None
        for flen in range(min(4, len(name) - 1), 1, -1):
            prefix = name[:flen]
            if prefix != core_family and prefix in known_families:
                diff_family = prefix
                break

        if diff_family:
            given_part = name[len(diff_family):]
            # Check for prefecture suffix in given part → not a real name
            if given_part and _PREFECTURE_SUFFIX_RE.search(given_part[:2]):
                rec["name"] = None
                rec["name_latin"] = None
                n_nulled += 1
                continue
            fam_latin = _pykakasi_romanize_single(diff_family)
            giv_latin = _pykakasi_romanize_single(given_part) if given_part else None
            if fam_latin:
                if giv_latin and len(given_part) <= 6:
                    rec["name_latin"] = f"{fam_latin} {giv_latin}"
                elif len(given_part) > 6:
                    rec["name"] = None
                    rec["name_latin"] = None
                    n_nulled += 1
                    continue
                else:
                    rec["name_latin"] = fam_latin
                n_split += 1
        else:
            given_part = name
            if core_family and name.startswith(core_family) and len(name) > len(core_family):
                given_part = name[len(core_family):]
                # Check for prefecture suffix in given part
                if _PREFECTURE_SUFFIX_RE.search(given_part[:2]):
                    rec["name"] = None
                    rec["name_latin"] = None
                    n_nulled += 1
                    continue
                if len(given_part) > 6:
                    rec["name"] = None
                    rec["name_latin"] = None
                    n_nulled += 1
                    continue
                # Re-romanize with space between family and given
                fam_latin = _pykakasi_romanize_single(core_family)
                giv_latin = _pykakasi_romanize_single(given_part)
                if fam_latin and giv_latin:
                    rec["name_latin"] = f"{fam_latin} {giv_latin}"
                    n_split += 1
            elif len(given_part) > 6:
                rec["name"] = None
                rec["name_latin"] = None
                n_nulled += 1

        # Long romaji detection
        final_nl = rec.get("name_latin") or ""
        if final_nl and len(final_nl.replace(" ", "")) > 15:
            rec["name"] = None
            rec["name_latin"] = None
            n_nulled += 1

    return n_split, n_nulled


def build_organizations(person_career, person_education,
                        person_family_career=None, person_family_education=None):
    """Deduplicate orgs, assign IDs, replace text with org_id."""
    org_set = set()
    for c in person_career:
        if c.get("organization"):
            org_set.add(c["organization"])
    for e in person_education:
        if e.get("institution"):
            org_set.add(e["institution"])
    # Pool family member orgs into the same deduplication
    for c in (person_family_career or []):
        if c.get("organization"):
            org_set.add(c["organization"])
    for e in (person_family_education or []):
        if e.get("institution"):
            org_set.add(e["institution"])

    org_map = {}
    orgs_table = []
    for i, name in enumerate(sorted(org_set), 1):
        oid = f"O{i}"
        org_map[name] = oid
        orgs_table.append({"organization_id": oid, "name": name})

    for c in person_career:
        if c.get("organization"):
            c["organization_id"] = org_map[c["organization"]]
        else:
            c["organization_id"] = None
        c.pop("organization", None)
    for e in person_education:
        if e.get("institution"):
            e["organization_id"] = org_map[e["institution"]]
        else:
            e["organization_id"] = None
        e.pop("institution", None)
    # Assign org_ids to family member tables
    for c in (person_family_career or []):
        if c.get("organization"):
            c["organization_id"] = org_map[c["organization"]]
        else:
            c["organization_id"] = None
        c.pop("organization", None)
    for e in (person_family_education or []):
        if e.get("institution"):
            e["organization_id"] = org_map[e["institution"]]
        else:
            e["organization_id"] = None
        e.pop("institution", None)

    return orgs_table


# ==========================================
# H. PLAUSIBILITY CHECKS
# ==========================================
def run_plausibility_checks(person_core, person_education, person_career, person_family, orgs_table,
                            person_family_education=None, person_family_career=None):
    """Apply plausibility corrections and document editorial changes."""
    editorial_changes = []

    # Build lookups
    gender_map = {c["person_id"]: c.get("gender") for c in person_core}
    org_name_map = {o["organization_id"]: o["name"] for o in orgs_table}

    # Check 1: Male subjects who STUDIED at girls' institutions → delete education record
    edu_to_remove = []
    for idx, edu in enumerate(person_education):
        pid = edu["person_id"]
        gender = gender_map.get(pid)
        org_id = edu.get("organization_id")
        org_name = org_name_map.get(org_id, "")
        if gender == "m" and "女" in org_name:
            editorial_changes.append({
                "person_id": pid,
                "change_type": "delete_education",
                "field": "education",
                "old_value": {"organization_id": org_id, "org_name": org_name},
                "new_value": None,
                "reason": f"Male subject unlikely to have studied at girls' institution '{org_name}'; "
                          "likely misattributed from family member record",
            })
            edu_to_remove.append(idx)

    # Remove flagged education records (reverse order to preserve indices)
    for idx in reversed(edu_to_remove):
        del person_education[idx]

    if edu_to_remove:
        logger.info(f"Plausibility: removed {len(edu_to_remove)} education records "
                     "(male at girls' institution)")

    # Check 2: Conflicting family ordinals (長男+一男, 長女+一女)
    CONFLICT_PAIRS = [("長男", "一男"), ("長女", "一女")]

    family_by_person = defaultdict(list)
    for fm in person_family:
        family_by_person[fm["person_id"]].append(fm)

    conflict_count = 0
    for pid, members in family_by_person.items():
        relations = {fm.get("relation") for fm in members if fm.get("relation")}
        for term_a, term_b in CONFLICT_PAIRS:
            if term_a in relations and term_b in relations:
                # NA both conflicting relations
                for fm in members:
                    if fm.get("relation") in (term_a, term_b):
                        old_rel = fm["relation"]
                        fm["relation"] = None
                        editorial_changes.append({
                            "person_id": pid,
                            "change_type": "null_conflicting_relation",
                            "field": "relation",
                            "old_value": old_rel,
                            "new_value": None,
                            "reason": f"Conflicting ordinals: family has both '{term_a}' and '{term_b}'",
                        })
                        conflict_count += 1

    if conflict_count:
        logger.info(f"Plausibility: nulled {conflict_count} conflicting family relations")

    # Check 3: Person cannot have both 妻 (wife) and 夫 (husband) family members
    gender_by_pid = {c["person_id"]: c.get("gender") for c in person_core}
    spouse_conflict_count = 0
    for pid, members in family_by_person.items():
        relations = [fm.get("relation") for fm in members if fm.get("relation")]
        has_wife = any("妻" in r for r in relations)
        has_husband = any("夫" in r for r in relations)
        if has_wife and has_husband:
            gender = gender_by_pid.get(pid)
            if gender == "m":
                # Keep wife, nullify husband
                for fm in members:
                    rel = fm.get("relation") or ""
                    if "夫" in rel:
                        old_rel = fm["relation"]
                        fm["relation"] = None
                        editorial_changes.append({
                            "person_id": pid,
                            "change_type": "null_conflicting_spouse",
                            "field": "relation",
                            "old_value": old_rel,
                            "new_value": None,
                            "reason": "Male subject cannot have both wife and husband; removed husband",
                        })
                        spouse_conflict_count += 1
            elif gender == "f":
                # Keep husband, nullify wife
                for fm in members:
                    rel = fm.get("relation") or ""
                    if "妻" in rel:
                        old_rel = fm["relation"]
                        fm["relation"] = None
                        editorial_changes.append({
                            "person_id": pid,
                            "change_type": "null_conflicting_spouse",
                            "field": "relation",
                            "old_value": old_rel,
                            "new_value": None,
                            "reason": "Female subject cannot have both wife and husband; removed wife",
                        })
                        spouse_conflict_count += 1
            else:
                # Unknown gender: nullify both
                for fm in members:
                    rel = fm.get("relation") or ""
                    if "妻" in rel or "夫" in rel:
                        old_rel = fm["relation"]
                        fm["relation"] = None
                        editorial_changes.append({
                            "person_id": pid,
                            "change_type": "null_conflicting_spouse",
                            "field": "relation",
                            "old_value": old_rel,
                            "new_value": None,
                            "reason": "Unknown gender subject has both wife and husband; removed both",
                        })
                        spouse_conflict_count += 1

    if spouse_conflict_count:
        logger.info(f"Plausibility: nulled {spouse_conflict_count} conflicting spouse relations")

    # Check 4: Education year cannot be before person turned 14;
    #           employment year cannot be before person turned 16
    birthyear_map = {c["person_id"]: c.get("birthyear") for c in person_core}

    edu_year_nulled = 0
    for edu in person_education:
        by = birthyear_map.get(edu["person_id"])
        yg = edu.get("year_graduated")
        if by and yg and yg < by + 14:
            editorial_changes.append({
                "person_id": edu["person_id"],
                "change_type": "null_implausible_edu_year",
                "field": "year_graduated",
                "old_value": yg,
                "new_value": None,
                "reason": f"Graduation year {yg} before person turned 14 (born {by})",
            })
            edu["year_graduated"] = None
            edu_year_nulled += 1

    career_year_nulled = 0
    for job in person_career:
        by = birthyear_map.get(job["person_id"])
        sy = job.get("start_year")
        if by and sy and sy < by + 16:
            editorial_changes.append({
                "person_id": job["person_id"],
                "change_type": "null_implausible_career_year",
                "field": "start_year",
                "old_value": sy,
                "new_value": None,
                "reason": f"Career start year {sy} before person turned 16 (born {by})",
            })
            job["start_year"] = None
            career_year_nulled += 1

    if edu_year_nulled or career_year_nulled:
        logger.info(f"Plausibility: nulled {edu_year_nulled} education years, {career_year_nulled} career years (too young)")

    # Check 5: Dates cannot exceed the publication year of the volume
    volume_map = {c["person_id"]: int(re.match(r"\d+", str(c.get("volume", "9999"))).group())
                  for c in person_core}
    future_edu_nulled = 0
    future_career_nulled = 0

    all_edu = list(person_education) + list(person_family_education or [])
    all_career = list(person_career) + list(person_family_career or [])

    for edu in all_edu:
        vol_year = volume_map.get(edu["person_id"], 9999)
        yg = edu.get("year_graduated")
        if yg and yg > vol_year:
            editorial_changes.append({
                "person_id": edu["person_id"],
                "change_type": "null_future_edu_year",
                "field": "year_graduated",
                "old_value": yg,
                "new_value": None,
                "reason": f"Graduation year {yg} exceeds volume publication year {vol_year}",
            })
            edu["year_graduated"] = None
            future_edu_nulled += 1

    for job in all_career:
        vol_year = volume_map.get(job["person_id"], 9999)
        sy = job.get("start_year")
        if sy and sy > vol_year:
            editorial_changes.append({
                "person_id": job["person_id"],
                "change_type": "null_future_career_year",
                "field": "start_year",
                "old_value": sy,
                "new_value": None,
                "reason": f"Career start year {sy} exceeds volume publication year {vol_year}",
            })
            job["start_year"] = None
            future_career_nulled += 1

    if future_edu_nulled or future_career_nulled:
        logger.info(f"Plausibility: nulled {future_edu_nulled} education years, {future_career_nulled} career years (exceed volume year)")

    logger.info(f"Plausibility checks: {len(editorial_changes)} editorial changes applied")
    return editorial_changes


# ==========================================
# I. IMPROVED ORG LOCATION ASSIGNMENT
# ==========================================
def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance between two lat/lon points in km."""
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def is_generic_org(org_name):
    """Check if org name looks like a generic government body (exists in many locations)."""
    generic_suffixes = ["省", "部", "局", "院", "廳", "庁"]
    if len(org_name) <= 4 and any(org_name.endswith(s) for s in generic_suffixes):
        return True
    return False


def assign_org_locations(orgs_table, person_career, locations_table, geonames_idx, mcgd_idx):
    """Assign location_id to organizations from career data with geographic clustering,
    and org-name-prefix fallback.  Results are cached so reruns skip recomputation."""
    cache_path = os.path.join(STRUCT_OUTPUT_DIR, "_cache_org_locations.json")
    current_org_ids = {org["organization_id"] for org in orgs_table}

    # Try loading from cache
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        cached_ids = set(cached.keys())
        if current_org_ids <= cached_ids:
            applied = 0
            for org in orgs_table:
                org["location_id"] = cached.get(org["organization_id"])
                if org["location_id"]:
                    applied += 1
            logger.info(f"Org locations loaded from cache: {applied}/{len(orgs_table)} assigned")
            return

    # Build location coordinate lookup
    loc_coords = {}
    for loc in locations_table:
        loc_coords[loc["location_id"]] = (loc["latitude"], loc["longitude"])

    # Gather location_ids from career records
    org_location_map = defaultdict(list)
    for c in person_career:
        oid = c.get("organization_id")
        lid = c.get("location_id")
        if oid and lid:
            org_location_map[oid].append(lid)

    assigned_career = 0
    assigned_prefix = 0
    skipped_generic = 0
    skipped_spread = 0

    total_orgs = len(orgs_table)
    for org_idx, org in enumerate(orgs_table, 1):
        if org_idx % 5000 == 0:
            logger.info(f"  Org locations: {org_idx}/{total_orgs} processed")
        oid = org["organization_id"]
        org_name = org["name"]
        locs = org_location_map.get(oid, [])

        # Strategy A: from career locations
        if locs:
            if is_generic_org(org_name):
                org["location_id"] = None
                skipped_generic += 1
                continue

            # Check geographic spread
            coords = [loc_coords[lid] for lid in locs if lid in loc_coords]
            if len(coords) >= 2:
                max_dist = 0
                for ci in range(len(coords)):
                    for cj in range(ci + 1, len(coords)):
                        d = haversine_km(coords[ci][0], coords[ci][1],
                                         coords[cj][0], coords[cj][1])
                        if d > max_dist:
                            max_dist = d
                if max_dist > STRUCT_ORG_LOCATION_MAX_SPREAD_KM:
                    org["location_id"] = None
                    skipped_spread += 1
                    continue

            org["location_id"] = Counter(locs).most_common(1)[0][0]
            assigned_career += 1
        else:
            # Strategy B: match org name prefix to a known location
            org_cjk = "".join(CJK_CHAR.findall(org_name))
            matched_lid = None
            for length in range(min(4, len(org_cjk)), 1, -1):
                prefix = org_cjk[:length]
                result = match_place(prefix, None, geonames_idx, mcgd_idx)
                if result and length >= 2:
                    lat, lon = result[0], result[1]
                    # Find existing location entry with similar coords
                    for existing_loc in locations_table:
                        if (round(existing_loc["latitude"], 1) == round(lat, 1) and
                                round(existing_loc["longitude"], 1) == round(lon, 1)):
                            matched_lid = existing_loc["location_id"]
                            break
                    break

            org["location_id"] = matched_lid
            if matched_lid:
                assigned_prefix += 1

    logger.info(f"Org locations: {assigned_career} from career data, {assigned_prefix} from name prefix, "
                f"{skipped_generic} skipped (generic), {skipped_spread} skipped (spread > {STRUCT_ORG_LOCATION_MAX_SPREAD_KM}km)")

    # Save cache
    cache_data = {org["organization_id"]: org.get("location_id") for org in orgs_table}
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False)
    logger.info(f"Org location cache saved ({len(cache_data)} entries)")


# ==========================================
# J. WRITE OUTPUT
# ==========================================
def write_jsonl(filepath, records):
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(records)} records to {filepath}")


def main():
    logger.info("=== Step 6: Structure Biographies ===")

    # 1. Load
    records = load_all_volume_records()

    # 2. Build tables
    (person_core, person_career, person_education,
     person_hobbies, person_ranks, person_religions, person_political_parties,
     person_family, person_family_education, person_family_career,
     name_pairs) = build_tables(records)

    # Merge kyūjitai/shinjitai hobby variants (e.g. 謠曲↔謡曲)
    n_hobby_dedup = _dedup_hobby_variants(person_hobbies)
    if n_hobby_dedup:
        logger.info(f"person_hobbies: {n_hobby_dedup} kyūjitai-variant duplicates merged")

    logger.info(f"Built: {len(person_core)} persons, {len(person_career)} careers, "
                f"{len(person_education)} education, {len(person_hobbies)} hobbies, "
                f"{len(person_ranks)} ranks, {len(person_religions)} religions, "
                f"{len(person_political_parties)} political parties, "
                f"{len(person_family)} family members, "
                f"{len(person_family_education)} family edu, {len(person_family_career)} family career")

    # Drop phantom family members (all identity fields null)
    n_phantom_fm, n_phantom_fe, n_phantom_fc = _drop_phantom_family_members(
        person_family, person_family_education, person_family_career)
    if n_phantom_fm:
        logger.info(f"Dropped {n_phantom_fm} phantom family members "
                    f"({n_phantom_fe} orphaned edu, {n_phantom_fc} orphaned career)")

    # Null garbled era-year and deictic place values (before geocoding)
    n_garbled = _fix_garbled_and_deictic_places(person_core, person_family,
                                                 person_career, person_family_career)
    if n_garbled:
        logger.info(f"Nulled {n_garbled} garbled era-year / deictic / number place values")

    # 3. Geolocation (before gender/romanization so spatial domain can override)
    logger.info("--- Step 3: Geolocation ---")
    geonames_idx = load_geonames_index()
    mcgd_idx = load_mcgd_index()
    locations_table = build_locations(person_core, person_career, geonames_idx, mcgd_idx,
                                     person_family, person_family_career)

    # 3b. Spatial join → assigns province + country from GeoJSON boundaries
    logger.info("--- Step 3b: Spatial join (province/country) ---")
    fill_provinces(locations_table)

    # 3c. Override text-based domain with authoritative spatial result
    logger.info("--- Step 3c: Spatial domain override ---")
    domain_updates = resolve_domain_from_spatial(person_core, locations_table, person_career)

    # 3c2. Fix zh-domain 1-char surnames that aren't real Chinese surnames
    logger.info("--- Step 3c2: Fix zh-domain 1-char surname splits ---")
    onechar_fixes = fix_zh_onechar_surnames(person_core)
    domain_updates += onechar_fixes

    # 3d. Rebuild name_pairs if any domains changed (affects romanization system)
    if domain_updates:
        logger.info("--- Step 3d: Rebuilding name pairs after domain updates ---")
        name_pairs = []
        for core in person_core:
            nf, ng, dom = core["name_family"], core["name_given"], core["domain"]
            origin = core.get("origin_place")
            if nf and ng:
                name_pairs.append((nf, ng, dom, origin))
            elif dom == "other" and nf:
                name_pairs.append((nf, None, dom, origin))
        # Also rebuild family name pairs
        core_by_pid = {c["person_id"]: c for c in person_core}
        for fm in person_family:
            fm_name = fm.get("name")
            if fm_name:
                fm_name_clean = clean_name(fm_name)
                if fm_name_clean:
                    fm_name_clean = strip_kinship_noise(fm_name_clean)
                    pid = fm["person_id"]
                    # Find subject's domain and family name
                    subj = core_by_pid.get(pid)
                    if subj:
                        fm_family_ctx = subj["name_family"] if len(fm_name_clean) <= 2 else None
                        fm["_subject_name_family"] = fm_family_ctx
                        name_pairs.append((fm_family_ctx, fm_name_clean, subj["domain"], subj.get("origin_place")))

    # 3e. Admin hierarchy (before translations so admin names are available)
    logger.info("--- Step 3e: Admin hierarchy parsing ---")
    parse_admin_hierarchy(locations_table)

    # 4. Gender identification
    logger.info("--- Step 4: Gender classification ---")
    classify_gender(person_core, person_family, person_education, person_career, person_ranks)

    # 5. Romanize names
    logger.info("--- Step 5: Name romanization ---")
    roman_cache = romanize_names_batch(name_pairs)
    apply_romanization(person_core, person_family, roman_cache)

    # 5b. Split romanization for family members with different family names,
    #     null out implausibly long given names (>6 chars)
    logger.info("--- Step 5b: Family name split + length fix ---")
    known_families = {c["name_family"] for c in person_core
                      if c.get("name_family") and len(c["name_family"]) >= 2}
    core_family_by_pid = {c["person_id"]: c.get("name_family", "")
                          for c in person_core}
    n_split, n_nulled = _fix_family_name_split_and_length(
        person_family, known_families, core_family_by_pid)
    logger.info(f"Family name splits: {n_split} split, {n_nulled} nulled (given >6 chars)")

    # 5c. Convert ou → ō in ja-domain romaji names
    _ou_re = re.compile(r"ou(?![aeiou])", re.I)
    def _replace_ou(m):
        return "Ō" if m.group()[0].isupper() else "ō"
    n_ou = 0
    for core in person_core:
        if core.get("domain") != "ja":
            continue
        for field in ("name_family_latin", "name_given_latin"):
            val = core.get(field)
            if val and _ou_re.search(val):
                core[field] = _ou_re.sub(_replace_ou, val)
                n_ou += 1
    ja_pids = {c["person_id"] for c in person_core if c.get("domain") == "ja"}
    for fm in person_family:
        if fm.get("person_id") not in ja_pids:
            continue
        val = fm.get("name_latin")
        if val and _ou_re.search(val):
            fm["name_latin"] = _ou_re.sub(_replace_ou, val)
            n_ou += 1
    logger.info(f"ou → ō: {n_ou} fields converted")

    # 6. Translate location names (including admin levels)
    logger.info("--- Step 6: Location name translation ---")
    loc_names = [loc["name"] for loc in locations_table]
    loc_translations = translate_batch_ollama(loc_names, "location")
    for loc in locations_table:
        loc["name_en"] = loc_translations.get(loc["name"])

    # Translate admin-level names
    logger.info("--- Step 6b: Admin region name translation ---")
    admin_names = set()
    for loc in locations_table:
        for field in ("admin1", "admin2", "admin3"):
            if loc.get(field):
                admin_names.add(loc[field])
    if admin_names:
        admin_translations = translate_batch_ollama(list(admin_names), "administrative region")
        for loc in locations_table:
            loc["admin1_en"] = admin_translations.get(loc.get("admin1")) if loc.get("admin1") else None
            loc["admin2_en"] = admin_translations.get(loc.get("admin2")) if loc.get("admin2") else None
            loc["admin3_en"] = admin_translations.get(loc.get("admin3")) if loc.get("admin3") else None

    # 6c. Override Xinjing admin1_en (LLM mistranslates 新京)
    _XINJING_ADMIN1_MAP = {
        "新京特別市": "Hsinking Special City",
        "新 京特別市": "Hsinking Special City",
        "新京市": "Hsinking City",
        "新京別市": "Hsinking Special City",
        "新京都": "Hsinking",
    }
    xj_fixed = 0
    for loc in locations_table:
        admin1 = loc.get("admin1") or ""
        if admin1 in _XINJING_ADMIN1_MAP:
            loc["admin1_en"] = _XINJING_ADMIN1_MAP[admin1]
            xj_fixed += 1
        elif admin1.startswith("新京") or admin1.startswith("新 京"):
            loc["admin1_en"] = "Hsinking Special City"
            xj_fixed += 1
    if xj_fixed:
        logger.info(f"Xinjing admin1_en override: {xj_fixed} fixes")

    # 6d. Remove girls' school education for male family members
    _girls_school_re = re.compile(r"高女|女學校|女学校|女子.*學校|女子.*学校|女子専門|女子師範")
    fm_gender_by_key = {(fm["person_id"], fm.get("relation_id")): fm.get("gender")
                        for fm in person_family}
    before_fe = len(person_family_education)
    person_family_education = [
        e for e in person_family_education
        if not (fm_gender_by_key.get((e["person_id"], e.get("relation_id"))) == "m"
                and e.get("institution") and _girls_school_re.search(e["institution"]))
    ]
    n_girls_removed = before_fe - len(person_family_education)
    if n_girls_removed:
        logger.info(f"Removed {n_girls_removed} girls' school records for male family members")

    # 6e. Null deictic organization names (本省/現社/本縣 = "this ministry/company/prefecture")
    _deictic_orgs = frozenset({"本省", "現社", "本縣", "地方", "本府", "本店", "現地"})
    n_deictic = 0
    for c in person_career:
        if c.get("organization") in _deictic_orgs:
            c["organization"] = None
            n_deictic += 1
    for e in person_education:
        if e.get("institution") in _deictic_orgs:
            e["institution"] = None
            n_deictic += 1
    for c in (person_family_career or []):
        if c.get("organization") in _deictic_orgs:
            c["organization"] = None
            n_deictic += 1
    for e in (person_family_education or []):
        if e.get("institution") in _deictic_orgs:
            e["institution"] = None
            n_deictic += 1
    if n_deictic:
        logger.info(f"Nulled {n_deictic} deictic org names (本省/現社/本縣)")

    # 7. Organization deduplication
    logger.info("--- Step 7: Organization deduplication ---")
    orgs_table = build_organizations(person_career, person_education,
                                     person_family_career, person_family_education)
    logger.info(f"Deduplicated {len(orgs_table)} unique organizations")

    # 7b. Null org refs for family-relation org names (長男, 二女, 養子, etc.)
    n_relation_org = _null_relation_org_refs(
        orgs_table, person_education, person_family_education,
        person_career, person_family_career)
    if n_relation_org:
        logger.info(f"Nulled {n_relation_org} refs to family-relation org names")

    # 7c. Merge corporate-suffix variant orgs (横浜船渠 + 横浜船渠株 → one)
    n_corp = _merge_corporate_suffix_orgs(
        orgs_table, person_career, person_education,
        person_family_career, person_family_education)
    if n_corp:
        logger.info(f"Corporate suffix org merge: {n_corp} orgs merged, "
                    f"{len(orgs_table)} remain")

    # 8. Plausibility checks (after org dedup so we have org names)
    logger.info("--- Step 8: Plausibility checks ---")
    editorial_changes = run_plausibility_checks(
        person_core, person_education, person_career, person_family, orgs_table,
        person_family_education, person_family_career
    )
    if editorial_changes:
        write_jsonl(os.path.join(STRUCT_OUTPUT_DIR, "editorial_changes.jsonl"), editorial_changes)

    # 9. Org / job title translation → deferred to step 7 (after org merge)

    # 10. Improved org location assignment (geographic clustering + name-prefix fallback)
    logger.info("--- Step 10: Org location assignment ---")
    assign_org_locations(orgs_table, person_career, locations_table, geonames_idx, mcgd_idx)

    # 10b. Null org location_ids pointing to locations with org-type names
    n_orgname_loc = _null_orgname_location_refs(locations_table, orgs_table)
    if n_orgname_loc:
        logger.info(f"Nulled {n_orgname_loc} org location_ids pointing to org-name locations")

    # 10c. Fix ケ/ヶ misgeocoding + 東大曽根 wrong-ward
    n_ke_fix = _fix_misgeocoded_ke_locations(locations_table)
    if n_ke_fix:
        logger.info(f"Fixed {n_ke_fix} misgeocoded locations (ケ/ヶ stripping + 東大曽根 wrong ward)")

    # 11. Write output
    logger.info("--- Step 11: Writing output ---")
    write_jsonl(os.path.join(STRUCT_OUTPUT_DIR, "person_core.jsonl"), person_core)
    write_jsonl(os.path.join(STRUCT_OUTPUT_DIR, "person_career.jsonl"), person_career)
    write_jsonl(os.path.join(STRUCT_OUTPUT_DIR, "person_education.jsonl"), person_education)
    write_jsonl(os.path.join(STRUCT_OUTPUT_DIR, "person_hobbies.jsonl"), person_hobbies)
    write_jsonl(os.path.join(STRUCT_OUTPUT_DIR, "person_ranks.jsonl"), person_ranks)
    write_jsonl(os.path.join(STRUCT_OUTPUT_DIR, "person_religions.jsonl"), person_religions)
    write_jsonl(os.path.join(STRUCT_OUTPUT_DIR, "person_political_parties.jsonl"), person_political_parties)
    write_jsonl(os.path.join(STRUCT_OUTPUT_DIR, "person_family_members.jsonl"), person_family)
    write_jsonl(os.path.join(STRUCT_OUTPUT_DIR, "person_family_education.jsonl"), person_family_education)
    write_jsonl(os.path.join(STRUCT_OUTPUT_DIR, "person_family_career.jsonl"), person_family_career)
    write_jsonl(os.path.join(STRUCT_OUTPUT_DIR, "organizations.jsonl"), orgs_table)
    write_jsonl(os.path.join(STRUCT_OUTPUT_DIR, "locations.jsonl"), locations_table)

    logger.info("=== Step 6 complete ===")


if __name__ == "__main__":
    main()
