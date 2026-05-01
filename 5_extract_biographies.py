import json
import re
import os
import sys
import argparse
import yaml
import time
import logging
from json_repair import repair_json
from difflib import SequenceMatcher
import requests
from pathlib import Path
from config import (
    BIO_INPUT_ROOT, BIO_OUTPUT_FILE, BIO_OUTPUT_PATTERN, VOLUME_ID,
    BIO_MODEL_NAME, BIO_OLLAMA_BASE_URL, BIO_MAX_TOKENS,
    DEDUP_SIMILARITY_THRESHOLD,
)

OLLAMA_CHAT_URL = BIO_OLLAMA_BASE_URL.replace("/v1", "").rstrip("/") + "/api/chat"

logger = logging.getLogger(__name__)

# --- 1. SORTING HELPER ---
def get_sort_key(filepath):
    numbers = re.findall(r'\d+', str(filepath))
    return [int(n) for n in numbers]

# --- 2. THE STITCHING GENERATOR ---
def stream_stitched_entries(root_dir):
    json_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith("segmented_output.json"):
                json_files.append(Path(root) / file)

    json_files.sort(key=get_sort_key)
    print(f"Found {len(json_files)} source files.")

    last_valid_entry = None
    for filepath in json_files:
        source_page = filepath.parent.name  # page directory = original image name

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                segments = json.load(f)
        except Exception as e:
            print(f"Skipping corrupt file {filepath}: {e}")
            continue

        for segment in segments:
            segment['source_page'] = source_page
            seg_type = segment.get('type')
            body_text = segment.get('body_text', '')
            if seg_type == 'orphan':
                if last_valid_entry:
                    last_valid_entry['body_text'] += " " + body_text
            elif seg_type == 'standard':
                if last_valid_entry:
                    # Deduplicate: skip if this segment's text is near-identical
                    # to the previous entry (same row detected twice by YOLO)
                    new_body = body_text.strip()
                    old_body = last_valid_entry.get('body_text', '').strip()
                    if old_body and new_body:
                        sim = SequenceMatcher(None, old_body, new_body).ratio()
                        if sim >= DEDUP_SIMILARITY_THRESHOLD:
                            logger.info(
                                "Skipping duplicate row: %s/%s (%.0f%% similar to previous)",
                                source_page, segment.get('source_image'), sim * 100
                            )
                            continue
                    yield last_valid_entry
                last_valid_entry = segment

    if last_valid_entry:
        yield last_valid_entry

# --- 3. POST-PROCESSING ---
KINSHIP_SUFFIXES = re.compile(
    r'(?:長女|二女|三女|四女|五女|長男|二男|三男|四男|五男|'
    r'妹|姉|弟|兄|養子|養女)$'
)

def fix_family_names(parsed_json):
    """Null out family member names that are compound descriptors (e.g. '高木金之助三女')."""
    for fm in parsed_json.get("family_member", []):
        if not isinstance(fm, dict):
            continue
        name = fm.get("name")
        if name and len(name) > 2 and KINSHIP_SUFFIXES.search(name):
            fm["name"] = None
        # Recurse into nested family
        if "family_member" in fm:
            fix_family_names(fm)

KINSHIP_LIST = [
    "養父", "養母", "養子", "養女", "叔父", "叔母", "伯父", "伯母",
    "義父", "義母", "義兄", "義弟", "義姉", "義妹", "娘婿", "婿養子",
    "長男", "二男", "三男", "四男", "五男", "六男", "七男",
    "長女", "二女", "三女", "四女", "五女", "六女", "七女",
    "父", "母", "妻", "夫", "兄", "弟", "姉", "妹", "孫",
]

def _extract_kinship(raw):
    """Extract a single kinship term from a potentially messy relation string."""
    if not raw or len(raw) <= 2:
        return raw
    # 1. Strip parenthetical content
    cleaned = re.sub(r'[（(][^）)]*[）)]', '', raw).strip()
    if not cleaned:
        cleaned = raw
    # 2. Try END match (most common: "...三女")
    for term in KINSHIP_LIST:
        if cleaned.endswith(term):
            return term
    # 3. Try START match ("養父...", "妻の...")
    for term in KINSHIP_LIST:
        if cleaned.startswith(term):
            return term
    # 4. Find ANY match anywhere
    for term in KINSHIP_LIST:
        if term in cleaned:
            return term
    # 5. Fallback
    return raw[:2]

def _normalize_relations(family_members):
    """Recursively fix relation strings in family_member arrays."""
    for fm in family_members:
        if not isinstance(fm, dict):
            continue
        r = fm.get("relation")
        if r and len(r) > 2:
            fm["relation"] = _extract_kinship(r)
        if "family_member" in fm:
            _normalize_relations(fm["family_member"])

def _filter_dicts_recursive(items):
    """Filter a list to dicts only, and recursively filter nested family_member/career/education."""
    result = []
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in ("family_member", "career", "education"):
            nested = item.get(key)
            if isinstance(nested, list):
                item[key] = _filter_dicts_recursive(nested)
        result.append(item)
    return result

def normalize_schema(parsed_json):
    """Fix common type violations in extracted JSON."""
    # phone_number: list → joined string (flatten nested lists/dicts)
    pn = parsed_json.get("phone_number")
    if isinstance(pn, list):
        flat = []
        for item in pn:
            if isinstance(item, list):
                flat.extend(str(x) for x in item)
            elif isinstance(item, dict):
                flat.extend(str(v) for v in item.values() if v)
            else:
                flat.append(str(item))
        parsed_json["phone_number"] = "・".join(flat)
    elif isinstance(pn, (int, float)):
        parsed_json["phone_number"] = str(pn)
    # tax_amount: list → joined string (flatten nested lists/dicts)
    pn = parsed_json.get("tax_amount")
    if isinstance(pn, list):
        flat = []
        for item in pn:
            if isinstance(item, list):
                flat.extend(str(x) for x in item)
            elif isinstance(item, dict):
                flat.extend(str(v) for v in item.values() if v)
            else:
                flat.append(str(item))
        parsed_json["tax_amount"] = "・".join(flat)
    elif isinstance(pn, (int, float)):
        parsed_json["tax_amount"] = str(pn)
    # hobbies: ensure list
    hobbies = parsed_json.get("hobbies")
    if not isinstance(hobbies, list):
        parsed_json["hobbies"] = [hobbies] if isinstance(hobbies, str) and hobbies else []
    # religion: list → first element
    rel = parsed_json.get("religion")
    if isinstance(rel, list):
        parsed_json["religion"] = rel[0] if rel else None
    # political_party: list → first element
    pp = parsed_json.get("political_party")
    if isinstance(pp, list):
        parsed_json["political_party"] = pp[0] if pp else None
    # Normalize career, education, and family_member to lists of dicts only
    career_raw = parsed_json.get("career")
    parsed_json["career"] = [c for c in career_raw if isinstance(c, dict)] if isinstance(career_raw, list) else []
    edu_raw = parsed_json.get("education")
    parsed_json["education"] = [e for e in edu_raw if isinstance(e, dict)] if isinstance(edu_raw, list) else []
    fm_raw = parsed_json.get("family_member")
    parsed_json["family_member"] = _filter_dicts_recursive(fm_raw) if isinstance(fm_raw, list) else []
    # Fix string fields that LLM sometimes returns as lists (e.g. place_name)
    for career in parsed_json["career"]:
        for key in ("place_name", "job_title", "organization"):
            val = career.get(key)
            if isinstance(val, list):
                career[key] = val[0] if val else None
    for edu in parsed_json["education"]:
        for key in ("institution", "major_of_study"):
            val = edu.get(key)
            if isinstance(val, list):
                edu[key] = val[0] if val else None
    # Remove top-level relation key (belongs only inside family_member)
    parsed_json.pop("relation", None)
    # Strip any keys the LLM invented (e.g. "Son_of", "publications")
    _ALLOWED_TOP_KEYS = {
        "name", "rank", "place", "phone_number", "tax_amount", "political_party",
        "birth_year", "birth_year_raw", "origin_place", "hobbies", "religion",
        "education", "career", "family_member",
    }
    for key in list(parsed_json.keys()):
        if key not in _ALLOWED_TOP_KEYS:
            del parsed_json[key]
    # Fix relation strings inside family members (recursive)
    _normalize_relations(parsed_json.get("family_member", []))

# --- 3b. FIX ERA DATES IN WRONG FIELDS ---
# Abbreviated era dates like "明四", "大三" sometimes end up in institution, organization, or name fields.

_ERA_DATE_ONLY = re.compile(r'^(明|大|昭|同)[一二三四五六七八九十〇\d・]+$')

def fix_era_in_wrong_fields(parsed_json):
    """Relocate era-date strings misplaced in non-date fields."""
    for edu in parsed_json.get("education") or []:
        if not isinstance(edu, dict):
            continue
        inst = edu.get("institution")
        if inst and _ERA_DATE_ONLY.match(inst):
            if not edu.get("year_raw"):
                edu["year_raw"] = inst
            edu["institution"] = None
    for career in parsed_json.get("career") or []:
        if not isinstance(career, dict):
            continue
        org = career.get("organization")
        if org and _ERA_DATE_ONLY.match(org):
            if not career.get("start_year_raw"):
                career["start_year_raw"] = org
            career["organization"] = None
    _fix_era_in_family(parsed_json.get("family_member") or [])

def _fix_era_in_family(family_members):
    """Recursively fix era-date strings in family member fields."""
    for fm in family_members:
        if not isinstance(fm, dict):
            continue
        name = fm.get("name")
        if name and _ERA_DATE_ONLY.match(name):
            if not fm.get("birth_year_raw"):
                fm["birth_year_raw"] = name
            fm["name"] = None
        for edu in fm.get("education") or []:
            if not isinstance(edu, dict):
                continue
            inst = edu.get("institution")
            if inst and _ERA_DATE_ONLY.match(inst):
                if not edu.get("year_raw"):
                    edu["year_raw"] = inst
                edu["institution"] = None
        for career in fm.get("career") or []:
            if not isinstance(career, dict):
                continue
            org = career.get("organization")
            if org and _ERA_DATE_ONLY.match(org):
                if not career.get("start_year_raw"):
                    career["start_year_raw"] = org
                career["organization"] = None
        if "family_member" in fm:
            _fix_era_in_family(fm["family_member"])

# --- 3c. ERA DATE POST-PROCESSOR ---
# Deterministic conversion of raw era strings to Western years,
# overriding the LLM's (often wrong) arithmetic.

ERA_TABLE = {
    # Japanese eras
    "明治": 1867, "大正": 1911, "昭和": 1925, "慶應": 1864,
    "嘉永": 1847, "安政": 1853, "文久": 1860, "元治": 1863, "萬延": 1859,
    "天保": 1829, "弘化": 1843, "文政": 1817, "文化": 1803,
    # Manchukuo (満州国)
    "大同": 1931, "康德": 1933, "康徳": 1933,
    # Korean Empire (大韓帝國)
    "建陽": 1895, "光武": 1896, "隆熙": 1906,
    # Qing dynasty (清)
    "道光": 1820, "咸豐": 1850, "咸丰": 1850,
    "同治": 1861, "光緒": 1874, "光绪": 1874, "宣統": 1908, "宣统": 1908,
    # Republic of China (民國)
    "民國": 1911, "民国": 1911,
    # Abbreviated (single-kanji) forms for Japanese eras
    "明": 1867, "大": 1911, "昭": 1925,
}

# Kanji numeral → int lookup
_KANJI_DIGITS = {
    "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    "十": 10, "百": 100, "廿": 20, "卅": 30,
    "元": 1,  # 元年 = year 1
}

def _kanji_to_int(s):
    """Convert a kanji numeral string like '十四', '三十三', '元' to int."""
    if not s:
        return None
    s = s.strip()
    if s == "元":
        return 1
    # Try Arabic numerals first
    if s.isdigit():
        return int(s)
    # Parse kanji numeral with positional values
    result = 0
    current = 0
    for ch in s:
        if ch in ("十",):
            result += (current if current else 1) * 10
            current = 0
        elif ch == "廿":
            result += 20
            current = 0
        elif ch == "卅":
            result += 30
            current = 0
        elif ch == "百":
            result += (current if current else 1) * 100
            current = 0
        elif ch in _KANJI_DIGITS:
            current = _KANJI_DIGITS[ch]
        else:
            return None  # unrecognized character
    result += current
    return result if result > 0 else None

# Pattern: era name (1-2 chars) + number (kanji or Arabic) + 年
# Also handles 同 (same era as previous)
_ERA_PATTERN = re.compile(
    r'(同|' + '|'.join(sorted(ERA_TABLE.keys(), key=len, reverse=True)) + r')'
    r'\s*'
    r'(元|[〇一二三四五六七八九十廿卅百]+|\d+)'
    r'\s*年?'
)

def _parse_era_raw(raw_str):
    """Parse a raw era string and return (era_name, year_number) or (None, None).

    Returns the era name (or '同') and the integer year number.
    """
    if not raw_str or not isinstance(raw_str, str):
        return None, None
    m = _ERA_PATTERN.search(raw_str)
    if not m:
        return None, None
    era_name = m.group(1)
    num = _kanji_to_int(m.group(2))
    if num is None:
        return None, None
    return era_name, num

def _convert_era_to_western(era_name, num, prev_era=None):
    """Convert era name + number to Western year. Returns (western_year, resolved_era_name)."""
    if era_name == "同":
        if prev_era and prev_era in ERA_TABLE:
            return ERA_TABLE[prev_era] + num, prev_era
        return None, prev_era
    base = ERA_TABLE.get(era_name)
    if base is None:
        return None, era_name
    return base + num, era_name

def fix_era_dates(parsed_json):
    """Override LLM year conversions using deterministic era parsing.

    Processes all year fields in the extraction, resolving '同' references
    by tracking the most recently seen era name in document order.
    """
    prev_era = None  # track last era for 同 resolution

    # 1. Subject birth year
    raw = parsed_json.get("birth_year_raw")
    if raw:
        era_name, num = _parse_era_raw(raw)
        if era_name:
            western, prev_era = _convert_era_to_western(era_name, num, prev_era)
            if western:
                parsed_json["birth_year"] = western

    # 2. Education
    for edu in parsed_json.get("education") or []:
        raw = edu.get("year_raw")
        if raw:
            era_name, num = _parse_era_raw(raw)
            if era_name:
                western, prev_era = _convert_era_to_western(era_name, num, prev_era)
                if western:
                    edu["year_graduated"] = western

    # 3. Career
    for career in parsed_json.get("career") or []:
        raw = career.get("start_year_raw")
        if raw:
            era_name, num = _parse_era_raw(raw)
            if era_name:
                western, prev_era = _convert_era_to_western(era_name, num, prev_era)
                if western:
                    career["start_year"] = western

    # 4. Family members (recursive)
    _fix_era_dates_family(parsed_json.get("family_member") or [], prev_era)

def _fix_era_dates_family(family_members, prev_era):
    """Recursively fix era dates in family member records."""
    for fm in family_members:
        if not isinstance(fm, dict):
            continue
        raw = fm.get("birth_year_raw")
        if raw:
            era_name, num = _parse_era_raw(raw)
            if era_name:
                western, prev_era = _convert_era_to_western(era_name, num, prev_era)
                if western:
                    fm["birth_year"] = western
        for edu in fm.get("education") or []:
            raw = edu.get("year_raw")
            if raw:
                era_name, num = _parse_era_raw(raw)
                if era_name:
                    western, prev_era = _convert_era_to_western(era_name, num, prev_era)
                    if western:
                        edu["year_graduated"] = western
        for career in fm.get("career") or []:
            raw = career.get("start_year_raw")
            if raw:
                era_name, num = _parse_era_raw(raw)
                if era_name:
                    western, prev_era = _convert_era_to_western(era_name, num, prev_era)
                    if western:
                        career["start_year"] = western
        # Recurse into nested family
        if "family_member" in fm:
            _fix_era_dates_family(fm["family_member"], prev_era)


# --- 4. VALIDATION CHECK ---
def run_validation_check(original_text, parsed_json):
    career_markers = [
        "取締役", "社長", "部長", "課長", "事務官", "書記官", "技師",
        "會員", "勤務", "就任", "歷任", "歷職", "理事", "知事", "局長",
        "警部", "屬", "判官", "檢事", "參事", "主事", "囑託",
    ]
    text_hits = sum(original_text.count(m) for m in career_markers)
    json_count = len(parsed_json.get("career") or [])
    if text_hits > 5 and json_count < 3:
        return False, f"Suspected skipping ({text_hits} markers, {json_count} jobs)"
    return True, "Passed"

# --- 4. PROMPT FACTORY ---
def create_prompt(header, body_text):
    full_example_extraction = {
        "name": "安達左京",
        "rank": "從五勳六",
        "place": "臺北市幸町",
        "phone_number": "2801",
        "birth_year": 1869,
        "birth_year_raw": "明治二年",
        "origin_place": "大分縣北海部郡丹生村",
        "hobbies": ["碁", "打球"],
        "religion": "眞宗",
        "education": [{"institution": "東大", "major_of_study": "法科", "year_graduated": 1925, "year_raw": "大正十四年"}],
        "career": [
            {"job_title": "書記官", "organization": "總督府", "start_year": None, "start_year_raw": None, "place_name": "臺北市", "current": True},
            {"job_title": "課長", "organization": "財務局會計課", "start_year": None, "start_year_raw": None, "place_name": None},
            {"job_title": "警務課長", "organization": "總督府税關", "start_year": None, "start_year_raw": None, "place_name": "高雄州"},
            {"job_title": "地方課長", "organization": "總督府税關", "start_year": None, "start_year_raw": None, "place_name": "高雄州"},
            {"job_title": "事務官", "organization": "總督府", "start_year": None, "start_year_raw": None, "place_name": None},
            {"job_title": "庶務課長", "organization": "中央研究所", "start_year": None, "start_year_raw": None, "place_name": None},
            {"job_title": "勤務", "organization": "殖産局鑛務課", "start_year": None, "start_year_raw": None, "place_name": None},
            {"job_title": "總務課長", "organization": "工業研究所", "start_year": None, "start_year_raw": None, "place_name": None},
            {"job_title": "勤務", "organization": "殖產局商工課", "start_year": None, "start_year_raw": None, "place_name": None},
            {"job_title": "會計課長", "organization": "總督官房", "start_year": 1942, "start_year_raw": "昭和十七年", "place_name": None}
        ],
        "family_member": [
            {"name": "峯三郎", "birth_year": None, "birth_year_raw": None, "place": "大分縣", "relation": "父"},
            {"name": "マチ", "birth_year": 1873, "birth_year_raw": "明治六年", "relation": "母"},
            {"name": "河瀬四郎", "relation": "娘婿"},
            {
                "name": "キヨ",
                "education": [{"institution": "臺北二女"}],
                "birth_year": 1871,
                "birth_year_raw": "明治四年",
                "place": "徳島",
                "relation": "妻",
                "career": [{"job_title": "鑑定業者", "organization": "水先案内海軍"}]
            }
        ]
    }

    return f"""
You are a precision data entry robot. Your task is to extract Japanese biographical data.

### RULES
1. **NO OMISSIONS:** List EVERY job title, organization, and family member. Do not summarize or skip.
   Count every job title keyword (取締役, 社長, 部長, 課長, 事務官, 書記官, 技師, 局長, 勤務, etc.) in the source text. Your career array must have at least that many entries. Do not summarize or skip any position.
2. **ERA CONVERSION:** Convert all dates to Western Years (YYYY).
   - Formulas: 明/明治 N = 1867+N | 大/大正 N = 1911+N | 昭/昭和 N = 1925+N | 慶應 N = 1864+N
   - Additional Japanese eras: 嘉永 N = 1847+N | 安政 N = 1853+N | 文久 N = 1860+N | 元治 N = 1863+N | 萬延 N = 1859+N | 天保 N = 1829+N | 弘化 N = 1843+N | 文政 N = 1817+N | 文化 N = 1803+N
   - Manchukuo: 大同 N = 1931+N | 康德 N = 1933+N
   - Korean Empire: 建陽 N = 1895+N | 光武 N = 1896+N | 隆熙 N = 1906+N
   - Qing dynasty: 道光 N = 1820+N | 咸豐 N = 1850+N | 同治 N = 1861+N | 光緒 N = 1874+N | 宣統 N = 1908+N
   - Republic of China: 民國 N = 1911+N
   - Verify: Meiji dates → 1868–1912, Taisho → 1912–1926, Showa → 1926–1989. If a birth year falls outside 1820–1920, double-check your arithmetic.
   - **RAW ERA DATE:** For every year field, also output the raw era string exactly as it appears in the source
     text. Use these companion fields: `birth_year_raw`, `year_raw` (education), `start_year_raw` (career).
     Copy the era name + number verbatim, e.g. "大正九年", "明治三十三年", "同十二年", "昭二".
     If the source uses "同" (meaning "same era as previous"), output it as-is, e.g. "同十二年".
     If there is no date in the source, set the raw field to null.
3. **PARSING LOGIC:**
   - **place**: Current residence. Remove street numbers (e.g., "幸町一五〇" -> "幸町").
   - **birth_year**: The subject's birth year is stated in the narrative, typically after "の男", "の長男",
    "の弟", or similar kinship phrases. Look for an era name followed by a year number, e.g.:
    "明治十三年六月を以て生る" → 1867+13 = 1880, "慶應元年七月" → 1865+1-1 = 1865.
    The "生る/生れ" verb may be missing due to OCR errors — extract the year regardless.
    Additional era: 慶應 N = 1864+N.
   - **place_name**: If an organization starts with a location, use that as place_name (e.g.
    organization "臺灣製糖", set place_name to "臺灣", organization "東京高商", set place_name to "東京").
    If place_name is blank, check again if the extracted organization starts with a potential place_name.
    Sanity check for plausible place_names, for example, the full string "雲原商會常任監査基隆輕鐡常務臺陽鑛業監査等歷職"
    is clearly too long and not the name of an existing place.
   - **job_title**: Derive from compound titles (e.g. "日報編輯長" -> Org: "日報", Title: "編輯長").
   - **era abbreviations are dates, not names**: Short strings that are just an era
     abbreviation followed by numbers (e.g., "明四", "大三", "大一", "昭二", "明三〇")
     are abbreviated dates (明治四年, 大正三年, etc.), NOT institution, organization, or
     personal names. Place them in the appropriate date field based on context
     (year_raw, start_year_raw, birth_year_raw). Real names contain descriptive characters
     beyond the era prefix: "明治大學", "明大", "大正大學", "臺北二女", "明治學院".
   - **phone_number**: Convert Kanji numerals (〇-九) to digits (0-9).
   - **tax_amount**: Convert Kanji numerals (〇-九) to digits (0-9).
   - **family_member**: Include family career/education inside the family object. The name entry of family members
    must be personal names only (given names like "マチ", "キヨ", "茂", "和子") — never compound descriptors.
    Watch for this pattern: "Surname+GivenName+relation" describes WHO someone is relative to, not their name.
    Examples:
      • "梅本仲藏長女" → 梅本仲藏 is the father's name, 長女 means "eldest daughter". The wife's personal name
        is stated elsewhere or is null. Do NOT use "梅本仲藏長女" as the name field.
      • "高木金之助三女" → 高木金之助 is a parent's name, 三女 = "third daughter". Name is null or found elsewhere.
      • "木村兵右衛門四女" → same pattern. 四女 = relation, name is separate.
      • "長女子" → "長女" is the relation, "子" or the following text is the name.
    If the text only says "SomeoneのN女" without giving a personal name, set name to null and use
    origin_person info in the place field if relevant.
    RED FLAG: If a name field ends with a kinship/relation word such as 長女, 二女, 三女, 四女, 五女,
    長男, 二男, 三男, 四男, 五男, 妹, 姉, 弟, 兄, 養子, 養女, etc., it is ALWAYS wrong — these
    describe a relationship to someone, not a personal name. Set name to null.
    Examples of WRONG names → correct action:
      • "高木金之助三女" → name=null (means "third daughter of 高木金之助")
      • "藤村佐太郎妹" → name=null (means "sister of 藤村佐太郎")
      • "加藤ティの養子" → name=null (means "adopted child of 加藤ティ")
      • "新田增太郎長男" → name=null (means "eldest son of 新田增太郎")
      • "木村齊次長女" → name=null (means "eldest daughter of 木村齊次")
    Similarly, institution/school names (e.g., "東洋英和女學校卒") are NOT personal names — set name to null.
    Other invalid names: "次彥昭四" ("昭四" is a birth_year), "敦子天葚臺北一高女在" (contains institution "臺北一高女").
   - **nested family**: When a family member's own spouse, children, or relatives are described
    (e.g., "叔父X の妻Y" or "叔父X 妻Y"), nest them inside that family member's object using a
    `family_member` array. Only direct relatives of the subject go in the top-level family_member list.
   - **relation**: must be EXACTLY one kinship term: 父, 母, 妻, 長男, 二男, 三男, 長女, 二女, 三女, 養子, 叔父, etc.
    Maximum 2 characters. Never include names or parentheticals — strip '養父先代定吉三女' down to '三女'.
   - **current**: Set to true on the career entry that represents the subject's current position (現職).
    Omit or set to false on all other career entries.
4. **TYPE CONSTRAINTS:**
   - `phone_number`: must be a single string, never an array. Multiple numbers → join with "・".
   - `tax_amount`: must be a single string, never an array.
   - `hobbies`: must be an array, use `[]` if none mentioned, never `null`.
   - `political_party`: must be a single string, never an array. Extract the party name if mentioned.
   - `religion`: must be a single string, never an array.
5. **NO NEW OBSERVATION TYPES:** The only allowed categories are name, rank, place, phone_number, tax_amount,
    political_party, birth_year, birth_year_raw, origin_place, hobbies, religion,
    education (institution, major_of_study, year_graduated, year_raw),
    career (job_title, organization, start_year, start_year_raw, place_name, current),
    family_member (name, birth_year, birth_year_raw, place, relation,
    education (institution, major_of_study, year_graduated, year_raw),
    career (job_title, organization, start_year, start_year_raw, place_name),
    family_member (recursive, for nested relatives)).
6. **LIGHT OCR GUESSWORK:** The text you analyze comes from the non-perfect OCR of an old document.
    Some light corrections are allowed if it is clear from the context that a Chinese character was
    originally misrecognized in the context of the string of which is is part of.
7. **NO ENUMERATIONS:** There cannot be more than one string per observation. In other words, there
    cannot be any commas or semicolons. Likewise, "等", "兼", and whitespaces are indicators for enumerations.
    In these cases, divide the observations. The character "各" often indicates enumerations preceding "各",
    for example, "打狗烏樹林北門嶼臺南嘉義各支局" means 5 different 支局, located respectively in 打狗島, 樹林,
    北門嶼, 臺南, and 嘉義. Likewise, "臺中高雄各地方法院" means two different 法院: one in 臺中 and one in 高雄.
    IMPORTANT: Even without "各", consecutive place names before a shared title must be split into separate
    entries. For example, "埼玉新潟縣理事官島根廣島各縣警察部長" contains FOUR career entries:
    埼玉縣 理事官, 新潟縣 理事官, 島根縣 警察部長, 廣島縣 警察部長. Each place name gets its own career entry
    with the applicable job title.
8. **SANITY CHECK: NO UNLIKELY STRINGS:** If a detected string is over 20 characters long, you must try to
    divide the string further into different observation types. If you cannot, skip that entry. Overly long
    strings are an indicator that a string may not yet have been divided into sufficient observation types.
    For example, "天)京大卒 三男 徹男天八)臺北大卒總督府動 四男正美天" is clearly not the name of a person. There are
    two persons here, one with family relation "三男" and one with the family relation "四男".

### EXAMPLE (ONE-SHOT)
INPUT HEADER: 安達左京
INPUT BODY: 從五勳六 總督府書 記官 財務局會計課 臺北市幸町一五〇官舎 電二八〇一 [閲歴]大分縣峯三郎の男明治二年四月十七日同縣北海部郡丹生村に生る大正十四年東大法科卒業總督府に入り税關屬高雄州警務課長同地方課長稅關事務官等歷職 總督府事務官中央研究所庶務課長兼殖産局鑛務課勤務工業研究所總務課長兼殖產局商工課勤務總督官房會計課長を経て昭和十七年十一月現職 宗教眞宗 趣味碁 打球【家庭]母マチ(明大) 妻キヨ(明四)徳島 水先案内海軍鑑定業者 臺北二女卒

OUTPUT JSON:
{json.dumps(full_example_extraction, ensure_ascii=False, indent=2)}

### TARGET DATA
HEADER: {header}
BODY: {body_text}
"""

_BIO_RESTART_RE = re.compile(
    r'(?:從[一二三四五六七八九\d]+位|正[一二三四五六七八九\d]+位|'
    r'勲[一二三四五六七八九\d]+等|大勲位|'
    r'所得[稅税]|'
    r'電[一二三四五六七八九〇\d]+)'
)

def truncate_merged_entries(body_text):
    """Cut off a second biography that was accidentally merged in.

    After the family section (家庭) + buffer, if biography-opening markers
    (court rank, decoration, tax_amount) reappear, truncate before them.
    """
    if len(body_text) < 2000:
        return body_text
    family_pos = body_text.find('家庭')
    if family_pos == -1:
        return body_text
    match = _BIO_RESTART_RE.search(body_text, family_pos + 50)
    if match:
        return body_text[:match.start()].rstrip()
    return body_text


def clean_json_string(text):
    # Strip thinking tags (Qwen 3.5 may emit these despite think:false)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Strip markdown code fences first so brace counting sees clean text
    text = re.sub(r'```(?:json)?', '', text)

    # Find the last balanced top-level { ... } using string-aware bracket counting.
    last_obj = None
    i = 0
    while i < len(text):
        if text[i] == '{':
            stack = ['}']
            start = i
            i += 1
            in_str = False
            esc = False
            while i < len(text) and stack:
                ch = text[i]
                if esc:
                    esc = False
                elif ch == '\\' and in_str:
                    esc = True
                elif ch == '"':
                    in_str = not in_str
                elif not in_str:
                    if ch == '{': stack.append('}')
                    elif ch == '[': stack.append(']')
                    elif ch in ('}', ']'):
                        if stack[-1] == ch:
                            stack.pop()
                        else:
                            break  # mismatch — not truly balanced
                i += 1
            if not stack:
                last_obj = text[start:i]
        else:
            i += 1
    if last_obj:
        text = last_obj
    else:
        # No balanced {} found — try to repair truncated JSON
        text = _repair_truncated_json(text)
    return text.replace("None", "null").replace("True", "true").replace("False", "false").strip()


def _repair_truncated_json(text):
    """Close unclosed brackets/braces in truncated JSON so it can still parse."""
    start = text.find('{')
    if start == -1:
        return text
    text = text[start:]

    # Track last position where an element is definitely complete:
    # after , (previous element done), after { or [ (container opened),
    # after } or ] (container closed).
    stack = []
    in_string = False
    escape = False
    last_cut = 1  # right after opening {

    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{': stack.append('}'); last_cut = i + 1
        elif ch == '[': stack.append(']'); last_cut = i + 1
        elif ch in ('}', ']'):
            if stack:
                if stack[-1] == ch:
                    stack.pop()
                else:
                    # Mismatched closer (e.g. } when ] expected): auto-close expected first
                    stack.pop()
                    if stack and stack[-1] == ch:
                        stack.pop()
            last_cut = i + 1
        elif ch == ',':
            last_cut = i + 1

    if not stack:
        return text  # already balanced

    # Truncate to last safe point, strip trailing comma, recount stack
    text = text[:last_cut].rstrip().rstrip(',')
    stack = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{': stack.append('}')
        elif ch == '[': stack.append(']')
        elif ch in ('}', ']') and stack:
            if stack[-1] == ch:
                stack.pop()
            else:
                stack.pop()
                if stack and stack[-1] == ch:
                    stack.pop()

    text += ''.join(reversed(stack))
    return text


def parse_llm_json(raw_content):
    """Parse LLM output into a Python dict, with multi-stage fallback.

    Pipeline: raw json.loads -> clean json.loads -> repair_json(clean)
              -> repair_json(raw) -> yaml.safe_load
    """
    # Stage 0: try raw content directly (format:"json" should produce valid JSON)
    raw_stripped = raw_content.strip()
    if raw_stripped.startswith('{'):
        try:
            return json.loads(raw_stripped, strict=False)
        except json.JSONDecodeError:
            pass

    # Stage 1: clean then parse
    clean_text = clean_json_string(raw_content)
    if not clean_text:
        raise ValueError(f"No JSON in response. Raw: {raw_content[:300]!r}")
    try:
        return json.loads(clean_text, strict=False)
    except json.JSONDecodeError:
        pass

    # Stage 2: repair cleaned text
    try:
        result = repair_json(clean_text, return_objects=True)
        if isinstance(result, dict):
            return result
    except Exception:
        pass

    # Stage 3: repair raw text (clean_json_string may have mangled it)
    try:
        result = repair_json(raw_stripped, return_objects=True)
        if isinstance(result, dict):
            return result
    except Exception:
        pass

    # Stage 4: YAML fallback
    result = yaml.safe_load(clean_text)
    if not isinstance(result, dict):
        raise ValueError(f"All parsers failed. clean_text[:200]: {clean_text[:200]!r}")
    return result


def call_ollama(system_prompt, user_prompt, timeout_sec=45):
    """Call Ollama API with a hard wall-clock timeout via Popen + poll loop.

    TIMEOUT STRATEGY — v8-popen-poll-debug (do NOT change without testing):
    Previous approaches that FAILED on this WSL2/GDrive system:
      1. signal.SIGALRM — can't interrupt C-level blocking I/O
      2. socket.settimeout — fragile across urllib3 versions
      3. stream=True + deadline — check only runs between chunks
      4. subprocess.communicate(timeout=) — hangs on WSL2
      5. os.system + Linux timeout — works once then hangs on 2nd call
      6. Popen + poll + file I/O — Entry 1 OK, Entry 40 hangs (investigating)
    """
    import subprocess, tempfile, signal

    cfg = {
        "url": OLLAMA_CHAT_URL,
        "model": BIO_MODEL_NAME,
        "system": system_prompt,
        "user": user_prompt,
        "max_tokens": BIO_MAX_TOKENS,
    }
    cfg_path = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(cfg, cfg_path)
    cfg_path.close()

    inline_code = (
        'import json,sys,requests\n'
        'with open(sys.argv[1]) as f: cfg=json.load(f)\n'
        'import os; os.unlink(sys.argv[1])\n'
        'r=requests.post(cfg["url"],json={"model":cfg["model"],'
        '"messages":[{"role":"system","content":cfg["system"]},'
        '{"role":"user","content":cfg["user"]}],'
        '"format":"json","think":False,"stream":False,'
        '"options":{"temperature":0.0,"num_predict":cfg["max_tokens"]}},timeout=300)\n'
        'r.raise_for_status()\n'
        'print(r.json()["message"]["content"])\n'
    )

    out_path = tempfile.NamedTemporaryFile(mode='w', suffix='.out', delete=False).name
    err_path = tempfile.NamedTemporaryFile(mode='w', suffix='.err', delete=False).name

    fout = open(out_path, 'w')
    ferr = open(err_path, 'w')
    try:
        proc = subprocess.Popen(
            [sys.executable, '-c', inline_code, cfg_path.name],
            stdout=fout, stderr=ferr,
            start_new_session=True,
        )
    except Exception:
        fout.close(); ferr.close()
        for p in (cfg_path.name, out_path, err_path):
            try: os.unlink(p)
            except OSError: pass
        raise

    deadline = time.time() + timeout_sec
    while proc.poll() is None:
        if time.time() >= deadline:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    proc.kill()
                except (ProcessLookupError, PermissionError):
                    pass
            proc.wait()
            fout.close(); ferr.close()
            for p in (cfg_path.name, out_path, err_path):
                try: os.unlink(p)
                except OSError: pass
            raise TimeoutError(f"Ollama request killed after {timeout_sec}s")
        time.sleep(0.5)

    fout.close(); ferr.close()
    exit_code = proc.returncode

    with open(out_path, 'r') as f:
        stdout = f.read()
    with open(err_path, 'r') as f:
        stderr = f.read()
    for p in (cfg_path.name, out_path, err_path):
        try: os.unlink(p)
        except OSError: pass

    if exit_code != 0:
        raise RuntimeError(f"Ollama failed (rc={exit_code}): {stderr[:500]}")

    return stdout


# --- 5. EXECUTION ---
def load_completed_indices(filepath):
    """Read existing output file and return the set of entry_indices
    that completed successfully (have 'extraction' key).
    Also rewrites the file to remove error-only records."""
    if not os.path.exists(filepath):
        return set()

    successful = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if 'extraction' in rec:
                    successful.append(line)
            except json.JSONDecodeError:
                continue

    # Rewrite file with only successful records
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in successful:
            f.write(line + '\n')

    return {json.loads(line)['entry_index'] for line in successful}


def process_biographies(volume=None):
    if volume is None:
        volume = VOLUME_ID
    output_file = BIO_OUTPUT_PATTERN.format(volume=volume)
    # --- GPU status check ---
    try:
        ps = requests.get(OLLAMA_CHAT_URL.rsplit("/", 1)[0] + "/ps", timeout=5).json()
        for m in ps.get("models", []):
            vram = m.get("size_vram", 0)
            total = m.get("size", 0)
            pct = (vram / total * 100) if total else 0
            device = "GPU" if pct > 50 else "CPU"
            print(f"Ollama model: {m['name']}  |  VRAM: {vram/1e9:.1f}/{total/1e9:.1f} GB ({pct:.0f}%)  |  device: {device}")
        if not ps.get("models"):
            print("Warning: No models loaded in Ollama yet (will load on first call).")
    except Exception as e:
        print(f"Warning: Could not query Ollama status: {e}")

    print(f"Starting Extraction Loop: {BIO_MODEL_NAME} [timeout: v8-debug]")
    print(f"Output file: {output_file}")

    completed = load_completed_indices(output_file)
    if completed:
        print(f"Found {len(completed)} completed records, will retry any previous errors.")

    with open(output_file, 'a', encoding='utf-8') as out_f:
        processed, errors = 0, 0
        for i, entry in enumerate(stream_stitched_entries(BIO_INPUT_ROOT)):
            if i in completed:
                continue

            header = entry.get('header_ocr', '').replace('\n', '')
            body = entry.get('body_text', '').replace('\n', ' ')
            if len(body) < 5:
                continue
            body = truncate_merged_entries(body)

            display_header = re.sub(r'\s+', '', header)[:12]
            print(f"Entry {i+1}: {display_header}...", end=" ", flush=True)

            try:
                raw_content = call_ollama(
                    "You are a robotic data parser. Output valid JSON only. Follow conversion rules strictly. Do not invent new fields.",
                    create_prompt(header, body),
                )

                parsed_json = parse_llm_json(raw_content)
                fix_family_names(parsed_json)
                normalize_schema(parsed_json)
                fix_era_in_wrong_fields(parsed_json)
                fix_era_dates(parsed_json)

                # Retry if LLM returned wrong schema (no name key)
                if "name" not in parsed_json:
                    print("RETRY (wrong schema)...", end=" ", flush=True)
                    raw_content = call_ollama(
                        "You are a robotic data parser. Output valid JSON only. Follow conversion rules strictly. Do not invent new fields.",
                        create_prompt(header, body),
                    )
                    parsed_json = parse_llm_json(raw_content)
                    fix_family_names(parsed_json)
                    normalize_schema(parsed_json)
                    fix_era_in_wrong_fields(parsed_json)
                    fix_era_dates(parsed_json)
                    if "name" not in parsed_json:
                        raise ValueError("Wrong schema after retry")

                # Name fallback from header OCR
                if not parsed_json.get("name") and header:
                    parsed_json["name"] = header

                passed_val, val_msg = run_validation_check(body, parsed_json)

                record = {
                    "entry_index": i,
                    "source_image": entry.get('source_image'),
                    "source_page": entry.get('source_page'),
                    "header_ocr": header,
                    "validation": val_msg,
                    "extraction": parsed_json
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                out_f.flush()
                processed += 1
                print(f"{'OK' if passed_val else 'WARN'} ({val_msg})")

            except Exception as e:
                errors += 1
                print(f"ERROR: {e}")

    print(f"Done. {processed} extracted, {errors} errors (will be retried on next run).")

def reprocess_existing(volume=None):
    """Re-run post-processing on all existing records without re-calling LLM."""
    if volume is None:
        volume = VOLUME_ID
    output_file = BIO_OUTPUT_PATTERN.format(volume=volume)
    if not os.path.exists(output_file):
        print(f"No output file found: {output_file}")
        return

    records = []
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"Reprocessing {len(records)} records from {output_file}")
    fixed_relations, fixed_names, flagged = 0, 0, 0

    for rec in records:
        extraction = rec.get("extraction")
        if not extraction or not isinstance(extraction, dict):
            rec["needs_reextraction"] = True
            flagged += 1
            continue

        # Re-run post-processing
        fix_family_names(extraction)
        normalize_schema(extraction)
        fix_era_in_wrong_fields(extraction)
        fix_era_dates(extraction)

        # Name fallback from header_ocr
        header = rec.get("header_ocr", "")
        if not extraction.get("name") and header:
            extraction["name"] = header
            fixed_names += 1

        # Flag wrong-schema records
        if "name" not in extraction:
            rec["needs_reextraction"] = True
            flagged += 1

    # Write back
    with open(output_file, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"Done. Fixed {fixed_names} empty names, flagged {flagged} for re-extraction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract biographies from OCR results")
    parser.add_argument("--volume", default=VOLUME_ID, help=f"Volume identifier (default: {VOLUME_ID})")
    parser.add_argument("--reprocess", action="store_true", help="Re-run post-processing on existing records")
    args = parser.parse_args()
    if args.reprocess:
        reprocess_existing(volume=args.volume)
    else:
        process_biographies(volume=args.volume)
