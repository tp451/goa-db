"""
Step 8: Classify occupations (HISCO) and industries (ISIC).

Part A — Qwen via Ollama: assigns HISCO major group codes to each career record.
Part B — Qwen via Ollama: assigns ISIC Rev 4 section codes to each organization.
Part C — Writes enriched JSONL back to disambiguated/.
"""
import json
import os
import re
import logging
import time

import requests

from config import (
    DISAMBIG_OUTPUT_DIR,
    BIO_OLLAMA_BASE_URL, BIO_MODEL_NAME, ISIC_LLM_MAX_TOKENS,
)

OLLAMA_CHAT_URL = BIO_OLLAMA_BASE_URL.replace("/v1", "").rstrip("/") + "/api/chat"

CAREER_PATH = os.path.join(DISAMBIG_OUTPUT_DIR, "person_career.jsonl")
ORG_PATH = os.path.join(DISAMBIG_OUTPUT_DIR, "organizations.jsonl")
HISCO_CACHE_PATH = os.path.join(DISAMBIG_OUTPUT_DIR, "_cache_hisco.json")
ISIC_CACHE_PATH = os.path.join(DISAMBIG_OUTPUT_DIR, "_cache_isic.json")

# ==========================================
# LOGGING
# ==========================================
os.makedirs(DISAMBIG_OUTPUT_DIR, exist_ok=True)
log_dir = os.path.join(DISAMBIG_OUTPUT_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"classify_{time.strftime('%Y%m%d_%H%M%S')}.log")

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
# ISIC REV 4 SECTIONS
# ==========================================
ISIC_SECTIONS = {
    "A": "Agriculture, forestry and fishing",
    "B": "Mining and quarrying",
    "C": "Manufacturing",
    "D": "Electricity, gas, steam and air conditioning supply",
    "E": "Water supply; sewerage, waste management and remediation activities",
    "F": "Construction",
    "G": "Wholesale and retail trade; repair of motor vehicles and motorcycles",
    "H": "Transportation and storage",
    "I": "Accommodation and food service activities",
    "J": "Information and communication",
    "K": "Financial and insurance activities",
    "L": "Real estate activities",
    "M": "Professional, scientific and technical activities",
    "N": "Administrative and support service activities",
    "O": "Public administration and defence; compulsory social security",
    "P": "Education",
    "Q": "Human health and social work activities",
    "R": "Arts, entertainment and recreation",
    "S": "Other service activities",
    "T": "Activities of households as employers",
    "U": "Activities of extraterritorial organizations and bodies",
}

ISIC_LIST_STR = "\n".join(f"{k} - {v}" for k, v in ISIC_SECTIONS.items())

# ==========================================
# I/O
# ==========================================
def load_jsonl(path) -> list[dict]:
    records = []
    if not os.path.exists(path):
        logger.warning(f"File not found: {path}")
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(records)} records to {path}")


def load_cache(path) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(path, cache):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ==========================================
# HISCO MINOR GROUPS (ISCO-68 based)
# ==========================================
# Major groups 0/1 share "Professional, Technical and Related Workers"
# Source: ILO ISCO-68, van Leeuwen/Maas/Miles (2002)
HISCO_MINOR_GROUPS = {
    "01": "Physical scientists and related technicians",
    "02": "Architects, engineers and related technicians",
    "03": "Aircraft and ship officers",
    "04": "Life scientists and related technicians",
    "05": "Medical, dental, veterinary and related workers",
    "06": "Statisticians, mathematicians, systems analysts",
    "07": "Economists",
    "08": "Accountants",
    "09": "Jurists",
    "11": "Teachers",
    "12": "Workers in religion",
    "13": "Authors, journalists and related writers",
    "14": "Sculptors, painters, photographers, creative artists",
    "15": "Composers, performing artists",
    "16": "Athletes, sportsmen and related workers",
    "17": "Professional workers not elsewhere classified",
    "20": "Legislative officials and government administrators",
    "21": "General managers",
    "30": "Clerical supervisors",
    "31": "Government executive officials",
    "32": "Stenographers, typists and teletypists",
    "33": "Bookkeepers, cashiers and related workers",
    "34": "Computing machine operators",
    "35": "Transport and communications supervisors",
    "36": "Transport conductors",
    "37": "Mail distribution clerks",
    "38": "Telephone and telegraph operators",
    "39": "Clerical workers not elsewhere classified",
    "40": "Managers (wholesale and retail trade)",
    "41": "Working proprietors (wholesale and retail trade)",
    "42": "Sales supervisors and buyers",
    "43": "Technical salesmen, commercial travellers",
    "44": "Insurance, real estate, securities, business services salesmen",
    "45": "Salesmen, shop assistants and related workers",
    "49": "Sales workers not elsewhere classified",
    "50": "Managers (catering and lodging services)",
    "51": "Working proprietors (catering and lodging services)",
    "52": "Housekeeping and related service supervisors",
    "53": "Cooks, waiters, bartenders and related workers",
    "54": "Maids and related housekeeping service workers",
    "55": "Building caretakers, charworkers, cleaners",
    "56": "Launderers, dry-cleaners and pressers",
    "57": "Hairdressers, barbers, beauticians",
    "58": "Protective service workers (police, fire, guards)",
    "59": "Service workers not elsewhere classified",
    "60": "Farm managers and supervisors",
    "61": "Farmers",
    "62": "Agricultural and animal husbandry workers",
    "63": "Forestry workers",
    "64": "Fishermen, hunters and related workers",
    "70": "Production supervisors and general foremen",
    "71": "Miners, quarrymen, well drillers",
    "72": "Metal processers",
    "73": "Wood preparation workers, paper makers",
    "74": "Chemical processers and related workers",
    "75": "Spinners, weavers, knitters, dyers",
    "76": "Tanners, fellmongers, pelt dressers",
    "77": "Food and beverage processers",
    "78": "Tobacco preparers and tobacco product makers",
    "79": "Tailors, dressmakers, sewers, upholsterers",
    "80": "Shoemakers and leather goods makers",
    "81": "Cabinetmakers, woodworkers",
    "82": "Stone cutters and carvers",
    "83": "Blacksmiths, toolmakers, machine-tool operators",
    "84": "Machinery fitters, machine assemblers",
    "85": "Electrical fitters and related workers",
    "86": "Broadcasting station, sound equipment operators",
    "87": "Plumbers, welders, sheet metal and structural metal workers",
    "88": "Jewellery and precious metal workers",
    "89": "Glass formers, potters, related workers",
    "90": "Rubber and plastics product makers",
    "91": "Paper and paperboard products makers",
    "92": "Printers and related workers",
    "93": "Painters",
    "94": "Production workers not elsewhere classified",
    "95": "Bricklayers, carpenters, construction workers",
    "96": "Stationary engine and related equipment operators",
    "97": "Material handling and related equipment operators",
    "98": "Transport equipment operators",
    "99": "Labourers not elsewhere classified",
}

HISCO_MAJOR_LABELS = {
    "0": "Professional/Technical",
    "1": "Professional/Technical",
    "2": "Administrative/Managerial",
    "3": "Clerical",
    "4": "Sales",
    "5": "Service",
    "6": "Agricultural/Forestry/Fishing",
    "7": "Production/Transport",
    "8": "Production/Transport",
    "9": "Production/Transport",
}

HISCO_LIST_STR = "\n".join(f"{k} - {v}" for k, v in HISCO_MINOR_GROUPS.items())


# ==========================================
# PART A: HISCO via Qwen / Ollama
# ==========================================
CLASSIFY_BATCH_SIZE = 10

_HISCO_SYSTEM_PROMPT = (
    "You are an occupation classification expert. "
    "For each numbered job title, assign exactly one HISCO minor group code (two digits). "
    "Reply with ONLY the number, the two-digit code, and the group name on one line.\n"
    "Example:\n1. 21 - General managers\n2. 11 - Teachers\n\n"
    "Important distinctions:\n"
    "- Code 20 for senior government leaders: governors, mayors, ministers, ambassadors, "
    "prefects, legislators, commissioners, directors of government bureaus.\n"
    "- Code 21 for department heads, section chiefs, branch managers, "
    "and supervisory roles in organisations.\n"
    "- Code 31 only for mid-level government clerks handling routine "
    "administrative paperwork (not senior officials).\n"
    "- Code 58 for military officers, police, guards, and defense roles: "
    "generals, admirals, commanders, adjutants, military attachés.\n"
    "- Code 39 is a last resort — prefer a specific code when possible.\n\n"
    f"HISCO minor groups:\n{HISCO_LIST_STR}"
)


def classify_hisco(career_records: list[dict]) -> list[dict]:
    """Add hisco_code and hisco_major to each career record via batched LLM."""
    cache = load_cache(HISCO_CACHE_PATH)

    # Collect unique titles not yet cached
    all_titles = {r.get("job_title_en", "") for r in career_records}
    all_titles.discard("")
    all_titles.discard(None)
    uncached = [t for t in all_titles if t not in cache]
    total = len(uncached)

    if uncached:
        logger.info(f"Classifying {total} unique titles via LLM ({len(all_titles)} total unique, "
                     f"{len(all_titles) - total} cached)")

        num_batches = (total + CLASSIFY_BATCH_SIZE - 1) // CLASSIFY_BATCH_SIZE
        for batch_idx in range(num_batches):
            start = batch_idx * CLASSIFY_BATCH_SIZE
            batch = uncached[start:start + CLASSIFY_BATCH_SIZE]

            user_content = "\n".join(f"{i}. {title}" for i, title in enumerate(batch, 1))

            try:
                resp = requests.post(OLLAMA_CHAT_URL, json={
                    "model": BIO_MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": _HISCO_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "think": False,
                    "stream": False,
                    "options": {"temperature": 0.0, "top_k": 1,
                                "num_predict": len(batch) * 50},
                }, timeout=120)
                resp.raise_for_status()
                content = resp.json()["message"].get("content", "")

                for line in content.splitlines():
                    m = re.match(r"(\d+)\.\s*(\d{2})\b", line)
                    if m:
                        idx = int(m.group(1)) - 1
                        code = m.group(2)
                        if 0 <= idx < len(batch) and code in HISCO_MINOR_GROUPS:
                            cache[batch[idx]] = code

            except Exception as e:
                logger.error(f"HISCO batch {batch_idx + 1}/{num_batches} failed: {e}")

            # Mark unparsed items
            for title in batch:
                if title not in cache:
                    cache[title] = ""

            done = min((batch_idx + 1) * CLASSIFY_BATCH_SIZE, total)
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
                logger.info(f"HISCO: {done}/{total} titles classified")
                save_cache(HISCO_CACHE_PATH, cache)

        save_cache(HISCO_CACHE_PATH, cache)
        logger.info(f"HISCO cache saved ({len(cache)} entries)")
    else:
        logger.info(f"All {len(all_titles)} unique titles already cached")

    # Apply to records
    for rec in career_records:
        title = rec.get("job_title_en", "")
        code = cache.get(title, "")
        rec["hisco_code"] = code
        rec["hisco_major"] = code[0] if code else ""

    classified = sum(1 for r in career_records if r["hisco_code"])
    logger.info(f"HISCO: {classified}/{len(career_records)} career records classified")
    return career_records


# ==========================================
# PART B: ISIC via Qwen / Ollama
# ==========================================
_ISIC_SYSTEM_PROMPT = (
    "You are an industry classification expert. "
    "For each numbered organization, assign exactly one ISIC Rev 4 section code (A-U). "
    "Reply with ONLY the number, the letter code, and the section name on one line.\n"
    "Example:\n1. C - Manufacturing\n2. P - Education\n\n"
    f"ISIC Rev 4 sections:\n{ISIC_LIST_STR}"
)


def classify_isic(org_records: list[dict]) -> list[dict]:
    """Add isic_section and isic_label to each organization record via batched LLM."""
    cache = load_cache(ISIC_CACHE_PATH)

    # Build list of (index, cache_key, display_name) for uncached orgs
    uncached = []
    for i, org in enumerate(org_records):
        name = org.get("name", "")
        name_en = org.get("name_en", "")
        cache_key = name or name_en
        if cache_key and cache_key in cache:
            org["isic_section"] = cache[cache_key]["section"]
            org["isic_label"] = cache[cache_key]["label"]
        else:
            org_display = f"{name} ({name_en})" if name_en else name
            uncached.append((i, cache_key, org_display))

    total = len(uncached)
    logger.info(f"ISIC: {len(org_records)} orgs, {len(org_records) - total} cached, {total} remaining")

    if uncached:
        num_batches = (total + CLASSIFY_BATCH_SIZE - 1) // CLASSIFY_BATCH_SIZE
        for batch_idx in range(num_batches):
            start = batch_idx * CLASSIFY_BATCH_SIZE
            batch = uncached[start:start + CLASSIFY_BATCH_SIZE]

            user_content = "\n".join(f"{j}. {display}" for j, (_, _, display) in enumerate(batch, 1))

            results = [("", "")] * len(batch)
            try:
                resp = requests.post(OLLAMA_CHAT_URL, json={
                    "model": BIO_MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": _ISIC_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "think": False,
                    "stream": False,
                    "options": {"temperature": 0.0, "top_k": 1,
                                "num_predict": len(batch) * 50},
                }, timeout=120)
                resp.raise_for_status()
                content = resp.json()["message"].get("content", "")

                for line in content.splitlines():
                    m = re.match(r"(\d+)\.\s*([A-U])\b", line)
                    if m:
                        idx = int(m.group(1)) - 1
                        section = m.group(2)
                        if 0 <= idx < len(batch):
                            results[idx] = (section, ISIC_SECTIONS.get(section, "Unknown"))

            except Exception as e:
                logger.error(f"ISIC batch {batch_idx + 1}/{num_batches} failed: {e}")

            # Apply results
            for j, (org_idx, cache_key, _) in enumerate(batch):
                section, label = results[j]
                org_records[org_idx]["isic_section"] = section
                org_records[org_idx]["isic_label"] = label
                if cache_key:
                    cache[cache_key] = {"section": section, "label": label}

            done = min((batch_idx + 1) * CLASSIFY_BATCH_SIZE, total)
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
                logger.info(f"ISIC: {done}/{total} orgs classified")
                save_cache(ISIC_CACHE_PATH, cache)

        save_cache(ISIC_CACHE_PATH, cache)

    classified = sum(1 for o in org_records if o.get("isic_section"))
    logger.info(f"ISIC: {classified}/{len(org_records)} organizations classified")
    return org_records


# ==========================================
# MAIN
# ==========================================
def main():
    logger.info("=" * 60)
    logger.info("Step 8: Classify occupations (HISCO) & industries (ISIC)")
    logger.info("=" * 60)

    # Load data
    career_records = load_jsonl(CAREER_PATH)
    org_records = load_jsonl(ORG_PATH)
    logger.info(f"Loaded {len(career_records)} career records, {len(org_records)} organizations")

    if not career_records and not org_records:
        logger.warning("No data found — nothing to classify")
        return

    # Part A: HISCO
    if career_records:
        logger.info("-" * 40)
        logger.info("Part A: HISCO occupation classification")
        logger.info("-" * 40)
        career_records = classify_hisco(career_records)
        write_jsonl(CAREER_PATH, career_records)

    # Part B: ISIC
    if org_records:
        logger.info("-" * 40)
        logger.info("Part B: ISIC industry classification")
        logger.info("-" * 40)
        org_records = classify_isic(org_records)
        write_jsonl(ORG_PATH, org_records)

    logger.info("=" * 60)
    logger.info("Classification complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
