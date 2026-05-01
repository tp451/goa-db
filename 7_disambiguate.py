"""
Step 7: Cross-volume entity disambiguation.

Finds persons appearing across multiple volumes (matched by name + birthyear),
merges their records with canonical IDs, and produces fuzzy org match candidates.
"""
import csv
import json
import os
import re
import logging
import time
from collections import defaultdict

import pykakasi
import requests

from config import (
    STRUCT_OUTPUT_DIR, DISAMBIG_OUTPUT_DIR, DISAMBIG_ORG_FUZZY_THRESHOLD,
    BIO_OLLAMA_BASE_URL, BIO_MODEL_NAME, BIO_MAX_TOKENS, DISAMBIG_LLM_MAX_TOKENS,
    DISAMBIG_ORG_HIERARCHY_LLM_MAX_TOKENS,
    DISAMBIG_ORG_HIERARCHY_SKIP_PARENTS,
    STRUCT_GEONAMES_FILES, STRUCT_GEONAMES_CLASSES,
    STRUCT_ORG_TRANSLATION_MAX_LEN,
)

OLLAMA_CHAT_URL = BIO_OLLAMA_BASE_URL.replace("/v1", "").rstrip("/") + "/api/chat"

# ==========================================
# LOGGING
# ==========================================
os.makedirs(DISAMBIG_OUTPUT_DIR, exist_ok=True)
log_dir = os.path.join(DISAMBIG_OUTPUT_DIR, "logs")
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


# ==========================================
# TRANSLATION (moved here from step 6 — runs after org merge to avoid wasted work)
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
}
_TRANSLATE_SYSTEM_DEFAULT = (
    "You are a translation engine for historical East Asian names (1900s-1940s). "
    "For each numbered input, output ONLY the number and the English translation on one line. "
    "Example:\n1. Tokyo\n2. Berlin"
)


def translate_batch_ollama(items, item_type):
    """Translate a list of CJK strings to English via Ollama. Returns dict name→english.

    Sends batches of items per LLM call for efficiency.
    """
    if not items:
        return {}
    unique = list(set(items))

    # Load cache (shared with step 6 so prior translations carry over)
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
    num_batches = (len(todo) + TRANSLATE_BATCH_SIZE - 1) // TRANSLATE_BATCH_SIZE

    for batch_idx in range(num_batches):
        start = batch_idx * TRANSLATE_BATCH_SIZE
        batch = todo[start:start + TRANSLATE_BATCH_SIZE]

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
        done = min((batch_idx + 1) * TRANSLATE_BATCH_SIZE, len(todo))
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
            logger.info(f"  {item_type}: {done}/{len(todo)} translated")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)

    return results


# ==========================================
# JARO-WINKLER SIMILARITY
# ==========================================
def jaro_similarity(s1, s2):
    """Compute Jaro similarity between two strings."""
    if s1 == s2:
        return 1.0
    len_s1, len_s2 = len(s1), len(s2)
    if len_s1 == 0 or len_s2 == 0:
        return 0.0

    match_distance = max(len_s1, len_s2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len_s1
    s2_matches = [False] * len_s2
    matches = 0
    transpositions = 0

    for i in range(len_s1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len_s2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len_s1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (matches / len_s1 + matches / len_s2 + (matches - transpositions / 2) / matches) / 3
    return jaro


def jaro_winkler_similarity(s1, s2, p=0.1):
    """Compute Jaro-Winkler similarity."""
    jaro = jaro_similarity(s1, s2)
    # Common prefix up to 4 chars
    prefix_len = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break
    return jaro + prefix_len * p * (1 - jaro)


# ==========================================
# KYŪJITAI → SHINJITAI NORMALIZATION
# ==========================================
# Mapping of traditional (旧字体) → simplified (新字体) kanji forms.
# Used for grouping during person and org disambiguation — original names are preserved.
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


def _normalize_kyujitai(name):
    """Normalize 旧字体 → 新字体 for grouping purposes only."""
    return name.translate(_KYUJITAI_TABLE) if name else name


# ==========================================
# PERSON DISAMBIGUATION
# ==========================================
def disambiguate_persons(person_core):
    """Find persons with same (name, birthyear) across volumes.

    Groups by kyūjitai-normalized name so that traditional/simplified kanji
    variants (e.g. 衞/衛, 藏/蔵, 澤/沢) are treated as the same person.

    Returns:
        id_remap: dict mapping merged person_id → canonical person_id
        mappings: list of entity_mapping records
    """
    # Group by (normalized_name, birthyear) where both are non-null
    groups = defaultdict(list)
    for rec in person_core:
        name = rec.get("name")
        birthyear = rec.get("birthyear")
        if name and birthyear:
            groups[(_normalize_kyujitai(name), birthyear)].append(rec)

    id_remap = {}
    mappings = []

    for (norm_name, birthyear), members in groups.items():
        # Only care about groups spanning 2+ volumes
        volumes = set()
        for m in members:
            v = m.get("volume", "unknown")
            if isinstance(v, list):
                volumes.update(str(x) for x in v)
            else:
                volumes.add(str(v))
        if len(volumes) < 2:
            continue

        # Check for origin_place conflicts
        origin_places = set()
        for m in members:
            op = m.get("origin_place")
            if op:
                origin_places.add(op)

        conflicts = {}
        if len(origin_places) > 1:
            conflicts["place"] = sorted(origin_places)

        # Canonical = earliest volume's record
        members_sorted = sorted(members, key=lambda m: m.get("volume", "") if isinstance(m.get("volume"), str) else str(m.get("volume", [""])[0]) if isinstance(m.get("volume"), list) else "")
        canonical = members_sorted[0]
        canonical_id = canonical["person_id"]
        merged_ids = []

        for m in members_sorted[1:]:
            mid = m["person_id"]
            id_remap[mid] = canonical_id
            merged_ids.append(mid)

        if merged_ids:
            # Use original name from canonical record for match_reason
            orig_name = canonical.get("name", norm_name)
            mapping = {
                "entity_type": "person",
                "canonical_id": canonical_id,
                "merged_ids": merged_ids,
                "match_reason": f"name={orig_name}, birthyear={birthyear}",
            }
            if conflicts:
                mapping["conflicts"] = conflicts
            mappings.append(mapping)

    logger.info(f"Person disambiguation: {len(id_remap)} IDs remapped across {len(mappings)} groups")
    return id_remap, mappings


def _birthyears_compatible(by1, by2, max_diff):
    """Check if two birthyears are compatible (both None, one None, or within max_diff)."""
    if by1 is None or by2 is None:
        return True
    return abs(by1 - by2) <= max_diff


def disambiguate_persons_extended(person_core, person_career, id_remap):
    """Additional person disambiguation tiers beyond exact (name, birthyear) match.

    Operates on persons NOT already merged. Modifies id_remap in-place.

    Tier 1: Same name + shared phone, birthyears within 3 years or missing
    Tier 2: Same name + 2+ shared career orgs, birthyear within 3 years or missing
    Tier 3: Same name + career overlap + exactly 10-year birthyear diff (OCR error)

    Returns list of new entity_mapping records.
    """
    already_merged = set(id_remap.keys())
    remaining = [r for r in person_core if r["person_id"] not in already_merged]

    # Group remaining by normalized name → only keep groups spanning 2+ volumes
    name_groups = defaultdict(list)
    for rec in remaining:
        name = rec.get("name")
        if name:
            name_groups[_normalize_kyujitai(name)].append(rec)

    # Filter to multi-volume groups only
    def _volumes_of(members):
        vols = set()
        for m in members:
            v = m.get("volume", "unknown")
            if isinstance(v, list):
                vols.update(str(x) for x in v)
            else:
                vols.add(str(v))
        return vols

    name_groups = {
        name: members for name, members in name_groups.items()
        if len(_volumes_of(members)) >= 2
    }
    logger.info(f"Extended disambiguation: {len(name_groups)} name groups spanning 2+ volumes "
                f"({sum(len(v) for v in name_groups.values())} persons)")

    # Build lookup structures
    career_by_pid = defaultdict(set)
    for c in person_career:
        oid = c.get("organization_id")
        if oid:
            career_by_pid[c["person_id"]].add(oid)

    mappings = []
    tier1_count = 0
    tier2_count = 0
    tier3_count = 0

    # ------ Tier 1: Phone match ------
    newly_merged_t1 = set()
    for name, members in name_groups.items():
        # Group by phone (non-null only)
        phone_groups = defaultdict(list)
        for m in members:
            if m["person_id"] in newly_merged_t1:
                continue
            phone = m.get("phone")
            if phone:
                phone_groups[phone].append(m)

        for phone, pg in phone_groups.items():
            volumes = set(m.get("volume", "unknown") for m in pg)
            if len(volumes) < 2:
                continue
            # Check all pairwise birthyears compatible within 3 years
            all_compatible = True
            for i in range(len(pg)):
                for j in range(i + 1, len(pg)):
                    if not _birthyears_compatible(pg[i].get("birthyear"), pg[j].get("birthyear"), 3):
                        all_compatible = False
                        break
                if not all_compatible:
                    break
            if not all_compatible:
                continue

            pg_sorted = sorted(pg, key=lambda m: m.get("volume", ""))
            canonical_id = pg_sorted[0]["person_id"]
            merged_ids = []
            for m in pg_sorted[1:]:
                mid = m["person_id"]
                id_remap[mid] = canonical_id
                merged_ids.append(mid)
                newly_merged_t1.add(mid)
            if merged_ids:
                tier1_count += len(merged_ids)
                mappings.append({
                    "entity_type": "person",
                    "canonical_id": canonical_id,
                    "merged_ids": merged_ids,
                    "match_reason": f"name={name}, phone={phone}",
                    "match_tier": "phone",
                })

    logger.info(f"Tier 1 (phone): {tier1_count} IDs merged")

    # ------ Tier 2: Career overlap (2+ shared orgs, birthyear within 3 years) ------
    newly_merged_t2 = set()
    for name, members in name_groups.items():
        # Exclude persons already merged in tier 1
        pool = [m for m in members if m["person_id"] not in newly_merged_t1 and m["person_id"] not in newly_merged_t2]
        if len(pool) < 2:
            continue
        volumes = set(m.get("volume", "unknown") for m in pool)
        if len(volumes) < 2:
            continue

        # Try to merge cross-volume pairs with 2+ shared orgs
        # Use union-find style: iteratively merge compatible pairs
        merged_here = set()
        for i in range(len(pool)):
            if pool[i]["person_id"] in merged_here:
                continue
            cluster = [pool[i]]
            for j in range(i + 1, len(pool)):
                if pool[j]["person_id"] in merged_here:
                    continue
                # Must be from a different volume than all current cluster members
                j_vol = pool[j].get("volume", "unknown")
                cluster_vols = set(c.get("volume", "unknown") for c in cluster)
                if j_vol in cluster_vols:
                    continue
                # Check career overlap with any cluster member
                j_orgs = career_by_pid[pool[j]["person_id"]]
                has_overlap = any(
                    len(j_orgs & career_by_pid[c["person_id"]]) >= 2
                    for c in cluster
                )
                if not has_overlap:
                    continue
                # Check birthyear compatibility with all cluster members
                all_ok = all(
                    _birthyears_compatible(pool[j].get("birthyear"), c.get("birthyear"), 3)
                    for c in cluster
                )
                if all_ok:
                    cluster.append(pool[j])

            if len(cluster) >= 2:
                cluster_sorted = sorted(cluster, key=lambda m: m.get("volume", ""))
                canonical_id = cluster_sorted[0]["person_id"]
                merged_ids = []
                for m in cluster_sorted[1:]:
                    mid = m["person_id"]
                    id_remap[mid] = canonical_id
                    merged_ids.append(mid)
                    merged_here.add(mid)
                    newly_merged_t2.add(mid)
                n_shared = len(career_by_pid[canonical_id] & career_by_pid[cluster_sorted[1]["person_id"]])
                tier2_count += len(merged_ids)
                mappings.append({
                    "entity_type": "person",
                    "canonical_id": canonical_id,
                    "merged_ids": merged_ids,
                    "match_reason": f"name={name}, shared_orgs>={n_shared}",
                    "match_tier": "career_overlap",
                })

    logger.info(f"Tier 2 (career overlap): {tier2_count} IDs merged")

    # ------ Tier 3: 10-year birthyear diff + career overlap (OCR decade error) ------
    all_prev_merged = newly_merged_t1 | newly_merged_t2
    for name, members in name_groups.items():
        pool = [m for m in members if m["person_id"] not in all_prev_merged]
        if len(pool) < 2:
            continue

        merged_here = set()
        for i in range(len(pool)):
            if pool[i]["person_id"] in merged_here:
                continue
            by_i = pool[i].get("birthyear")
            if by_i is None:
                continue
            for j in range(i + 1, len(pool)):
                if pool[j]["person_id"] in merged_here:
                    continue
                by_j = pool[j].get("birthyear")
                if by_j is None:
                    continue
                # Must be from different volumes
                if pool[i].get("volume", "") == pool[j].get("volume", ""):
                    continue
                # Exactly 10-year difference
                if abs(by_i - by_j) != 10:
                    continue
                # At least 1 shared career org
                shared = career_by_pid[pool[i]["person_id"]] & career_by_pid[pool[j]["person_id"]]
                if len(shared) < 1:
                    continue

                # Merge: earlier volume is canonical
                pair = sorted([pool[i], pool[j]], key=lambda m: m.get("volume", ""))
                canonical_id = pair[0]["person_id"]
                mid = pair[1]["person_id"]
                id_remap[mid] = canonical_id
                merged_here.add(mid)
                tier3_count += 1
                mappings.append({
                    "entity_type": "person",
                    "canonical_id": canonical_id,
                    "merged_ids": [mid],
                    "match_reason": f"name={name}, birthyear_10yr_ocr_error ({by_i} vs {by_j}), shared_orgs={len(shared)}",
                    "match_tier": "ocr_decade_error",
                })

    logger.info(f"Tier 3 (10yr OCR error): {tier3_count} IDs merged")
    logger.info(f"Extended disambiguation total: {tier1_count + tier2_count + tier3_count} additional IDs merged")

    return mappings


def _merge_family_into(kept, other, relation_id_remap):
    """Merge fields from `other` family member into `kept`."""
    _fold_provenance_into(kept, other)
    if not kept.get("relation") and other.get("relation"):
        kept["relation"] = other["relation"]
    if not kept.get("birth_year") and other.get("birth_year"):
        kept["birth_year"] = other["birth_year"]
    if not kept.get("place") and other.get("place"):
        kept["place"] = other["place"]
    if not kept.get("location_id") and other.get("location_id"):
        kept["location_id"] = other["location_id"]
    if not kept.get("name_latin") and other.get("name_latin"):
        kept["name_latin"] = other["name_latin"]
    # Map old relation_id → canonical
    pid = kept["person_id"]
    canonical_rid = kept.get("relation_id")
    old_rid = other.get("relation_id")
    if old_rid and canonical_rid and old_rid != canonical_rid:
        relation_id_remap[(pid, old_rid)] = canonical_rid


def deduplicate_family_members(person_family):
    """Fuzzy-deduplicate family members within each person.

    Two passes:
    1. Exact name match: members with identical (person_id, name) are merged.
    2. OCR-variant merge: members with identical (person_id, relation) whose
       names share a prefix (one is prefix of other, or differ in last char
       only) are merged.

    Returns (deduped_list, relation_id_remap) where relation_id_remap maps
    (person_id, old_relation_id) → canonical_relation_id.
    """
    # --- Pass 1: exact name match ---
    groups = defaultdict(list)
    order = []
    for fm in person_family:
        name = fm.get("name")
        if not name:
            order.append(None)
            groups[id(fm)] = [fm]
            continue
        key = (fm["person_id"], name)
        if key not in groups:
            order.append(key)
        groups[key].append(fm)

    result = []
    relation_id_remap = {}

    for key in order:
        if key is None:
            continue
        members = groups[key]
        if isinstance(key, int):
            result.append(members[0])
            continue
        if len(members) == 1:
            result.append(members[0])
            continue

        kept = members[0]
        for other in members[1:]:
            _merge_family_into(kept, other, relation_id_remap)
        result.append(kept)

    n_exact = len(person_family) - len(result)
    if n_exact:
        logger.info(f"Family member dedup (exact): {len(person_family)} → {len(result)} "
                     f"({n_exact} merged)")

    # --- Pass 2: OCR-variant merge (same person_id + relation, similar names) ---
    rel_groups = defaultdict(list)
    for i, fm in enumerate(result):
        rel = fm.get("relation") or ""
        if not rel:
            continue
        rel_groups[(fm["person_id"], rel)].append(i)

    remove_indices = set()
    n_ocr = 0
    for key, indices in rel_groups.items():
        if len(indices) < 2:
            continue
        entries = [(i, result[i]) for i in indices]
        merged = set()
        for a in range(len(entries)):
            if a in merged:
                continue
            ia, ra = entries[a]
            na = ra.get("name") or ""
            if not na:
                continue
            cluster = [a]
            for b in range(a + 1, len(entries)):
                if b in merged:
                    continue
                nb = entries[b][1].get("name") or ""
                if not nb:
                    continue
                shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
                if (longer.startswith(shorter)
                        or (len(na) == len(nb) >= 2 and na[:-1] == nb[:-1])):
                    cluster.append(b)
            if len(cluster) < 2:
                continue
            # Keep longest name
            cluster.sort(key=lambda c: len(entries[c][1].get("name") or ""), reverse=True)
            kept_a = cluster[0]
            kept_idx, kept = entries[kept_a]
            for drop_a in cluster[1:]:
                drop_idx, drop = entries[drop_a]
                _merge_family_into(kept, drop, relation_id_remap)
                remove_indices.add(drop_idx)
                n_ocr += 1
            merged.update(cluster)

    if remove_indices:
        result = [r for i, r in enumerate(result) if i not in remove_indices]
        logger.info(f"Family member dedup (OCR variant): {n_ocr} merged")

    return result, relation_id_remap


def _appearance_from(rec):
    return {
        "volume": rec.get("volume", "unknown"),
        "source_page": rec.get("source_page"),
        "source_image": rec.get("source_image"),
        "entry_index": rec.get("entry_index"),
    }


def merge_person_records(canonical, merged_records):
    """Merge scalar fields from multiple person_core records.

    Later volume records override scalar fields (more recent data).
    The volume field becomes a list of all source volumes, and an `appearances`
    list aligns volume / source_page / source_image / entry_index per source row.
    """
    result = dict(canonical)
    volumes = [canonical.get("volume", "unknown")]
    appearances = [_appearance_from(canonical)]

    for rec in merged_records:
        vol = rec.get("volume", "unknown")
        if vol not in volumes:
            volumes.append(vol)
        appearances.append(_appearance_from(rec))
        # Later volume overrides non-null scalar fields
        for key in ["phone", "place", "origin_place", "rank", "gender"]:
            if rec.get(key) is not None:
                result[key] = rec[key]
        # Only override domain + latin names if domains agree (avoids
        # wrong-domain romanization from overwriting correct values)
        if rec.get("domain") == result.get("domain"):
            for key in ["name_family_latin", "name_given_latin"]:
                if rec.get(key) is not None:
                    result[key] = rec[key]

    result["volume"] = volumes
    result["appearances"] = appearances
    return result


# ==========================================
# TABLE REMAPPING
# ==========================================
def remap_table(records, id_field, remap):
    """Remap person_id references in a table using the id_remap dict."""
    for rec in records:
        old_id = rec.get(id_field)
        if old_id in remap:
            rec[id_field] = remap[old_id]
    return records


PROVENANCE_FIELDS = ("source_volume", "source_page")


def _fold_provenance_into(kept, incoming):
    """Merge incoming row's source_volume/source_page into the kept row's lists.

    Promotes scalar→list when the second value differs. Order-preserving dedup so
    the original source's volume/page comes first.
    """
    for f in PROVENANCE_FIELDS:
        kv = kept.get(f)
        nv = incoming.get(f)
        if kv is None and nv is None:
            continue
        kl = kv if isinstance(kv, list) else [kv]
        nl = nv if isinstance(nv, list) else [nv]
        merged = list(dict.fromkeys(kl + nl))
        kept[f] = merged if len(merged) > 1 else merged[0]


def deduplicate_entries(records, key_fields):
    """Collapse rows with identical key_fields, folding source_volume/source_page provenance.

    When two rows share the same key_fields, the first is kept and the second's
    source_volume/source_page values are merged into the kept row (lists if they
    differ). Preserves cross-volume corroboration that pure dedup would discard.
    """
    by_key = {}
    order = []
    for rec in records:
        key = tuple(tuple(v) if isinstance(v, list) else v for f in key_fields for v in [rec.get(f)])
        if key in by_key:
            _fold_provenance_into(by_key[key], rec)
        else:
            by_key[key] = rec
            order.append(key)
    return [by_key[k] for k in order]


def remove_subset_records(records, group_fields, detail_fields):
    """Remove records that are strict subsets of another record in the same group.

    Two records sharing the same *group_fields* are compared on *detail_fields*.
    A record whose non-null detail values are a subset of another record's
    non-null detail values is redundant and gets dropped.
    E.g. career entry (person, org, title=None, year=None) is a subset of
    (person, org, title="clerk", year=1920) and adds no new information.

    When a subset row is dropped, its source_volume/source_page provenance is
    folded into the chosen superset survivor so cross-volume corroboration is
    preserved.
    """
    from collections import defaultdict

    grouped = defaultdict(list)
    for rec in records:
        gkey = tuple(rec.get(f) for f in group_fields)
        grouped[gkey].append(rec)

    result = []
    for gkey, group in grouped.items():
        if len(group) == 1:
            result.append(group[0])
            continue

        # For each record, compute its set of (field, value) pairs that are non-null
        details = []
        for rec in group:
            d = set()
            for f in detail_fields:
                v = rec.get(f)
                if v is not None and v != "" and v != "NULL":
                    d.add((f, tuple(v) if isinstance(v, list) else v))
            details.append(d)

        # First pass: classify each record as kept or dropped (subset of another).
        kept_idxs = []
        absorber_for = {}  # subset_idx -> idx of the chosen superset survivor
        for i, rec in enumerate(group):
            superset_idx = None
            for j, other in enumerate(group):
                if i == j:
                    continue
                if details[i] < details[j]:
                    superset_idx = j
                    break
            if superset_idx is None:
                kept_idxs.append(i)
            else:
                absorber_for[i] = superset_idx

        # Resolve absorbers that themselves got absorbed (chain through to a kept row)
        kept_set = set(kept_idxs)
        for sub_i, abs_j in list(absorber_for.items()):
            seen = {sub_i}
            while abs_j not in kept_set and abs_j in absorber_for and abs_j not in seen:
                seen.add(abs_j)
                abs_j = absorber_for[abs_j]
            absorber_for[sub_i] = abs_j

        # Fold provenance from each dropped row into its absorber (if absorber survived)
        for sub_i, abs_j in absorber_for.items():
            if abs_j in kept_set:
                _fold_provenance_into(group[abs_j], group[sub_i])

        for i in kept_idxs:
            result.append(group[i])

    return result


# ==========================================
# ORG FUZZY MATCHING
# ==========================================
def find_fuzzy_org_matches(organizations, threshold):
    """Find org name pairs above the Jaro-Winkler similarity threshold.

    Returns list of candidate match dicts for manual review.
    Caches results to a JSONL checkpoint; invalidates when threshold or org count changes.
    """
    checkpoint_path = os.path.join(DISAMBIG_OUTPUT_DIR, "org_fuzzy_checkpoint.jsonl")
    num_orgs = len(organizations)

    # Try loading from checkpoint (complete run → return immediately; partial → resume below)
    if os.path.exists(checkpoint_path):
        cached = load_jsonl(checkpoint_path)
        if cached and cached[0].get("_meta"):
            meta = cached[0]
            if meta.get("threshold") == threshold and meta.get("num_orgs") == num_orgs:
                if meta.get("complete"):
                    candidates = cached[1:]
                    logger.info(f"Org fuzzy checkpoint loaded: {len(candidates)} cached candidates "
                                f"(threshold={threshold}, num_orgs={num_orgs})")
                    return candidates
                # Partial checkpoint — will be resumed below
            else:
                logger.info(f"Org fuzzy checkpoint invalidated (threshold: {meta.get('threshold')}→{threshold}, "
                            f"num_orgs: {meta.get('num_orgs')}→{num_orgs}), recomputing...")

    candidates = []
    names = [(o["organization_id"], o["name"], len(o["name"])) for o in organizations]
    # Sort by name length so the inner loop can break early
    names.sort(key=lambda x: x[2])
    total_pairs = len(names) * (len(names) - 1) // 2
    logger.info(f"Org fuzzy matching: comparing {len(names)} org names ({total_pairs} total pairs, threshold={threshold})...")

    # Try loading partial checkpoint to resume from
    start_i = 0
    if os.path.exists(checkpoint_path):
        cached = load_jsonl(checkpoint_path)
        if cached and cached[0].get("_meta"):
            meta = cached[0]
            if meta.get("threshold") == threshold and meta.get("num_orgs") == num_orgs:
                resumed_i = meta.get("progress_i")
                if resumed_i is not None and not meta.get("complete"):
                    candidates = cached[1:]
                    start_i = resumed_i
                    logger.info(f"Resuming Jaro matching from org {start_i}/{len(names)} "
                                f"({len(candidates)} candidates so far)")

    skipped_len = 0
    checked = 0
    for i in range(start_i, len(names)):
        if i > start_i and i % 500 == 0:
            logger.info(f"Org fuzzy matching progress: {i}/{len(names)} orgs processed, "
                        f"{checked} pairs checked, {skipped_len} skipped by length, "
                        f"{len(candidates)} candidates found")
            # Incremental checkpoint
            meta_record = {"_meta": True, "threshold": threshold, "num_orgs": num_orgs,
                           "progress_i": i, "complete": False}
            write_jsonl(checkpoint_path, [meta_record] + candidates)
        id_a, name_a, len_a = names[i]
        for j in range(i + 1, len(names)):
            id_b, name_b, len_b = names[j]
            # Length pre-filter: max possible JW score from lengths alone.
            # max_jaro = (2 + len_a/len_b) / 3  (since len_a <= len_b after sort)
            # max_jw = max_jaro + min(4, len_a) * 0.1 * (1 - max_jaro)
            r = len_a / len_b if len_b > 0 else 1.0
            max_jaro = (2.0 + r) / 3.0
            max_jw = max_jaro + min(4, len_a) * 0.1 * (1.0 - max_jaro)
            if max_jw < threshold:
                skipped_len += len(names) - j
                break
            if name_a == name_b:
                continue
            checked += 1
            sim = jaro_winkler_similarity(name_a, name_b)
            if sim >= threshold:
                candidates.append({
                    "entity_type": "organization",
                    "canonical_id": id_a,
                    "merged_ids": [id_b],
                    "match_reason": f"fuzzy_match: '{name_a}' ~ '{name_b}' (sim={sim:.3f})",
                    "name_a": name_a,
                    "name_b": name_b,
                    "similarity": round(sim, 3),
                })

    logger.info(f"Org fuzzy matching: {len(candidates)} candidate pairs above threshold {threshold} "
                f"({checked} pairs checked, {skipped_len} skipped by length filter)")

    # Save final checkpoint (complete)
    meta_record = {"_meta": True, "threshold": threshold, "num_orgs": num_orgs,
                   "progress_i": len(names), "complete": True}
    write_jsonl(checkpoint_path, [meta_record] + candidates)

    return candidates


DISAMBIG_ORG_LLM_BATCH_SIZE = 20

_ORG_VERIFY_SYSTEM_PROMPT = (
    "You are a historical East Asian organization name expert. "
    "For each numbered pair, answer YES if they refer to the same organization "
    "(considering traditional/simplified character variants, OCR errors, "
    "abbreviations, and historical name changes) or NO if they are different "
    "organizations (including subdivisions, branches, or related but distinct "
    "entities). Reply with ONLY the number and YES or NO for each pair, one per line. "
    "Example:\n1. YES\n2. NO\n3. YES"
)


def verify_org_matches_llm(candidates):
    """Use LLM to verify whether fuzzy-matched org name pairs are true duplicates.

    Sends batches of pairs per LLM call for efficiency.
    Deduplicates by name pair so each unique pair is only checked once.
    Checkpoints results to disk after each batch so progress survives interruptions.
    """
    checkpoint_path = os.path.join(DISAMBIG_OUTPUT_DIR, "org_llm_checkpoint.jsonl")

    # Load existing checkpoint results
    checked = {}  # (name_a, name_b) → bool
    if os.path.exists(checkpoint_path):
        for rec in load_jsonl(checkpoint_path):
            checked[(rec["name_a"], rec["name_b"])] = rec["llm_confirmed"]
        logger.info(f"LLM checkpoint loaded: {len(checked)} pairs already checked, resuming...")

    # Deduplicate: group candidates by (name_a, name_b)
    unique_pairs = {}  # (name_a, name_b) → first candidate
    for cand in candidates:
        key = (cand["name_a"], cand["name_b"])
        if key not in unique_pairs:
            unique_pairs[key] = cand

    total_unique = len(unique_pairs)
    total_candidates = len(candidates)

    # Filter to pairs not yet checked
    todo = [(k, v) for k, v in unique_pairs.items() if k not in checked]
    logger.info(f"LLM verification: {total_candidates} org candidates, {total_unique} unique pairs, "
                f"{total_unique - len(todo)} cached, {len(todo)} remaining")

    if todo:
        # Process in batches
        num_batches = (len(todo) + DISAMBIG_ORG_LLM_BATCH_SIZE - 1) // DISAMBIG_ORG_LLM_BATCH_SIZE
        confirmed_count = 0

        with open(checkpoint_path, "a", encoding="utf-8") as ckpt:
            for batch_idx in range(num_batches):
                start = batch_idx * DISAMBIG_ORG_LLM_BATCH_SIZE
                batch = todo[start:start + DISAMBIG_ORG_LLM_BATCH_SIZE]

                # Build numbered prompt
                lines = []
                for i, ((na, nb), _cand) in enumerate(batch, 1):
                    lines.append(f"{i}. {na} | {nb}")
                user_content = "\n".join(lines)

                payload = {
                    "model": BIO_MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": _ORG_VERIFY_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "think": False,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "top_k": 1,
                        "num_predict": len(batch) * 15,
                    },
                }

                # Parse response — default to NO for unparsed entries
                results = [False] * len(batch)
                try:
                    resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=120)
                    resp.raise_for_status()
                    content = resp.json()["message"].get("content", "")
                    for line in content.splitlines():
                        m = re.match(r"(\d+)\.\s*(YES|NO)", line, re.IGNORECASE)
                        if m:
                            idx = int(m.group(1)) - 1
                            if 0 <= idx < len(batch):
                                results[idx] = m.group(2).upper() == "YES"
                except Exception as e:
                    logger.warning(f"Batch {batch_idx + 1}/{num_batches} failed: {e}")

                # Write checkpoint and update checked dict
                batch_yes = 0
                for i, ((na, nb), _cand) in enumerate(batch):
                    is_match = results[i]
                    checked[(na, nb)] = is_match
                    if is_match:
                        batch_yes += 1
                    ckpt.write(json.dumps({
                        "name_a": na, "name_b": nb, "llm_confirmed": is_match
                    }, ensure_ascii=False) + "\n")
                ckpt.flush()
                confirmed_count += batch_yes

                if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == num_batches:
                    logger.info(f"[batch {batch_idx + 1}/{num_batches}] "
                                f"{(batch_idx + 1) * DISAMBIG_ORG_LLM_BATCH_SIZE} pairs processed, "
                                f"{confirmed_count} confirmed so far")

    # Apply results to all candidates
    verified = [cand for cand in candidates if checked.get((cand["name_a"], cand["name_b"]), False)]

    logger.info(f"LLM verified: {len(verified)}/{total_candidates} candidates confirmed "
                f"({total_unique} unique pairs checked)")
    return verified


def apply_org_overrides(organizations, all_tables_with_org_id):
    """Apply manual org ID overrides from org_overrides.jsonl if present."""
    overrides_path = "org_overrides.jsonl"
    if not os.path.exists(overrides_path):
        logger.info("No org_overrides.jsonl found, skipping org remapping")
        return organizations, {}

    overrides = load_jsonl(overrides_path)
    org_remap = {}
    for entry in overrides:
        canonical = entry.get("canonical_id")
        for mid in entry.get("merged_ids", []):
            org_remap[mid] = canonical

    if not org_remap:
        return organizations, {}

    # Resolve transitive chains (A->B, B->C becomes A->C)
    changed = True
    while changed:
        changed = False
        for mid in org_remap:
            cid = org_remap[mid]
            if cid in org_remap:
                org_remap[mid] = org_remap[cid]
                changed = True

    # Remove merged orgs from the table
    merged_ids = set(org_remap.keys())
    organizations = [o for o in organizations if o["organization_id"] not in merged_ids]

    # Remap org references in other tables
    for table in all_tables_with_org_id:
        for rec in table:
            if rec.get("organization_id") in org_remap:
                rec["organization_id"] = org_remap[rec["organization_id"]]

    logger.info(f"Applied {len(org_remap)} org overrides")
    return organizations, org_remap


# ==========================================
# ORG HIERARCHY DETECTION
# ==========================================
CJK_CHAR = re.compile(r"[\u4E00-\u9FFF]")


def load_geonames_placenames():
    """Load a set of CJK place names from GeoNames files for filtering geographic prefixes."""
    placenames = set()
    for filename in STRUCT_GEONAMES_FILES:
        if not os.path.exists(filename):
            continue
        with open(filename, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 8:
                    continue
                if row[6] not in STRUCT_GEONAMES_CLASSES:
                    continue
                for token in row[3].split(","):
                    token = token.strip()
                    cjk_only = "".join(CJK_CHAR.findall(token))
                    if 2 <= len(cjk_only) <= 4:
                        placenames.add(cjk_only)
    logger.info(f"Loaded {len(placenames)} CJK place names for hierarchy filtering")
    return placenames


HIERARCHY_LLM_BATCH_SIZE = 20

_HIERARCHY_SYSTEM_PROMPT = (
    "You are a historical East Asian organization expert (1900s-1940s). "
    "For each numbered pair, answer YES if the child organization is a subsidiary, "
    "branch, division, or affiliate of the parent organization. Answer NO if they "
    "are independent organizations that merely share a name prefix. "
    "Reply with ONLY the number and YES or NO for each pair, one per line. "
    "Example:\n1. YES\n2. NO\n3. YES"
)


def detect_org_hierarchy(organizations, placenames):
    """Detect parent-child relationships among organizations.

    Candidates: org A's name is a prefix of org B's name, remainder >= 2 chars.
    Filters out geographic prefixes (日本, 東京, etc.) using geonames data.
    Uses batched LLM to verify remaining candidates.
    Checkpoints LLM results.

    Returns list of verified hierarchy dicts.
    """
    # Build name → org lookup for O(n*L) prefix search
    org_by_name = {}
    for org in organizations:
        org_by_name.setdefault(org["name"], org)

    # Pre-filter geographic/skip parents
    filtered_parents = set()
    for name in org_by_name:
        if len(name) < 2:
            continue
        cjk = "".join(CJK_CHAR.findall(name))
        if cjk in placenames or cjk in DISAMBIG_ORG_HIERARCHY_SKIP_PARENTS:
            filtered_parents.add(name)

    # Find candidates: for each org, find its longest matching prefix (direct parent only)
    candidates = []
    total_orgs = len(organizations)
    for idx, org in enumerate(organizations):
        if idx > 0 and idx % 5000 == 0:
            logger.info(f"  Org hierarchy candidate search: {idx}/{total_orgs} orgs, "
                        f"{len(candidates)} candidates found")
        name = org["name"]
        if len(name) < 4:  # need prefix >= 2 AND remainder >= 2
            continue
        # Search longest prefix first — only keep the direct parent
        for prefix_len in range(len(name) - 2, 1, -1):
            prefix = name[:prefix_len]
            if prefix in filtered_parents:
                continue
            if prefix in org_by_name and prefix != name:
                parent = org_by_name[prefix]
                candidates.append({
                    "child_id": org["organization_id"],
                    "parent_id": parent["organization_id"],
                    "child_name": name,
                    "parent_name": prefix,
                })
                break  # longest match found, skip shorter prefixes

    logger.info(f"Org hierarchy: {len(candidates)} candidate pairs after geographic filtering")

    if not candidates:
        return []

    # LLM verification with batching and checkpointing
    checkpoint_path = os.path.join(DISAMBIG_OUTPUT_DIR, "org_hierarchy_checkpoint.jsonl")
    checked = {}  # (parent_name, child_name) → bool
    if os.path.exists(checkpoint_path):
        for rec in load_jsonl(checkpoint_path):
            checked[(rec["parent_name"], rec["child_name"])] = rec["llm_confirmed"]
        logger.info(f"Org hierarchy checkpoint: {len(checked)} pairs already checked")

    # Filter to unchecked candidates
    todo = [c for c in candidates if (c["parent_name"], c["child_name"]) not in checked]
    logger.info(f"Org hierarchy LLM verification: {len(candidates)} total, "
                f"{len(candidates) - len(todo)} cached, {len(todo)} remaining")

    if todo:
        num_batches = (len(todo) + HIERARCHY_LLM_BATCH_SIZE - 1) // HIERARCHY_LLM_BATCH_SIZE
        confirmed_count = 0

        with open(checkpoint_path, "a", encoding="utf-8") as ckpt:
            for batch_idx in range(num_batches):
                start = batch_idx * HIERARCHY_LLM_BATCH_SIZE
                batch = todo[start:start + HIERARCHY_LLM_BATCH_SIZE]

                # Build numbered prompt
                lines = []
                for i, cand in enumerate(batch, 1):
                    lines.append(f"{i}. Parent: {cand['parent_name']} | Child: {cand['child_name']}")
                user_content = "\n".join(lines)

                results = [False] * len(batch)
                try:
                    resp = requests.post(OLLAMA_CHAT_URL, json={
                        "model": BIO_MODEL_NAME,
                        "messages": [
                            {"role": "system", "content": _HIERARCHY_SYSTEM_PROMPT},
                            {"role": "user", "content": user_content},
                        ],
                        "think": False,
                        "stream": False,
                        "options": {"temperature": 0.0, "top_k": 1,
                                    "num_predict": len(batch) * 15},
                    }, timeout=120)
                    resp.raise_for_status()
                    content = resp.json()["message"].get("content", "")
                    for line in content.splitlines():
                        m = re.match(r"(\d+)\.\s*(YES|NO)", line, re.IGNORECASE)
                        if m:
                            idx = int(m.group(1)) - 1
                            if 0 <= idx < len(batch):
                                results[idx] = m.group(2).upper() == "YES"
                except Exception as e:
                    logger.warning(f"Hierarchy batch {batch_idx + 1}/{num_batches} failed: {e}")

                # Write checkpoint
                batch_yes = 0
                for i, cand in enumerate(batch):
                    is_match = results[i]
                    checked[(cand["parent_name"], cand["child_name"])] = is_match
                    if is_match:
                        batch_yes += 1
                    ckpt.write(json.dumps({
                        "parent_name": cand["parent_name"],
                        "child_name": cand["child_name"],
                        "llm_confirmed": is_match,
                    }, ensure_ascii=False) + "\n")
                ckpt.flush()
                confirmed_count += batch_yes

                if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == num_batches:
                    logger.info(f"[batch {batch_idx + 1}/{num_batches}] "
                                f"{min((batch_idx + 1) * HIERARCHY_LLM_BATCH_SIZE, len(todo))} pairs processed, "
                                f"{confirmed_count} confirmed so far")

    verified = [c for c in candidates if checked.get((c["parent_name"], c["child_name"]), False)]
    logger.info(f"Org hierarchy: {len(verified)}/{len(candidates)} pairs confirmed")
    return verified


# ==========================================
# FAMILY MEMBER RE-ROMANIZATION
# ==========================================
_kakasi = pykakasi.kakasi()

# Pinyin-telltale patterns that don't appear in Hepburn romanization
_PINYIN_MARKERS = re.compile(
    r"(?:zh|(?<![aeiouns])x[aeiou]|(?<![aeiou])q[iuü]|"
    r"(?<![aeiou])c[aeiou](?!h))", re.I
)


def _pykakasi_romanize(text):
    """Romanize a name part via pykakasi. Returns capitalized string or None."""
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


def fix_family_name_split_and_length(family_records, known_families, core_family_by_pid):
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
            fam_latin = _pykakasi_romanize(diff_family)
            giv_latin = _pykakasi_romanize(given_part) if given_part else None
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
                fam_latin = _pykakasi_romanize(core_family)
                giv_latin = _pykakasi_romanize(given_part)
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


def fix_family_member_romanization(family_records, ja_person_ids):
    """Re-romanize name_latin for family members of ja-domain persons
    whose name_latin contains Chinese pinyin patterns."""
    fixed = 0
    for rec in family_records:
        if rec.get("person_id") not in ja_person_ids:
            continue
        nl = rec.get("name_latin") or ""
        name = rec.get("name") or ""
        if not nl or not name:
            continue
        if not _PINYIN_MARKERS.search(nl.lower()):
            continue
        result = _pykakasi_romanize(name)
        if result:
            rec["name_latin"] = result
            fixed += 1
    return fixed


# ==========================================
# FALSE JOB REMOVAL
# ==========================================
_RELATION_EXACT = frozenset({
    "前名", "舊名", "旧名",
    "長男", "二男", "三男", "四男", "五男", "六男", "七男", "八男",
    "長女", "二女", "三女", "四女", "五女", "六女",
    "養子", "養女", "嫡男", "嫡子", "庶子",
    "入夫", "婿養子", "養嗣子", "嗣子", "養妹",
    "妻", "夫", "男", "女", "子", "兄", "弟", "姉", "妹",
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


def is_false_job(job_title):
    """Return True if job_title is actually a family relation, not a job."""
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


def fix_false_jobs(records):
    """Remove records whose job_title is a family relation, not a real job."""
    before = len(records)
    records[:] = [r for r in records if not is_false_job(r.get("job_title"))]
    return before - len(records)


# ==========================================
# XINJING ADMIN1_EN FIX
# ==========================================
_XINJING_ADMIN1_MAP = {
    "新京特別市": "Hsinking Special City",
    "新 京特別市": "Hsinking Special City",
    "新京市": "Hsinking City",
    "新京別市": "Hsinking Special City",
    "新京都": "Hsinking",
}

LOCATION_OVERRIDES = {
    9930571: {"latitude": 43.88, "longitude": 125.32,
              "province": "Jilin", "country": "zh"},
}


_JA_ADMIN_PATTERN = re.compile(
    r"郡.{1,10}(?:町|村)"  # 郡+町/村 (gun + machi/mura)
    r"|[區区]"             # 區/区 (Japanese ward — bare match, no 町 required)
    r"|丁目"               # chōme block numbering
    r"|番地"               # banchi lot numbering
    r"|番町"               # banchō
)


_COLONIAL_CITY_PREFIXES = re.compile(
    r"^(?:新京|奉天|大連|哈爾|長春|吉林|齊齊|安東市|鞍山|撫順|營口|鐵嶺|"
    r"四平|牡丹|本溪|佳木|錦州|旅順|瀋陽|通化|熱河|承德|張家口|"
    r"包頭|厚和|鳳城|龍井|間島|延吉|圖們|琿春|敦化|海拉爾|"
    r"興安|三江省|綏遠|察哈爾|蒙疆|北安省|龍江省|濱江省|黑河省|青島|"
    r"釜山|京城|大邱|平壌|仁川|元山|清津|羅南|"
    r"慶尙|全羅|忠清|平安|黃海|咸鏡|京畿|江原)"
)


# Regex for place fields that end with a family relation type (not a location)
_RELATION_SUFFIX_RE = re.compile(
    r"(?:長男|二男|三男|四男|五男|六男|七男|八男"
    r"|長女|二女|三女|四女|五女|六女"
    r"|養子|養女|嫡男|嫡子|庶子"
    r"|入夫|婿養子|養嗣子|嗣子|養妹|娘婿|後裔"
    r"|[の之](?:妻|夫|男|女|子|兄|弟|姉|妹|孫))$"
)


# Organization-type suffixes that should not appear in location names
_ORG_PLACE_SUFFIXES = ("領事館", "鐵道", "鉄道", "中學", "中学", "造船部")


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
        norm = _normalize_kyujitai(org["name"])
        if norm not in norm_to_org:
            norm_to_org[norm] = org

    remap = {}
    for org in org_records:
        norm = _normalize_kyujitai(org["name"])
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


def fix_mislocated_ja_locations(records):
    """Null geolocation for locations whose name has Japanese admin patterns
    but whose GeoNames match is non-JP, OR whose name is actually a relation
    description or organization name (not a real place).

    Mirrors match_place() logic: when the place name contains Japanese admin
    suffixes, only JP GeoNames entries should be used.

    Exceptions: Korean (country='ko') and Manchurian/colonial-era locations
    whose names start with known city/province prefixes are kept.
    """
    _NULL_GEO_FIELDS = (
        "geonameid", "latitude", "longitude", "province", "country",
        "admin1", "admin1_en", "admin2", "admin2_en", "admin3", "admin3_en",
        "admin1_norm", "name_en",
    )
    nulled = 0
    for rec in records:
        name = rec.get("name") or ""
        if not name:
            continue

        # Location names that are actually relation descriptions or org names
        # — null geo regardless of country.
        # Check both name and admin1: some names have extra OCR garbage after
        # the org suffix (e.g. "丸子鐵道務小"), but admin1 is clean ("丸子鐵道").
        admin1 = rec.get("admin1") or ""
        if (name.endswith(_ORG_PLACE_SUFFIXES)
                or admin1.endswith(_ORG_PLACE_SUFFIXES)
                or _RELATION_SUFFIX_RE.search(name)):
            if rec.get("geonameid") is not None or rec.get("latitude") is not None:
                for k in _NULL_GEO_FIELDS:
                    rec[k] = None
                nulled += 1
            continue

        country = rec.get("country")
        if country == "ja" or country == "ko":
            continue
        if not rec.get("geonameid"):
            continue
        if not _JA_ADMIN_PATTERN.search(name):
            continue
        if _COLONIAL_CITY_PREFIXES.match(name):
            continue
        for k in _NULL_GEO_FIELDS:
            rec[k] = None
        nulled += 1
    return nulled


# -------------------------------------------------------------------------
# Fix Manchukuo/Mengjiang/colonial-era locations matched to wrong regions
# -------------------------------------------------------------------------
# Historical place names (1930s–1940s) not in GeoNames.  The geocoder fell
# through to shorter CJK prefixes that exist elsewhere in Japan/China/Taiwan.
_MANCHUKUO_PLACES = {
    "興安西省": (44.00, 118.50, "Inner Mongolia", "zh"),
    "興安北省": (49.22, 119.73, "Inner Mongolia", "zh"),
    "興安東省": (48.00, 122.74, "Inner Mongolia", "zh"),
    "興安南省": (46.07, 122.07, "Inner Mongolia", "zh"),
    "興安總省": (46.00, 121.00, "Inner Mongolia", "zh"),
    "厚和":     (40.84, 111.75, "Suiyuan", "zh"),
    "三江省":   (46.80, 130.37, "Heilongjiang", "zh"),
    "青島":     (36.06, 120.38, "Shandong", "zh"),
    "間島省":   (42.88, 129.47, "Jilin", "zh"),
    "安東省":   (40.12, 124.39, "Liaoning", "zh"),
    "安東縣":   (40.12, 124.39, "Liaoning", "zh"),
    "安東県":   (40.12, 124.39, "Liaoning", "zh"),
    "山海關":   (40.00, 119.77, "Hebei", "zh"),
}

_MANCHUKUO_OK_PROVINCES = {
    "興安": {"Inner Mongolia", "Heilongjiang", "Jilin", "Liaoning", None},
    "厚和": {"Inner Mongolia", "Suiyuan", "Chahaer", None},
    "三江省": {"Heilongjiang", None},
    "青島": {"Shandong", None},
    "間島": {"Jilin", None},
    "安東省": {"Liaoning", None},
    "安東縣": {"Liaoning", None},
    "安東県": {"Liaoning", None},
    "山海關": {"Hebei", None},
}

_MANCHUKUO_NULL_IF_BAD = {
    "蒙古": {"Inner Mongolia", "Chahaer", "Suiyuan", "Mongolia", None},
    "蒙疆": {"Inner Mongolia", "Chahaer", "Suiyuan", None},
    "綏遠": {"Inner Mongolia", "Suiyuan", None},
}

_MANCHUKUO_NULL_IF_STARTSWITH = {
    "安東": {"Liaoning", "Gyeongsangbuk-do", "Gyeongsangbuk-do\t",
             "Mie", "Changhua", None},
}


def fix_manchukuo_geocoding(loc_records):
    """Fix locations with Manchukuo/colonial-era names matched to wrong regions.

    Three modes:
    1. Known coordinates: re-geocode with correct approximate coordinates.
    2. Null-if-bad: null geo data for vague historical names (蒙古, 蒙疆, 綏遠).
    3. Null-if-startswith: null for names starting with ambiguous prefixes (安東)
       that have too many substring false positives for mode 2.
    """
    _NULL_GEO_FIELDS = (
        "geonameid", "latitude", "longitude", "province", "country",
        "admin1", "admin1_en", "admin2", "admin2_en", "admin3", "admin3_en",
        "admin1_norm", "name_en",
    )

    fixed = 0
    for rec in loc_records:
        name = rec.get("name") or ""
        if not rec.get("geonameid"):
            continue
        province = rec.get("province")

        # Mode 1: known coordinates
        matched = None
        for prefix in sorted(_MANCHUKUO_PLACES, key=len, reverse=True):
            if name.startswith(prefix) or prefix in name:
                matched = prefix
                break

        if matched:
            ok_key = None
            for key in sorted(_MANCHUKUO_OK_PROVINCES, key=len, reverse=True):
                if key in matched or matched.startswith(key):
                    ok_key = key
                    break
            if ok_key and province not in _MANCHUKUO_OK_PROVINCES[ok_key]:
                lat, lon, prov, cc = _MANCHUKUO_PLACES[matched]
                rec["geonameid"] = None
                rec["latitude"] = lat
                rec["longitude"] = lon
                rec["province"] = prov
                rec["country"] = cc
                fixed += 1
                continue

        # Mode 2: null if clearly wrong (vague historical references)
        for prefix, ok_provs in _MANCHUKUO_NULL_IF_BAD.items():
            if prefix not in name:
                continue
            idx = name.find(prefix)
            after = name[idx + len(prefix):] if idx >= 0 else ""
            if after.startswith("路") or after.startswith("通"):
                continue
            if province not in ok_provs:
                for k in _NULL_GEO_FIELDS:
                    rec[k] = None
                fixed += 1
                break
        else:
            # Mode 3: startswith-only null (安東 — too many substring false positives)
            for prefix, ok_provs in _MANCHUKUO_NULL_IF_STARTSWITH.items():
                if not name.startswith(prefix):
                    continue
                if province not in ok_provs:
                    for k in _NULL_GEO_FIELDS:
                        rec[k] = None
                    fixed += 1
                    break

    return fixed


# -------------------------------------------------------------------------
# Null geo for university-abbreviation location names
# -------------------------------------------------------------------------
_UNI_PREFIXES = ("東北大", "東大", "北大", "京大", "阪大", "帝大", "九大", "名大", "慶大", "早大")
_UNI_SAFE_PLACE_CHARS = frozenset("阪曽曾崎久門海野浦杉")


def fix_university_abbrev_locations(records):
    """Null geo data for locations whose names are university abbreviations
    followed by academic terms — not real places.

    OCR/segmentation sometimes attaches a university abbreviation (東大, 北大,
    etc.) to adjacent tokens, producing strings like "東大法科卒" or "北大農學部"
    that the geocoder matches against unrelated MCGD/GeoNames entries.

    Keeps real place names: 東大阪 (Higashi-Osaka), 東大曽根町 (Nagoya),
    北大海村 (Aomori), etc.
    """
    _NULL_GEO_FIELDS = (
        "geonameid", "latitude", "longitude", "province", "country",
        "admin1", "admin1_en", "admin2", "admin2_en", "admin3", "admin3_en",
        "admin1_norm", "name_en", "mcgd_locid",
    )
    nulled = 0
    for rec in records:
        name = rec.get("name") or ""
        if not name:
            continue
        matched_prefix = None
        for pfx in _UNI_PREFIXES:
            if name.startswith(pfx):
                matched_prefix = pfx
                break
        if not matched_prefix:
            continue
        remainder = name[len(matched_prefix):]
        if remainder and remainder[0] in _UNI_SAFE_PLACE_CHARS:
            continue
        if rec.get("geonameid") is not None or rec.get("latitude") is not None:
            for k in _NULL_GEO_FIELDS:
                rec[k] = None
            nulled += 1
    return nulled


def fix_location_overrides(locations):
    """Apply coordinate and admin1_en fixes for known mislocated places."""
    fixed_coords = 0
    fixed_names = 0
    for rec in locations:
        gid = rec.get("geonameid")
        if gid not in LOCATION_OVERRIDES:
            continue
        override = LOCATION_OVERRIDES[gid]
        for key, val in override.items():
            if rec.get(key) != val:
                rec[key] = val
        fixed_coords += 1
        admin1 = rec.get("admin1") or ""
        if admin1 in _XINJING_ADMIN1_MAP:
            rec["admin1_en"] = _XINJING_ADMIN1_MAP[admin1]
            fixed_names += 1
        elif admin1.startswith("新京") or admin1.startswith("新 京"):
            rec["admin1_en"] = "Hsinking Special City"
            fixed_names += 1
        # Null wrong name_en translations for 新京 locations
        rec["name_en"] = None
    return fixed_coords, fixed_names


# ==========================================
# ORG LOCATION ASSIGNMENT BY CITY-PREFIX REGEX
# ==========================================
_CITY_PREFIXES = [
    # Longer prefixes first to avoid partial matches
    ("名古屋", "名古屋"), ("北九州", "北九州"),
    ("鹿児島", "鹿児島"), ("鹿兒島", "鹿児島"),
    ("横須賀", "横須賀"),
    ("哈爾濱", "哈爾濱"),
    # 2-char cities (kyūjitai variants grouped)
    ("横浜", "横浜"), ("横濱", "横浜"), ("橫濱", "横浜"),
    ("東京", "東京"), ("大阪", "大阪"), ("京都", "京都"),
    ("神戸", "神戸"), ("神戶", "神戸"),
    ("札幌", "札幌"), ("仙台", "仙台"),
    ("福岡", "福岡"), ("廣島", "広島"), ("広島", "広島"),
    ("長崎", "長崎"), ("新潟", "新潟"),
    ("金沢", "金沢"), ("金澤", "金沢"),
    ("千葉", "千葉"), ("埼玉", "埼玉"),
    ("奈良", "奈良"), ("岡山", "岡山"), ("熊本", "熊本"),
    ("静岡", "静岡"), ("靜岡", "静岡"),
    ("浜松", "浜松"), ("濱松", "浜松"),
    ("小樽", "小樽"), ("函館", "函館"),
    # Colonial / overseas
    ("台北", "台北"), ("臺北", "台北"),
    ("京城", "京城"), ("新京", "新京"),
    ("大連", "大連"), ("奉天", "奉天"),
    ("釜山", "釜山"), ("平壌", "平壌"), ("平壤", "平壌"),
    ("台南", "台南"), ("臺南", "台南"),
    ("台中", "台中"), ("臺中", "台中"),
    ("上海", "上海"),
]
# Same list sorted longest-first, for suffix matching (city before 支店)
_CITY_PREFIXES_BY_LEN = sorted(_CITY_PREFIXES, key=lambda x: -len(x[0]))

# Canonical city names that should match existing location records
_CANONICAL_CITIES = sorted({c for _, c in _CITY_PREFIXES})

# Fallback coordinates for cities that have no existing location record
_CITY_FALLBACK = {
    "奉天": (41.80, 123.43, "Liaoning", "zh", "奉天"),
    "京城": (37.5665, 126.978, "Gyeonggi-do", "ko", "京城"),
}

_DEICTIC_ORG_RE = re.compile(
    r"^(同校|同社|同省|同局|同會|同院|同廳|高女|本省|現社|本縣|本府|本店|現地)$"
)
_PERSON_IN_ORG_RE = re.compile(r"(長男|長女|次男|次女|三男|養子|妻|後裔)")
_DEICTIC_MID = ("同店", "同社", "同校", "同省", "同局", "同廳")


def _build_city_to_lid(locations):
    """Build a mapping of canonical city name → location_id from existing locations.

    Looks up each city by exact name or name+市.  Falls back to _CITY_FALLBACK
    for cities with no existing location (creates new records).
    Returns (city_to_lid dict, n_new_locations_created).
    """
    city_to_lid = {}
    loc_by_name = defaultdict(list)
    for loc in locations:
        if loc.get("latitude") is not None:
            loc_by_name[loc.get("name", "")].append(loc)

    for city in _CANONICAL_CITIES:
        if city in _CITY_FALLBACK:
            continue  # handled below — existing records are wrong
        for candidate_name in (city, city + "市"):
            matches = loc_by_name.get(candidate_name, [])
            if matches:
                city_to_lid[city] = matches[0]["location_id"]
                break

    # Create location records for cities with no match or known-wrong matches
    max_lid = max(int(loc["location_id"][1:]) for loc in locations)
    n_new = 0
    for city in _CANONICAL_CITIES:
        if city not in city_to_lid and city in _CITY_FALLBACK:
            lat, lon, province, country, name = _CITY_FALLBACK[city]
            max_lid += 1
            lid = f"L{max_lid}"
            locations.append({
                "location_id": lid,
                "name": name,
                "latitude": lat,
                "longitude": lon,
                "province": province,
                "country": country,
                "geonameid": None,
                "name_en": None,
            })
            city_to_lid[city] = lid
            n_new += 1
            logger.info(f"Created location {lid} for {name} ({lat}, {lon})")

    return city_to_lid, n_new


def assign_org_locations_by_regex(organizations, locations):
    """Assign location_id to organizations whose name starts with a known city.

    Two passes:
      1. Names starting with a city prefix (excluding 支店/各/卒/deictic).
      2. Names ending with [City]支店 — uses the city directly before 支店
         as the branch location.

    Creates new location records for cities not found in existing locations
    (e.g. 奉天, 京城).  Returns (n_assigned, n_new_locations).
    """
    city_to_lid, n_new = _build_city_to_lid(locations)

    n_assigned = 0
    for org in organizations:
        if org.get("location_id"):
            continue
        name = org["name"]
        if _DEICTIC_ORG_RE.match(name):
            continue

        # Pass 2: names ending with [City]支店 — extract city before 支店
        if name.endswith("支店") and "各" not in name:
            before = name[:-2]
            for prefix, canonical in _CITY_PREFIXES_BY_LEN:
                if before.endswith(prefix):
                    lid = city_to_lid.get(canonical)
                    if lid:
                        org["location_id"] = lid
                        n_assigned += 1
                    break
            continue  # handled (matched or not), skip pass 1

        # Pass 1: names starting with a city prefix
        for prefix, canonical in _CITY_PREFIXES:
            if not name.startswith(prefix):
                continue
            if "各" in name:
                break
            if len(name) <= len(prefix):
                break
            if name.endswith("卒"):
                break
            if any(d in name for d in _DEICTIC_MID):
                break
            if _PERSON_IN_ORG_RE.search(name[len(prefix):]):
                break
            lid = city_to_lid.get(canonical)
            if lid:
                org["location_id"] = lid
                n_assigned += 1
            break

    return n_assigned, n_new


# ==========================================
# MAIN
# ==========================================
def main():
    logger.info("=== Step 7: Cross-Volume Disambiguation ===")

    # Load structured tables
    person_core = load_jsonl(os.path.join(STRUCT_OUTPUT_DIR, "person_core.jsonl"))
    person_career = load_jsonl(os.path.join(STRUCT_OUTPUT_DIR, "person_career.jsonl"))
    person_education = load_jsonl(os.path.join(STRUCT_OUTPUT_DIR, "person_education.jsonl"))
    person_hobbies = load_jsonl(os.path.join(STRUCT_OUTPUT_DIR, "person_hobbies.jsonl"))
    person_ranks = load_jsonl(os.path.join(STRUCT_OUTPUT_DIR, "person_ranks.jsonl"))
    person_religions = load_jsonl(os.path.join(STRUCT_OUTPUT_DIR, "person_religions.jsonl"))
    person_political_parties = load_jsonl(
        os.path.join(STRUCT_OUTPUT_DIR, "person_political_parties.jsonl"))
    person_family = load_jsonl(os.path.join(STRUCT_OUTPUT_DIR, "person_family_members.jsonl"))
    person_family_education = load_jsonl(os.path.join(STRUCT_OUTPUT_DIR, "person_family_education.jsonl"))
    person_family_career = load_jsonl(os.path.join(STRUCT_OUTPUT_DIR, "person_family_career.jsonl"))
    organizations = load_jsonl(os.path.join(STRUCT_OUTPUT_DIR, "organizations.jsonl"))
    locations = load_jsonl(os.path.join(STRUCT_OUTPUT_DIR, "locations.jsonl"))

    logger.info(f"Loaded: {len(person_core)} persons, {len(organizations)} orgs, {len(locations)} locations")

    # 0. Drop phantom family members (all identity fields null)
    n_phantom_fm, n_phantom_fe, n_phantom_fc = _drop_phantom_family_members(
        person_family, person_family_education, person_family_career)
    if n_phantom_fm:
        logger.info(f"Dropped {n_phantom_fm} phantom family members "
                    f"({n_phantom_fe} orphaned edu, {n_phantom_fc} orphaned career)")

    # 0b. Null garbled era-year and deictic place values
    n_garbled = _fix_garbled_and_deictic_places(person_core, person_family,
                                                 person_career, person_family_career)
    if n_garbled:
        logger.info(f"Nulled {n_garbled} garbled era-year / deictic / number place values")

    # 1. Person disambiguation
    id_remap, person_mappings = disambiguate_persons(person_core)

    # 1b. Extended disambiguation tiers (phone, career overlap, OCR decade error)
    extended_mappings = disambiguate_persons_extended(person_core, person_career, id_remap)
    person_mappings.extend(extended_mappings)

    if id_remap:
        # Build merged person_core
        canonical_ids = set(id_remap.values())
        merged_ids = set(id_remap.keys())

        # Group merged records by canonical ID
        core_by_id = {rec["person_id"]: rec for rec in person_core}
        merged_groups = defaultdict(list)
        for mid, cid in id_remap.items():
            if mid in core_by_id:
                merged_groups[cid].append(core_by_id[mid])

        # Merge and replace canonical records
        new_core = []
        for rec in person_core:
            pid = rec["person_id"]
            if pid in merged_ids:
                continue  # skip merged-away records
            if pid in merged_groups:
                rec = merge_person_records(rec, merged_groups[pid])
            new_core.append(rec)
        person_core = new_core

        # Remap person_id in all related tables
        person_career = remap_table(person_career, "person_id", id_remap)
        person_education = remap_table(person_education, "person_id", id_remap)
        person_hobbies = remap_table(person_hobbies, "person_id", id_remap)
        person_ranks = remap_table(person_ranks, "person_id", id_remap)
        person_religions = remap_table(person_religions, "person_id", id_remap)
        person_political_parties = remap_table(person_political_parties, "person_id", id_remap)
        person_family = remap_table(person_family, "person_id", id_remap)
        person_family_education = remap_table(person_family_education, "person_id", id_remap)
        person_family_career = remap_table(person_family_career, "person_id", id_remap)

        # Deduplicate entries that became identical after merge
        person_career = deduplicate_entries(person_career, ["person_id", "job_title", "organization_id", "start_year"])
        person_education = deduplicate_entries(person_education, ["person_id", "organization_id", "major_of_study", "year_graduated"])
        person_hobbies = deduplicate_entries(person_hobbies, ["person_id", "hobby"])
        # Merge kyūjitai/shinjitai hobby variants (e.g. 謠曲↔謡曲)
        n_hobby_dedup = _dedup_hobby_variants(person_hobbies)
        if n_hobby_dedup:
            logger.info(f"person_hobbies: {n_hobby_dedup} kyūjitai-variant duplicates merged")
        person_ranks = deduplicate_entries(person_ranks, ["person_id", "rank"])
        person_religions = deduplicate_entries(person_religions, ["person_id", "religion"])
        person_political_parties = deduplicate_entries(person_political_parties, ["person_id", "political_party"])
        # Fuzzy-dedup family members: same (person_id, name) = same person,
        # regardless of relation label or birth_year differences between volumes.
        person_family, relation_id_remap = deduplicate_family_members(person_family)

        if relation_id_remap:
            for rec in person_family_education:
                key = (rec["person_id"], rec.get("relation_id"))
                if key in relation_id_remap:
                    rec["relation_id"] = relation_id_remap[key]
            for rec in person_family_career:
                key = (rec["person_id"], rec.get("relation_id"))
                if key in relation_id_remap:
                    rec["relation_id"] = relation_id_remap[key]
            logger.info(f"Relation ID remap: {len(relation_id_remap)} relation_ids unified across volumes")

        person_family_education = deduplicate_entries(person_family_education,
            ["person_id", "relation_id", "organization_id", "major_of_study", "year_graduated"])
        person_family_career = deduplicate_entries(person_family_career,
            ["person_id", "relation_id", "job_title", "organization_id", "start_year"])

    # Remove records that are strict subsets of a more specific record
    n_career_before = len(person_career)
    person_career = remove_subset_records(
        person_career,
        group_fields=["person_id", "organization_id"],
        detail_fields=["job_title", "start_year", "place_name", "current"],
    )
    n_edu_before = len(person_education)
    person_education = remove_subset_records(
        person_education,
        group_fields=["person_id", "organization_id"],
        detail_fields=["major_of_study", "year_graduated"],
    )
    n_fm_career_before = len(person_family_career)
    person_family_career = remove_subset_records(
        person_family_career,
        group_fields=["person_id", "relation_id", "organization_id"],
        detail_fields=["job_title", "start_year", "place_name", "current"],
    )
    n_fm_edu_before = len(person_family_education)
    person_family_education = remove_subset_records(
        person_family_education,
        group_fields=["person_id", "relation_id", "organization_id"],
        detail_fields=["major_of_study", "year_graduated"],
    )
    logger.info(
        f"Subset removal: career {n_career_before}→{len(person_career)}, "
        f"education {n_edu_before}→{len(person_education)}, "
        f"family career {n_fm_career_before}→{len(person_family_career)}, "
        f"family education {n_fm_edu_before}→{len(person_family_education)}"
    )

    # 2a. Kyūjitai org merge — exact-match after normalization (no LLM needed)
    org_by_norm = defaultdict(list)
    for org in organizations:
        norm = _normalize_kyujitai(org["name"])
        org_by_norm[norm].append(org)
    kyujitai_org_remap = {}
    for norm, group in org_by_norm.items():
        if len(group) < 2:
            continue
        # Only merge if names actually differ (i.e. kyujitai variant exists)
        names = {o["name"] for o in group}
        if len(names) < 2:
            continue
        canonical = group[0]
        for dup in group[1:]:
            if dup["name"] != canonical["name"]:
                kyujitai_org_remap[dup["organization_id"]] = canonical["organization_id"]
    if kyujitai_org_remap:
        # Resolve transitive chains
        changed = True
        while changed:
            changed = False
            for mid in kyujitai_org_remap:
                cid = kyujitai_org_remap[mid]
                if cid in kyujitai_org_remap:
                    kyujitai_org_remap[mid] = kyujitai_org_remap[cid]
                    changed = True
        merged_ids_kyu = set(kyujitai_org_remap.keys())
        organizations = [o for o in organizations if o["organization_id"] not in merged_ids_kyu]
        for table in [person_career, person_education, person_family_career, person_family_education]:
            for rec in table:
                if rec.get("organization_id") in kyujitai_org_remap:
                    rec["organization_id"] = kyujitai_org_remap[rec["organization_id"]]
        logger.info(f"Kyūjitai org merge: {len(kyujitai_org_remap)} orgs merged "
                    f"(exact match after normalization), {len(organizations)} remain")
    else:
        logger.info("Kyūjitai org merge: no variant pairs found")

    # 2a-ii. Corporate suffix merge (横浜船渠 + 横浜船渠株 → one)
    n_corp = _merge_corporate_suffix_orgs(
        organizations, person_career, person_education,
        person_family_career, person_family_education)
    if n_corp:
        logger.info(f"Corporate suffix org merge: {n_corp} orgs merged, "
                    f"{len(organizations)} remain")

    # 2b. Organization fuzzy matching
    org_candidates = find_fuzzy_org_matches(organizations, DISAMBIG_ORG_FUZZY_THRESHOLD)
    if org_candidates:
        org_candidates = verify_org_matches_llm(org_candidates)

    # Apply LLM-verified org merges
    org_remap_auto = {}
    if org_candidates:
        for cand in org_candidates:
            canonical = cand["canonical_id"]
            for mid in cand["merged_ids"]:
                org_remap_auto[mid] = canonical
        # Resolve transitive chains (A->B, B->C becomes A->C)
        changed = True
        while changed:
            changed = False
            for mid in org_remap_auto:
                cid = org_remap_auto[mid]
                if cid in org_remap_auto:
                    org_remap_auto[mid] = org_remap_auto[cid]
                    changed = True
        merged_ids_auto = set(org_remap_auto.keys())
        organizations = [o for o in organizations if o["organization_id"] not in merged_ids_auto]
        for table in [person_career, person_education, person_family_career, person_family_education]:
            for rec in table:
                if rec.get("organization_id") in org_remap_auto:
                    rec["organization_id"] = org_remap_auto[rec["organization_id"]]
        # Deduplicate career/education entries that became identical after org merge
        person_career = deduplicate_entries(person_career, ["person_id", "job_title", "organization_id", "start_year"])
        person_education = deduplicate_entries(person_education, ["person_id", "organization_id", "major_of_study", "year_graduated"])
        person_family_career = deduplicate_entries(person_family_career,
            ["person_id", "relation_id", "job_title", "organization_id", "start_year"])
        person_family_education = deduplicate_entries(person_family_education,
            ["person_id", "relation_id", "organization_id", "major_of_study", "year_graduated"])
        # Remove subset records created by org merge
        person_career = remove_subset_records(
            person_career,
            group_fields=["person_id", "organization_id"],
            detail_fields=["job_title", "start_year", "place_name", "current"],
        )
        person_education = remove_subset_records(
            person_education,
            group_fields=["person_id", "organization_id"],
            detail_fields=["major_of_study", "year_graduated"],
        )
        person_family_career = remove_subset_records(
            person_family_career,
            group_fields=["person_id", "relation_id", "organization_id"],
            detail_fields=["job_title", "start_year", "place_name", "current"],
        )
        person_family_education = remove_subset_records(
            person_family_education,
            group_fields=["person_id", "relation_id", "organization_id"],
            detail_fields=["major_of_study", "year_graduated"],
        )
        logger.info(f"Auto-applied {len(org_remap_auto)} LLM-verified org merges, {len(organizations)} orgs remain")

    # Apply manual org overrides if available (on top of auto merges)
    organizations, org_remap = apply_org_overrides(
        organizations, [person_career, person_education, person_family_career, person_family_education]
    )

    # 3. Organization hierarchy detection
    placenames = load_geonames_placenames()
    org_hierarchy = detect_org_hierarchy(organizations, placenames)

    # Apply parent_organization_id to organizations
    parent_map = {}
    for link in org_hierarchy:
        if link["child_id"] not in parent_map:
            parent_map[link["child_id"]] = link["parent_id"]
    for org in organizations:
        org["parent_organization_id"] = parent_map.get(org["organization_id"])

    # 4a. Remove false job entries + null 現地/現職 (before translation)
    logger.info("--- Removing false job entries + 現地/現職 ---")
    n_false_c = fix_false_jobs(person_career)
    n_false_fc = fix_false_jobs(person_family_career)
    logger.info(f"Removed {n_false_c} false jobs from person_career, "
                f"{n_false_fc} from person_family_career")
    # Null deictic placeholders
    for table in [person_career, person_family_career]:
        for rec in table:
            if rec.get("place_name") == "現地":
                rec["place_name"] = None
            if rec.get("job_title") in ("現職", "在職", "在任", "在職者"):
                rec["job_title"] = None
                rec["job_title_en"] = None
    for rec in person_core:
        if rec.get("place") == "現地":
            rec["place"] = None
        if rec.get("origin_place") == "現地":
            rec["origin_place"] = None
    for rec in person_family:
        if rec.get("place") == "現地":
            rec["place"] = None

    # Remove girls' school education for male family members
    _girls_school_re = re.compile(
        r"高女|女學校|女学校|女子.*學校|女子.*学校|女子専門|女子師範")
    org_name_by_id = {o["organization_id"]: o["name"] for o in organizations}
    fm_gender_by_key = {(fm["person_id"], fm.get("relation_id")): fm.get("gender")
                        for fm in person_family}
    before_fe = len(person_family_education)
    person_family_education = [
        e for e in person_family_education
        if not (fm_gender_by_key.get((e["person_id"], e.get("relation_id"))) == "m"
                and _girls_school_re.search(org_name_by_id.get(e.get("organization_id"), "")))
    ]
    n_girls_removed = before_fe - len(person_family_education)
    if n_girls_removed:
        logger.info(f"Removed {n_girls_removed} girls' school records for male family members")

    # Mapping: place field → corresponding location_id field
    _place_locid = {"origin_place": "origin_location_id", "place": "location_id"}

    # Convert 支店 (branch office) place fields to undated career entries
    n_shiten = 0
    for rec in person_core:
        for field in ("origin_place", "place"):
            val = rec.get(field)
            if val and val.endswith("支店"):
                person_career.append({
                    "person_id": rec["person_id"],
                    "job_title": None,
                    "organization": val,
                    "organization_id": None,
                    "start_year": None,
                    "place_name": None,
                    "current": None,
                    "source_volume": rec.get("source_volume") or rec.get("volume"),
                    "source_page": rec.get("source_page"),
                })
                rec[field] = None
                rec[_place_locid[field]] = None
                n_shiten += 1
    for rec in person_family:
        val = rec.get("place")
        if val and val.endswith("支店"):
            person_family_career.append({
                "person_id": rec["person_id"],
                "relation_id": rec.get("relation_id"),
                "job_title": None,
                "organization": val,
                "organization_id": None,
                "place_name": None,
                "start_year": None,
                "current": None,
                "source_volume": rec.get("source_volume"),
                "source_page": rec.get("source_page"),
            })
            rec["place"] = None
            rec["location_id"] = None
            n_shiten += 1
    if n_shiten:
        logger.info(f"Converted {n_shiten} 支店 place fields → career entries")

    # Null out place fields ending with family relation or organization suffix
    _org_place_suffixes = ("領事館", "鐵道", "鉄道", "中學", "中学", "造船部")
    def _is_bad_place(val):
        return _RELATION_SUFFIX_RE.search(val) or val.endswith(_org_place_suffixes)
    n_bad_place = 0
    for rec in person_core:
        for field in ("origin_place", "place"):
            val = rec.get(field)
            if val and _is_bad_place(val):
                rec[field] = None
                rec[_place_locid[field]] = None
                n_bad_place += 1
    for rec in person_family:
        val = rec.get("place")
        if val and _is_bad_place(val):
            rec["place"] = None
            rec["location_id"] = None
            n_bad_place += 1
    if n_bad_place:
        logger.info(f"Nulled {n_bad_place} place fields (relation/org suffix)")

    # Null org references to deictic org names (本省/現社/本縣)
    _deictic_orgs = frozenset({"本省", "現社", "本縣", "地方", "本府", "本店", "現地"})
    deictic_org_ids = {o["organization_id"] for o in organizations
                       if o["name"] in _deictic_orgs}
    if deictic_org_ids:
        n_deictic = 0
        for table in (person_career, person_family_career,
                      person_education, person_family_education):
            for rec in table:
                if rec.get("organization_id") in deictic_org_ids:
                    rec["organization_id"] = None
                    n_deictic += 1
        if n_deictic:
            logger.info(f"Nulled {n_deictic} org references to deictic names "
                        f"(本省/現社/本縣)")

    # 4b. Re-romanize family members with pinyin names (post-merge domain correction)
    logger.info("--- Re-romanizing family member names ---")
    ja_person_ids = {r["person_id"] for r in person_core if r.get("domain") == "ja"}
    n_fm_roman = fix_family_member_romanization(person_family, ja_person_ids)
    logger.info(f"Re-romanized {n_fm_roman} family member names (pinyin→hepburn)")

    # 4b2. Split romanization for family members with different family names,
    #       null out implausibly long given names (>6 chars)
    logger.info("--- Fixing family member name splits ---")
    known_families = {r["name_family"] for r in person_core
                      if r.get("name_family") and len(r["name_family"]) >= 2}
    core_family_by_pid = {r["person_id"]: r.get("name_family", "")
                          for r in person_core}
    n_split, n_nulled = fix_family_name_split_and_length(
        person_family, known_families, core_family_by_pid)
    logger.info(f"Family name splits: {n_split} split, {n_nulled} nulled (given >6 chars)")

    # 4b3. Convert ou → ō and uu → ū in ja-domain romaji names
    _ou_re = re.compile(r"ou(?![aeiou])", re.I)
    _uu_re = re.compile(r"uu", re.I)
    def _replace_ou(m):
        return "Ō" if m.group()[0].isupper() else "ō"
    def _replace_uu(m):
        return "Ū" if m.group()[0].isupper() else "ū"
    n_ou = 0
    for rec in person_core:
        if rec.get("domain") != "ja":
            continue
        for field in ("name_family_latin", "name_given_latin"):
            val = rec.get(field)
            if val and _ou_re.search(val):
                rec[field] = _ou_re.sub(_replace_ou, val)
                n_ou += 1
                val = rec[field]
            if val and _uu_re.search(val):
                rec[field] = _uu_re.sub(_replace_uu, val)
                n_ou += 1
    for rec in person_family:
        if rec.get("person_id") not in ja_person_ids:
            continue
        val = rec.get("name_latin")
        if val and _ou_re.search(val):
            rec["name_latin"] = _ou_re.sub(_replace_ou, val)
            n_ou += 1
            val = rec["name_latin"]
        if val and _uu_re.search(val):
            rec["name_latin"] = _uu_re.sub(_replace_uu, val)
            n_ou += 1
    logger.info(f"ou → ō / uu → ū: {n_ou} fields converted")

    # 4c. Fix location overrides + mislocated JA locations
    logger.info("--- Fixing location overrides ---")
    n_loc_coords, n_loc_names = fix_location_overrides(locations)
    if n_loc_coords:
        logger.info(f"locations: {n_loc_coords} coordinate fixes, {n_loc_names} admin1_en fixes")
    n_misloc = fix_mislocated_ja_locations(locations)
    if n_misloc:
        logger.info(f"locations: {n_misloc} mislocated JA locations nulled")
    n_manchukuo = fix_manchukuo_geocoding(locations)
    if n_manchukuo:
        logger.info(f"locations: {n_manchukuo} Manchukuo/colonial locations re-geocoded or nulled")
    n_uni = fix_university_abbrev_locations(locations)
    if n_uni:
        logger.info(f"locations: {n_uni} university abbreviation locations nulled")

    # 4d. Translate org names and job titles (after merge — fewer unique items)
    logger.info("--- Translating organization names ---")
    org_names = [o["name"] for o in organizations]
    org_translations = translate_batch_ollama(org_names, "organization")
    for org in organizations:
        org["name_en"] = org_translations.get(org["name"])

    # Fix wrong romanizations of 本縣/本府 prefix in name_en
    _honken_prefix_re = re.compile(
        r"^(?:Honken|Hon-ken|Honshu|Honten|Honma|Honjo|Hon )", re.IGNORECASE)
    _honfu_prefix_re = re.compile(
        r"^(?:Honfu|Hon-fu|Honfuro|Honjo|Honmachi|Honmura|Honno|Honshu|Hon )",
        re.IGNORECASE)
    n_deictic_en = 0
    for org in organizations:
        name_en = org.get("name_en")
        if not name_en:
            continue
        if org["name"].startswith("本縣"):
            m = _honken_prefix_re.match(name_en)
            if m:
                org["name_en"] = "Prefectural " + name_en[m.end():].lstrip()
                n_deictic_en += 1
        elif org["name"].startswith("本府"):
            m = _honfu_prefix_re.match(name_en)
            if m:
                org["name_en"] = "Prefectural " + name_en[m.end():].lstrip()
                n_deictic_en += 1
    if n_deictic_en:
        logger.info(f"Fixed {n_deictic_en} 本縣/本府 org name_en → Prefectural")

    # Fix 高女 name_en (高女 = 高等女學校 = Girls' High School, NOT "Takano")
    n_koujo = 0
    for org in organizations:
        if "高女" not in org["name"]:
            continue
        if org["name"] == "高女":
            if org.get("name_en") != "Girls' High School":
                org["name_en"] = "Girls' High School"
                n_koujo += 1
        elif org.get("name_en") and "Takano" in org["name_en"]:
            org["name_en"] = None
            n_koujo += 1
    if n_koujo:
        logger.info(f"Fixed {n_koujo} 高女 org name_en (Takano → Girls' High School)")

    # Replace wrong 新京 translations in org name_en with "Hsinking"
    # Skip 新京極 (Shinkyōgoku, Kyoto) and 新京阪 (Shin-Keihan Railway)
    _xinjing_wrong_re = re.compile(
        r"New (?:Chinchang|Chitose|Chongqing|Changhai|Chinchow|Capital|Jing|Beijing|"
        r"Kyoto|Tokyo|Changchun|Special City|Chonglin|Chinchengyaku|Chinkyo|Chingli|"
        r"Chongli|Chichibu|Chong|Chichang|Ching)"
        r"|Shinj(?:ing|yo|uku|ingyo)?(?!itsu)"
        r"|Shinkei|Shinky[oōou]?|Shinkin|Shinkyi|Shinch"
        r"|Shin-K(?:yo|oji|okyo|obe|ojima)"
        r"|Send?[gy]?ai|Shenyang|Sengyo|Mukden|Dalny|Siping|Sapporo|Harbin|Chinkwang"
        r"|X(?:in|ing)j?g?ing|Xinhua"
        r"|Changchun|Manchuria(?= (?:Branch|Office))"
        r"|Showkin",
        re.IGNORECASE,
    )
    n_xinjing = 0
    for org in organizations:
        name = org["name"]
        ne = org.get("name_en")
        if "新京" not in name or not ne:
            continue
        idx = name.index("新京")
        after = name[idx + 2: idx + 3]
        if after in ("極", "阪"):
            continue
        new_ne = _xinjing_wrong_re.sub("Hsinking", ne)
        if new_ne != ne:
            org["name_en"] = new_ne
            n_xinjing += 1
    if n_xinjing:
        logger.info(f"Fixed {n_xinjing} org name_en: 新京 → Hsinking")

    logger.info("--- Translating job titles ---")
    all_job_titles = [c["job_title"] for c in person_career if c.get("job_title")]
    all_job_titles += [c["job_title"] for c in person_family_career if c.get("job_title")]
    job_translations = translate_batch_ollama(all_job_titles, "job title")
    for career in person_career:
        if career.get("job_title"):
            career["job_title_en"] = job_translations.get(career["job_title"])
    for fc in person_family_career:
        if fc.get("job_title"):
            fc["job_title_en"] = job_translations.get(fc["job_title"])

    # 5. Build entity_mappings
    all_mappings = person_mappings + org_candidates

    # Ensure every person_core row has an `appearances` list (single-element for
    # unmerged single-volume rows; multi-element for cross-volume merged rows).
    # Derive a flat person_appearances table at the same time.
    person_appearances = []
    for rec in person_core:
        if "appearances" not in rec:
            rec["appearances"] = [_appearance_from(rec)]
        for appx in rec["appearances"]:
            person_appearances.append({
                "person_id": rec["person_id"],
                "volume": appx.get("volume"),
                "source_page": appx.get("source_page"),
                "source_image": appx.get("source_image"),
                "entry_index": appx.get("entry_index"),
            })

    # Fix ケ/ヶ misgeocoding + 東大曽根 wrong-ward
    n_ke_fix = _fix_misgeocoded_ke_locations(locations)
    if n_ke_fix:
        logger.info(f"Fixed {n_ke_fix} misgeocoded locations (ケ/ヶ stripping + 東大曽根 wrong ward)")

    # Null stale location_id refs pointing to geo-nulled locations
    hollow_lids = set()
    for loc in locations:
        if (loc.get("latitude") is None and loc.get("longitude") is None
                and loc.get("geonameid") is None and loc.get("province") is None
                and loc.get("country") is None):
            hollow_lids.add(loc["location_id"])
    if hollow_lids:
        n_stale = 0
        for org in organizations:
            if org.get("location_id") in hollow_lids:
                org["location_id"] = None
                n_stale += 1
        for rec in person_core:
            if rec.get("location_id") in hollow_lids:
                rec["location_id"] = None
                n_stale += 1
            if rec.get("origin_location_id") in hollow_lids:
                rec["origin_location_id"] = None
                n_stale += 1
        for table in (person_career, person_family_career, person_family):
            for rec in table:
                if rec.get("location_id") in hollow_lids:
                    rec["location_id"] = None
                    n_stale += 1
        if n_stale:
            logger.info(f"Nulled {n_stale} stale location_id refs to geo-nulled locations")

    # 4f. Regex-based org location assignment
    n_org_loc, n_new_loc = assign_org_locations_by_regex(organizations, locations)
    if n_org_loc:
        logger.info(f"Assigned {n_org_loc} org locations by city-prefix regex "
                    f"({n_new_loc} new location records created)")

    # 4g. Null org refs for family-relation org names (長男, 二女, 養子, etc.)
    n_relation_org = _null_relation_org_refs(
        organizations, person_education, person_family_education,
        person_career, person_family_career)
    if n_relation_org:
        logger.info(f"Nulled {n_relation_org} refs to family-relation org names")

    # 4h. Null org location_ids pointing to locations with org-type names
    n_orgname_loc = _null_orgname_location_refs(locations, organizations)
    if n_orgname_loc:
        logger.info(f"Nulled {n_orgname_loc} org location_ids pointing to org-name locations")

    # 5. Write output
    write_jsonl(os.path.join(DISAMBIG_OUTPUT_DIR, "person_core.jsonl"), person_core)
    write_jsonl(os.path.join(DISAMBIG_OUTPUT_DIR, "person_appearances.jsonl"), person_appearances)
    write_jsonl(os.path.join(DISAMBIG_OUTPUT_DIR, "person_career.jsonl"), person_career)
    write_jsonl(os.path.join(DISAMBIG_OUTPUT_DIR, "person_education.jsonl"), person_education)
    write_jsonl(os.path.join(DISAMBIG_OUTPUT_DIR, "person_hobbies.jsonl"), person_hobbies)
    write_jsonl(os.path.join(DISAMBIG_OUTPUT_DIR, "person_ranks.jsonl"), person_ranks)
    write_jsonl(os.path.join(DISAMBIG_OUTPUT_DIR, "person_religions.jsonl"), person_religions)
    write_jsonl(os.path.join(DISAMBIG_OUTPUT_DIR, "person_political_parties.jsonl"), person_political_parties)
    write_jsonl(os.path.join(DISAMBIG_OUTPUT_DIR, "person_family_members.jsonl"), person_family)
    write_jsonl(os.path.join(DISAMBIG_OUTPUT_DIR, "person_family_education.jsonl"), person_family_education)
    write_jsonl(os.path.join(DISAMBIG_OUTPUT_DIR, "person_family_career.jsonl"), person_family_career)
    write_jsonl(os.path.join(DISAMBIG_OUTPUT_DIR, "organizations.jsonl"), organizations)
    write_jsonl(os.path.join(DISAMBIG_OUTPUT_DIR, "locations.jsonl"), locations)
    write_jsonl(os.path.join(DISAMBIG_OUTPUT_DIR, "entity_mappings.jsonl"), all_mappings)

    logger.info(f"=== Step 7 complete: {len(id_remap)} persons merged, "
                f"{len(org_remap_auto)} orgs auto-merged, "
                f"{len(org_hierarchy)} org hierarchy links ===")


if __name__ == "__main__":
    main()
