#!/usr/bin/env python3
"""compute_paper_stats.py — Descriptive statistics, gold-standard sampling,
scoring, and SLM-vs-cloud-LLM extraction comparison for the archival Japanese
digitisation pipeline.

OCR evaluation is intentionally NOT here — it lives in eval_ocr.py, which does
the 4-way Ollama × Google × {original, resize} comparison with proper CER,
precision, recall, and F1 against a manual gold standard.

Creates output in paper_stats/.  Does **not** modify any existing files.

Usage:
    python compute_paper_stats.py --stats
    python compute_paper_stats.py --export-gold [--bio-dir PATH --ocr-dir PATH]
    python compute_paper_stats.py --score-gold
    python compute_paper_stats.py --run-cloud --cloud-provider google --cloud-model gemini-2.0-flash
    python compute_paper_stats.py --compare-hisco
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

# Increase CSV field-size limit for large JSON blobs in extraction samples
csv.field_size_limit(2 ** 24)

# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def load_jsonl(path):
    """Load a JSONL file, skipping meta-header records."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("_meta"):
                continue
            records.append(rec)
    return records


def save_csv(path, rows, fieldnames=None):
    """Write *rows* (list[dict]) to *path* as CSV."""
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {path} ({len(rows)} rows)")


def save_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"  Wrote {path}")


def _safe_div(a, b, default=0.0):
    return a / b if b else default


def _pct(a, b):
    return f"{_safe_div(a, b) * 100:.1f}%"


def _iter_volumes(rec):
    """Yield each volume string for a record. Step 7 turns ``volume`` into a
    ``list[str]`` for cross-volume merges; pre-disambiguation rows keep it as
    ``str``. None values pass through untouched."""
    v = rec.get("volume")
    if isinstance(v, list):
        yield from v
    else:
        yield v


def _data_path(data_dir, filename):
    """Resolve *filename* in *data_dir*; fall back to sibling ``disambiguated/``
    and ``structured/`` directories at the script root. Lets ``--stats`` pull
    checkpoints (only in ``disambiguated/``) and ``editorial_changes.jsonl``
    (only in ``structured/``) without an extra flag. Returns the primary path
    when the file is absent everywhere so existence checks stay readable."""
    primary = os.path.join(data_dir, filename)  # noqa: not the helper itself
    if os.path.exists(primary):
        return primary
    script_root = os.path.dirname(os.path.abspath(__file__))
    for sibling in ("disambiguated", "structured"):
        cand = os.path.join(script_root, sibling, filename)
        if os.path.exists(cand):
            return cand
    return primary


# ─── Shinjitai → Kyūjitai normalisation for strict comparison ──────────────
# Models occasionally emit shinjitai (modern simplified Japanese: 国, 学, 区)
# while gold annotations and other models use kyūjitai (historical: 國, 學, 區)
# — or vice versa — even though the prompt asks for kyūjitai preservation.
# Normalising both sides to kyūjitai before strict equality removes this
# orthographic noise. Only includes 1:1 mappings (no ambiguous conversions
# like 弁 ← 辨/瓣/辮).
SHIN_TO_KY = {
    "真":"眞","国":"國","区":"區","実":"實","学":"學","会":"會","広":"廣","芸":"藝",
    "楽":"樂","沢":"澤","静":"靜","医":"醫","鉄":"鐵","台":"臺","辺":"邊","変":"變",
    "寿":"壽","豊":"豐","応":"應","営":"營","検":"檢","駅":"驛","体":"體","続":"續",
    "尽":"盡","当":"當","声":"聲","万":"萬","気":"氣","参":"參","与":"與","励":"勵",
    "栄":"榮","団":"團","円":"圓","弥":"彌","稲":"稻","払":"拂","県":"縣","独":"獨",
    "桜":"櫻","条":"條","抜":"拔","廃":"廢","闘":"鬪","歯":"齒","塩":"鹽","峰":"峯",
    "竜":"龍","彦":"彥","壱":"壹","弐":"貳","継":"繼","聴":"聽","霊":"靈","顕":"顯",
    "験":"驗","発":"發","仮":"假","雑":"雜","旧":"舊","即":"卽","挙":"擧","勲":"勳",
    "鉱":"礦","産":"產","剤":"劑","随":"隨","蔵":"藏","号":"號","塁":"壘","双":"雙",
    "両":"兩","売":"賣","読":"讀","対":"對","総":"總","奨":"獎","従":"從","駆":"驅",
    "図":"圖","囲":"圍","検":"檢","機":"機","乱":"亂","糸":"絲","絵":"繪","姉":"姊",
    "斎":"齋","斉":"齊","児":"兒","写":"寫","処":"處","渋":"澁","収":"收","粛":"肅",
    "称":"稱","従":"從","証":"證","乗":"乘","状":"狀","畳":"疊","条":"條","嬢":"孃",
    "壌":"壤","譲":"讓","醸":"釀","触":"觸","嘱":"囑","属":"屬","続":"續","堕":"墮",
    "対":"對","帯":"帶","滞":"滯","台":"颱","沢":"澤","担":"擔","胆":"膽","団":"團",
    "弾":"彈","遅":"遲","昼":"晝","虫":"蟲","鋳":"鑄","聴":"聽","懲":"懲","勅":"敕",
    "鎮":"鎭","逓":"遞","鉄":"鐵","点":"點","伝":"傳","当":"當","党":"黨","盗":"盜",
    "灯":"燈","闘":"鬪","徳":"德","独":"獨","読":"讀","届":"屆","縄":"繩","軟":"軟",
    "弐":"貳","悩":"惱","脳":"腦","廃":"廢","売":"賣","麦":"麥","発":"發","髪":"髮",
    "抜":"拔","繁":"繁","蛮":"蠻","卑":"卑","秘":"祕","浜":"濱","宝":"寶","豊":"豐",
    "翻":"飜","毎":"毎","万":"萬","満":"滿","免":"免","訳":"譯","薬":"藥","与":"與",
    "余":"餘","誉":"譽","揺":"搖","謡":"謠","様":"樣","羅":"羅","頼":"賴","乱":"亂",
    "覧":"覽","欄":"欄","竜":"龍","虜":"虜","両":"兩","猟":"獵","糧":"糧","励":"勵",
    "礼":"禮","炉":"爐","労":"勞","郎":"郞","録":"錄","湾":"灣",
}


def _normalize_chars(s):
    """Apply shinjitai → kyūjitai map for fair strict comparison."""
    if not isinstance(s, str):
        return s
    return s.translate(str.maketrans(SHIN_TO_KY))


# Free-text scalar fields where partial (substring) match is meaningful.
PARTIAL_SCALAR_FIELDS = {"name", "place", "origin_place", "rank", "religion"}


# ─── BERTScore (lazy, opt-in) ───────────────────────────────────────────────
# Used by score_extraction when --bertscore is passed. Adds semantic-similarity
# F1 alongside the strict-equality scoring so the paper can report both rigorous
# and lenient numbers (e.g. variant kanji 佛教/仏教 fail strict but score ~1.0
# on BERTScore).
DEFAULT_BERT_MODEL = "cl-tohoku/bert-base-japanese-v3"
BERT_F1_MATCH = 0.85   # threshold for "counts as a match" in match-rate stats

_BERT_MODEL_NAME = DEFAULT_BERT_MODEL
_BERT_CACHE = {}       # (cand, ref) -> F1


def bert_score_batch(pairs):
    """Score (cand, ref) string pairs in one batched call. Returns list[float] of F1.

    Empty strings on either side return F1=0.0 without invoking the model.
    Caches per (cand, ref) so repeated rows (e.g. when a string appears in both
    SLM and cloud comparisons) don't re-encode.
    """
    if not pairs:
        return []
    out = [0.0] * len(pairs)
    todo_idx, todo_cand, todo_ref = [], [], []
    for i, (c, r) in enumerate(pairs):
        c, r = (c or "").strip(), (r or "").strip()
        if not c or not r:
            continue
        key = (c, r)
        if key in _BERT_CACHE:
            out[i] = _BERT_CACHE[key]
        else:
            todo_idx.append(i); todo_cand.append(c); todo_ref.append(r)
    if todo_cand:
        from bert_score import score as _bs   # lazy import
        # bert_score's model2layers table is hardcoded; newer models like
        # cl-tohoku/bert-base-japanese-v3 aren't in it. Try the table first,
        # then fall back to the BERT-base convention (layer 9 of 12) — works
        # for any 12-layer Japanese BERT-base variant.
        try:
            _, _, F1 = _bs(todo_cand, todo_ref, model_type=_BERT_MODEL_NAME,
                           lang="ja", verbose=False, rescale_with_baseline=False)
        except KeyError:
            _, _, F1 = _bs(todo_cand, todo_ref, model_type=_BERT_MODEL_NAME,
                           num_layers=9,
                           lang="ja", verbose=False, rescale_with_baseline=False)
        for j, i in enumerate(todo_idx):
            f = float(F1[j])
            _BERT_CACHE[(todo_cand[j], todo_ref[j])] = f
            out[i] = f
    return out


# ═══════════════════════════════════════════════════════════════════════════
# SECTION A — Descriptive Statistics  (7 tables)
# ═══════════════════════════════════════════════════════════════════════════

def _dist_stats(counter, n_total):
    """Mean / median / max for a Counter of per-entity counts, zero-filling."""
    vals = list(counter.values()) + [0] * max(0, n_total - len(counter))
    if not vals:
        return {"mean": 0, "median": 0, "max": 0}
    return {
        "mean": round(statistics.mean(vals), 2),
        "median": round(statistics.median(vals), 1),
        "max": max(vals),
    }


def compute_dataset_overview(data_dir, out_dir):
    """Table 1 — record counts, per-person distributions."""
    print("\n=== Table 1: Dataset Overview ===")
    persons  = load_jsonl(_data_path(data_dir, "person_core.jsonl"))
    careers  = load_jsonl(_data_path(data_dir, "person_career.jsonl"))
    edu      = load_jsonl(_data_path(data_dir, "person_education.jsonl"))
    family   = load_jsonl(_data_path(data_dir, "person_family_members.jsonl"))
    orgs     = load_jsonl(_data_path(data_dir, "organizations.jsonl"))
    locs     = load_jsonl(_data_path(data_dir, "locations.jsonl"))

    optional = {}
    for tbl in ("person_hobbies", "person_ranks", "person_religions",
                "person_political_parties"):
        p = _data_path(data_dir, f"{tbl}.jsonl")
        optional[tbl] = load_jsonl(p) if os.path.exists(p) else []

    n = len(persons)
    vol_counts = Counter(v for p in persons for v in _iter_volumes(p))
    cs = _dist_stats(Counter(c["person_id"] for c in careers), n)
    es = _dist_stats(Counter(e["person_id"] for e in edu), n)
    fs = _dist_stats(Counter(f["person_id"] for f in family), n)

    rows = [{"metric": "Persons (total)", "value": n}]
    for v in sorted(vol_counts, key=str):
        rows.append({"metric": f"Persons (volume {v})", "value": vol_counts[v]})
    if sum(vol_counts.values()) > n:
        rows.append({"metric": "Persons (volume sum > total)",
                     "value": "cross-volume merges counted in each appearing volume"})
    rows += [
        {"metric": "Career records",          "value": len(careers)},
        {"metric": "Education records",        "value": len(edu)},
        {"metric": "Family member records",    "value": len(family)},
        {"metric": "Hobby records",            "value": len(optional["person_hobbies"])},
        {"metric": "Rank records",             "value": len(optional["person_ranks"])},
        {"metric": "Religion records",         "value": len(optional["person_religions"])},
        {"metric": "Political party records",  "value": len(optional["person_political_parties"])},
        {"metric": "Organisations",            "value": len(orgs)},
        {"metric": "Locations",                "value": len(locs)},
        {"metric": "Careers/person (mean)",    "value": cs["mean"]},
        {"metric": "Careers/person (median)",  "value": cs["median"]},
        {"metric": "Careers/person (max)",     "value": cs["max"]},
        {"metric": "Education/person (mean)",  "value": es["mean"]},
        {"metric": "Education/person (median)","value": es["median"]},
        {"metric": "Family/person (mean)",     "value": fs["mean"]},
        {"metric": "Family/person (median)",   "value": fs["median"]},
    ]

    # Coverage: persons with at least one record of each type
    n_with_career = len(set(c["person_id"] for c in careers))
    n_with_edu    = len(set(e["person_id"] for e in edu))
    n_with_family = len(set(f["person_id"] for f in family))
    n_with_hobby  = len(set(h["person_id"] for h in optional["person_hobbies"]))
    n_with_religion = len(set(r["person_id"] for r in optional["person_religions"]))
    n_with_party  = len(set(p["person_id"] for p in optional["person_political_parties"]))
    rows += [
        {"metric": "Persons with career",      "value": f"{n_with_career} ({_pct(n_with_career, n)})"},
        {"metric": "Persons with education",   "value": f"{n_with_edu} ({_pct(n_with_edu, n)})"},
        {"metric": "Persons with family data", "value": f"{n_with_family} ({_pct(n_with_family, n)})"},
        {"metric": "Persons with hobbies",     "value": f"{n_with_hobby} ({_pct(n_with_hobby, n)})"},
        {"metric": "Persons with religion",    "value": f"{n_with_religion} ({_pct(n_with_religion, n)})"},
        {"metric": "Persons with political party", "value": f"{n_with_party} ({_pct(n_with_party, n)})"},
    ]
    save_csv(os.path.join(out_dir, "tables", "table1_overview.csv"), rows)
    return rows


# Dashboard default filter — mirrors website/app.R's main view (post-2026-04-30).
# Two gates only: usable Latin family name + biographical-period birthyear.
# If app.R's defaults change, update DASHBOARD_BIRTHYEAR_RANGE below.
DASHBOARD_BIRTHYEAR_MIN = 1845
DASHBOARD_BIRTHYEAR_MAX = 1945
DASHBOARD_INCLUDE_UNKNOWN_BIRTHYEAR = True  # default switch in app.R


def _apply_dashboard_filter(persons):
    """Return persons that pass the dashboard's default filter:
      - name_family_latin present and ≥ 2 chars
      - birthyear in [1845, 1945] OR null (with include-unknown ON)"""
    kept = []
    for p in persons:
        nfl = p.get("name_family_latin") or ""
        if len(nfl) < 2:
            continue
        by = p.get("birthyear")
        if by is None:
            if not DASHBOARD_INCLUDE_UNKNOWN_BIRTHYEAR:
                continue
        elif not (DASHBOARD_BIRTHYEAR_MIN <= by <= DASHBOARD_BIRTHYEAR_MAX):
            continue
        kept.append(p)
    return kept


def compute_dashboard_filtered_counts(data_dir, out_dir):
    """Table 1b — record counts under the same default filter as
    website/app.R's main view. Reproduces the headline numbers a user sees
    when opening the Shiny dashboard with no filters touched."""
    print("\n=== Table 1b: Dashboard Default Filter ===")
    persons = load_jsonl(_data_path(data_dir, "person_core.jsonl"))
    careers = load_jsonl(_data_path(data_dir, "person_career.jsonl"))
    edu     = load_jsonl(_data_path(data_dir, "person_education.jsonl"))
    family  = load_jsonl(_data_path(data_dir, "person_family_members.jsonl"))
    orgs    = load_jsonl(_data_path(data_dir, "organizations.jsonl"))
    locs    = load_jsonl(_data_path(data_dir, "locations.jsonl"))

    optional = {}
    for tbl in ("person_hobbies", "person_ranks", "person_religions",
                "person_political_parties"):
        p = _data_path(data_dir, f"{tbl}.jsonl")
        optional[tbl] = load_jsonl(p) if os.path.exists(p) else []

    f_persons = _apply_dashboard_filter(persons)
    kept_ids = {p["person_id"] for p in f_persons}

    f_careers = [c for c in careers if c.get("person_id") in kept_ids]
    f_edu     = [e for e in edu     if e.get("person_id") in kept_ids]
    f_family  = [r for r in family  if r.get("person_id") in kept_ids]
    f_hobbies = [r for r in optional["person_hobbies"]
                 if r.get("person_id") in kept_ids]
    f_ranks = [r for r in optional["person_ranks"]
               if r.get("person_id") in kept_ids]
    f_religions = [r for r in optional["person_religions"]
                   if r.get("person_id") in kept_ids]
    f_parties = [r for r in optional["person_political_parties"]
                 if r.get("person_id") in kept_ids]

    # Organisations linked to at least one kept person's career (matches
    # app.R's behaviour: orgs are only "visible" via filtered persons).
    visible_org_ids = {c.get("organization_id") for c in f_careers
                       if c.get("organization_id")}
    f_orgs = [o for o in orgs
              if o.get("organization_id") in visible_org_ids
              and (o.get("name") or "").strip()]

    # Locations referenced by any visible record.
    visible_loc_ids = set()
    for p in f_persons:
        for k in ("location_id", "origin_location_id"):
            v = p.get(k)
            if v:
                visible_loc_ids.add(v)
    for c in f_careers:
        v = c.get("location_id")
        if v:
            visible_loc_ids.add(v)
    for o in f_orgs:
        v = o.get("location_id")
        if v:
            visible_loc_ids.add(v)
    f_locs = [l for l in locs if l.get("location_id") in visible_loc_ids]

    n_raw = len(persons)
    n_f = len(f_persons)

    rows = [
        {"metric": "Filter spec",
         "value": f"name_family_latin≥2 AND birthyear∈"
                  f"[{DASHBOARD_BIRTHYEAR_MIN},{DASHBOARD_BIRTHYEAR_MAX}]∪null"},
        {"metric": "Persons (raw)",                    "value": n_raw},
        {"metric": "Persons (dashboard default)",      "value": f"{n_f} ({_pct(n_f, n_raw)})"},
        {"metric": "Career records",                   "value": len(f_careers)},
        {"metric": "Education records",                "value": len(f_edu)},
        {"metric": "Family member records",            "value": len(f_family)},
        {"metric": "Hobby records",                    "value": len(f_hobbies)},
        {"metric": "Rank records",                     "value": len(f_ranks)},
        {"metric": "Religion records",                 "value": len(f_religions)},
        {"metric": "Political party records",          "value": len(f_parties)},
        {"metric": "Organisations (linked)",           "value": len(f_orgs)},
        {"metric": "Locations (referenced)",           "value": len(f_locs)},
    ]

    # Reasons for exclusion (counted independently — overlap is possible).
    excl_no_latin = sum(1 for p in persons
                        if len(p.get("name_family_latin") or "") < 2)
    excl_birthyear = sum(
        1 for p in persons
        if p.get("birthyear") is not None
        and not (DASHBOARD_BIRTHYEAR_MIN <= p["birthyear"]
                 <= DASHBOARD_BIRTHYEAR_MAX)
    )
    rows += [
        {"metric": "Excluded: name_family_latin <2",
         "value": f"{excl_no_latin} ({_pct(excl_no_latin, n_raw)})"},
        {"metric": f"Excluded: birthyear outside "
                   f"[{DASHBOARD_BIRTHYEAR_MIN},{DASHBOARD_BIRTHYEAR_MAX}]",
         "value": f"{excl_birthyear} ({_pct(excl_birthyear, n_raw)})"},
    ]

    save_csv(os.path.join(out_dir, "tables",
                          "table1b_dashboard_default.csv"), rows)
    return rows


def compute_demographics(data_dir, out_dir):
    """Table 2 — birth year, gender, domain, origin province."""
    print("\n=== Table 2: Demographics ===")
    persons = load_jsonl(_data_path(data_dir, "person_core.jsonl"))
    locs    = load_jsonl(_data_path(data_dir, "locations.jsonl"))
    loc_map = {l["location_id"]: l for l in locs}
    n = len(persons)
    rows = []

    # Birth year
    bys = [p["birthyear"] for p in persons if p.get("birthyear")]
    if bys:
        rows += [
            {"metric": "Birth year (N with data)", "value": len(bys)},
            {"metric": "Birth year (mean)",  "value": round(statistics.mean(bys), 1)},
            {"metric": "Birth year (SD)",    "value": round(statistics.stdev(bys), 1) if len(bys) > 1 else "N/A"},
            {"metric": "Birth year (min)",   "value": min(bys)},
            {"metric": "Birth year (max)",   "value": max(bys)},
        ]
        s = sorted(bys)
        for d in range(1, 11):
            rows.append({"metric": f"Birth year (P{d*10})",
                         "value": s[int(len(s) * d / 10) - 1]})

    # Gender (total + per volume)
    gc = Counter(p.get("gender") for p in persons)
    for g in sorted(gc, key=str):
        rows.append({"metric": f"Gender: {g}", "value": f"{gc[g]} ({_pct(gc[g], n)})"})
    vg = defaultdict(Counter)
    for p in persons:
        for v in _iter_volumes(p):
            vg[v][p.get("gender")] += 1
    for vol in sorted(vg, key=str):
        vn = sum(vg[vol].values())
        for g in sorted(vg[vol], key=str):
            rows.append({"metric": f"Gender: {g} (vol {vol})",
                         "value": f"{vg[vol][g]} ({_pct(vg[vol][g], vn)})"})

    # Domain
    dc = Counter(p.get("domain") for p in persons)
    for d in sorted(dc, key=str):
        rows.append({"metric": f"Domain: {d}", "value": f"{dc[d]} ({_pct(dc[d], n)})"})

    # Origin province top-10
    pc = Counter()
    for p in persons:
        olid = p.get("origin_location_id")
        if olid and olid in loc_map:
            prov = loc_map[olid].get("admin1")
            if prov:
                pc[prov] += 1
    for prov, cnt in pc.most_common(10):
        rows.append({"metric": f"Origin province: {prov}", "value": cnt})

    save_csv(os.path.join(out_dir, "tables", "table2_demographics.csv"), rows)
    return rows


def compute_careers(data_dir, out_dir):
    """Table 3 — HISCO, ISIC, temporal, top job titles & orgs."""
    print("\n=== Table 3: Careers & Occupations ===")
    careers = load_jsonl(_data_path(data_dir, "person_career.jsonl"))
    orgs    = load_jsonl(_data_path(data_dir, "organizations.jsonl"))
    rows = []

    # HISCO major
    hm = Counter(c.get("hisco_major") for c in careers if c.get("hisco_major"))
    th = sum(hm.values())
    for code in sorted(hm, key=str):
        rows.append({"metric": f"HISCO major: {code}",
                      "value": f"{hm[code]} ({_pct(hm[code], th)})"})
    rows.append({"metric": "HISCO coverage", "value": _pct(th, len(careers))})

    # HISCO minor top-20  +  full distribution CSV
    hmin = Counter(c.get("hisco_code") for c in careers if c.get("hisco_code"))
    save_csv(os.path.join(out_dir, "slm_metrics", "hisco_distribution.csv"),
             [{"hisco_code": c, "count": n} for c, n in hmin.most_common()])
    for code, cnt in hmin.most_common(20):
        rows.append({"metric": f"HISCO minor top-20: {code}", "value": cnt})

    # ISIC
    ic = Counter(o.get("isic_section") for o in orgs if o.get("isic_section"))
    ti = sum(ic.values())
    for sec in sorted(ic):
        rows.append({"metric": f"ISIC section: {sec}",
                      "value": f"{ic[sec]} ({_pct(ic[sec], ti)})"})
    rows.append({"metric": "ISIC coverage", "value": _pct(ti, len(orgs))})
    save_csv(os.path.join(out_dir, "slm_metrics", "isic_distribution.csv"),
             [{"isic_section": s, "count": c} for s, c in sorted(ic.items())])

    # Temporal
    sy = [c["start_year"] for c in careers if c.get("start_year")]
    if sy:
        rows += [
            {"metric": "Career start_year (min)",  "value": min(sy)},
            {"metric": "Career start_year (max)",  "value": max(sy)},
            {"metric": "Career start_year (mean)", "value": round(statistics.mean(sy), 1)},
        ]

    # Top-20 job titles
    jt = Counter(c.get("job_title") for c in careers if c.get("job_title"))
    for title, cnt in jt.most_common(20):
        rows.append({"metric": f"Top job title: {title}", "value": cnt})

    # Top-20 orgs by career count
    oc = Counter(c.get("organization_id") for c in careers if c.get("organization_id"))
    om = {o["organization_id"]: o.get("name", "") for o in orgs}
    for oid, cnt in oc.most_common(20):
        rows.append({"metric": f"Top org: {om.get(oid, oid)}", "value": cnt})

    save_csv(os.path.join(out_dir, "tables", "table3_careers.csv"), rows)
    return rows


def compute_family(data_dir, out_dir):
    """Table 4 — family structure."""
    print("\n=== Table 4: Family Structure ===")
    family  = load_jsonl(_data_path(data_dir, "person_family_members.jsonl"))
    persons = load_jsonl(_data_path(data_dir, "person_core.jsonl"))
    n = len(persons)
    fp = Counter(f["person_id"] for f in family)
    sizes = list(fp.values()) + [0] * max(0, n - len(fp))
    rows = [
        {"metric": "Family size/person (mean)",   "value": round(statistics.mean(sizes), 2) if sizes else 0},
        {"metric": "Family size/person (median)",  "value": round(statistics.median(sizes), 1) if sizes else 0},
        {"metric": "Family size/person (max)",     "value": max(sizes) if sizes else 0},
        {"metric": "Persons with family data",     "value": f"{len(fp)} ({_pct(len(fp), n)})"},
    ]
    rc = Counter(f.get("relation") for f in family)
    for rel, cnt in rc.most_common():
        rows.append({"metric": f"Relation: {rel}", "value": cnt})
    wby = sum(1 for f in family if f.get("birth_year"))
    rows.append({"metric": "Family with birth_year", "value": f"{wby} ({_pct(wby, len(family))})"})
    fg = Counter(f.get("gender") for f in family)
    for g in sorted(fg, key=str):
        rows.append({"metric": f"Family gender: {g}",
                      "value": f"{fg[g]} ({_pct(fg[g], len(family))})"})

    # Family member education & career totals
    fm_edu_path = _data_path(data_dir, "person_family_education.jsonl")
    fm_car_path = _data_path(data_dir, "person_family_career.jsonl")
    fm_edu = load_jsonl(fm_edu_path) if os.path.exists(fm_edu_path) else []
    fm_car = load_jsonl(fm_car_path) if os.path.exists(fm_car_path) else []
    nf = len(family)
    fm_with_edu = len(set(e["relation_id"] for e in fm_edu))
    fm_with_career = len(set(c["relation_id"] for c in fm_car))
    fm_with_place = sum(1 for f in family if f.get("place"))
    fm_with_name = sum(1 for f in family if f.get("name"))
    rows += [
        {"metric": "Family education records",         "value": len(fm_edu)},
        {"metric": "Family career records",            "value": len(fm_car)},
        {"metric": "Family members with name",         "value": f"{fm_with_name} ({_pct(fm_with_name, nf)})"},
        {"metric": "Family members with education",    "value": f"{fm_with_edu} ({_pct(fm_with_edu, nf)})"},
        {"metric": "Family members with career",       "value": f"{fm_with_career} ({_pct(fm_with_career, nf)})"},
        {"metric": "Family members with place",        "value": f"{fm_with_place} ({_pct(fm_with_place, nf)})"},
    ]

    save_csv(os.path.join(out_dir, "tables", "table4_family.csv"), rows)
    return rows


def compute_geography(data_dir, out_dir):
    """Table 5 — geographic coverage."""
    print("\n=== Table 5: Geographic Coverage ===")
    persons = load_jsonl(_data_path(data_dir, "person_core.jsonl"))
    locs    = load_jsonl(_data_path(data_dir, "locations.jsonl"))
    n, nl = len(persons), len(locs)
    wl = sum(1 for p in persons if p.get("location_id"))
    wo = sum(1 for p in persons if p.get("origin_location_id"))
    rows = [
        {"metric": "Person geocoding rate (location_id)", "value": f"{wl} ({_pct(wl, n)})"},
        {"metric": "Person origin geocoding rate",        "value": f"{wo} ({_pct(wo, n)})"},
    ]
    pc = Counter(l.get("province") for l in locs if l.get("province"))
    for prov, cnt in pc.most_common(15):
        rows.append({"metric": f"Province: {prov}", "value": cnt})
    for lvl in ("admin1", "admin2", "admin3"):
        w = sum(1 for l in locs if l.get(lvl))
        rows.append({"metric": f"Locations with {lvl}", "value": f"{w} ({_pct(w, nl)})"})
    wne = sum(1 for l in locs if l.get("name_en"))
    rows.append({"metric": "Location name_en coverage", "value": f"{wne} ({_pct(wne, nl)})"})
    save_csv(os.path.join(out_dir, "tables", "table5_geography.csv"), rows)
    return rows


def compute_org_network(data_dir, out_dir):
    """Table 6 — organisation network."""
    print("\n=== Table 6: Organisation Network ===")
    orgs = load_jsonl(_data_path(data_dir, "organizations.jsonl"))
    n = len(orgs)
    we  = sum(1 for o in orgs if o.get("name_en"))
    wp  = sum(1 for o in orgs if o.get("parent_organization_id"))
    wlc = sum(1 for o in orgs if o.get("location_id"))
    rows = [
        {"metric": "Total organisations",           "value": n},
        {"metric": "With English translation",       "value": f"{we} ({_pct(we, n)})"},
        {"metric": "With parent org (hierarchy)",    "value": f"{wp} ({_pct(wp, n)})"},
        {"metric": "With location",                  "value": f"{wlc} ({_pct(wlc, n)})"},
        {"metric": "Hierarchy links",                "value": wp},
    ]
    em_path = _data_path(data_dir, "entity_mappings.jsonl")
    if os.path.exists(em_path):
        em = load_jsonl(em_path)
        rows.append({"metric": "Cross-volume org merges",
                      "value": sum(1 for m in em if m.get("entity_type") == "organization")})
        rows.append({"metric": "Cross-volume person merges",
                      "value": sum(1 for m in em if m.get("entity_type") == "person")})
    save_csv(os.path.join(out_dir, "tables", "table6_org_network.csv"), rows)
    return rows


def compute_data_quality(data_dir, out_dir):
    """Table 7 — null rates, editorial changes, romanization."""
    print("\n=== Table 7: Data Quality ===")
    persons = load_jsonl(_data_path(data_dir, "person_core.jsonl"))
    n = len(persons)
    rows = []
    for fld in ("name", "name_family", "name_given", "name_family_latin",
                "name_given_latin", "domain", "birthyear", "phone", "tax_amount",
                "place", "origin_place", "rank", "gender", "location_id",
                "origin_location_id"):
        nn = sum(1 for p in persons if p.get(fld) not in (None, ""))
        rows.append({"metric": f"person_core.{fld} (non-null)",
                      "value": f"{nn} ({_pct(nn, n)})"})
    wl = sum(1 for p in persons if p.get("name_family_latin"))
    rows.append({"metric": "Name romanization coverage", "value": _pct(wl, n)})
    ec_path = _data_path(data_dir, "editorial_changes.jsonl")
    if os.path.exists(ec_path):
        ec = load_jsonl(ec_path)
        rows.append({"metric": "Editorial changes (total)", "value": len(ec)})
        for ct, cnt in Counter(c.get("change_type") for c in ec).most_common():
            rows.append({"metric": f"Editorial: {ct}", "value": cnt})
    save_csv(os.path.join(out_dir, "tables", "table7_data_quality.csv"), rows)
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# SECTION B — SLM Pipeline Metrics
# ═══════════════════════════════════════════════════════════════════════════

_ERA_PREFIXES = {
    "明治": "Meiji", "明": "Meiji",
    "大正": "Taisho", "大": "Taisho",
    "昭和": "Showa",  "昭": "Showa",
    "平成": "Heisei", "令和": "Reiwa",
    "光緒": "Guangxu", "宣統": "Xuantong",
    "民國": "Minguo", "民国": "Minguo",
    "建武": "Keonmu", "開國": "Gaeguk",
    "光武": "Gwangmu", "隆熙": "Yunghui",
    "大同": "Datong", "康德": "Kangde",
}


def compute_era_date_stats(data_dir, bio_dir, out_dir):
    """Era-date override rates from biographies_extracted_*.jsonl."""
    print("\n=== SLM Metric: Era Date Conversion ===")
    if not bio_dir:
        print("  Skipped (no --bio-dir provided)")
        return []
    import glob as _glob
    bio_files = _glob.glob(os.path.join(bio_dir, "biographies_extracted_*.jsonl"))
    if not bio_files:
        print(f"  Skipped (no files in {bio_dir})")
        return []

    total = with_raw = 0
    era_counts = Counter()
    for bf in bio_files:
        for rec in load_jsonl(bf):
            total += 1
            raw = (rec.get("extraction") or {}).get("birth_year_raw")
            if raw:
                with_raw += 1
                matched = "unknown"
                for pfx in sorted(_ERA_PREFIXES, key=len, reverse=True):
                    if pfx in str(raw):
                        matched = _ERA_PREFIXES[pfx]
                        break
                era_counts[matched] += 1

    rows = [
        {"metric": "Total biographies",               "value": total},
        {"metric": "With era date (birth_year_raw)",   "value": f"{with_raw} ({_pct(with_raw, total)})"},
    ]
    for era, cnt in era_counts.most_common():
        rows.append({"metric": f"Era: {era}", "value": cnt})
    save_csv(os.path.join(out_dir, "slm_metrics", "era_date_stats.csv"), rows)
    return rows


def compute_org_matching_stats(data_dir, out_dir):
    """Fuzzy matching + LLM verification stats."""
    print("\n=== SLM Metric: Org Matching Pipeline ===")
    fuzzy_path = _data_path(data_dir, "org_fuzzy_checkpoint.jsonl")
    llm_path   = _data_path(data_dir, "org_llm_checkpoint.jsonl")
    if not (os.path.exists(fuzzy_path) and os.path.exists(llm_path)):
        print("  Skipped (checkpoint files not found)")
        return []

    fuzzy = load_jsonl(fuzzy_path)
    llm   = load_jsonl(llm_path)
    conf  = [r for r in llm if r.get("llm_confirmed")]
    rej   = [r for r in llm if not r.get("llm_confirmed")]

    rows = [
        {"metric": "Fuzzy candidate pairs",  "value": len(fuzzy)},
        {"metric": "LLM-verified pairs",     "value": len(llm)},
        {"metric": "LLM confirmed",          "value": len(conf)},
        {"metric": "LLM rejected",           "value": len(rej)},
        {"metric": "LLM confirmation rate",  "value": _pct(len(conf), len(llm))},
    ]

    # Similarity breakdown
    llm_lk = {(r.get("name_a"), r.get("name_b")): r.get("llm_confirmed") for r in llm}
    sc, sr = [], []
    sim_dist = []
    for rec in fuzzy:
        sim = rec.get("similarity")
        if sim is None:
            continue
        key = (rec.get("name_a"), rec.get("name_b"))
        confirmed = llm_lk.get(key)
        (sc if confirmed else sr).append(sim)
        sim_dist.append({"name_a": rec.get("name_a", ""),
                         "name_b": rec.get("name_b", ""),
                         "similarity": sim, "llm_confirmed": confirmed})
    if sc:
        rows.append({"metric": "Mean similarity (confirmed)",
                      "value": round(statistics.mean(sc), 4)})
    if sr:
        rows.append({"metric": "Mean similarity (rejected)",
                      "value": round(statistics.mean(sr), 4)})

    save_csv(os.path.join(out_dir, "slm_metrics", "org_matching_stats.csv"), rows)
    save_csv(os.path.join(out_dir, "slm_metrics", "similarity_distribution.csv"), sim_dist)
    return rows


def compute_org_hierarchy_stats(data_dir, out_dir):
    """Org hierarchy detection stats."""
    print("\n=== SLM Metric: Org Hierarchy ===")
    hp = _data_path(data_dir, "org_hierarchy_checkpoint.jsonl")
    if not os.path.exists(hp):
        print("  Skipped (checkpoint not found)")
        return []
    hier = load_jsonl(hp)
    conf = [r for r in hier if r.get("llm_confirmed")]
    rows = [
        {"metric": "Hierarchy candidates",  "value": len(hier)},
        {"metric": "LLM confirmed",         "value": len(conf)},
        {"metric": "LLM rejected",          "value": len(hier) - len(conf)},
        {"metric": "Confirmation rate",      "value": _pct(len(conf), len(hier))},
    ]
    # Depth distribution
    op = _data_path(data_dir, "organizations.jsonl")
    if os.path.exists(op):
        orgs = load_jsonl(op)
        pm = {o["organization_id"]: o.get("parent_organization_id") for o in orgs}
        dc = Counter()
        for oid in pm:
            depth, cur, seen = 0, oid, set()
            while pm.get(cur) and cur not in seen:
                seen.add(cur); cur = pm[cur]; depth += 1
            dc[depth] += 1
        for d in sorted(dc):
            rows.append({"metric": f"Hierarchy depth {d}", "value": dc[d]})
    save_csv(os.path.join(out_dir, "slm_metrics", "org_hierarchy_stats.csv"), rows)
    return rows


def compute_disambiguation_stats(data_dir, out_dir):
    """Cross-volume disambiguation (meaningful once multi-volume)."""
    print("\n=== SLM Metric: Disambiguation ===")
    ep = _data_path(data_dir, "entity_mappings.jsonl")
    if not os.path.exists(ep):
        print("  Skipped (entity_mappings.jsonl not found)")
        return []
    em = load_jsonl(ep)
    persons = load_jsonl(_data_path(data_dir, "person_core.jsonl"))
    vc = Counter(v for p in persons for v in _iter_volumes(p))
    rows = [
        {"metric": "Person merges",
         "value": sum(1 for m in em if m.get("entity_type") == "person")},
        {"metric": "Organisation merges",
         "value": sum(1 for m in em if m.get("entity_type") == "organization")},
    ]
    for v in sorted(vc, key=str):
        rows.append({"metric": f"Persons in volume {v}", "value": vc[v]})
    save_csv(os.path.join(out_dir, "slm_metrics", "disambiguation_stats.csv"), rows)
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# SECTION C — Gold Standard Sample Export
# ═══════════════════════════════════════════════════════════════════════════

def _stratified_sample(items, key_fn, n, rng):
    """Proportionally-stratified random sample of *n* items."""
    by_stratum = defaultdict(list)
    for it in items:
        by_stratum[key_fn(it)].append(it)
    total = len(items)
    sample = []
    for k in sorted(by_stratum, key=str):
        grp = by_stratum[k]
        take = max(1, round(n * len(grp) / total))
        take = min(take, len(grp))
        sample.extend(rng.sample(grp, take))
    if len(sample) > n:
        sample = rng.sample(sample, n)
    rng.shuffle(sample)
    return sample


def _tertile_sample(items, sort_key, n, rng):
    """Sample *n* items stratified into tertiles by *sort_key*."""
    items = sorted(items, key=sort_key)
    t = len(items) // 3
    strata = [items[:t], items[t:2*t], items[2*t:]]
    per = n // 3
    rem = n - per * 3
    sample = []
    for i, s in enumerate(strata):
        take = min(per + (1 if i < rem else 0), len(s))
        sample.extend(rng.sample(s, take))
    rng.shuffle(sample)
    return sample


def _volume_balanced_sample(items, item_to_volume, n, sub_key_fn, rng):
    """Two-level stratified sample: split *n* equally across volumes, then
    apply proportional sub-stratification (HISCO major / ISIC section) within
    each volume's slice. If a volume has fewer items than its quota, the
    deficit is topped up from any remaining unsampled items so the final
    sample size still equals *n* whenever possible."""
    by_vol = defaultdict(list)
    for it in items:
        by_vol[item_to_volume(it)].append(it)
    if not by_vol:
        return []
    sorted_vols = sorted(by_vol, key=str)
    n_vols = len(sorted_vols)
    base = n // n_vols
    extra = n - base * n_vols
    quotas = {v: base + (1 if i < extra else 0)
              for i, v in enumerate(sorted_vols)}

    sample = []
    sampled_ids = set()
    for v in sorted_vols:
        items_v = by_vol[v]
        quota = min(quotas[v], len(items_v))
        if quota <= 0:
            continue
        sub = _stratified_sample(items_v, sub_key_fn, quota, rng)
        sample.extend(sub)
        sampled_ids.update(id(s) for s in sub)

    if len(sample) < n:
        remaining = [it for it in items if id(it) not in sampled_ids]
        if remaining:
            need = min(n - len(sample), len(remaining))
            sample.extend(rng.sample(remaining, need))
    rng.shuffle(sample)
    return sample


def _build_pid_to_volume(persons):
    """person_id -> primary (canonical/earliest) volume string."""
    out = {}
    for p in persons:
        v = p.get("volume")
        if isinstance(v, list):
            out[p["person_id"]] = v[0] if v else "?"
        else:
            out[p["person_id"]] = v or "?"
    return out


def _build_oid_to_volume(careers, pid_vol):
    """organization_id -> most-common volume among linked careers."""
    counts = defaultdict(Counter)
    for c in careers:
        oid = c.get("organization_id")
        if not oid:
            continue
        v = pid_vol.get(c.get("person_id"))
        if v:
            counts[oid][v] += 1
    return {oid: vc.most_common(1)[0][0] for oid, vc in counts.items()}


def export_extraction_gold_sample(data_dir, bio_dir, out_dir, ocr_dir=None,
                                   n=50, seed=42):
    """Gold 2 — extraction biographies. Needs --bio-dir.

    Body for each sampled row is resolved at export time via
    _resolve_body_for_row (header-matched, single-bio). The `slm_json`
    column is left EMPTY here — populate it with a fresh Qwen run via
    `--run-slm-alt --alt-model qwen3.5:9b --alt-column slm_json`. We
    intentionally do not copy `extraction` from the bio JSONL because that
    content was produced by step 5's stitched body, which differs from
    the corrected single-bio body used by every other comparison column.
    """
    print("\n=== Export: Extraction Gold Sample ===")
    if not bio_dir:
        print("  Skipped (need --bio-dir)")
        return

    import glob as _g
    bio_files = _g.glob(os.path.join(bio_dir, "biographies_extracted_*.jsonl"))
    if not bio_files:
        print(f"  No biographies_extracted_*.jsonl in {bio_dir}")
        return

    all_bios = []
    for bf in bio_files:
        for rec in load_jsonl(bf):
            all_bios.append({
                "entry_index": rec.get("entry_index"),
                "source_page": rec.get("source_page", ""),
                "source_image": rec.get("source_image", ""),
                "header": rec.get("header_ocr", ""),
                # body + slm_json filled below
                "body": "",
                "body_len": 0,
            })

    # Resolve body via header-matched single-bio lookup (same path used by
    # _resolve_body_for_row). Builds the bio image index + OCR root list once.
    bio_idx = _build_bio_image_index(Path(__file__).parent)
    next_bio_idx = _build_next_bio_index(Path(__file__).parent)
    ocr_roots = _detect_ocr_roots(Path(__file__).parent, primary=ocr_dir)
    if ocr_roots:
        print(f"  resolving body from OCR roots:")
        for r in ocr_roots:
            print(f"    {r}")
        for bio in all_bios:
            row_like = {
                "source_page": bio["source_page"],
                "entry_index": bio["entry_index"],
                "header": bio["header"],
                "body": "",
            }
            body = _resolve_body_for_row(row_like, ocr_roots, bio_idx, next_bio_idx)
            if body:
                bio["body"] = body
                bio["body_len"] = len(body)
    else:
        print("  no OCR roots found; bodies left empty")

    if not all_bios:
        print("  No biographies found")
        return
    n = min(n, len(all_bios))
    rng = random.Random(seed)
    sample = _tertile_sample(all_bios, lambda r: r["body_len"], n, rng)

    save_csv(os.path.join(out_dir, "gold_samples", "extraction_gold_sample.csv"),
             [{"entry_index": r["entry_index"], "source_page": r["source_page"],
               "header": r["header"], "body": r["body"],
               "slm_json": "", "cloud_json": "", "gold_json": ""}
              for r in sample])


def export_hisco_gold_sample(data_dir, out_dir, n=50, seed=42):
    """Gold 3 — HISCO, stratified by volume × HISCO major group."""
    print("\n=== Export: HISCO Gold Sample ===")
    careers = load_jsonl(_data_path(data_dir, "person_career.jsonl"))
    persons = load_jsonl(_data_path(data_dir, "person_core.jsonl"))
    eligible = [c for c in careers if c.get("job_title_en") and c.get("hisco_code")]
    if not eligible:
        print("  No eligible career records"); return
    pid_vol = _build_pid_to_volume(persons)
    rng = random.Random(seed)
    sample = _volume_balanced_sample(
        eligible,
        item_to_volume=lambda c: pid_vol.get(c.get("person_id"), "?"),
        n=n,
        sub_key_fn=lambda c: c.get("hisco_major", "?"),
        rng=rng,
    )
    save_csv(os.path.join(out_dir, "gold_samples", "hisco_gold_sample.csv"),
             [{"source_volume": pid_vol.get(c.get("person_id"), "?"),
               "person_id": c["person_id"], "job_title": c.get("job_title", ""),
               "job_title_en": c.get("job_title_en", ""),
               "organization_id": c.get("organization_id", ""),
               "qwen_hisco": c.get("hisco_code", ""),
               "qwen_hisco_major": c.get("hisco_major", ""),
               "gold_hisco": ""} for c in sample])


def export_isic_gold_sample(data_dir, out_dir, n=50, seed=42):
    """Gold 4 — ISIC, stratified by volume × ISIC section."""
    print("\n=== Export: ISIC Gold Sample ===")
    orgs    = load_jsonl(_data_path(data_dir, "organizations.jsonl"))
    careers = load_jsonl(_data_path(data_dir, "person_career.jsonl"))
    persons = load_jsonl(_data_path(data_dir, "person_core.jsonl"))
    eligible = [o for o in orgs if o.get("isic_section") and o.get("name")]
    if not eligible:
        print("  No eligible organisations"); return
    pid_vol = _build_pid_to_volume(persons)
    oid_vol = _build_oid_to_volume(careers, pid_vol)
    rng = random.Random(seed)
    sample = _volume_balanced_sample(
        eligible,
        item_to_volume=lambda o: oid_vol.get(o.get("organization_id"), "?"),
        n=n,
        sub_key_fn=lambda o: o["isic_section"],
        rng=rng,
    )
    save_csv(os.path.join(out_dir, "gold_samples", "isic_gold_sample.csv"),
             [{"source_volume": oid_vol.get(o.get("organization_id"), "?"),
               "organization_id": o["organization_id"],
               "org_name": o.get("name", ""), "name_en": o.get("name_en", ""),
               "qwen_isic": o.get("isic_section", ""),
               "qwen_isic_label": o.get("isic_label", ""),
               "gold_isic": ""} for o in sample])


def export_org_match_gold_sample(data_dir, out_dir, n=50, seed=42):
    """Gold 5 — org match verification: 50 confirmed + 50 rejected."""
    print("\n=== Export: Org Match Gold Sample ===")
    llm_path = _data_path(data_dir, "org_llm_checkpoint.jsonl")
    fuzzy_path = _data_path(data_dir, "org_fuzzy_checkpoint.jsonl")
    if not os.path.exists(llm_path):
        print("  Skipped (checkpoint not found)"); return

    llm = load_jsonl(llm_path)
    sim_map = {}
    if os.path.exists(fuzzy_path):
        for rec in load_jsonl(fuzzy_path):
            sim_map[(rec.get("name_a"), rec.get("name_b"))] = rec.get("similarity", 0)

    conf = [r for r in llm if r.get("llm_confirmed")]
    rej  = [r for r in llm if not r.get("llm_confirmed")]
    rng  = random.Random(seed)
    half = n // 2
    sc = rng.sample(conf, min(half, len(conf)))
    sr = rng.sample(rej,  min(n - len(sc), len(rej)))
    sample = sc + sr
    rng.shuffle(sample)

    save_csv(os.path.join(out_dir, "gold_samples", "org_match_gold_sample.csv"),
             [{"name_a": r.get("name_a", ""), "name_b": r.get("name_b", ""),
               "similarity": sim_map.get((r.get("name_a"), r.get("name_b")), ""),
               "llm_confirmed": r.get("llm_confirmed"), "gold_same": ""}
              for r in sample])


def export_org_hierarchy_gold_sample(data_dir, out_dir, n=50, seed=42):
    """Gold 6 — org hierarchy: 50 confirmed + 50 rejected."""
    print("\n=== Export: Org Hierarchy Gold Sample ===")
    hp = _data_path(data_dir, "org_hierarchy_checkpoint.jsonl")
    if not os.path.exists(hp):
        print("  Skipped (checkpoint not found)"); return
    hier = load_jsonl(hp)
    conf = [r for r in hier if r.get("llm_confirmed")]
    rej  = [r for r in hier if not r.get("llm_confirmed")]
    rng  = random.Random(seed)
    half = n // 2
    sc = rng.sample(conf, min(half, len(conf)))
    sr = rng.sample(rej,  min(n - len(sc), len(rej)))
    sample = sc + sr
    rng.shuffle(sample)
    save_csv(os.path.join(out_dir, "gold_samples", "org_hierarchy_gold_sample.csv"),
             [{"parent_name": r.get("parent_name", ""),
               "child_name": r.get("child_name", ""),
               "llm_confirmed": r.get("llm_confirmed"),
               "gold_parent_child": ""} for r in sample])


def write_annotation_guide(out_dir):
    """Write ANNOTATION_GUIDE.txt explaining what to fill in each gold CSV."""
    save_text(os.path.join(out_dir, "gold_samples", "ANNOTATION_GUIDE.txt"), """\
GOLD STANDARD ANNOTATION GUIDE
===============================

Each CSV file below has pre-filled columns (the pipeline's output) and one
empty column that YOU fill in.  Open in Excel / Google Sheets / LibreOffice,
fill the empty column, save as CSV, then run:

    python compute_paper_stats.py --score-gold


NOTE: OCR evaluation is handled by a separate, more comprehensive script —
      eval_ocr.py.  It does the 4-way Ollama × Google × {original, resize}
      comparison with proper CER / precision / recall / F1 against a manual
      gold standard.  This script only handles biography extraction, HISCO,
      ISIC, organisation matching, and organisation hierarchy.


1. extraction_gold_sample.csv  (exported with --bio-dir)
   ---------------------------------------------------------
   Columns:  entry_index | source_page | header | body |
             slm_json | cloud_json | gold_json
             (and any other *_json columns from --run-slm-alt / --run-hf)

   - `body` is populated at export time via header-matched single-bio
     resolution from the OCR segment files (no orphan stitching).
   - `slm_json` starts EMPTY. Populate it with a fresh Qwen run on the
     same body (so it's directly comparable to cloud_json/ministral_json/
     llama_json):
         python compute_paper_stats.py --run-slm-alt --alt-model qwen3.5:9b
                                       --alt-column slm_json
   - Read the `header` and `body` to write your `gold_json` annotation.
   - Schema fields: name, birth_year, birth_year_raw, place, phone_number,
     origin_place, rank, religion, hobbies[], education[], career[],
     family_member[].
   - Career entries: {job_title, organization, start_year, start_year_raw,
     place_name, current}.
   - Tip: after running --run-slm-alt for slm_json, copy slm_json into
     gold_json, then fix only the errors.
   - `cloud_json` is filled by --run-cloud or you can paste Gemini results.

2. hisco_gold_sample.csv
   ----------------------
   Columns:  source_volume | person_id | job_title | job_title_en |
             organization_id | qwen_hisco | qwen_hisco_major | gold_hisco

   - Sample is stratified by volume (≈ n/6 per volume) × HISCO major group.
   - Fill `gold_hisco` with the correct 2-digit HISCO minor-group code
     (ISCO-68 taxonomy) for the job title.
   - Reference: https://historyofwork.iisg.nl/list_minor.php
   - Use the ENGLISH translation in `job_title_en` to decide the code.
   - Examples:  President → 20,  Teacher → 13,  Physician → 06

3. isic_gold_sample.csv
   ---------------------
   Columns:  source_volume | organization_id | org_name | name_en |
             qwen_isic | qwen_isic_label | gold_isic

   - Sample is stratified by volume (≈ n/6 per volume) × ISIC section.
   - Fill `gold_isic` with the correct ISIC Rev 4 section letter (A–U).
   - Reference: https://unstats.un.org/unsd/classifications/Econ/isic
   - Examples:  Bank → K,  Hospital → Q,  Railway → H,  University → P

4. org_match_gold_sample.csv
   --------------------------
   Columns:  name_a | name_b | similarity | llm_confirmed | gold_same

   - Do `name_a` and `name_b` refer to the SAME real-world organisation?
   - Fill `gold_same` with:  True  or  False
   - Ignore similarity score — judge by meaning, not string resemblance.

5. org_hierarchy_gold_sample.csv
   ------------------------------
   Columns:  parent_name | child_name | llm_confirmed | gold_parent_child

   - Is `child_name` a department/division/subsidiary of `parent_name`?
   - Fill `gold_parent_child` with:  True  or  False


WORKFLOW
--------
1. Run:  python compute_paper_stats.py --export-gold [--bio-dir ... --ocr-dir ...]
2. Fill the gold_* columns in each CSV.
3. Run:  python compute_paper_stats.py --score-gold
4. Optionally run --run-cloud first to fill cloud_json columns,
   then --score-gold will include the SLM-vs-Gemini comparison automatically.

CLOUD COMPARISON (Gemini extraction)
-------------------------------------
    python compute_paper_stats.py --run-cloud --cloud-provider google --cloud-model gemini-2.0-flash

Requires:
  - pip install google-generativeai
  - GOOGLE_API_KEY or GEMINI_API_KEY env var

Cloud extraction uses Gemini with the identical extraction prompt from
5_extract_biographies.py for an apples-to-apples comparison.
""")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION D — Gold Standard Scoring
# ═══════════════════════════════════════════════════════════════════════════

def _load_gold_csv(path):
    """Load a gold-standard CSV with two defences against editor mangling:
      1) Strip a UTF-8 BOM (Excel adds one when saving) by reading utf-8-sig.
      2) Detect the "Excel-reset-the-header" pattern (column names become
         Column1..ColumnN and the original header is pushed down to row 1
         of data) and recover the real header from that first data row."""
    if not os.path.exists(path):
        print(f"  Not found: {path}")
        return None
    with open(path, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return rows
    cols = list(rows[0].keys())
    if all(str(c).startswith("Column") for c in cols):
        first_vals = [str(v) for v in rows[0].values()]
        if any("gold_" in v or v in ("source_volume", "person_id",
                                     "organization_id", "name_a",
                                     "parent_name", "entry_index")
               for v in first_vals):
            print(f"  Note: detected Excel-style header reset in "
                  f"{os.path.basename(path)}; recovering header from row 2")
            new_header = first_vals
            data_rows = rows[1:]
            recovered = []
            for r in data_rows:
                vals = list(r.values())
                recovered.append({new_header[i]: vals[i]
                                  for i in range(min(len(new_header), len(vals)))})
            return recovered
    return rows


def _cohens_kappa(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    n = len(y_true)
    if n == 0:
        return 0.0
    cm = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        cm[(t, p)] += 1
    po = sum(cm[(l, l)] for l in labels) / n
    pe = sum((sum(cm[(l, l2)] for l2 in labels) / n) *
             (sum(cm[(l2, l)] for l2 in labels) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1.0 else 1.0


def _confusion_csv(y_true, y_pred, labels=None):
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred), key=str)
    cm = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        cm[(t, p)] += 1
    return [dict([("true\\pred", tl)] + [(str(pl), cm[(tl, pl)]) for pl in labels])
            for tl in labels]


def _mcnemar(correct_a, correct_b):
    """Returns (chi2, p_approx)."""
    b = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if not a and bb)
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    try:
        p = math.erfc(math.sqrt(chi2 / 2))
    except (ValueError, OverflowError):
        p = 0.0
    return chi2, p


# ── OCR scoring lives in eval_ocr.py — see that script for the 4-way
#    Ollama × Google × {original, resize} comparison with gold standard. ──


# ── D2: Extraction ──

def score_extraction(out_dir, use_bert=False):
    print("\n=== Score: Extraction ===")
    data = _load_gold_csv(os.path.join(out_dir, "gold_samples",
                                       "extraction_gold_sample.csv"))
    if not data:
        return
    ann = [r for r in data if r.get("gold_json", "").strip()]
    if not ann:
        print("  No annotated rows (gold_json empty)"); return

    # Free-text scalars get BERTScore; numeric scalars stay strict-only.
    SCALAR_TEXT  = ["name", "place", "origin_place", "rank", "religion"]
    SCALAR_EXACT = ["birth_year", "phone_number"]
    SCALAR = SCALAR_TEXT + SCALAR_EXACT
    # Sub-fields concatenated to form an item-string for BERTScore set matching.
    LIST_FIELDS = {
        "career":        ["job_title", "organization", "place_name"],
        "education":     ["institution", "major_of_study"],
        "family_member": ["name", "relation", "place"],
        "hobbies":       None,   # list of plain strings
    }
    LIST = list(LIST_FIELDS.keys())

    if use_bert:
        try:
            import bert_score  # noqa: F401  -- fast-fail if missing
        except ImportError:
            raise SystemExit(
                "--bertscore requested but 'bert_score' is not installed.\n"
                "Install with: pip install bert_score torch transformers fugashi unidic-lite"
            )

    def _item_to_str(item, sub_fields):
        if sub_fields is None:
            return str(item or "")
        if not isinstance(item, dict):
            return str(item or "")
        return " | ".join(str(item.get(f) or "") for f in sub_fields)

    def _coerce_to_dict(obj, target_name=None):
        """Some models wrap their output as [{...}] or {"data": {...}}.
        Try to pull a single dict out so scoring can proceed.

        When *target_name* is given (typically the row's `header`), we use it
        to pick the matching bio out of a multi-bio list — Gemini sometimes
        extracts every bio on the OCR page, not just the requested one."""
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list):
            dicts = [x for x in obj if isinstance(x, dict)]
            if not dicts:
                return None
            if target_name:
                tgt = "".join(str(target_name).split())  # strip whitespace
                # 1. Exact name match
                for d in dicts:
                    nm = "".join(str(d.get("name") or "").split())
                    if nm and nm == tgt:
                        return d
                # 2. Substring either way (handles slight OCR variations)
                for d in dicts:
                    nm = "".join(str(d.get("name") or "").split())
                    if nm and (nm in tgt or tgt in nm):
                        return d
                # 3. Highest-similarity fuzzy match (kyūjitai/shinjitai
                #    variants like 豐/豊 or OCR slips like 熹/熾). Threshold
                #    0.5 keeps us from picking a totally unrelated person
                #    when the target name simply isn't in the list at all.
                from difflib import SequenceMatcher
                best_ratio = 0.0
                best_dict = None
                for d in dicts:
                    nm = "".join(str(d.get("name") or "").split())
                    if not nm:
                        continue
                    ratio = SequenceMatcher(None, tgt, nm).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_dict = d
                if best_ratio >= 0.5 and best_dict is not None:
                    return best_dict
            return dicts[0]
        return None

    def _lenient_json_parse(s):
        """json.loads with fallback: strip ```json fences, then try to
        extract the last balanced {...} block. Saves rows where Llama-style
        models wrap the JSON in markdown or echo prompt prefixes."""
        if not isinstance(s, str):
            return s
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        import re
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # Last balanced top-level {...}
        depth = 0
        start = -1
        candidate = None
        for i, ch in enumerate(s):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    chunk = s[start:i + 1]
                    try:
                        candidate = json.loads(chunk)
                    except json.JSONDecodeError:
                        pass
        if candidate is not None:
            return candidate
        raise json.JSONDecodeError("Could not lenient-parse", s, 0)

    def _strict_score_pair(pred_str, gold_str, target_name=None):
        """Per-field scoring. Returns (strict_correct, total, partial_correct,
        list_item_match, pred_dict, gold_dict).
        - 'strict' = exact equality after kyūjitai normalisation.
        - 'partial' = strict OR substring match either way (free-text scalars).
        - list_item_match[field] = (pred_matched, gold_matched, n_pred, n_gold)
          for content-level partial matching on LIST fields, accumulated
          across the bio."""
        fc, ft, fp = defaultdict(int), defaultdict(int), defaultdict(int)
        lim = defaultdict(lambda: [0, 0, 0, 0])  # pred_match, gold_match, n_pred, n_gold
        try:
            pred_raw = _lenient_json_parse(pred_str) if isinstance(pred_str, str) else pred_str
            gold_raw = _lenient_json_parse(gold_str) if isinstance(gold_str, str) else gold_str
        except (json.JSONDecodeError, TypeError):
            return fc, ft, fp, lim, None, None
        pred = _coerce_to_dict(pred_raw, target_name=target_name)
        gold = _coerce_to_dict(gold_raw, target_name=target_name)
        if pred is None or gold is None:
            return fc, ft, fp, lim, None, None
        for f in SCALAR:
            gv, pv = gold.get(f), pred.get(f)
            if gv is not None or pv is not None:
                ft[f] += 1
                # Normalise both sides to kyūjitai before comparing so that
                # 国 vs 國 etc. don't count as mismatches.
                sg = _normalize_chars(str(gv) if gv is not None else "")
                sp = _normalize_chars(str(pv) if pv is not None else "")
                if sg == sp:
                    fc[f] += 1
                    fp[f] += 1
                elif f in PARTIAL_SCALAR_FIELDS and sg and sp and (
                        sg in sp or sp in sg):
                    fp[f] += 1
        for f in LIST:
            gl = gold.get(f) or []
            pl = pred.get(f) or []
            if not isinstance(gl, list):
                gl = []
            if not isinstance(pl, list):
                pl = []
            if gl or pl:
                ft[f"{f}_count"] += 1
                if len(gl) == len(pl):
                    fc[f"{f}_count"] += 1
                    fp[f"{f}_count"] += 1
                # Item-level partial match (kyūjitai-normalised substring).
                # An item "matches" if its concatenated sub-field string is
                # a substring of, or contains, any item on the other side.
                sub = LIST_FIELDS[f]
                ps = [_normalize_chars(_item_to_str(p, sub)) for p in pl]
                gs = [_normalize_chars(_item_to_str(g, sub)) for g in gl]
                ps = [s for s in ps if s.strip()]
                gs = [s for s in gs if s.strip()]
                p_match = sum(1 for x in ps if any(
                    x == y or x in y or y in x for y in gs))
                g_match = sum(1 for y in gs if any(
                    x == y or x in y or y in x for x in ps))
                lim[f][0] += p_match
                lim[f][1] += g_match
                lim[f][2] += len(ps)
                lim[f][3] += len(gs)
        return fc, ft, fp, lim, pred, gold

    # Source columns: "slm" (Qwen baseline) is always first. Any other column
    # ending in _json is auto-discovered so a later --run-slm-alt / --run-hf /
    # --run-cloud invocation appears in the report without code changes.
    sources = [("slm", "slm_json", ann)]
    SOURCE_LABELS = {"cloud_json": "cloud",
                     "ministral_json": "ministral",
                     "llama_json": "llama"}
    extra_cols = [k for k in ann[0].keys()
                  if k.endswith("_json") and k not in ("slm_json", "gold_json")]
    for col in extra_cols:
        label = SOURCE_LABELS.get(col, col[:-len("_json")])
        rows_for_col = [r for r in ann if r.get(col, "").strip()
                        and not r.get(col, "").lstrip().startswith('{"error"')]
        if rows_for_col:
            sources.append((label, col, rows_for_col))

    # ── Pass 1: strict + partial accuracy + (if use_bert) collect string pairs ──
    strict_totals = {src: (defaultdict(int), defaultdict(int), defaultdict(int),
                           defaultdict(lambda: [0, 0, 0, 0]))
                     for src, _, _ in sources}
    bert_pairs = []     # [{src,row,field,kind,p_idx,g_idx,cand,ref}, ...]
    bert_meta  = []     # [{src,row,field,n_p,n_g}] — captures rows even if one side empty
    for src_label, json_field, rows in sources:
        c_acc, t_acc, p_acc, lim_acc = strict_totals[src_label]
        for ri, r in enumerate(rows):
            pred_str = r.get(json_field, "{}")
            gold_str = r["gold_json"]
            c, t, p, lim, pred, gold = _strict_score_pair(
                pred_str, gold_str, target_name=r.get("header"))
            for k in t:
                c_acc[k] += c[k]; t_acc[k] += t[k]; p_acc[k] += p[k]
            for f, vals in lim.items():
                acc = lim_acc[f]
                acc[0] += vals[0]; acc[1] += vals[1]
                acc[2] += vals[2]; acc[3] += vals[3]
            if not (use_bert and pred is not None and gold is not None):
                continue
            for f in SCALAR_TEXT:
                gv, pv = gold.get(f), pred.get(f)
                if gv is None and pv is None:
                    continue
                bert_pairs.append({"src": src_label, "row": ri, "field": f,
                                   "kind": "scalar",
                                   "cand": str(pv or ""), "ref": str(gv or "")})
            for f, sub in LIST_FIELDS.items():
                gl = gold.get(f) or []
                pl = pred.get(f) or []
                if not (gl or pl):
                    continue
                ps = [_item_to_str(p, sub) for p in pl]
                gs = [_item_to_str(g, sub) for g in gl]
                bert_meta.append({"src": src_label, "row": ri, "field": f,
                                  "n_p": len(ps), "n_g": len(gs)})
                for i, p in enumerate(ps):
                    for j, g in enumerate(gs):
                        bert_pairs.append({"src": src_label, "row": ri, "field": f,
                                           "kind": "item", "p_idx": i, "g_idx": j,
                                           "cand": p, "ref": g})

    # ── BERTScore batch call (single model load) ──
    if use_bert and bert_pairs:
        scores = bert_score_batch([(t["cand"], t["ref"]) for t in bert_pairs])
        for t, s in zip(bert_pairs, scores):
            t["f1"] = s

    # ── Pass 2: aggregate BERTScore per (src, field) ──
    scalar_f1s = defaultdict(list)              # (src,field) -> [f1, ...]
    list_matrices = defaultdict(lambda: defaultdict(dict))
    # list_matrices[(src,field)][row][(i,j)] = f1
    list_dims = defaultdict(dict)               # (src,field)[row] = (n_p, n_g)
    if use_bert:
        for t in bert_pairs:
            if t["kind"] == "scalar":
                scalar_f1s[(t["src"], t["field"])].append(t["f1"])
            else:
                list_matrices[(t["src"], t["field"])][t["row"]][(t["p_idx"], t["g_idx"])] = t["f1"]
        for m in bert_meta:
            list_dims[(m["src"], m["field"])][m["row"]] = (m["n_p"], m["n_g"])

    def _bert_scalar(src, field):
        f1s = scalar_f1s.get((src, field))
        if not f1s:
            return None
        matches = [f for f in f1s if f >= BERT_F1_MATCH]
        cond_f1 = sum(matches) / len(matches) if matches else 0.0
        return {
            "bert_f1_mean":    round(sum(f1s) / len(f1s), 4),
            "bert_match_rate": _pct(len(matches), len(f1s)),
            # Mean F1 conditional on the prediction landing in the gold
            # neighbourhood (F1 ≥ threshold). Always ≥ threshold by
            # construction; useful as a "given a hit, how close?" signal.
            "bert_cond_f1":    round(cond_f1, 4) if matches else None,
            "bert_n":          len(f1s),
        }

    def _bert_list(src, field):
        rows = list_dims.get((src, field))
        if not rows:
            return None
        all_p_max, all_g_max = [], []
        mats = list_matrices[(src, field)]
        for row, (n_p, n_g) in rows.items():
            mat = mats.get(row, {})
            for i in range(n_p):
                scores = [mat.get((i, j), 0.0) for j in range(n_g)]
                all_p_max.append(max(scores) if scores else 0.0)
            for j in range(n_g):
                scores = [mat.get((i, j), 0.0) for i in range(n_p)]
                all_g_max.append(max(scores) if scores else 0.0)
        prec = sum(all_p_max) / len(all_p_max) if all_p_max else 0.0
        rec  = sum(all_g_max) / len(all_g_max) if all_g_max else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        # Threshold-based metrics, split by side. The combined version is
        # kept for back-compat in the CSV but the printed report uses the
        # split versions, which have cleaner interpretations:
        #   pred-side match rate = "of items I predicted, what % landed
        #                            in the gold neighbourhood"  (precision-style)
        #   gold-side match rate = "of gold items, what % were recovered
        #                            with high quality"          (recall-style)
        match_p = [s for s in all_p_max if s >= BERT_F1_MATCH]
        match_g = [s for s in all_g_max if s >= BERT_F1_MATCH]
        cond_p = sum(match_p) / len(match_p) if match_p else 0.0
        cond_r = sum(match_g) / len(match_g) if match_g else 0.0
        return {
            "bert_precision":       round(prec, 4),
            "bert_recall":          round(rec, 4),
            "bert_f1":              round(f1, 4),
            "bert_match_rate":      _pct(len(match_p) + len(match_g),
                                         len(all_p_max) + len(all_g_max)),
            "bert_match_rate_pred": _pct(len(match_p), len(all_p_max)),
            "bert_match_rate_gold": _pct(len(match_g), len(all_g_max)),
            "bert_cond_p":          round(cond_p, 4) if match_p else None,
            "bert_cond_r":          round(cond_r, 4) if match_g else None,
            "bert_n_pred":          len(all_p_max),
            "bert_n_gold":          len(all_g_max),
        }

    # ── Render text + CSV (strict numbers always; BERTScore columns when on) ──
    lines = [f"Extraction Gold Standard ({len(ann)} bios)\n{'='*60}\n"]
    if use_bert:
        lines.append(f"BERTScore: model={_BERT_MODEL_NAME}, match-threshold F1≥{BERT_F1_MATCH}\n")
    pf_rows = []
    for src_label, _, rows in sources:
        c_acc, t_acc, p_acc, lim_acc = strict_totals[src_label]
        n = len(rows)
        lines.append(f"\n{src_label.upper()} per-field accuracy ({n} bios):")
        for f in SCALAR:
            t = t_acc.get(f, 0); c = c_acc.get(f, 0); p = p_acc.get(f, 0)
            acc = _pct(c, t) if t else "N/A"
            row = {"field": f, "correct": c, "total": t,
                   "accuracy": acc, "source": src_label,
                   "correct_partial": p,
                   "accuracy_partial": _pct(p, t) if t else "N/A"}
            partial_str = ""
            if t and f in PARTIAL_SCALAR_FIELDS and p > c:
                partial_str = f"  partial: {_pct(p, t)} ({p}/{t})"
            extra = ""
            if use_bert and f in SCALAR_TEXT:
                bs = _bert_scalar(src_label, f)
                if bs:
                    row.update(bs)
                    cond = (f", cond_F1={bs['bert_cond_f1']}"
                            if bs.get("bert_cond_f1") is not None else "")
                    extra = (f"  [BERT F1={bs['bert_f1_mean']}, "
                             f"match≥{BERT_F1_MATCH}: {bs['bert_match_rate']}"
                             f"{cond}]")
            lines.append(f"  {f}: {acc} ({c}/{t}){partial_str}{extra}")
            pf_rows.append(row)
        for f in LIST:
            count_key = f"{f}_count"
            t = t_acc.get(count_key, 0); c = c_acc.get(count_key, 0)
            acc = _pct(c, t) if t else "N/A"
            pf_rows.append({"field": count_key, "correct": c, "total": t,
                            "accuracy": acc, "source": src_label})
            lines.append(f"  {count_key}: {acc} ({c}/{t})")
            # Item-level partial-match rates (substring, kyūjitai-normalised).
            #   pred-side = "of items I predicted, what % found a substring
            #                match in gold"  (precision-style)
            #   gold-side = "of gold items, what % were recovered by some
            #                pred"            (recall-style)
            lim = lim_acc.get(f)
            if lim and (lim[2] or lim[3]):
                pm, gm, np, ng = lim
                p_rate = _pct(pm, np) if np else "N/A"
                g_rate = _pct(gm, ng) if ng else "N/A"
                pf_rows.append({"field": f"{f}_items_partial", "source": src_label,
                                "accuracy": "N/A", "correct": "", "total": "",
                                "items_pred_match": pm, "items_pred_total": np,
                                "items_pred_rate": p_rate,
                                "items_gold_match": gm, "items_gold_total": ng,
                                "items_gold_rate": g_rate})
                lines.append(
                    f"        items partial: pred {p_rate} ({pm}/{np}), "
                    f"gold {g_rate} ({gm}/{ng})"
                )
            if use_bert:
                bs = _bert_list(src_label, f)
                if bs:
                    pf_rows.append({"field": f, "source": src_label,
                                    "accuracy": "N/A", "correct": "", "total": "",
                                    **bs})
                    cp = (f" cond_P={bs['bert_cond_p']}"
                          if bs.get("bert_cond_p") is not None else "")
                    cr = (f" cond_R={bs['bert_cond_r']}"
                          if bs.get("bert_cond_r") is not None else "")
                    lines.append(
                        f"  {f}: BERT P={bs['bert_precision']} R={bs['bert_recall']} "
                        f"F1={bs['bert_f1']} (pred={bs['bert_n_pred']}, "
                        f"gold={bs['bert_n_gold']})\n"
                        f"        match≥{BERT_F1_MATCH}: "
                        f"pred {bs['bert_match_rate_pred']}, "
                        f"gold {bs['bert_match_rate_gold']}"
                        f"{cp}{cr}"
                    )

    txt = "\n".join(lines)
    save_text(os.path.join(out_dir, "scores", "extraction_scores.txt"), txt)
    save_csv(os.path.join(out_dir, "scores", "extraction_per_field.csv"), pf_rows)
    print(txt)


# ── D3: HISCO ──

def _norm_hisco_minor(v):
    """Coerce a CSV cell into a canonical 2-digit HISCO minor-group string.
    Handles users typing '4' instead of '04' (Excel strips leading zeros on
    save), float coercion ('4.0'), and non-digit junk. Empty string passes
    through unchanged so 'unannotated' / 'no prediction' rows stay flaggable."""
    if v is None:
        return ""
    s = str(v).strip()
    if not s:
        return ""
    if "." in s:
        try:
            s = str(int(float(s)))
        except ValueError:
            pass
    return s.zfill(2)[:2] if s.isdigit() else s


def score_hisco(out_dir):
    print("\n=== Score: HISCO ===")
    data = _load_gold_csv(os.path.join(out_dir, "gold_samples",
                                       "hisco_gold_sample.csv"))
    if not data:
        return
    ann = [r for r in data if r.get("gold_hisco", "").strip()]
    if not ann:
        print("  No annotated rows"); return

    yg  = [_norm_hisco_minor(r.get("gold_hisco")) for r in ann]
    yq  = [_norm_hisco_minor(r.get("qwen_hisco")) for r in ann]
    ygm = [g[:1] for g in yg]
    yqm = [q[:1] for q in yq]

    mc = sum(g == q for g, q in zip(yg, yq))
    jc = sum(g == q for g, q in zip(ygm, yqm))
    lines = [
        f"HISCO Gold Standard ({len(ann)} records)", "=" * 60,
        f"\nQwen SLM:",
        f"  Minor (2-digit) accuracy: {_pct(mc, len(ann))} ({mc}/{len(ann)})",
        f"  Major (1-digit) accuracy: {_pct(jc, len(ann))} ({jc}/{len(ann)})",
        f"  Cohen's kappa (minor): {_cohens_kappa(yg, yq):.3f}",
        f"  Cohen's kappa (major): {_cohens_kappa(ygm, yqm):.3f}",
    ]

    # OccCANINE (if column filled)
    if any(r.get("occcanine_hisco", "").strip() for r in ann):
        yo  = [_norm_hisco_minor(r.get("occcanine_hisco")) for r in ann]
        wp  = sum(bool(o) for o in yo)
        oc  = sum(g == o for g, o in zip(yg, yo) if o)
        om  = [o[:1] if o else "" for o in yo]
        omc = sum(g == o for g, o in zip(ygm, om) if o)
        lines += [
            f"\nOccCANINE:",
            f"  Coverage: {wp}/{len(ann)} ({_pct(wp, len(ann))})",
            f"  Minor accuracy (of classified): {_pct(oc, wp)}",
            f"  Major accuracy (of classified): {_pct(omc, wp)}",
        ]
        paired = [(g, q, o) for g, q, o in zip(yg, yq, yo) if o]
        if paired:
            # Minor-level (full 2-digit match)
            qa_min = [g == q for g, q, o in paired]
            oa_min = [g == o for g, q, o in paired]
            chi2_min, p_min = _mcnemar(qa_min, oa_min)
            # Major-level (1-digit match — "good enough" agreement)
            qa_maj = [g[:1] == q[:1] for g, q, o in paired]
            oa_maj = [g[:1] == o[:1] for g, q, o in paired]
            chi2_maj, p_maj = _mcnemar(qa_maj, oa_maj)
            lines += [
                f"\nMcNemar (Qwen vs OccCANINE, n={len(paired)}):",
                f"  Minor (2-digit): chi2={chi2_min:.3f}  p={p_min:.4f}",
                f"  Major (1-digit): chi2={chi2_maj:.3f}  p={p_maj:.4f}",
            ]

    txt = "\n".join(lines)
    save_text(os.path.join(out_dir, "scores", "hisco_scores.txt"), txt)
    save_csv(os.path.join(out_dir, "scores", "hisco_confusion_major.csv"),
             _confusion_csv(ygm, yqm))
    save_csv(os.path.join(out_dir, "scores", "hisco_confusion_minor.csv"),
             _confusion_csv(yg, yq))
    print(txt)


# ── D4: ISIC ──

def score_isic(out_dir):
    print("\n=== Score: ISIC ===")
    data = _load_gold_csv(os.path.join(out_dir, "gold_samples",
                                       "isic_gold_sample.csv"))
    if not data:
        return
    ann = [r for r in data if r.get("gold_isic", "").strip()]
    if not ann:
        print("  No annotated rows"); return
    yg = [r["gold_isic"].strip() for r in ann]
    yq = [r.get("qwen_isic", "").strip() for r in ann]
    c  = sum(g == q for g, q in zip(yg, yq))
    txt = "\n".join([
        f"ISIC Gold Standard ({len(ann)} records)", "=" * 60,
        f"  Accuracy: {_pct(c, len(ann))} ({c}/{len(ann)})",
        f"  Cohen's kappa: {_cohens_kappa(yg, yq):.3f}",
    ])
    save_text(os.path.join(out_dir, "scores", "isic_scores.txt"), txt)
    save_csv(os.path.join(out_dir, "scores", "isic_confusion.csv"),
             _confusion_csv(yg, yq))
    print(txt)


# ── D5 + D6: binary scoring (org match / hierarchy) ──

def _score_binary(out_dir, name, csv_name, pred_col, gold_col):
    print(f"\n=== Score: {name} ===")
    data = _load_gold_csv(os.path.join(out_dir, "gold_samples", csv_name))
    if not data:
        return
    ann = [r for r in data if r.get(gold_col, "").strip()]
    if not ann:
        print("  No annotated rows"); return

    def _bool(v):
        return str(v).strip().lower() in ("true", "1", "yes", "y")

    tp = fp = fn = tn = 0
    for r in ann:
        pr, gl = _bool(r.get(pred_col, "")), _bool(r.get(gold_col, ""))
        if pr and gl:     tp += 1
        elif pr:          fp += 1
        elif gl:          fn += 1
        else:             tn += 1
    prec = _safe_div(tp, tp + fp)
    rec  = _safe_div(tp, tp + fn)
    f1   = _safe_div(2 * prec * rec, prec + rec)
    txt = "\n".join([
        f"{name} ({len(ann)} pairs)", "=" * 60,
        f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}",
        f"  Precision: {prec:.3f}",
        f"  Recall:    {rec:.3f}",
        f"  F1:        {f1:.3f}",
        f"  Accuracy:  {_safe_div(tp + tn, tp+fp+fn+tn):.3f}",
    ])
    slug = name.lower().replace(" ", "_")
    save_text(os.path.join(out_dir, "scores", f"{slug}_scores.txt"), txt)
    print(txt)


def score_org_match(out_dir):
    _score_binary(out_dir, "Org Match", "org_match_gold_sample.csv",
                  "llm_confirmed", "gold_same")

def score_org_hierarchy(out_dir):
    _score_binary(out_dir, "Org Hierarchy", "org_hierarchy_gold_sample.csv",
                  "llm_confirmed", "gold_parent_child")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION E — Cloud LLM Comparison (extraction only) + OccCANINE
# ═══════════════════════════════════════════════════════════════════════════

def _cloud_client(provider):
    """Return (client, provider_name) or (None, None) for biography
    extraction comparison. OCR comparison is in eval_ocr.py, not here."""
    if provider == "openai":
        try:
            import openai
        except ImportError:
            print("  pip install openai  is required"); return None, None
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            print("  Set OPENAI_API_KEY"); return None, None
        return openai.OpenAI(api_key=key), "openai"
    if provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            print("  pip install anthropic  is required"); return None, None
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            print("  Set ANTHROPIC_API_KEY"); return None, None
        return anthropic.Anthropic(api_key=key), "anthropic"
    if provider == "google":
        try:
            import google.generativeai as genai
        except ImportError:
            print("  pip install google-generativeai  is required"); return None, None
        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            print("  Set GOOGLE_API_KEY or GEMINI_API_KEY"); return None, None
        genai.configure(api_key=key)
        return {"genai": genai}, "google"
    print(f"  Unknown provider: {provider}")
    return None, None


def _load_create_prompt():
    """Try to import create_prompt from 5_extract_biographies.py."""
    script = Path(__file__).parent / "5_extract_biographies.py"
    if not script.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_ext_bio", str(script))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "create_prompt", None)
    except Exception as exc:
        print(f"  Warning: could not import create_prompt: {exc}")
        return None


def _cloud_extract(client, prov, model, header, body, create_prompt_fn):
    sys_msg = ("You are a robotic data parser. Output valid JSON only. "
               "Follow conversion rules strictly. Do not invent new fields.")
    if create_prompt_fn:
        usr_msg = create_prompt_fn(header, body)
    else:
        usr_msg = (f"Extract structured biographical data.\n"
                   f"Header: {header}\nBody: {body}\nReturn valid JSON.")
    if prov == "openai":
        r = client.chat.completions.create(
            model=model, temperature=0, max_tokens=3500,
            messages=[{"role": "system", "content": sys_msg},
                      {"role": "user", "content": usr_msg}])
        return _clean_llm_json_output(r.choices[0].message.content)
    if prov == "anthropic":
        r = client.messages.create(
            model=model, max_tokens=3500, temperature=0, system=sys_msg,
            messages=[{"role": "user", "content": usr_msg}])
        return _clean_llm_json_output(r.content[0].text)
    if prov == "google":
        genai = client.get("genai") if isinstance(client, dict) else None
        if not genai:
            return '{"error": "google-generativeai not configured"}'
        gm = genai.GenerativeModel(
            model_name=model,
            system_instruction=sys_msg,
            # 32768 fits Gemini 2.5 Pro's mandatory "thinking" tokens AND
            # the JSON output. With 8192 the model burned the whole budget
            # on thinking and emitted no text. If a row still hits
            # MAX_TOKENS, switch to gemini-2.5-flash (cheaper, less
            # aggressive thinking) via --cloud-model.
            generation_config={"temperature": 0, "max_output_tokens": 32768,
                               "response_mime_type": "application/json"})
        r = gm.generate_content(usr_msg)
        # Don't use r.text — it raises when finish_reason != STOP. Pull the
        # text from candidates[0].content.parts and report the finish_reason
        # explicitly so MAX_TOKENS / SAFETY / RECITATION are diagnosable.
        cands = getattr(r, "candidates", None) or []
        if not cands:
            raise RuntimeError(f"Gemini returned no candidates "
                               f"(prompt_feedback={getattr(r, 'prompt_feedback', None)})")
            # finish_reason: 1=STOP, 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION, 5=OTHER
        fr = getattr(cands[0], "finish_reason", None)
        parts = getattr(getattr(cands[0], "content", None), "parts", []) or []
        text = "".join(getattr(p, "text", "") for p in parts).strip()
        if not text:
            fr_name = {1: "STOP", 2: "MAX_TOKENS", 3: "SAFETY",
                       4: "RECITATION", 5: "OTHER"}.get(int(fr) if fr else 0,
                                                        f"finish_reason={fr}")
            raise RuntimeError(f"Gemini produced no text ({fr_name})")
        return _clean_llm_json_output(text)


def _clean_llm_json_output(raw):
    """Best-effort: turn an LLM's raw response into a clean, single-line JSON
    string. Used by _cloud_extract / _ollama_extract / run_hf_extraction so
    every CSV cell stores parseable JSON whenever possible — preventing the
    markdown-fence + flatten breakage we hit before.

    Stages:
      1. json.loads → re-serialize compact (already-clean output).
      2. Strip ```json ... ``` markdown fences → retry.
      3. Find the last balanced {...} block → retry.
      4. Give up: return the original string (caller stores it raw, scoring
         will still try _lenient_json_parse later)."""
    if not isinstance(raw, str):
        return raw
    s = raw.strip()
    if not s:
        return s
    # 1
    try:
        return json.dumps(json.loads(s), ensure_ascii=False,
                          separators=(", ", ": "))
    except (json.JSONDecodeError, TypeError):
        pass
    # 2
    import re
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    if m:
        try:
            return json.dumps(json.loads(m.group(1)), ensure_ascii=False,
                              separators=(", ", ": "))
        except json.JSONDecodeError:
            pass
    # 3 — last balanced {...}
    depth = 0
    start = -1
    last = None
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    last = json.loads(s[start:i + 1])
                except json.JSONDecodeError:
                    pass
    if last is not None:
        return json.dumps(last, ensure_ascii=False, separators=(", ", ": "))
    return s


def _is_dummy_extraction(s):
    """A previously-written value should be retried (not skipped) if it is
    empty / whitespace / "{}" / a {"error": ...}-wrapper. Anything else
    counts as a real prior result and is preserved."""
    if not s:
        return True
    t = s.strip()
    if not t or t == "{}":
        return True
    try:
        d = json.loads(t)
    except (json.JSONDecodeError, TypeError):
        return False
    if isinstance(d, dict) and (not d or set(d.keys()) == {"error"}):
        return True
    return False


def _build_bio_image_index(project_root):
    """(source_page, entry_index_int) → source_image, by scanning
    biographies_extracted_*.jsonl. Used to resolve the OCR segment-file
    path when the gold CSV's body is empty."""
    import glob
    out = {}
    for bf in sorted(glob.glob(str(Path(project_root) / "biographies_extracted_*.jsonl"))):
        with open(bf, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sp, ei, si = (rec.get("source_page"), rec.get("entry_index"),
                              rec.get("source_image"))
                if sp is not None and ei is not None and si:
                    out[(sp, ei)] = si
    return out


def _build_next_bio_index(project_root):
    """(source_page, entry_index_int) → (next_source_page, next_source_image)
    in step-5 stream order. Used by _resolve_body_for_row to walk forward
    and append leading orphan segments from the next bio's row-crop file —
    replicating 5_extract_biographies.stream_stitched_entries' behaviour
    of attaching orphan continuations to the bio that started just before."""
    import glob
    out = {}
    for bf in sorted(glob.glob(str(Path(project_root) / "biographies_extracted_*.jsonl"))):
        entries = []
        with open(bf, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sp = rec.get("source_page")
                ei = rec.get("entry_index")
                si = rec.get("source_image")
                if sp is None or ei is None or not si:
                    continue
                entries.append((sp, ei, si))
        entries.sort(key=lambda x: x[1])  # by entry_index (step 5 yield order)
        for i, (sp, ei, _) in enumerate(entries[:-1]):
            nxt = entries[i + 1]
            out[(sp, ei)] = (nxt[0], nxt[2])
    return out


EXTRA_OCR_SEARCH_PATHS = [
    # Optional extra search paths for per-volume OCR archives stored outside
    # the project tree. Read-only references; never written to. Populate with
    # absolute paths to local OCR result directories if needed.
]


def _detect_ocr_roots(project_root, primary=None):
    """Return a list of OCR-result directories to try, in priority order.
    The gold sample spans multiple volumes and each volume's OCR may live
    in its own dir (ocr_results/, ocr_results_1927/, ocr_results_1943e/, …).
    Also walks EXTRA_OCR_SEARCH_PATHS so per-volume archives outside the
    project root (e.g. ~/taishu/ocr_results_1935/) get picked up."""
    roots = []

    def add(path):
        s = str(path)
        if path.is_dir() and s not in roots:
            roots.append(s)

    base = Path(project_root)
    if primary:
        p = Path(primary)
        if not p.is_absolute():
            p = base / primary
        add(p)

    add(base / "ocr_results")
    for sub in sorted(base.glob("ocr_results_*")):
        add(sub)

    for extra in EXTRA_OCR_SEARCH_PATHS:
        ep = Path(extra)
        if not ep.is_dir():
            continue
        add(ep / "ocr_results")
        for sub in sorted(ep.glob("ocr_results_*")):
            add(sub)
    return roots


def _resolve_body_for_row(row, ocr_roots, bio_image_index, next_bio_index=None):
    """Return body text for a gold row, replicating step 5's stitcher
    behaviour. Prefers row['body'] if non-empty; otherwise:

      1. Find the row's `type='standard'` segment in its own row-crop file,
         matched by `header_ocr` against row['header'] (with fuzzy fallback).
      2. If the row's standard is the LAST standard in its own file, walk
         forward through subsequent row-crop files (in step-5 order, looked
         up via *next_bio_index*) and append each file's leading orphan
         segments. Stop at the first 'standard' encountered. This recovers
         continuation content that flowed across row crops — which is what
         step 5 actually fed to Qwen for long bios like 永田良介.
      3. If the row's standard is followed by other standards in the same
         file, no orphans are appended (the next standard ends accumulation).

    *ocr_roots* may be a string or a list. *next_bio_index* is optional;
    omitting it disables orphan stitching (keeps single-bio behaviour).
    """
    body = (row.get("body") or "").strip()
    if body:
        return body
    sp = (row.get("source_page") or "").strip()
    if not sp or not ocr_roots:
        return ""
    try:
        ei = int(str(row.get("entry_index", "")).strip())
    except (ValueError, TypeError):
        return ""
    src_image = bio_image_index.get((sp, ei))
    if not src_image:
        return ""
    stem = src_image.rsplit(".", 1)[0]
    candidates = [ocr_roots] if isinstance(ocr_roots, str) else list(ocr_roots)
    target = "".join(str(row.get("header") or "").split())

    def _norm(s):
        return "".join(str(s or "").split())

    def _read_segs(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def _find_seg_path(page, image_stem):
        """Return the first existing seg-file path across candidate roots."""
        fname = f"{image_stem}_segmented_output.json"
        for root in candidates:
            p = Path(root) / page / fname
            if p.exists():
                return p
        return None

    seg_path = _find_seg_path(sp, stem)
    if not seg_path:
        return ""
    segs = _read_segs(seg_path)
    if segs is None:
        return ""

    std_segs = [s for s in segs
                if s.get("type") == "standard"
                and (s.get("body_text") or "").strip()]
    if not std_segs:
        for s in segs:
            t = (s.get("body_text") or "").strip()
            if t:
                return t
        return ""

    # 1. Find target's standard segment by header match
    target_idx = None
    if target:
        for i, s in enumerate(std_segs):
            if _norm(s.get("header_ocr")) == target:
                target_idx = i
                break
        if target_idx is None:
            for i, s in enumerate(std_segs):
                hdr = _norm(s.get("header_ocr"))
                if hdr and (hdr in target or target in hdr):
                    target_idx = i
                    break
        if target_idx is None:
            from difflib import SequenceMatcher
            best_ratio = 0.0
            best_idx = None
            for i, s in enumerate(std_segs):
                hdr = _norm(s.get("header_ocr"))
                if not hdr:
                    continue
                ratio = SequenceMatcher(None, target, hdr).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_idx = i
            if best_ratio >= 0.5:
                target_idx = best_idx
    if target_idx is None:
        target_idx = 0

    body_parts = [std_segs[target_idx]["body_text"].strip()]

    # 2. If target is NOT the last standard in this file, the next standard
    # in the same file ends the bio — no orphans to append.
    if target_idx < len(std_segs) - 1:
        return body_parts[0]

    # 3. Target IS the last standard. Walk forward to the next bio's file
    # (via next_bio_index) and append leading orphans there. Repeat through
    # consecutive all-orphan files in the rare case that orphan content
    # spans multiple row crops.
    if next_bio_index is None:
        return body_parts[0]

    visited = set()
    cur_sp, cur_ei = sp, ei
    while True:
        nxt = next_bio_index.get((cur_sp, cur_ei))
        if not nxt:
            break
        next_sp, next_si = nxt
        next_stem = next_si.rsplit(".", 1)[0]
        if (next_sp, next_stem) in visited:
            break
        visited.add((next_sp, next_stem))
        next_path = _find_seg_path(next_sp, next_stem)
        if not next_path:
            break
        next_segs = _read_segs(next_path)
        if next_segs is None:
            break
        # Take leading orphans, stop at first 'standard'.
        added_any = False
        hit_standard = False
        for s in next_segs:
            if s.get("type") == "standard":
                hit_standard = True
                break
            bt = (s.get("body_text") or "").strip()
            if bt:
                body_parts.append(bt)
                added_any = True
        if hit_standard or not added_any:
            break
        # File had orphans only — continue to the file after it.
        # We need the entry_index of the bio that "owns" the next file.
        # Use the bio_image_index in reverse: find entry_index for next_sp/next_si.
        next_ei = None
        for (ksp, kei), ksi in bio_image_index.items():
            if ksp == next_sp and ksi == next_si:
                next_ei = kei
                break
        if next_ei is None:
            break
        cur_sp, cur_ei = next_sp, next_ei

    return "\n".join(body_parts).strip()


def run_cloud_extraction(out_dir, provider, model, ocr_dir=None):
    print(f"\n=== Cloud Extraction ({provider}/{model}) ===")
    path = os.path.join(out_dir, "gold_samples", "extraction_gold_sample.csv")
    data = _load_gold_csv(path)
    if not data:
        return
    client, prov = _cloud_client(provider)
    if not client:
        return
    cp = _load_create_prompt()
    if cp:
        print("  Using create_prompt from 5_extract_biographies.py")
    else:
        print("  Warning: using fallback prompt (create_prompt not found)")
    bio_idx = _build_bio_image_index(Path(__file__).parent)
    next_bio_idx = _build_next_bio_index(Path(__file__).parent)
    ocr_roots = _detect_ocr_roots(Path(__file__).parent, primary=ocr_dir)
    print(f"  body fallback: {len(bio_idx)} bio entries indexed; OCR roots:")
    for r in ocr_roots:
        print(f"    {r}")
    n_filled = n_no_body = n_ran = n_err = 0
    for i, row in enumerate(data):
        existing = row.get("cloud_json", "")
        if not _is_dummy_extraction(existing):
            n_filled += 1
            print(f"  [skip {i+1}/{len(data)}] cloud_json already filled")
            continue
        header = row.get("header", "")
        body = _resolve_body_for_row(row, ocr_roots, bio_idx, next_bio_idx)
        if not body:
            n_no_body += 1
            print(f"  [skip {i+1}/{len(data)}] no body "
                  f"(entry_index={row.get('entry_index')}, "
                  f"source_page={row.get('source_page')})")
            row["cloud_json"] = ""  # leave empty rather than dummy {}
            continue
        print(f"  [run  {i+1}/{len(data)}] {header[:30]}  body_len={len(body)}")
        try:
            raw = _cloud_extract(client, prov, model, header, body, cp)
            try:
                row["cloud_json"] = json.dumps(json.loads(raw), ensure_ascii=False)
            except json.JSONDecodeError:
                row["cloud_json"] = raw
            n_ran += 1
        except Exception as e:
            n_err += 1
            row["cloud_json"] = json.dumps({"error": str(e)})
            print(f"    error: {e}")
    save_csv(path, data, list(data[0].keys()))
    print(f"  Summary: ran={n_ran}, skipped(filled)={n_filled}, "
          f"skipped(no body)={n_no_body}, errors={n_err}")


def _ollama_extract(model, header, body, create_prompt_fn, base_url, max_tokens):
    """Run a single extraction via Ollama with the same options used in
    5_extract_biographies.py: temperature 0, format=json, think=False,
    num_predict=BIO_MAX_TOKENS, identical system prompt and create_prompt body."""
    import requests
    sys_msg = ("You are a robotic data parser. Output valid JSON only. "
               "Follow conversion rules strictly. Do not invent new fields.")
    usr_msg = create_prompt_fn(header, body) if create_prompt_fn else (
        f"Extract structured biographical data.\nHeader: {header}\n"
        f"Body: {body}\nReturn valid JSON.")
    chat_url = base_url.replace("/v1", "").rstrip("/") + "/api/chat"
    r = requests.post(chat_url, json={
        "model": model,
        "messages": [{"role": "system", "content": sys_msg},
                     {"role": "user", "content": usr_msg}],
        "format": "json", "think": False, "stream": False,
        "options": {"temperature": 0.0, "num_predict": max_tokens},
    }, timeout=300)
    r.raise_for_status()
    return _clean_llm_json_output(r.json()["message"]["content"])


def run_slm_alt_extraction(out_dir, model, column, ocr_dir=None):
    """Run an alternative Ollama model (e.g. ministral-3:8b) for SLM comparison.
    Mirrors run_cloud_extraction but goes through local Ollama with the same
    prompt and options as 5_extract_biographies.py. Writes to *column*."""
    print(f"\n=== Alt SLM Extraction (ollama/{model}) → {column} ===")
    path = os.path.join(out_dir, "gold_samples", "extraction_gold_sample.csv")
    data = _load_gold_csv(path)
    if not data:
        return
    try:
        spec = importlib.util.spec_from_file_location(
            "_cfg", str(Path(__file__).parent / "config.py"))
        cfg = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)
        base_url = getattr(cfg, "BIO_OLLAMA_BASE_URL", "http://localhost:11434/v1")
        max_tok  = getattr(cfg, "BIO_MAX_TOKENS", 8000)
    except Exception as exc:
        print(f"  Could not import config.py: {exc}; using defaults"); return
    cp = _load_create_prompt()
    if cp:
        print("  Using create_prompt from 5_extract_biographies.py")
    else:
        print("  Warning: using fallback prompt (create_prompt not found)")
    fields = list(data[0].keys())
    if column not in fields:
        fields.append(column)
        for row in data:
            row.setdefault(column, "")
    bio_idx = _build_bio_image_index(Path(__file__).parent)
    next_bio_idx = _build_next_bio_index(Path(__file__).parent)
    ocr_roots = _detect_ocr_roots(Path(__file__).parent, primary=ocr_dir)
    print(f"  body fallback: {len(bio_idx)} bio entries indexed; OCR roots:")
    for r in ocr_roots:
        print(f"    {r}")
    n_filled = n_no_body = n_ran = n_err = 0
    for i, row in enumerate(data):
        existing = row.get(column, "")
        if not _is_dummy_extraction(existing):
            n_filled += 1
            print(f"  [skip {i+1}/{len(data)}] {column} already filled")
            continue
        header = row.get("header", "")
        body = _resolve_body_for_row(row, ocr_roots, bio_idx, next_bio_idx)
        if not body:
            n_no_body += 1
            print(f"  [skip {i+1}/{len(data)}] no body "
                  f"(entry_index={row.get('entry_index')}, "
                  f"source_page={row.get('source_page')})")
            row[column] = ""
            continue
        print(f"  [run  {i+1}/{len(data)}] {header[:30]}  body_len={len(body)}")
        try:
            raw = _ollama_extract(model, header, body, cp, base_url, max_tok)
            try:
                row[column] = json.dumps(json.loads(raw), ensure_ascii=False)
            except json.JSONDecodeError:
                row[column] = raw
            n_ran += 1
        except Exception as e:
            n_err += 1
            row[column] = json.dumps({"error": str(e)})
            print(f"    error: {e}")
    save_csv(path, data, fields)
    print(f"  Summary: ran={n_ran}, skipped(filled)={n_filled}, "
          f"skipped(no body)={n_no_body}, errors={n_err}")


def run_hf_extraction(out_dir, model, column, token, ocr_dir=None):
    """Run a HuggingFace Inference API model (e.g. meta-llama/Llama-3.1-8B-Instruct)
    for SLM comparison. Uses chat-completion via huggingface_hub.InferenceClient
    with the identical system + create_prompt user message. Writes to *column*."""
    print(f"\n=== HF Extraction ({model}) → {column} ===")
    path = os.path.join(out_dir, "gold_samples", "extraction_gold_sample.csv")
    data = _load_gold_csv(path)
    if not data:
        return
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        print("  pip install huggingface_hub  is required"); return
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print("  Pass --hf-token or set HF_TOKEN env var"); return
    client = InferenceClient(token=token)
    cp = _load_create_prompt()
    if cp:
        print("  Using create_prompt from 5_extract_biographies.py")
    else:
        print("  Warning: using fallback prompt (create_prompt not found)")
    sys_msg = ("You are a robotic data parser. Output valid JSON only. "
               "Follow conversion rules strictly. Do not invent new fields.")
    fields = list(data[0].keys())
    if column not in fields:
        fields.append(column)
        for row in data:
            row.setdefault(column, "")
    bio_idx = _build_bio_image_index(Path(__file__).parent)
    next_bio_idx = _build_next_bio_index(Path(__file__).parent)
    ocr_roots = _detect_ocr_roots(Path(__file__).parent, primary=ocr_dir)
    print(f"  body fallback: {len(bio_idx)} bio entries indexed; OCR roots:")
    for r in ocr_roots:
        print(f"    {r}")
    n_filled = n_no_body = n_ran = n_err = 0
    for i, row in enumerate(data):
        existing = row.get(column, "")
        if not _is_dummy_extraction(existing):
            n_filled += 1
            print(f"  [skip {i+1}/{len(data)}] {column} already filled")
            continue
        header = row.get("header", "")
        body = _resolve_body_for_row(row, ocr_roots, bio_idx, next_bio_idx)
        if not body:
            n_no_body += 1
            print(f"  [skip {i+1}/{len(data)}] no body "
                  f"(entry_index={row.get('entry_index')}, "
                  f"source_page={row.get('source_page')})")
            row[column] = ""
            continue
        usr_msg = cp(header, body) if cp else (
            f"Extract structured biographical data.\nHeader: {header}\n"
            f"Body: {body}\nReturn valid JSON.")
        print(f"  [run  {i+1}/{len(data)}] {header[:30]}  body_len={len(body)}")
        try:
            resp = client.chat_completion(
                model=model,
                messages=[{"role": "system", "content": sys_msg},
                          {"role": "user", "content": usr_msg}],
                max_tokens=3500, temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            row[column] = _clean_llm_json_output(raw)
            n_ran += 1
        except Exception as e:
            n_err += 1
            row[column] = json.dumps({"error": str(e)})
            print(f"    error: {e}")
    save_csv(path, data, fields)
    print(f"  Summary: ran={n_ran}, skipped(filled)={n_filled}, "
          f"skipped(no body)={n_no_body}, errors={n_err}")


def run_occcanine_comparison(out_dir):
    """Supplementary: OccCANINE on HISCO gold sample.

    OccCANINE (package name ``histocc``) is NOT on PyPI — install from a local
    clone of https://github.com/christianvedels/OccCANINE:
        git clone https://github.com/christianvedels/OccCANINE.git
        pip install ./OccCANINE
    """
    print("\n=== OccCANINE Comparison ===")
    try:
        from histocc import OccCANINE
        import pandas as pd
    except ImportError:
        print("  histocc not installed.  OccCANINE is not on PyPI; install via:")
        print("    git clone https://github.com/christianvedels/OccCANINE.git")
        print("    pip install ./OccCANINE")
        return

    path = os.path.join(out_dir, "gold_samples", "hisco_gold_sample.csv")
    data = _load_gold_csv(path)
    if not data:
        return

    threshold = 0.22
    try:
        spec = importlib.util.spec_from_file_location(
            "_cfg", str(Path(__file__).parent / "config.py"))
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        threshold = getattr(cfg, "HISCO_CONFIDENCE_THRESHOLD", 0.22)
    except Exception:
        pass
    print(f"  Confidence threshold: {threshold}")

    # Map row index → job title; predict in a single batched call
    titles = [row.get("job_title_en", "") for row in data]
    valid_idx = [i for i, t in enumerate(titles) if t]
    if not valid_idx:
        print("  No rows have job_title_en — nothing to predict")
        return

    model = OccCANINE()
    try:
        df = model.predict([titles[i] for i in valid_idx], lang="en")
    except Exception as e:
        print(f"  Prediction failed: {e}")
        return

    # Output DataFrame columns observed: occ1, hisco_1, desc_1, plus either
    # `conf` (single-confidence) or `prob_1`/`prob_2`/... when top-k is on.
    conf_col = next((c for c in ("conf", "prob_1") if c in df.columns), None)
    hisco_col = "hisco_1" if "hisco_1" in df.columns else None
    if hisco_col is None:
        print(f"  Unexpected predict() output columns: {list(df.columns)}")
        return

    for j, i in enumerate(valid_idx):
        row_pred = df.iloc[j]
        raw = row_pred[hisco_col]
        if pd.isna(raw):
            code = ""
        else:
            s = str(raw).strip()
            # Numeric coercion can yield "79190.0" — drop trailing .0
            if "." in s:
                try:
                    s = str(int(float(s)))
                except ValueError:
                    pass
            # OccCANINE returns 5-digit HISCO codes; zero-pad then take the
            # first 2 digits (HISCO minor group) so it matches qwen_hisco
            # and gold_hisco, which are stored as 2-digit minors.
            code = s.zfill(5)[:2] if s.isdigit() else ""
        if conf_col is not None and code:
            try:
                conf_val = float(row_pred[conf_col])
            except (TypeError, ValueError):
                conf_val = 0.0
            if conf_val < threshold:
                code = ""
        data[i]["occcanine_hisco"] = code
    for i in range(len(data)):
        data[i].setdefault("occcanine_hisco", "")

    fns = list(data[0].keys())
    if "occcanine_hisco" not in fns:
        fns.append("occcanine_hisco")
    save_csv(path, data, fns)
    print(f"  Done ({len(valid_idx)}/{len(data)} records had job_title_en)")


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

def generate_summary(out_dir):
    import glob as _g
    lines = ["=" * 70, "PAPER STATISTICS SUMMARY", "=" * 70, ""]
    for d, skip in [("tables", set()), ("slm_metrics",
                     {"similarity_distribution", "hisco_distribution",
                      "isic_distribution"})]:
        dp = os.path.join(out_dir, d)
        if not os.path.isdir(dp):
            continue
        for cf in sorted(_g.glob(os.path.join(dp, "*.csv"))):
            stem = os.path.splitext(os.path.basename(cf))[0]
            if stem in skip:
                continue
            lines.append(f"\n--- {stem} ---")
            with open(cf, encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if "metric" in row:
                        lines.append(f"  {row['metric']}: {row.get('value','')}")
    sd = os.path.join(out_dir, "scores")
    if os.path.isdir(sd):
        for tf in sorted(_g.glob(os.path.join(sd, "*.txt"))):
            lines += ["", open(tf, encoding="utf-8").read()]
    save_text(os.path.join(out_dir, "summary.txt"), "\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Paper statistics for the archival Japanese digitisation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples
        --------
          python compute_paper_stats.py --stats
          python compute_paper_stats.py --export-gold --bio-dir ./bios --ocr-dir ./ocr_results
          python compute_paper_stats.py --score-gold
          python compute_paper_stats.py --run-cloud --cloud-provider google --cloud-model gemini-2.0-flash
          python compute_paper_stats.py --compare-hisco
        """))

    ap.add_argument("--stats",        action="store_true", help="Sections A+B")
    ap.add_argument("--export-gold",  action="store_true", help="Section C")
    ap.add_argument("--score-gold",   action="store_true", help="Section D")
    ap.add_argument("--run-cloud",    action="store_true", help="Section E (cloud)")
    ap.add_argument("--run-slm-alt",  action="store_true",
                    help="Section E: run an alternative Ollama model (default "
                         "ministral-3:8b) on the extraction gold sample, using "
                         "the same prompt and options as 5_extract_biographies.py")
    ap.add_argument("--run-hf",       action="store_true",
                    help="Section E: run a HuggingFace Inference model (default "
                         "meta-llama/Llama-3.1-8B-Instruct) on the extraction gold sample")
    ap.add_argument("--compare-hisco",action="store_true", help="Section E (OccCANINE)")

    ap.add_argument("--cloud-provider", default="google",
                    choices=["openai", "anthropic", "google"])
    ap.add_argument("--cloud-model",    default="gemini-2.0-flash",
                    help="Cloud model (default: gemini-2.0-flash) — used for "
                         "extraction comparison only; OCR comparison lives in "
                         "eval_ocr.py")
    ap.add_argument("--alt-model",   default="ministral-3:8b",
                    help="Ollama model name for --run-slm-alt (default: ministral-3:8b)")
    ap.add_argument("--alt-column",  default="ministral_json",
                    help="Gold-CSV column name to write alt-SLM output to "
                         "(default: ministral_json)")
    ap.add_argument("--hf-model",    default="meta-llama/Llama-3.1-8B-Instruct",
                    help="HF model id for --run-hf (default: meta-llama/Llama-3.1-8B-Instruct). "
                         "Note: base models without a chat template may need the "
                         "Instruct variant for reliable JSON output.")
    ap.add_argument("--hf-column",   default="llama_json",
                    help="Gold-CSV column name to write HF output to (default: llama_json)")
    ap.add_argument("--hf-token",    default=None,
                    help="HuggingFace token. Falls back to $HF_TOKEN / "
                         "$HUGGINGFACE_HUB_TOKEN if omitted.")

    ap.add_argument("--data-dir",   default=None,
                    help="Data directory (default: website/data/ relative to script)")
    ap.add_argument("--bio-dir",    default=None,
                    help="biographies_extracted_*.jsonl directory "
                         "(default: script directory)")
    ap.add_argument("--ocr-dir",    default=None,
                    help="ocr_results/ segment files directory — used to fill "
                         "body_text in extraction gold samples when missing "
                         "from the bio JSONL")
    ap.add_argument("--output-dir", default=None,
                    help="Output directory (default: paper_stats/)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    ap.add_argument("--gold-n", type=int, default=50,
                    help="Sample size per gold-standard CSV (default: 50). "
                         "HISCO and ISIC are also volume-stratified, so each "
                         "of the 6 volumes gets ≈ n/6 entries.")

    ap.add_argument("--bertscore", action="store_true",
                    help="Add BERTScore F1 columns to extraction scoring "
                         "(needs: pip install bert_score torch transformers fugashi unidic-lite)")
    ap.add_argument("--bert-model", default=DEFAULT_BERT_MODEL,
                    help=f"HuggingFace model for BERTScore (default: {DEFAULT_BERT_MODEL})")

    args = ap.parse_args()
    if not any([args.stats, args.export_gold, args.score_gold,
                args.run_cloud, args.run_slm_alt, args.run_hf,
                args.compare_hisco]):
        ap.print_help(); sys.exit(1)

    root    = Path(__file__).parent
    data_dir = args.data_dir or str(root / "website" / "data")
    bio_dir  = args.bio_dir  or str(root)
    out_dir  = args.output_dir or str(root / "paper_stats")
    os.makedirs(out_dir, exist_ok=True)

    # ── A + B ──
    if args.stats:
        print("\n" + "=" * 70 + "\nSECTION A: Descriptive Statistics\n" + "=" * 70)
        compute_dataset_overview(data_dir, out_dir)
        compute_dashboard_filtered_counts(data_dir, out_dir)
        compute_demographics(data_dir, out_dir)
        compute_careers(data_dir, out_dir)
        compute_family(data_dir, out_dir)
        compute_geography(data_dir, out_dir)
        compute_org_network(data_dir, out_dir)
        compute_data_quality(data_dir, out_dir)

        print("\n" + "=" * 70 + "\nSECTION B: SLM Pipeline Metrics\n" + "=" * 70)
        compute_era_date_stats(data_dir, bio_dir, out_dir)
        compute_org_matching_stats(data_dir, out_dir)
        compute_org_hierarchy_stats(data_dir, out_dir)
        compute_disambiguation_stats(data_dir, out_dir)
        generate_summary(out_dir)

    # ── C ──
    if args.export_gold:
        print("\n" + "=" * 70 + "\nSECTION C: Gold Standard Export\n" + "=" * 70)
        export_extraction_gold_sample(data_dir, bio_dir, out_dir,
                                      ocr_dir=args.ocr_dir,
                                      n=args.gold_n, seed=args.seed)
        export_hisco_gold_sample(data_dir, out_dir,
                                 n=args.gold_n, seed=args.seed)
        export_isic_gold_sample(data_dir, out_dir,
                                n=args.gold_n, seed=args.seed)
        export_org_match_gold_sample(data_dir, out_dir,
                                     n=args.gold_n, seed=args.seed)
        export_org_hierarchy_gold_sample(data_dir, out_dir,
                                         n=args.gold_n, seed=args.seed)
        write_annotation_guide(out_dir)

    # ── E (data-generating runs first, so D below sees fresh columns) ──
    if args.run_cloud:
        print("\n" + "=" * 70 + "\nSECTION E: Cloud LLM Comparison\n" + "=" * 70)
        run_cloud_extraction(out_dir, args.cloud_provider, args.cloud_model,
                             ocr_dir=args.ocr_dir)

    if args.run_slm_alt:
        print("\n" + "=" * 70 + "\nSECTION E: Alt-SLM (Ollama) Comparison\n" + "=" * 70)
        run_slm_alt_extraction(out_dir, args.alt_model, args.alt_column,
                               ocr_dir=args.ocr_dir)

    if args.run_hf:
        print("\n" + "=" * 70 + "\nSECTION E: HuggingFace Inference Comparison\n" + "=" * 70)
        run_hf_extraction(out_dir, args.hf_model, args.hf_column,
                          args.hf_token, ocr_dir=args.ocr_dir)

    if args.compare_hisco:
        run_occcanine_comparison(out_dir)

    # ── D (after E so scoring sees newly-populated cloud/ministral/llama columns) ──
    if args.score_gold:
        print("\n" + "=" * 70 + "\nSECTION D: Gold Standard Scoring\n" + "=" * 70)
        if args.bertscore:
            globals()["_BERT_MODEL_NAME"] = args.bert_model
        score_extraction(out_dir, use_bert=args.bertscore)
        score_hisco(out_dir)
        score_isic(out_dir)
        score_org_match(out_dir)
        score_org_hierarchy(out_dir)
        generate_summary(out_dir)

    print(f"\nDone. Output in: {out_dir}")


if __name__ == "__main__":
    main()
