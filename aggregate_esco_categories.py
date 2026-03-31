#!/usr/bin/env python3
"""
Aggregate ESCO-mapped skills by broad categories.

- Skills:    S1-S8 (top-level ESCO skill branches)
- Knowledge: 3-digit knowledge codes (e.g. 021, 041, 061)

Input:
  - east_asia_2026/esco_skills/jobads_*_esco.jsonl
  - comprehensive_esco.db

Output:
  - east_asia_2026/esco_category_shares.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ESCO_RESULTS_DIR = ROOT / "esco_skills"
ESCO_DB = ROOT.parent / "comprehensive_esco.db"
OUT_PATH = ROOT / "esco_category_shares.json"

COUNTRY_LABELS = {
    "in": "India",
    "jp": "Japan",
    "kr": "South Korea",
    "malaysia": "Malaysia",
    "mx": "Mexico",
    "ph": "Philippines",
    "pl": "Poland",
    "sg": "Singapore",
    "th": "Thailand",
    "tw": "Taiwan",
    "vn": "Vietnam",
}


def load_esco_lookup(db_path: Path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT code, uri, title, skill_type, level
        FROM esco_concepts
        """
    )
    rows = cur.fetchall()
    conn.close()

    by_uri = {}
    skill_meta = {}
    knowledge_meta = {}

    for code, uri, title, skill_type, level in rows:
        by_uri[uri] = {
            "code": code,
            "title": title,
            "skill_type": skill_type,
            "level": level,
        }
        if skill_type == "skills" and level == 1 and code.startswith("S"):
            skill_meta[code] = title
        if skill_type == "knowledge" and level == 2 and len(code) == 3 and code.isdigit():
            knowledge_meta[code] = title

    return by_uri, skill_meta, knowledge_meta


def skill_bucket(code: str) -> str | None:
    return code[:2] if code.startswith("S") else None


def knowledge_bucket(code: str) -> str | None:
    digits = "".join(ch for ch in code if ch.isdigit())
    return digits[:3] if len(digits) >= 3 else None


def compute_shares(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in sorted(counter.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.55)
    args = parser.parse_args()

    by_uri, skill_meta, knowledge_meta = load_esco_lookup(ESCO_DB)

    countries = {}
    knowledge_totals = Counter()

    for path in sorted(ESCO_RESULTS_DIR.glob("jobads_*_esco.jsonl")):
        code = path.stem.removeprefix("jobads_").removesuffix("_esco")
        if code not in COUNTRY_LABELS:
            continue

        skill_counts = Counter()
        knowledge_counts = Counter()
        other_type_counts = Counter()
        offers = 0
        unmapped = 0

        with open(path) as fh:
            for line in fh:
                offers += 1
                row = json.loads(line)
                for item in row.get("skills", []):
                    if item.get("score", 0) < args.threshold:
                        continue
                    meta = by_uri.get(item.get("uri"))
                    if not meta:
                        unmapped += 1
                        continue

                    if meta["skill_type"] == "skills":
                        bucket = skill_bucket(meta["code"])
                        if bucket:
                            skill_counts[bucket] += 1
                    elif meta["skill_type"] == "knowledge":
                        bucket = knowledge_bucket(meta["code"])
                        if bucket:
                            knowledge_counts[bucket] += 1
                            knowledge_totals[bucket] += 1
                    else:
                        other_type_counts[meta["skill_type"]] += 1

        countries[COUNTRY_LABELS[code]] = {
            "offers": offers,
            "skill_total": sum(skill_counts.values()),
            "knowledge_total": sum(knowledge_counts.values()),
            "other_total": sum(other_type_counts.values()),
            "unmapped_total": unmapped,
            "skill_counts": dict(skill_counts),
            "knowledge_counts": dict(knowledge_counts),
            "skill_shares": compute_shares(skill_counts),
            "knowledge_shares": compute_shares(knowledge_counts),
            "other_type_counts": dict(other_type_counts),
        }

    top_knowledge_codes = [code for code, _ in knowledge_totals.most_common(12)]

    payload = {
        "threshold": args.threshold,
        "country_order": list(countries.keys()),
        "skill_meta": skill_meta,
        "knowledge_meta": knowledge_meta,
        "top_knowledge_codes": top_knowledge_codes,
        "countries": countries,
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
