#!/usr/bin/env python3
"""
Aggregate offer shares with digital skills by level using the
"Digital skill classification_Chinen et al 2025.xlsx" taxonomy.

Method:
- map workbook ESCO labels -> comprehensive_esco.db codes/uris
- ignore transversal and OTHER
- classify offer as basic/intermediate/advanced if at least one mapped ESCO
  skill of that level appears in the offer's *_esco.jsonl output
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
ESCO_DB = ROOT.parent / "comprehensive_esco.db"
WORKBOOK = ROOT / "Digital skill classification_Chinen et al 2025.xlsx"
ESCO_RESULTS_DIR = ROOT / "esco_skills"
OUT_PATH = ROOT / "digital_skill_offer_shares.json"

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

LEVEL_MAP = {
    "Dig./Basic ICT": "Basic",
    "Dig./Intermediate ICT": "Intermediate",
    "Dig./Advanced ICT": "Advanced",
}


def normalize_label(s: str) -> str:
    return " ".join(str(s).strip().lower().replace("\u00a0", " ").split())


def load_esco_lookup():
    conn = sqlite3.connect(ESCO_DB)
    cur = conn.cursor()
    cur.execute("SELECT code, uri, title FROM esco_concepts")
    rows = cur.fetchall()
    conn.close()

    by_title = {}
    by_uri = {}
    for code, uri, title in rows:
        by_title[normalize_label(title)] = {"code": code, "uri": uri, "title": title}
        by_uri[uri] = {"code": code, "title": title}
    return by_title, by_uri


def build_level_sets(by_title):
    df = pd.read_excel(WORKBOOK, sheet_name="skilllist")
    level_sets = defaultdict(set)
    unmatched = []

    for _, row in df.iterrows():
        tax = row.get("DIGITAL_Taxonomy")
        if tax not in LEVEL_MAP:
            continue
        label = normalize_label(row.get("ESCO_SKILL_LABEL"))
        if not label or label == "nan":
            continue
        if label in by_title:
            level_sets[LEVEL_MAP[tax]].add(by_title[label]["uri"])
        else:
            unmatched.append(label)

    return level_sets, unmatched


def main():
    by_title, by_uri = load_esco_lookup()
    level_sets, unmatched = build_level_sets(by_title)

    rows = []
    for path in sorted(ESCO_RESULTS_DIR.glob("jobads_*_esco.jsonl")):
        key = path.stem.removeprefix("jobads_").removesuffix("_esco")
        if key not in COUNTRY_LABELS:
            continue

        totals = {"Basic": 0, "Intermediate": 0, "Advanced": 0, "Any": 0}
        offers = 0

        with open(path) as fh:
            for line in fh:
                offers += 1
                row = json.loads(line)
                uris = {item.get("uri") for item in row.get("skills", []) if item.get("uri")}

                offer_has_any = False
                for level in ("Basic", "Intermediate", "Advanced"):
                    if uris & level_sets[level]:
                        totals[level] += 1
                        offer_has_any = True
                if offer_has_any:
                    totals["Any"] += 1

        rows.append({
            "country": COUNTRY_LABELS[key],
            "offers": offers,
            "shares": {
                level: (totals[level] / offers) if offers else 0
                for level in ("Any", "Basic", "Intermediate", "Advanced")
            }
        })

    payload = {
        "levels": ["Any", "Basic", "Intermediate", "Advanced"],
        "rows": rows,
        "matched_taxonomy_counts": {k: len(v) for k, v in level_sets.items()},
        "unmatched_taxonomy_labels": len(unmatched),
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
