#!/usr/bin/env python3
"""Build digital_level_composition.json for the rolling-average mix chart.

For every country and month we compute the composition of digital-skill
*mentions* across the three ICT proficiency bands (Basic / Intermediate /
Advanced), which sum to 100% within each country-month. Only postings that
carry a month-level posting date are included; February 2026 is dropped as an
incomplete-collection month.

The composition conditions on a digital mention, so it is not distorted by how
many skills the pipeline extracts per posting (which varies by source language,
ad length, occupation, and processing). No skills are excluded beyond the
standard pipeline. The 3-month rolling average is applied client-side in the
report.
"""
from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ESCO_DIR = HERE / "esco_skills"
GEMMA_DIR = HERE / "gemma_results"
CLASS_CSV = HERE / "complete_esco_skills_classified_with_codes.csv"
DB = HERE / "east_asia_job_ads_gemma.db"
OUT = HERE / "digital_level_composition.json"
TMP = Path("/tmp/digital_level_composition.json")

THRESHOLD = 0.50
WIN_START, WIN_END = "2025-03", "2026-04"
UNRELIABLE = {"2026-02"}
LEVEL_MAP = {"Dig./Basic ICT": "Basic", "Dig./Intermediate ICT": "Intermediate", "Dig./Advanced ICT": "Advanced"}
LEVELS = ["Basic", "Intermediate", "Advanced"]
COUNTRIES = [
    ("jp", "Japan", "East Asia"), ("kr", "South Korea", "East Asia"), ("tw", "Taiwan", "East Asia"),
    ("th", "Thailand", "East Asia"), ("malaysia", "Malaysia", "East Asia"), ("sg", "Singapore", "East Asia"),
    ("id", "Indonesia", "East Asia"), ("vn", "Vietnam", "East Asia"), ("ph", "Philippines", "East Asia"),
    ("in", "India", "Benchmark"), ("mx", "Mexico", "Benchmark"), ("pl", "Poland", "Benchmark"),
]


def build_months():
    months, cur = [], WIN_START
    while cur <= WIN_END:
        if cur not in UNRELIABLE:
            months.append(cur)
        y, m = int(cur[:4]), int(cur[5:7])
        m += 1
        if m > 12:
            m, y = 1, y + 1
        cur = f"{y:04d}-{m:02d}"
    return months


def load_levels():
    out = {}
    with CLASS_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lv = LEVEL_MAP.get((r.get("label") or "").strip())
            nm = (r.get("skill_name") or "").strip().lower()
            if lv and nm:
                out[nm] = lv
    return out


def src(code):
    return [p for p in (ESCO_DIR / f"jobads_{code}_esco.jsonl", GEMMA_DIR / f"jobads_{code}_esco.jsonl") if p.exists()]


def offer_level_counts(code, name_levels):
    out, seen = {}, set()
    for p in src(code):
        with p.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rid = rec.get("id")
                try:
                    rid = int(rid)
                except (TypeError, ValueError):
                    pass
                if rid in seen:
                    continue
                seen.add(rid)
                c = Counter()
                for sk in rec.get("skills") or []:
                    if (sk.get("score", 0) or 0) < THRESHOLD:
                        continue
                    lv = name_levels.get((sk.get("label") or "").strip().lower())
                    if lv:
                        c[lv] += 1
                if c:
                    out[rid] = c
    return out


def main():
    months = build_months()
    month_set = set(months)
    name_levels = load_levels()
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    raw = {}
    for code, name, _grp in COUNTRIES:
        iddate = dict(cur.execute(f"SELECT id, date_posted FROM jobads_{code}"))
        olc = offer_level_counts(code, name_levels)
        cells = defaultdict(Counter)        # ym -> Counter(level)
        for rid, lv in olc.items():
            ym = (iddate.get(rid) or "")[:7]
            if ym not in month_set:
                continue
            cells[ym].update(lv)
        raw[name] = {}
        for ym in months:
            agg = cells.get(ym, Counter())
            tot = sum(agg.values())
            raw[name][ym] = {lv: (round(100 * agg[lv] / tot, 4) if tot else None) for lv in LEVELS}
    conn.close()

    payload = {
        "metric": "composition of digital skill mentions (Basic/Intermediate/Advanced sum to 100%)",
        "normalisations": {"raw": "pooled across all postings (month-level dated)"},
        "rolling_window_months": 3,
        "levels": LEVELS,
        "months": months,
        "countries": {name: {"group": grp} for _c, name, grp in COUNTRIES},
        "raw": raw,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    TMP.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    def adv_avg(name):
        vals = [raw[name][ym]["Advanced"] for ym in months if raw[name][ym]["Advanced"] is not None]
        return sum(vals) / len(vals) if vals else None
    print("Advanced share (avg of monthly):")
    for _c, name, _g in COUNTRIES:
        print(f"  {name:14s} {adv_avg(name):5.1f}%")
    print(f"Wrote {OUT} and {TMP}")


if __name__ == "__main__":
    main()
