#!/usr/bin/env python3
"""Export digital-skill co-occurrence analytics for the Quarto report.

Reads the dashboard's ``digital_links.db`` (produced by
``flask2/build_digital_links.py``) and writes a single JSON file that the
report embeds via ``FileAttachment``. Only two of the dashboard's three views
are carried over — the heavy co-occurrence *tree* is intentionally skipped:

  * Digital pairings    — NPMI heatmap nodes/edges among the top digital skills
                          (tagged with their ICT level).
  * Digital x areas      — premium (pp) of broad ESCO areas inside digital
                          offers vs. the all-offer baseline, by ICT level, plus
                          the concrete top co-occurring (non-digital) partners.

The output is keyed by the report's display country names so the OJS code can
look them up directly.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DB = HERE / "flask2" / "static" / "data" / "digital_links.db"
DEFAULT_OUT = HERE / "digital_cooccurrence_links.json"

# Report display name -> dashboard (app) country code. East Asia set, same
# order/coverage as the rest of the digital section in report.qmd.
COUNTRY_CODE = {
    "Japan": "jp",
    "South Korea": "kr",
    "Singapore": "sg",
    "Taiwan": "tw",
    "Malaysia": "my",
    "Thailand": "th",
    "Indonesia": "id",
    "Vietnam": "vn",
    "Philippines": "ph",
}


def export_country(conn: sqlite3.Connection, code: str) -> dict | None:
    conn.row_factory = sqlite3.Row
    nodes = [
        {"code": r["code"], "title": r["title"], "level": r["level"], "pct": r["pct"]}
        for r in conn.execute(
            "SELECT code,title,level,pct FROM dd_node WHERE country=? ORDER BY ord",
            (code,),
        )
    ]
    if not nodes:
        return None
    node_codes = {n["code"] for n in nodes}
    edges = [
        {"a": r["code_a"], "b": r["code_b"], "npmi": r["npmi"], "cooc": r["cooc"]}
        for r in conn.execute(
            "SELECT code_a,code_b,npmi,cooc FROM dd_edge WHERE country=?", (code,)
        )
        if r["code_a"] in node_codes and r["code_b"] in node_codes
    ]

    # Group premiums -> {ghead,glabel,pillar, by_level:{All/Basic/...}}
    groups: dict[str, dict] = {}
    for r in conn.execute(
        "SELECT level,ghead,glabel,pillar,base_share,dig_share,premium_pp,level_offers "
        "FROM group_premium WHERE country=?",
        (code,),
    ):
        g = groups.setdefault(
            r["ghead"],
            {"ghead": r["ghead"], "glabel": r["glabel"], "pillar": r["pillar"], "by_level": {}},
        )
        g["by_level"][r["level"]] = {
            "base": r["base_share"], "dig": r["dig_share"],
            "premium": r["premium_pp"], "level_offers": r["level_offers"],
        }
    groups_out = sorted(
        groups.values(),
        key=lambda g: g["by_level"].get("All", {}).get("premium", 0),
        reverse=True,
    )

    partners = [
        {
            "level": r["level"], "code": r["code"], "title": r["title"],
            "glabel": r["glabel"], "pillar": r["pillar"],
            "base_pct": r["base_pct"], "dig_pct": r["dig_pct"],
            "premium_pp": r["premium_pp"], "lift": r["lift"], "support": r["support"],
        }
        for r in conn.execute(
            "SELECT level,code,title,glabel,pillar,base_pct,dig_pct,premium_pp,lift,support "
            "FROM partner WHERE country=? ORDER BY premium_pp DESC",
            (code,),
        )
    ]

    tot = conn.execute(
        "SELECT * FROM country_totals WHERE country=?", (code,)
    ).fetchone()
    totals = dict(tot) if tot else {}

    return {"totals": totals, "nodes": nodes, "edges": edges,
            "groups": groups_out, "partners": partners}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    out: dict[str, dict] = {}
    for name, code in COUNTRY_CODE.items():
        res = export_country(conn, code)
        if res is None:
            print(f"  {name} ({code}): no data — skipped")
            continue
        out[name] = res
        t = res["totals"]
        print(
            f"  {name}: offers={t.get('total_offers', 0):,} "
            f"nodes={len(res['nodes'])} edges={len(res['edges'])} "
            f"groups={len(res['groups'])} partners={len(res['partners'])}"
        )
    conn.close()

    args.out.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    kb = args.out.stat().st_size / 1024
    print(f"Saved {args.out} ({kb:.1f} KB) — {len(out)} countries")


if __name__ == "__main__":
    main()
