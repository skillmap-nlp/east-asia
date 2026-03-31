#!/usr/bin/env python3
"""
1. Add 'source' column to all tables in east_asia_job_ads.db → value 'indeed'.
2. Import careerjet rows from offers.db, deduplicating by job_url.
   Columns are mapped to match east_asia_job_ads schema.
"""
import sqlite3
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EA_DB = str(SCRIPT_DIR / "east_asia_job_ads.db")
OFFERS_DB = str(SCRIPT_DIR.parent / "offers.db")

DOMAIN_TO_TABLE = {
    "www.opcionempleo.com.mx": "jobads_mx",
    "www.opcionempleo.cl":     "jobads_cl",
    "www.careerjet.jp":        "jobads_jp",
    "www.careerjet.pl":        "jobads_pl",
    "www.careerjet.sg":        "jobads_sg",
    "www.careerjet.id":        "jobads_id",
}

# offers.db col → east_asia_job_ads.db col
COL_MAP = {
    "job_url":               "job_url",
    "job_title":             "job_title",
    "employer":              "employer",
    "location_posting":      "workplace",
    "description":           "responsibilities",
    "date_posted":           "date_posted",
    "job_type":              "job_type",
    "min_amount":            "min_amount",
    "max_amount":            "max_amount",
    "company_num_employees": "company_num_employees",
    "company_revenue":       "company_revenue",
    "company_addresses":     "company_addresses",
    "company_industry":      "company_industry",
    "title_lang":            "job_title_lang",
    "responsibilities_lang": "responsibilities_lang",
    "description_english":   "description_english",
    "job_title_english":     "job_title_english",
    "skills":                "skills",
    "isic_section":          "isic_section",
    "region_iso":            "region_iso",
    "skill_match":           "skill_match",
}


def p(msg):
    print(msg, flush=True)


def get_domain(url):
    if not url:
        return None
    parts = url.split("/")
    return parts[2] if len(parts) >= 3 else None


def add_source_column(ea_conn):
    cur = ea_conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]

    for table in tables:
        cur.execute(f'PRAGMA table_info("{table}")')
        cols = {r[1] for r in cur.fetchall()}
        if "source" not in cols:
            cur.execute(f'ALTER TABLE "{table}" ADD COLUMN "source" TEXT')
            p(f"  Added 'source' column to {table}")
        cur.execute(f"UPDATE \"{table}\" SET source = 'indeed' WHERE source IS NULL")
        p(f"  {table}: set {cur.rowcount:,} rows → source='indeed'")

    ea_conn.commit()


def create_index_if_missing(ea_conn, table):
    idx = f"idx_{table}_job_url"
    ea_conn.execute(
        f'CREATE INDEX IF NOT EXISTS "{idx}" ON "{table}" ("job_url")'
    )
    ea_conn.commit()


def import_careerjet(ea_conn):
    offers_conn = sqlite3.connect(OFFERS_DB)
    ea_cur = ea_conn.cursor()

    # Create indexes on job_url for fast dedup lookups
    p("  Creating job_url indexes...")
    for table in set(DOMAIN_TO_TABLE.values()):
        create_index_if_missing(ea_conn, table)
    p("  Indexes ready.")

    # Get target columns per table
    target_cols_cache = {}
    for table in set(DOMAIN_TO_TABLE.values()):
        ea_cur.execute(f'PRAGMA table_info("{table}")')
        target_cols_cache[table] = {r[1] for r in ea_cur.fetchall()}

    # Read careerjet rows and bucket by target table
    p("  Reading careerjet rows from offers.db...")
    offers_cur = offers_conn.cursor()
    offers_cur.execute("SELECT * FROM offers WHERE source='careerjet'")
    src_cols = [d[0] for d in offers_cur.description]

    buckets: dict[str, list] = {t: [] for t in set(DOMAIN_TO_TABLE.values())}
    skipped_domain = 0

    for row in offers_cur:
        row_dict = dict(zip(src_cols, row))
        domain = get_domain(row_dict.get("job_url"))
        target = DOMAIN_TO_TABLE.get(domain)
        if not target:
            skipped_domain += 1
            continue

        target_cols = target_cols_cache[target]
        mapped = {}
        for src, dst in COL_MAP.items():
            if dst in target_cols and src in row_dict:
                mapped[dst] = row_dict[src]
        mapped["source"] = "careerjet"
        buckets[target].append(mapped)

    offers_conn.close()
    p(f"  Skipped (unknown domain): {skipped_domain:,}")

    # Insert with dedup via NOT EXISTS
    total_inserted = 0
    for table, rows in buckets.items():
        if not rows:
            p(f"  {table}: 0 careerjet rows")
            continue

        col_names = list(rows[0].keys())
        col_str = ", ".join(f'"{c}"' for c in col_names)
        placeholders = ", ".join(["?"] * len(col_names))

        insert_sql = (
            f'INSERT INTO "{table}" ({col_str}) '
            f'SELECT {placeholders} '
            f'WHERE NOT EXISTS '
            f'(SELECT 1 FROM "{table}" t2 WHERE t2.job_url = ?)'
        )

        url_idx = col_names.index("job_url")
        batch = []
        inserted = 0
        for r in rows:
            vals = tuple(r.get(c) for c in col_names)
            batch.append(vals + (vals[url_idx],))

        # Executemany with dedup subquery
        ea_cur.executemany(insert_sql, batch)
        inserted = ea_cur.rowcount
        ea_conn.commit()

        p(f"  {table}: {len(rows):,} careerjet → {inserted:,} inserted (rest duplicates)")
        total_inserted += inserted

    return total_inserted


def main():
    p("=" * 60)
    p("MERGE CAREERJET INTO EAST_ASIA_JOB_ADS")
    p("=" * 60)

    ea_conn = sqlite3.connect(EA_DB, timeout=60)
    ea_conn.execute("PRAGMA journal_mode=WAL")
    ea_conn.execute("PRAGMA synchronous=OFF")
    ea_conn.execute("PRAGMA cache_size=-64000")

    p("\n[1] Adding 'source' column (value='indeed')...")
    add_source_column(ea_conn)

    p("\n[2] Importing careerjet rows from offers.db...")
    total = import_careerjet(ea_conn)

    ea_conn.close()
    p(f"\n{'='*60}")
    p(f"DONE — inserted {total:,} careerjet rows total")
    p(f"{'='*60}")


if __name__ == "__main__":
    main()
