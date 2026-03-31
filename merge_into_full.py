#!/usr/bin/env python3
"""
Merge new 2026 job ads from east_asia_job_ads.db into east_asia_job_ads_full.db.
Deduplicates by job_url — only inserts rows whose job_url doesn't already exist.
"""
import sqlite3
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
NEW_DB = SCRIPT_DIR / "east_asia_job_ads.db"
FULL_DB = SCRIPT_DIR / "east_asia_job_ads_full.db"

TABLE_MAP = {
    "jobads_indeed_cl": "jobads_cl",
    "jobads_indeed_id": "jobads_id",
    "jobads_indeed_in": "jobads_in",
    "jobads_indeed_jp": "jobads_jp",
    "jobads_indeed_kr": "jobads_kr",
    "jobads_indeed_malaysia": "jobads_malaysia",
    "jobads_indeed_mx": "jobads_mx",
    "jobads_indeed_ph": "jobads_ph",
    "jobads_indeed_pl": "jobads_pl",
    "jobads_indeed_sg": "jobads_sg",
    "jobads_indeed_th": "jobads_th",
    "jobads_indeed_tw": "jobads_tw",
    "jobads_indeed_vn": "jobads_vn",
}


def get_columns(conn, table):
    cur = conn.execute(f'PRAGMA table_info("{table}")')
    return [r[1] for r in cur.fetchall()]


def merge():
    new_conn = sqlite3.connect(str(NEW_DB))
    full_conn = sqlite3.connect(str(FULL_DB))
    full_conn.execute("PRAGMA journal_mode=WAL")
    full_conn.execute("PRAGMA synchronous=OFF")
    full_conn.execute("PRAGMA cache_size=-64000")

    total_inserted = 0
    total_skipped = 0

    for new_table, full_table in TABLE_MAP.items():
        print(f"\n{'─'*60}")
        print(f"  {new_table}  →  {full_table}")
        print(f"{'─'*60}")

        new_cols = get_columns(new_conn, new_table)
        full_cols = get_columns(full_conn, full_table)

        # Only insert columns that exist in the target table (skip extra cols from new)
        common_cols = [c for c in new_cols if c in full_cols and c != "id"]
        if not common_cols:
            print("  ⚠ No common columns — skipping")
            continue

        if "job_url" not in common_cols:
            print("  ⚠ No job_url column — skipping")
            continue

        # Load existing job_urls from full DB for dedup
        cur = full_conn.execute(f'SELECT job_url FROM "{full_table}" WHERE job_url IS NOT NULL')
        existing_urls = {r[0] for r in cur.fetchall()}
        print(f"  Existing rows in full DB: {len(existing_urls):,}")

        # Read new rows
        col_list = ", ".join(f'"{c}"' for c in common_cols)
        new_rows = new_conn.execute(f'SELECT {col_list} FROM "{new_table}"').fetchall()
        print(f"  New rows to check:       {len(new_rows):,}")

        # Filter out duplicates by job_url
        url_idx = common_cols.index("job_url")
        to_insert = [r for r in new_rows if r[url_idx] and r[url_idx] not in existing_urls]

        skipped = len(new_rows) - len(to_insert)
        print(f"  Duplicates (skipped):    {skipped:,}")
        print(f"  New to insert:           {len(to_insert):,}")

        if to_insert:
            placeholders = ", ".join(["?"] * len(common_cols))
            insert_cols = ", ".join(f'"{c}"' for c in common_cols)
            full_conn.executemany(
                f'INSERT INTO "{full_table}" ({insert_cols}) VALUES ({placeholders})',
                to_insert
            )
            full_conn.commit()
            print(f"  ✓ Inserted {len(to_insert):,} rows")

        total_inserted += len(to_insert)
        total_skipped += skipped

    new_conn.close()
    full_conn.close()

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Total inserted: {total_inserted:,}")
    print(f"  Total skipped:  {total_skipped:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    merge()
