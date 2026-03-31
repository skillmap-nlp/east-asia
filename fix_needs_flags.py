#!/usr/bin/env python3
"""Rebuild needs_title_translation, needs_description_translation,
needs_skill_extraction columns from scratch based on actual data."""

import sqlite3
from pathlib import Path

DB_PATH = Path("data/jobs_ea_gemma.db")

TABLES = [
    "jobads_in", "jobads_jp", "jobads_kr", "jobads_malaysia",
    "jobads_mx", "jobads_ph", "jobads_pl", "jobads_sg",
    "jobads_th", "jobads_tw", "jobads_vn",
]

def main():
    conn = sqlite3.connect(str(DB_PATH), timeout=60)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = OFF")

    for table in TABLES:
        # needs_title_translation
        conn.execute(f"""
            UPDATE "{table}" SET needs_title_translation =
            CASE
                WHEN job_title IS NOT NULL
                 AND TRIM(job_title) != ''
                 AND (job_title_english IS NULL OR TRIM(job_title_english) = '')
                 AND (job_title_lang IS NULL OR job_title_lang != 'EN')
                THEN 1 ELSE 0
            END
        """)

        # needs_description_translation
        conn.execute(f"""
            UPDATE "{table}" SET needs_description_translation =
            CASE
                WHEN responsibilities IS NOT NULL
                 AND TRIM(responsibilities) != ''
                 AND (description_english IS NULL OR TRIM(description_english) = '')
                 AND (responsibilities_lang IS NULL OR responsibilities_lang != 'EN')
                THEN 1 ELSE 0
            END
        """)

        # needs_skill_extraction
        conn.execute(f"""
            UPDATE "{table}" SET needs_skill_extraction =
            CASE
                WHEN skills IS NULL OR TRIM(skills) = ''
                THEN 1 ELSE 0
            END
        """)

        conn.commit()

        # Report
        t = conn.execute(f'SELECT SUM(needs_title_translation) FROM "{table}"').fetchone()[0] or 0
        d = conn.execute(f'SELECT SUM(needs_description_translation) FROM "{table}"').fetchone()[0] or 0
        s = conn.execute(f'SELECT SUM(needs_skill_extraction) FROM "{table}"').fetchone()[0] or 0
        total = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        print(f"  {table:20s}  {total:>8,} rows | "
              f"title={t:>8,}  desc={d:>8,}  skills={s:>8,}", flush=True)

    conn.close()
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
