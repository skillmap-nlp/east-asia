#!/usr/bin/env python3
"""
Build a lightweight copy of east_asia_job_ads.db for translation / skills work on another server.

The output DB:
- keeps the same table names as the source DB,
- keeps only the columns needed for translation and skill extraction,
- includes all rows (also those that do not need translation),
- adds helper flags showing which fields still need translation / skills extraction.
"""

import sqlite3
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DB = SCRIPT_DIR / "east_asia_job_ads.db"
TARGET_DB = SCRIPT_DIR / "east_asia_job_ads_gemma.db"

# Keep source column names from east_asia_job_ads.
CORE_COLUMNS = [
    "id",
    "job_url",
    "job_title",
    "responsibilities",
    "job_title_lang",
    "responsibilities_lang",
    "job_title_english",
    "description_english",
    "skills",
]


def get_source_tables(conn):
    cur = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name LIKE 'jobads_%'
        ORDER BY name
        """
    )
    return [row[0] for row in cur.fetchall()]


def get_table_columns(conn, table_name):
    cur = conn.execute(f'PRAGMA table_info("{table_name}")')
    return {row[1]: row[2] or "TEXT" for row in cur.fetchall()}


def build_select_exprs(source_cols):
    exprs = []
    definitions = []

    for col in CORE_COLUMNS:
        if col in source_cols:
            exprs.append(f'"{col}"')
            definitions.append(f'"{col}" {source_cols[col]}')
        elif col == "source":
            exprs.append('"indeed" AS "source"')
            definitions.append('"source" TEXT')
        else:
            exprs.append(f'NULL AS "{col}"')
            definitions.append(f'"{col}" TEXT')

    helper_defs = [
        '"needs_title_translation" INTEGER',
        '"needs_description_translation" INTEGER',
        '"needs_skill_extraction" INTEGER',
    ]
    helper_exprs = [
        """
        CASE
            WHEN job_title IS NOT NULL
             AND TRIM(job_title) != ''
             AND (job_title_english IS NULL OR TRIM(job_title_english) = '')
             AND (job_title_lang IS NULL OR job_title_lang != 'EN')
            THEN 1 ELSE 0
        END AS "needs_title_translation"
        """.strip(),
        """
        CASE
            WHEN responsibilities IS NOT NULL
             AND TRIM(responsibilities) != ''
             AND (description_english IS NULL OR TRIM(description_english) = '')
             AND (responsibilities_lang IS NULL OR responsibilities_lang != 'EN')
            THEN 1 ELSE 0
        END AS "needs_description_translation"
        """.strip(),
        """
        CASE
            WHEN skills IS NULL OR TRIM(skills) = ''
            THEN 1 ELSE 0
        END AS "needs_skill_extraction"
        """.strip(),
    ]

    return definitions + helper_defs, exprs + helper_exprs


def build_table(src_conn, dst_conn, table_name):
    source_cols = get_table_columns(src_conn, table_name)
    definitions, select_exprs = build_select_exprs(source_cols)

    dst_conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    dst_conn.execute(f'CREATE TABLE "{table_name}" ({", ".join(definitions)})')

    src_cur = src_conn.execute(
        f'SELECT {", ".join(select_exprs)} FROM "{table_name}"'
    )

    insert_cols = CORE_COLUMNS + [
        "needs_title_translation",
        "needs_description_translation",
        "needs_skill_extraction",
    ]
    placeholders = ", ".join(["?"] * len(insert_cols))
    insert_sql = (
        f'INSERT INTO "{table_name}" '
        f'({", ".join(f"""\"{col}\"""" for col in insert_cols)}) '
        f'VALUES ({placeholders})'
    )

    while True:
        batch = src_cur.fetchmany(5000)
        if not batch:
            break
        dst_conn.executemany(insert_sql, batch)

    dst_conn.execute(
        f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_job_url" ON "{table_name}" ("job_url")'
    )
    dst_conn.execute(
        f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_needs_translate" '
        f'ON "{table_name}" ("needs_title_translation", "needs_description_translation")'
    )
    dst_conn.execute(
        f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_needs_skills" '
        f'ON "{table_name}" ("needs_skill_extraction")'
    )

    count = dst_conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
    needs_title = dst_conn.execute(
        f'SELECT COUNT(*) FROM "{table_name}" WHERE needs_title_translation = 1'
    ).fetchone()[0]
    needs_desc = dst_conn.execute(
        f'SELECT COUNT(*) FROM "{table_name}" WHERE needs_description_translation = 1'
    ).fetchone()[0]
    needs_skills = dst_conn.execute(
        f'SELECT COUNT(*) FROM "{table_name}" WHERE needs_skill_extraction = 1'
    ).fetchone()[0]

    print(
        f"{table_name}: {count:,} rows | "
        f"title={needs_title:,} | desc={needs_desc:,} | skills={needs_skills:,}"
    )


def main():
    if TARGET_DB.exists():
        TARGET_DB.unlink()

    src_conn = sqlite3.connect(SOURCE_DB)
    dst_conn = sqlite3.connect(TARGET_DB)
    dst_conn.execute("PRAGMA journal_mode = WAL")
    dst_conn.execute("PRAGMA synchronous = OFF")
    tables = get_source_tables(src_conn)

    print(f"Source: {SOURCE_DB}")
    print(f"Target: {TARGET_DB}\n")

    for table_name in tables:
        build_table(src_conn, dst_conn, table_name)
        dst_conn.commit()

    src_conn.close()
    dst_conn.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
