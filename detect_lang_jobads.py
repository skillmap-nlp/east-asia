#!/usr/bin/env python3
"""
Detect language for job_title and responsibilities in all jobads_* tables.
Stores results in job_title_lang and responsibilities_lang columns.
"""
import sqlite3
import time
from datetime import datetime

from lingua import LanguageDetectorBuilder

TARGET_DB = "/Users/michalpalinski/Desktop/east_asia/east_asia_2026/east_asia_job_ads.db"

TABLES = [
    "jobads_jp",
    "jobads_kr",
    "jobads_th",
    "jobads_malaysia",
    "jobads_mx",
    "jobads_pl",
    "jobads_tw",
    "jobads_ph",
    "jobads_cl",
    "jobads_vn",
    "jobads_id",
    "jobads_sg",
    "jobads_in",
]

TITLE_COL = "job_title"
RESP_COL = "responsibilities"
TITLE_LANG_COL = "job_title_lang"
RESP_LANG_COL = "responsibilities_lang"

BATCH_SIZE = 1000
PROGRESS_EVERY = 1000
MAX_TEXT_LEN = 300


def ensure_columns(conn, table_name):
    cursor = conn.cursor()
    cursor.execute(f'PRAGMA table_info("{table_name}")')
    existing = {row[1] for row in cursor.fetchall()}
    if TITLE_LANG_COL not in existing:
        cursor.execute(
            f'ALTER TABLE "{table_name}" ADD COLUMN "{TITLE_LANG_COL}" TEXT'
        )
    if RESP_LANG_COL not in existing:
        cursor.execute(
            f'ALTER TABLE "{table_name}" ADD COLUMN "{RESP_LANG_COL}" TEXT'
        )
    commit_with_retry(conn)


def normalize_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return text[:MAX_TEXT_LEN]


def detect_lang(detector, text):
    if not text:
        return None
    language = detector.detect_language_of(text)
    if language is None:
        return None
    iso = language.iso_code_639_1
    if iso is not None:
        return iso.name
    return language.iso_code_639_3.name


def commit_with_retry(conn, retries=10, base_sleep=0.5):
    for attempt in range(retries):
        try:
            conn.commit()
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower():
                raise
            time.sleep(base_sleep * (attempt + 1))
    conn.commit()


def executemany_with_retry(cursor, query, params, conn, retries=10, base_sleep=0.5):
    for attempt in range(retries):
        try:
            cursor.executemany(query, params)
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower():
                raise
            time.sleep(base_sleep * (attempt + 1))
    cursor.executemany(query, params)


def needs_update(value):
    """Return True if the lang column value is missing (NULL or empty string)."""
    return value is None or str(value).strip() == ""


def process_table(conn, detector, table_name):
    read_cursor = conn.cursor()
    write_cursor = conn.cursor()

    read_cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
    total_rows = read_cursor.fetchone()[0]

    # Count only rows that actually need processing
    read_cursor.execute(
        f'''
        SELECT COUNT(*) FROM "{table_name}"
        WHERE ("{TITLE_LANG_COL}" IS NULL OR TRIM("{TITLE_LANG_COL}") = '')
           OR ("{RESP_LANG_COL}"  IS NULL OR TRIM("{RESP_LANG_COL}")  = '')
        '''
    )
    pending_rows = read_cursor.fetchone()[0]

    print(f"  Total rows:   {total_rows:,}")
    print(f"  Pending rows: {pending_rows:,}")

    if pending_rows == 0:
        print("  Nothing to do — all rows already have lang values.")
        return

    print("  Detecting languages...")

    processed = 0
    updated = 0

    update_sql = (
        f'UPDATE "{table_name}" SET "{TITLE_LANG_COL}" = ?, '
        f'"{RESP_LANG_COL}" = ? WHERE id = ?'
    )

    last_id = 0
    while True:
        read_cursor.execute(
            f'''
            SELECT id,
                   "{TITLE_COL}",      "{RESP_COL}",
                   "{TITLE_LANG_COL}", "{RESP_LANG_COL}"
            FROM "{table_name}"
            WHERE id > ?
              AND (
                    ("{TITLE_LANG_COL}" IS NULL OR TRIM("{TITLE_LANG_COL}") = '')
                 OR ("{RESP_LANG_COL}"  IS NULL OR TRIM("{RESP_LANG_COL}")  = '')
              )
            ORDER BY id
            LIMIT ?
            ''',
            (last_id, BATCH_SIZE),
        )
        rows = read_cursor.fetchall()
        if not rows:
            break

        batch = []
        for row_id, title, resp, existing_title_lang, existing_resp_lang in rows:
            # Only detect when the column is missing; keep existing values otherwise
            if needs_update(existing_title_lang):
                title_lang = detect_lang(detector, normalize_text(title))
            else:
                title_lang = existing_title_lang

            if needs_update(existing_resp_lang):
                resp_lang = detect_lang(detector, normalize_text(resp))
            else:
                resp_lang = existing_resp_lang

            batch.append((title_lang, resp_lang, row_id))

        executemany_with_retry(write_cursor, update_sql, batch, conn)
        commit_with_retry(conn)

        processed += len(rows)
        updated += len(rows)
        last_id = rows[-1][0]

        if processed % PROGRESS_EVERY == 0:
            print(f"  Processed: {processed:,} / {pending_rows:,}")

    print(f"  Processed: {processed:,}")
    print(f"  Updated:   {updated:,}")


def main():
    print("=" * 80)
    print("LANGUAGE DETECTION (LINGUA) FOR JOB TITLES & RESPONSIBILITIES")
    print("=" * 80)
    print(f"Database: {TARGET_DB}")
    print(f"Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    detector = (
        LanguageDetectorBuilder.from_all_languages()
        .with_low_accuracy_mode()
        .build()
    )

    conn = sqlite3.connect(TARGET_DB, timeout=60)
    conn.execute("PRAGMA busy_timeout = 60000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    try:
        for table_name in TABLES:
            print("\n" + "─" * 80)
            print(f"Table: {table_name}")
            print("─" * 80)
            ensure_columns(conn, table_name)
            process_table(conn, detector, table_name)
    finally:
        conn.close()

    print("\n" + "=" * 80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
