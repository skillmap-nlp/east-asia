"""
1. Merge title & description translations from gemma_results/ JSONL into
   east_asia_job_ads_gemma.db (UPDATE rows that don't yet have a translation).
2. Export JSONL per country in the same format as gemma_results but with ALL
   translated records (old DB translations + newly merged ones).
"""

import json, sqlite3, time
from pathlib import Path

BASE      = Path(__file__).parent
DB_PATH   = BASE / "east_asia_job_ads_gemma.db"
RESULTS   = BASE / "gemma_results"
EXPORT    = BASE / "gemma_results_full"

TABLES = [
    "jobads_in", "jobads_jp", "jobads_kr", "jobads_malaysia", "jobads_mx",
    "jobads_ph", "jobads_pl", "jobads_sg", "jobads_th", "jobads_tw", "jobads_vn",
]

BATCH = 10_000


# ── Part 1: merge JSONL → DB ────────────────────────────────────────────────

def ensure_id_index(conn: sqlite3.Connection, table: str):
    idx_name = f"idx_{table}_id"
    conn.execute(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{table}" ("id")')
    conn.commit()


def merge_via_temp(conn: sqlite3.Connection, table: str, jsonl_path: Path,
                   target_col: str, label: str):
    if not jsonl_path.exists():
        print(f"  {table} {label}: no JSONL file", flush=True)
        return

    t0 = time.time()
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS _tmp_trans")
    cur.execute("CREATE TEMP TABLE _tmp_trans (id INTEGER PRIMARY KEY, value TEXT)")

    n = 0
    batch: list[tuple] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            val = rec.get("value", "")
            if val and str(val).strip():
                batch.append((int(rec["id"]), str(val).strip()))
                n += 1
            if len(batch) >= BATCH:
                cur.executemany(
                    "INSERT OR IGNORE INTO _tmp_trans (id, value) VALUES (?, ?)",
                    batch,
                )
                batch.clear()
    if batch:
        cur.executemany(
            "INSERT OR IGNORE INTO _tmp_trans (id, value) VALUES (?, ?)",
            batch,
        )

    print(f"  {table} {label}: {n:,} from JSONL", end="", flush=True)

    cur.execute(
        f'UPDATE "{table}" SET {target_col} = '
        f'(SELECT value FROM _tmp_trans WHERE _tmp_trans.id = "{table}".id) '
        f'WHERE id IN (SELECT id FROM _tmp_trans) '
        f'AND ({target_col} IS NULL OR TRIM({target_col}) = "")'
    )
    updated = cur.rowcount
    conn.commit()
    cur.execute("DROP TABLE IF EXISTS _tmp_trans")

    print(f" → {updated:,} new rows updated ({time.time()-t0:.1f}s)", flush=True)


def merge_translations(conn: sqlite3.Connection, table: str):
    merge_via_temp(conn, table, RESULTS / f"{table}_title.jsonl",
                   "job_title_english", "titles")
    merge_via_temp(conn, table, RESULTS / f"{table}_desc.jsonl",
                   "description_english", "descs ")


# ── Part 2: export full JSONL ────────────────────────────────────────────────

def export_full_jsonl(conn: sqlite3.Connection, table: str):
    EXPORT.mkdir(exist_ok=True)
    cur = conn.cursor()

    skill_filter = "AND needs_skill_extraction != 0"

    # Titles
    cur.execute(
        f'SELECT id, job_title_english FROM "{table}" '
        f'WHERE job_title_english IS NOT NULL AND TRIM(job_title_english) != "" '
        f'{skill_filter} '
        f'ORDER BY id'
    )
    title_out = EXPORT / f"{table}_title.jsonl"
    n = 0
    with open(title_out, "w", encoding="utf-8") as f:
        for row_id, val in cur:
            f.write(json.dumps({"id": row_id, "value": val}, ensure_ascii=False) + "\n")
            n += 1
    print(f"  {table} titles: {n:,} → {title_out.name}", flush=True)

    # Descriptions
    cur.execute(
        f'SELECT id, description_english FROM "{table}" '
        f'WHERE description_english IS NOT NULL AND TRIM(description_english) != "" '
        f'{skill_filter} '
        f'ORDER BY id'
    )
    desc_out = EXPORT / f"{table}_desc.jsonl"
    n = 0
    with open(desc_out, "w", encoding="utf-8") as f:
        for row_id, val in cur:
            f.write(json.dumps({"id": row_id, "value": val}, ensure_ascii=False) + "\n")
            n += 1
    print(f"  {table} descs:  {n:,} → {desc_out.name}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")

    # ── Step 1: merge ──
    print("=" * 60)
    print("STEP 1: Merging gemma_results → DB")
    print("=" * 60)
    t_all = time.time()
    print("Creating id indexes...", flush=True)
    for table in TABLES:
        ensure_id_index(conn, table)
    print("Indexes ready.\n", flush=True)
    for table in TABLES:
        merge_translations(conn, table)
    print(f"\nMerge done in {time.time()-t_all:.1f}s\n")

    # Post-merge stats
    print("─" * 60)
    print("Post-merge translation coverage:")
    print(f"{'Table':20s} {'total':>10s} {'title_eng':>10s} {'desc_eng':>10s}")
    print("─" * 60)
    for table in TABLES:
        total = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        titles = conn.execute(
            f'SELECT COUNT(*) FROM "{table}" '
            f'WHERE job_title_english IS NOT NULL AND TRIM(job_title_english) != ""'
        ).fetchone()[0]
        descs = conn.execute(
            f'SELECT COUNT(*) FROM "{table}" '
            f'WHERE description_english IS NOT NULL AND TRIM(description_english) != ""'
        ).fetchone()[0]
        print(f"{table:20s} {total:10,d} {titles:10,d} {descs:10,d}")

    # ── Step 2: export full JSONL ──
    print("\n" + "=" * 60)
    print("STEP 2: Exporting full JSONL → gemma_results_full/")
    print("=" * 60)
    t_all = time.time()
    for table in TABLES:
        export_full_jsonl(conn, table)
    print(f"\nExport done in {time.time()-t_all:.1f}s")

    conn.close()


if __name__ == "__main__":
    main()
