#!/usr/bin/env python3
"""
Translate job_title and responsibilities to English for ALL jobads_* tables.
Uses OpenAI Batch API (gpt-5.4-nano).

Workflow:
  1. Copy fields where lang is already EN (free, no API call).
  2. Build JSONL batch files for non-EN rows needing translation.
  3. Upload → create batch → poll → download results.
  4. Parse result JSONs and update DB.

Run:  python translate_all_batch.py           # full pipeline
      python translate_all_batch.py --apply   # only apply already-downloaded JSONs
"""
import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")
load_dotenv(SCRIPT_DIR.parent / ".env", override=False)

DB_PATH = str(SCRIPT_DIR / "east_asia_job_ads.db")
MODEL = "gpt-5.4-nano"
MAX_REQUESTS_PER_BATCH = 40_000  # OpenAI limit is 50K; leave headroom
OUT_DIR = SCRIPT_DIR / "batch_translate_output"

TABLES = [
    "jobads_cl", "jobads_id", "jobads_in", "jobads_jp", "jobads_kr",
    "jobads_malaysia", "jobads_mx", "jobads_ph", "jobads_pl",
    "jobads_sg", "jobads_th", "jobads_tw", "jobads_vn",
]

TITLE_COL = "job_title"
RESP_COL = "responsibilities"
TITLE_LANG = "job_title_lang"
RESP_LANG = "responsibilities_lang"
TITLE_EN = "job_title_english"
RESP_EN = "description_english"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    return key


def api_headers(api_key):
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA busy_timeout = 60000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


# ---------------------------------------------------------------------------
# Step 1 — copy already-English fields
# ---------------------------------------------------------------------------

def copy_english_fields():
    """For rows where lang=EN but English column is empty, copy the original."""
    conn = get_conn()
    cur = conn.cursor()
    total_copied = 0

    for table in TABLES:
        # title
        cur.execute(f'''
            UPDATE "{table}"
            SET "{TITLE_EN}" = "{TITLE_COL}"
            WHERE ("{TITLE_EN}" IS NULL OR TRIM("{TITLE_EN}") = '')
              AND "{TITLE_LANG}" = 'EN'
              AND "{TITLE_COL}" IS NOT NULL AND TRIM("{TITLE_COL}") != ''
        ''')
        t_count = cur.rowcount

        # resp
        cur.execute(f'''
            UPDATE "{table}"
            SET "{RESP_EN}" = "{RESP_COL}"
            WHERE ("{RESP_EN}" IS NULL OR TRIM("{RESP_EN}") = '')
              AND "{RESP_LANG}" = 'EN'
              AND "{RESP_COL}" IS NOT NULL AND TRIM("{RESP_COL}") != ''
        ''')
        r_count = cur.rowcount

        conn.commit()
        if t_count or r_count:
            print(f"  {table}: copied {t_count} titles, {r_count} descriptions (already EN)")
        total_copied += t_count + r_count

    conn.close()
    return total_copied


# ---------------------------------------------------------------------------
# Step 2 — build JSONL batch files
# ---------------------------------------------------------------------------

def collect_rows_needing_translation():
    """Return list of (table, id, field, text) for rows that need API translation."""
    conn = get_conn()
    cur = conn.cursor()
    items = []

    for table in TABLES:
        cur.execute(f'''
            SELECT id, "{TITLE_COL}", "{RESP_COL}",
                   "{TITLE_LANG}", "{RESP_LANG}",
                   "{TITLE_EN}", "{RESP_EN}"
            FROM "{table}"
            WHERE ("{TITLE_EN}" IS NULL OR TRIM("{TITLE_EN}") = '')
               OR ("{RESP_EN}"  IS NULL OR TRIM("{RESP_EN}")  = '')
        ''')
        for row_id, title, resp, tl, rl, te, re in cur:
            title = (title or "").strip()
            resp = (resp or "").strip()
            te = (te or "").strip()
            re = (re or "").strip()

            if not te and title and tl != "EN":
                items.append((table, row_id, "title", title, tl))
            if not re and resp and rl != "EN":
                items.append((table, row_id, "resp", resp, rl))

    conn.close()
    return items


def make_request_line(table, row_id, field, text, lang):
    if field == "title":
        system = (
            "Translate the job title to English. "
            "Return ONLY valid JSON: {\"job_title_english\": \"...\"}. "
            "If already English, return it unchanged."
        )
        user_payload = {"job_title": text, "detected_lang": lang}
    else:
        system = (
            "Translate the job description to English. "
            "Return ONLY valid JSON: {\"description_english\": \"...\"}. "
            "If already English, return it unchanged."
        )
        user_payload = {"responsibilities": text, "detected_lang": lang}

    return json.dumps({
        "custom_id": f"{table}:{row_id}:{field}",
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": MODEL,
            "text": {"format": {"type": "json_object"}},
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        },
    }, ensure_ascii=False)


def write_batch_files(items):
    """Split items into JSONL files of MAX_REQUESTS_PER_BATCH and save."""
    OUT_DIR.mkdir(exist_ok=True)
    paths = []
    for i in range(0, len(items), MAX_REQUESTS_PER_BATCH):
        chunk = items[i : i + MAX_REQUESTS_PER_BATCH]
        batch_num = i // MAX_REQUESTS_PER_BATCH
        path = OUT_DIR / f"batch_input_{batch_num:03d}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for table, row_id, field, text, lang in chunk:
                f.write(make_request_line(table, row_id, field, text, lang) + "\n")
        paths.append(path)
        print(f"  Wrote {path.name} ({len(chunk):,} requests)")
    return paths


# ---------------------------------------------------------------------------
# Step 3 — upload, create batch, poll, download
# ---------------------------------------------------------------------------

def upload_file(api_key, path):
    with open(path, "rb") as f:
        r = requests.post(
            "https://api.openai.com/v1/files",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": f},
            data={"purpose": "batch"},
            timeout=120,
        )
    r.raise_for_status()
    return r.json()["id"]


def create_batch(api_key, file_id):
    r = requests.post(
        "https://api.openai.com/v1/batches",
        headers=api_headers(api_key),
        json={"input_file_id": file_id, "endpoint": "/v1/responses", "completion_window": "24h"},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["id"]


def poll_batch(api_key, batch_id):
    while True:
        r = requests.get(
            f"https://api.openai.com/v1/batches/{batch_id}",
            headers=api_headers(api_key),
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        status = data["status"]
        completed = data.get("request_counts", {}).get("completed", "?")
        total = data.get("request_counts", {}).get("total", "?")
        print(f"    status={status}  completed={completed}/{total}")
        if status in {"completed", "failed", "cancelled", "expired"}:
            return data
        time.sleep(30)


def download_file(api_key, file_id, dest):
    r = requests.get(
        f"https://api.openai.com/v1/files/{file_id}/content",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=120,
    )
    r.raise_for_status()
    dest.write_bytes(r.content)


def submit_and_poll_batches(input_paths):
    """Submit each JSONL, poll, and download output. Returns list of output paths."""
    api_key = get_api_key()
    output_paths = []

    # Save batch state so we can resume if interrupted
    state_file = OUT_DIR / "batch_state.json"
    state = {}
    if state_file.exists():
        state = json.loads(state_file.read_text())

    for path in input_paths:
        key = path.name
        out_path = path.with_name(path.stem + "_output.jsonl")

        # Already downloaded?
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"  {out_path.name} already exists — skipping")
            output_paths.append(out_path)
            continue

        # Already submitted?
        batch_id = state.get(key, {}).get("batch_id")
        if not batch_id:
            print(f"  Uploading {key}...")
            file_id = upload_file(api_key, path)
            print(f"    file_id={file_id}")
            batch_id = create_batch(api_key, file_id)
            print(f"    batch_id={batch_id}")
            state[key] = {"file_id": file_id, "batch_id": batch_id}
            state_file.write_text(json.dumps(state, indent=2))
        else:
            print(f"  Resuming batch for {key}: {batch_id}")

        print(f"  Polling {batch_id}...")
        info = poll_batch(api_key, batch_id)
        if info["status"] != "completed":
            print(f"  ⚠ Batch {batch_id} ended with status={info['status']}")
            err_fid = info.get("error_file_id")
            if err_fid:
                err_path = path.with_name(path.stem + "_error.jsonl")
                download_file(api_key, err_fid, err_path)
                print(f"    Error file: {err_path}")
            continue

        out_fid = info.get("output_file_id")
        if out_fid:
            download_file(api_key, out_fid, out_path)
            print(f"  ✓ Downloaded {out_path.name}")
            output_paths.append(out_path)

    return output_paths


# ---------------------------------------------------------------------------
# Step 4 — parse result JSONs and write to DB
# ---------------------------------------------------------------------------

def parse_output_files(output_paths):
    """Parse all output JSONL files and return {custom_id: parsed_json}."""
    results = {}
    for path in output_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                cid = item.get("custom_id")
                response = item.get("response", {})
                body = response.get("body", {})
                output_list = body.get("output", []) if isinstance(body, dict) else []
                text = None
                for out_item in output_list:
                    for c in out_item.get("content", []):
                        if c.get("text"):
                            text = c["text"]
                            break
                    if text:
                        break
                if not text:
                    continue
                try:
                    results[cid] = json.loads(text)
                except json.JSONDecodeError:
                    pass
    return results


def apply_results(results):
    """Write parsed translations back to the database."""
    conn = get_conn()
    cur = conn.cursor()
    updated_title = 0
    updated_resp = 0

    for cid, payload in results.items():
        parts = cid.split(":")
        if len(parts) != 3:
            continue
        table, row_id, kind = parts

        if kind == "title":
            val = (payload.get("job_title_english") or "").strip()
            if val:
                cur.execute(f'''
                    UPDATE "{table}" SET "{TITLE_EN}" = ?
                    WHERE id = ? AND ("{TITLE_EN}" IS NULL OR TRIM("{TITLE_EN}") = '')
                ''', (val, row_id))
                updated_title += cur.rowcount
        elif kind == "resp":
            val = (payload.get("description_english") or "").strip()
            if val:
                cur.execute(f'''
                    UPDATE "{table}" SET "{RESP_EN}" = ?
                    WHERE id = ? AND ("{RESP_EN}" IS NULL OR TRIM("{RESP_EN}") = '')
                ''', (val, row_id))
                updated_resp += cur.rowcount

        if (updated_title + updated_resp) % 5000 == 0:
            conn.commit()

    conn.commit()
    conn.close()
    return updated_title, updated_resp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Skip batch submission; only apply already-downloaded output JSONLs")
    args = parser.parse_args()

    print("=" * 70)
    print("BATCH TRANSLATE TO ENGLISH — ALL TABLES")
    print(f"Model: {MODEL}")
    print(f"DB:    {DB_PATH}")
    print(f"Time:  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    # --- Step 1: copy EN fields ---
    print("\n[Step 1] Copying already-English fields...")
    copied = copy_english_fields()
    print(f"  Total copied: {copied:,}\n")

    if args.apply:
        # Only apply already-downloaded results
        output_paths = sorted(OUT_DIR.glob("*_output.jsonl"))
        if not output_paths:
            print("No output JSONL files found. Run without --apply first.")
            return
        print(f"[Step 4] Applying {len(output_paths)} output file(s)...")
        results = parse_output_files(output_paths)
        print(f"  Parsed {len(results):,} results")
        t, r = apply_results(results)
        print(f"  Updated: {t:,} titles, {r:,} descriptions")
        print(f"\nDone: {datetime.now():%Y-%m-%d %H:%M:%S}")
        return

    # --- Step 2: collect & build JSONL ---
    print("[Step 2] Collecting rows needing translation...")
    items = collect_rows_needing_translation()
    print(f"  Total translation requests: {len(items):,}")

    if not items:
        print("  Nothing to translate!")
        return

    print("\n  Writing batch JSONL files...")
    input_paths = write_batch_files(items)

    # --- Step 3: submit, poll, download ---
    print(f"\n[Step 3] Submitting {len(input_paths)} batch(es) to OpenAI...")
    output_paths = submit_and_poll_batches(input_paths)

    # --- Step 4: apply results ---
    if output_paths:
        print(f"\n[Step 4] Applying {len(output_paths)} output file(s)...")
        results = parse_output_files(output_paths)
        print(f"  Parsed {len(results):,} results")
        t, r = apply_results(results)
        print(f"  Updated: {t:,} titles, {r:,} descriptions")

    print(f"\nDone: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
