#!/usr/bin/env python3
"""
Translate job_title and responsibilities to English for jobads_jp
using OpenAI Batch API (gpt-5-mini). Updates only empty English fields.
"""
import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

DB_PATH = "/Users/michalpalinski/Desktop/east_asia/east_asia_2026/east_asia_job_ads.db"
TABLE = "jobads_jp"
MODEL = "gpt-5-mini"
LIMIT = 200

TITLE_COL = "job_title"
RESP_COL = "responsibilities"
TITLE_LANG_COL = "job_title_lang"
RESP_LANG_COL = "responsibilities_lang"
TITLE_EN_COL = "job_title_english"
RESP_EN_COL = "description_english"

OUT_DIR = Path("/Users/michalpalinski/Desktop/east_asia/east_asia_2026")
JSONL_PATH = OUT_DIR / "batch_translate_jobads_jp.jsonl"
OUTPUT_JSONL_PATH = OUT_DIR / "batch_translate_jobads_jp_output.jsonl"
ERROR_JSONL_PATH = OUT_DIR / "batch_translate_jobads_jp_error.jsonl"


def get_api_key():
    load_dotenv()
    load_dotenv("/Users/michalpalinski/Desktop/east_asia/.env")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found in environment.")
    return key


def select_rows():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT id, {TITLE_COL}, {RESP_COL}, {TITLE_LANG_COL}, {RESP_LANG_COL},
               {TITLE_EN_COL}, {RESP_EN_COL}
        FROM {TABLE}
        WHERE
            (
                {TITLE_LANG_COL} IS NULL OR {TITLE_LANG_COL} != 'EN'
                OR {RESP_LANG_COL} IS NULL OR {RESP_LANG_COL} != 'EN'
            )
            AND (
                {TITLE_EN_COL} IS NULL OR TRIM({TITLE_EN_COL}) = ''
                OR {RESP_EN_COL} IS NULL OR TRIM({RESP_EN_COL}) = ''
            )
        LIMIT ?
        """,
        (LIMIT,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def build_requests(rows):
    lines = []
    for row in rows:
        row_id, title, resp, title_lang, resp_lang, title_en, resp_en = row
        title = (title or "").strip()
        resp = (resp or "").strip()

        if (title_en is None or not str(title_en).strip()) and title:
            system_prompt = (
                "Translate the job title to English. "
                "Return ONLY valid JSON with key job_title_english. "
                "If the title is already English, return it unchanged."
            )
            user_payload = {
                "job_title": title,
                "job_title_lang": title_lang,
            }
            payload = {
                "model": MODEL,
                "text": {"format": {"type": "json_object"}},
                "input": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
            }
            lines.append(
                json.dumps(
                    {
                        "custom_id": f"{TABLE}:{row_id}:title",
                        "method": "POST",
                        "url": "/v1/responses",
                        "body": payload,
                    }
                )
            )

        if (resp_en is None or not str(resp_en).strip()) and resp:
            system_prompt = (
                "Translate the job description/responsibilities to English. "
                "Return ONLY valid JSON with key description_english. "
                "If the text is already English, return it unchanged."
            )
            user_payload = {
                "responsibilities": resp,
                "responsibilities_lang": resp_lang,
            }
            payload = {
                "model": MODEL,
                "text": {"format": {"type": "json_object"}},
                "input": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
            }
            lines.append(
                json.dumps(
                    {
                        "custom_id": f"{TABLE}:{row_id}:resp",
                        "method": "POST",
                        "url": "/v1/responses",
                        "body": payload,
                    }
                )
            )
    return lines


def write_jsonl(lines):
    JSONL_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def upload_file(api_key):
    with open(JSONL_PATH, "rb") as f:
        resp = requests.post(
            "https://api.openai.com/v1/files",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": f},
            data={"purpose": "batch"},
            timeout=60,
        )
    resp.raise_for_status()
    return resp.json()["id"]


def create_batch(api_key, file_id):
    payload = {
        "input_file_id": file_id,
        "endpoint": "/v1/responses",
        "completion_window": "24h",
    }
    resp = requests.post(
        "https://api.openai.com/v1/batches",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def poll_batch(api_key, batch_id):
    while True:
        resp = requests.get(
            f"https://api.openai.com/v1/batches/{batch_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        status = data["status"]
        print(f"  Batch status: {status}")
        if status in {"completed", "failed", "cancelled", "expired"}:
            return data
        time.sleep(20)


def download_output(api_key, file_id, path):
    resp = requests.get(
        f"https://api.openai.com/v1/files/{file_id}/content",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    resp.raise_for_status()
    path.write_bytes(resp.content)


def parse_output():
    results = {}
    with open(OUTPUT_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            custom_id = item.get("custom_id")
            response = item.get("response", {})
            output_text = None

            if "body" in response:
                body = response["body"]
                if isinstance(body, dict):
                    output = body.get("output", [])
                    if output:
                        content = output[0].get("content", [])
                        if content:
                            output_text = content[0].get("text")

            if not output_text:
                continue
            try:
                parsed = json.loads(output_text)
            except json.JSONDecodeError:
                continue
            results[custom_id] = parsed
    return results


def update_db(results):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    updated = 0
    for custom_id, payload in results.items():
        parts = custom_id.split(":")
        if len(parts) != 3:
            continue
        _, row_id, kind = parts

        if kind == "title":
            title_en = (payload.get("job_title_english") or "").strip()
            if title_en:
                cur.execute(
                    f"""
                    UPDATE {TABLE}
                    SET {TITLE_EN_COL} = COALESCE(NULLIF(TRIM({TITLE_EN_COL}), ''), ?)
                    WHERE id = ?
                    """,
                    (title_en, row_id),
                )
                updated += 1
        elif kind == "resp":
            resp_en = (payload.get("description_english") or "").strip()
            if resp_en:
                cur.execute(
                    f"""
                    UPDATE {TABLE}
                    SET {RESP_EN_COL} = COALESCE(NULLIF(TRIM({RESP_EN_COL}), ''), ?)
                    WHERE id = ?
                    """,
                    (resp_en, row_id),
                )
                updated += 1
    conn.commit()
    conn.close()
    return updated


def main():
    print("=" * 80)
    print("BATCH TRANSLATE TO ENGLISH (JOBADS_JP)")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    rows = select_rows()
    if not rows:
        print("No rows to translate.")
        return

    lines = build_requests(rows)
    write_jsonl(lines)
    print(f"Prepared {len(lines)} requests.")

    api_key = get_api_key()
    file_id = upload_file(api_key)
    print(f"Uploaded batch file: {file_id}")

    batch_id = create_batch(api_key, file_id)
    print(f"Created batch: {batch_id}")

    batch_info = poll_batch(api_key, batch_id)
    status = batch_info["status"]
    if status != "completed":
        print(f"Batch ended with status: {status}")
        return

    output_file_id = batch_info.get("output_file_id")
    error_file_id = batch_info.get("error_file_id")

    if not output_file_id:
        print("Batch completed but output_file_id is missing.")
        if error_file_id:
            print(f"Downloading error file: {error_file_id}")
            download_output(api_key, error_file_id, ERROR_JSONL_PATH)
            print(f"Saved error file to: {ERROR_JSONL_PATH}")
        else:
            print(f"Batch info: {json.dumps(batch_info, indent=2)}")
        return

    download_output(api_key, output_file_id, OUTPUT_JSONL_PATH)
    results = parse_output()
    updated = update_db(results)
    print(f"Updated rows: {updated}")

    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
