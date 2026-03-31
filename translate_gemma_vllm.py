#!/usr/bin/env python3
"""
Translate job descriptions to English using TranslateGemma 4B via vLLM.

Optimised for NVIDIA A40 (48 GB) + TranslateGemma 4B (~8 GB weights):
  • max_model_len=8192 → 99.92% descriptions translated in full
  • prefix caching  – shared prompt prefix computed once per language
  • greedy decoding (temperature=0)
  • sorted by (language, length) – maximises prefix-cache hits
  • tokenizer-aware truncation – only true outliers (~0.08%) are shortened

Usage:
    python translate_gemma_vllm.py                     # full pipeline (desc only)
    python translate_gemma_vllm.py --apply-only        # write JSONL → DB
    python translate_gemma_vllm.py --dry-run            # count pending rows
    python translate_gemma_vllm.py --table jobads_jp    # single table
    python translate_gemma_vllm.py --reset-desc         # clear old desc results & re-run
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path
from datetime import datetime

# ─── Config ──────────────────────────────────────────────────────────────────

DB_PATH          = Path("data/jobs_ea_gemma.db")
OUTPUT_DIR       = Path("gemma_results")
MODEL_ID         = "google/translategemma-4b-it"
MAX_MODEL_LEN    = 8192
DESC_MAX_TOKENS  = 2048
DEFAULT_CHUNK    = 50_000

TABLES = [
    "jobads_in", "jobads_jp", "jobads_kr", "jobads_malaysia",
    "jobads_mx", "jobads_ph", "jobads_pl", "jobads_sg",
    "jobads_th", "jobads_tw", "jobads_vn",
]

# ─── Language mappings ────────────────────────────────────────────────────────

TABLE_LANG = {
    "jobads_in": ("Hindi", "hi"),
    "jobads_jp": ("Japanese", "ja"),
    "jobads_kr": ("Korean", "ko"),
    "jobads_malaysia": ("Malay", "ms"),
    "jobads_mx": ("Spanish", "es"),
    "jobads_ph": ("Filipino", "fil"),
    "jobads_pl": ("Polish", "pl"),
    "jobads_sg": ("English", "en"),
    "jobads_th": ("Thai", "th"),
    "jobads_tw": ("Chinese", "zh-TW"),
    "jobads_vn": ("Vietnamese", "vi"),
    "jobads_cl": ("Spanish", "es"),
    "jobads_id": ("Indonesian", "id"),
}

LANG_MAP = {
    "ja": ("Japanese", "ja"),     "ko": ("Korean", "ko"),
    "zh": ("Chinese", "zh-CN"),   "vi": ("Vietnamese", "vi"),
    "th": ("Thai", "th"),         "id": ("Indonesian", "id"),
    "ms": ("Malay", "ms"),        "tl": ("Filipino", "fil"),
    "hi": ("Hindi", "hi"),        "mr": ("Marathi", "mr"),
    "ta": ("Tamil", "ta"),        "te": ("Telugu", "te"),
    "bn": ("Bangla", "bn"),       "gu": ("Gujarati", "gu"),
    "kn": ("Kannada", "kn"),      "ml": ("Malayalam", "ml"),
    "pa": ("Punjabi", "pa"),      "ur": ("Urdu", "ur"),
    "es": ("Spanish", "es"),      "pl": ("Polish", "pl"),
    "de": ("German", "de"),       "fr": ("French", "fr"),
    "pt": ("Portuguese", "pt-BR"),"it": ("Italian", "it"),
    "nl": ("Dutch", "nl"),        "ru": ("Russian", "ru"),
    "tr": ("Turkish", "tr"),      "ar": ("Arabic", "ar"),
    "en": ("English", "en"),
}


def _resolve_lang(lang_code, table: str):
    if lang_code:
        s = str(lang_code).strip().lower()
        if s in LANG_MAP:
            return LANG_MAP[s]
    return TABLE_LANG.get(table, ("English", "en"))


# ─── Prompt (official TranslateGemma format, arxiv 2601.09012 Figure 3) ──────

def _prompt_prefix(src_name: str, src_code: str) -> str:
    tgt_name, tgt_code = "English", "en-US"
    return (
        "<bos><start_of_turn>user\n"
        f"You are a professional {src_name} ({src_code}) to {tgt_name} "
        f"({tgt_code}) translator. Your goal is to accurately convey the "
        f"meaning and nuances of the original {src_name} text while adhering "
        f"to {tgt_name} grammar, vocabulary, and cultural sensitivities. "
        f"Produce only the {tgt_name} translation, without any additional "
        f"explanations or commentary. Please translate the following "
        f"{src_name} text into {tgt_name}:\n\n\n"
    )

_PROMPT_SUFFIX = "<end_of_turn>\n<start_of_turn>model\n"


def make_prompt(text: str, source_lang, table: str) -> str:
    src_name, src_code = _resolve_lang(source_lang, table)
    return _prompt_prefix(src_name, src_code) + text + _PROMPT_SUFFIX


# ─── DB helpers ──────────────────────────────────────────────────────────────

def get_conn():
    conn = sqlite3.connect(str(DB_PATH), timeout=60)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def fetch_pending_descs(conn, table: str):
    return conn.execute(
        f"""SELECT id, responsibilities, responsibilities_lang
            FROM "{table}"
            WHERE needs_description_translation = 1
              AND responsibilities IS NOT NULL
              AND TRIM(responsibilities) != '' """
    ).fetchall()


def count_pending(conn, table: str) -> int:
    r = conn.execute(
        f"""SELECT COUNT(*) FROM "{table}"
            WHERE needs_description_translation = 1
              AND responsibilities IS NOT NULL
              AND TRIM(responsibilities) != '' """
    ).fetchone()
    return r[0] or 0


# ─── Checkpoint helpers ──────────────────────────────────────────────────────

def _ckpt_path(table: str) -> Path:
    return OUTPUT_DIR / f"{table}_desc.jsonl"


def load_checkpoint(table: str) -> set:
    path = _ckpt_path(table)
    if not path.exists():
        return set()
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


def load_checkpoint_dict(table: str) -> dict:
    path = _ckpt_path(table)
    if not path.exists():
        return {}
    done = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done[obj["id"]] = obj["value"]
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def flush_checkpoint(path: Path, results: list[tuple[int, str]]):
    with open(path, "a", encoding="utf-8") as f:
        for row_id, value in results:
            f.write(json.dumps({"id": row_id, "value": value},
                               ensure_ascii=False) + "\n")


# ─── Truncation cache (persists across restarts) ─────────────────────────────

def _trunc_cache_path() -> Path:
    return OUTPUT_DIR / "truncation_cache.json"


def save_truncation_cache(truncated: dict[tuple, str], model_id: str):
    path = _trunc_cache_path()
    data = {
        "model": model_id,
        "truncated": {f"{t}\t{rid}": txt for (t, rid), txt in truncated.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"  Truncation cache saved ({len(truncated):,} entries)", flush=True)


def load_truncation_cache(model_id: str) -> dict[tuple, str] | None:
    path = _trunc_cache_path()
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("model") != model_id:
            print(f"  Truncation cache: model mismatch, will rebuild", flush=True)
            return None
        result = {}
        for key, txt in data["truncated"].items():
            t, rid = key.split("\t", 1)
            result[(t, int(rid))] = txt
        print(f"  Truncation cache loaded ({len(result):,} truncated texts)",
              flush=True)
        return result
    except Exception as e:
        print(f"  Truncation cache error ({e}), will rebuild", flush=True)
        return None


def delete_truncation_cache():
    path = _trunc_cache_path()
    if path.exists():
        path.unlink()
        print(f"  deleted {path.name}", flush=True)


# ─── Output parsing ──────────────────────────────────────────────────────────

def clean_translation(text: str) -> str | None:
    if not text:
        return None
    text = text.strip()
    for prefix in ("Translation:", "English:", "Translated:", "Result:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    for suffix in ("</s>", "<end_of_turn>", "<eos>"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    return text or None


# ─── Job collection ──────────────────────────────────────────────────────────

def collect_desc_jobs(tables: list[str]):
    """Collect (table, row_id, text, lang) for all pending descriptions."""
    conn = get_conn()
    jobs: list[tuple] = []

    for table in tables:
        done = load_checkpoint(table)
        rows = fetch_pending_descs(conn, table)
        n = 0
        for row_id, resp, resp_lang in rows:
            if row_id in done:
                continue
            r = (resp or "").strip()
            if r:
                jobs.append((table, row_id, r, resp_lang))
                n += 1
        print(f"  {table:20s}  descs={n:>8,}", flush=True)

    conn.close()
    jobs.sort(key=lambda j: (j[0], len(j[2])))
    return jobs


# ─── Tokenizer-aware prompt building with truncation ─────────────────────────

def build_prompts(jobs, tokenizer, model_id: str | None = None):
    """Build prompts, truncating only outlier texts that exceed token budget.

    Optimisations for fast restarts:
      1. Truncation cache — if model_id matches, previously-truncated texts
         are reused without touching the tokenizer.
      2. Char-length heuristic — texts shorter than token_budget//3 chars
         are guaranteed to fit (worst-case 3 tokens/char for CJK), so
         tokenisation is skipped entirely for those.
    """
    token_budget = MAX_MODEL_LEN - DESC_MAX_TOKENS - 16
    safe_chars = token_budget // 3

    trunc_cache = load_truncation_cache(model_id) if model_id else None

    overhead_cache: dict[tuple, int] = {}
    truncated_map: dict[tuple, str] = {}
    result = []
    n_truncated = 0
    n_fast = 0

    for table, row_id, text, lang in jobs:
        lang_key = _resolve_lang(lang, table)

        if trunc_cache is not None:
            cached = trunc_cache.get((table, row_id))
            if cached is not None:
                result.append((table, row_id, make_prompt(cached, lang, table)))
                n_truncated += 1
                continue

        if len(text) <= safe_chars:
            result.append((table, row_id, make_prompt(text, lang, table)))
            n_fast += 1
            continue

        if lang_key not in overhead_cache:
            prefix = _prompt_prefix(*lang_key) + _PROMPT_SUFFIX
            overhead_cache[lang_key] = len(tokenizer.encode(prefix))

        overhead = overhead_cache[lang_key]
        text_budget = token_budget - overhead

        text_tokens = tokenizer.encode(text, add_special_tokens=False)

        if len(text_tokens) > text_budget:
            text_tokens = text_tokens[:text_budget]
            text = tokenizer.decode(text_tokens, skip_special_tokens=True)
            n_truncated += 1
            truncated_map[(table, row_id)] = text

        result.append((table, row_id, make_prompt(text, lang, table)))

    pct = 100 * n_truncated / len(jobs) if jobs else 0
    fast_pct = 100 * n_fast / len(jobs) if jobs else 0
    print(f"  {len(result):,} prompts built, "
          f"{n_truncated:,} truncated ({pct:.2f}%), "
          f"{n_fast:,} skipped tokenisation ({fast_pct:.1f}%)",
          flush=True)

    if trunc_cache is None and model_id:
        save_truncation_cache(truncated_map, model_id)

    return result


# ─── Inference ────────────────────────────────────────────────────────────────

def run_inference(llm, jobs, sampling_params, chunk_size: int):
    total = len(jobs)
    if total == 0:
        return
    t0 = time.time()

    for start in range(0, total, chunk_size):
        chunk = jobs[start : start + chunk_size]
        prompts = [p for _, _, p in chunk]

        outputs = llm.generate(prompts, sampling_params)

        buf: dict[str, list[tuple[int, str]]] = {}
        for (table, row_id, _), out in zip(chunk, outputs):
            val = clean_translation(out.outputs[0].text)
            if val:
                buf.setdefault(table, []).append((row_id, val))

        for tbl, results in buf.items():
            flush_checkpoint(_ckpt_path(tbl), results)

        processed = min(start + chunk_size, total)
        elapsed = time.time() - t0
        speed = processed / elapsed if elapsed > 0 else 0
        eta_h = (total - processed) / speed / 3600 if speed > 0 else 0
        print(
            f"  {processed:>9,}/{total:,}  "
            f"({speed:,.0f} req/s, ETA {eta_h:.1f}h)",
            flush=True,
        )


# ─── Apply checkpoints → DB ─────────────────────────────────────────────────

def apply_checkpoints_to_db(tables: list[str]):
    conn = get_conn()
    conn.execute("PRAGMA synchronous = OFF")
    cur = conn.cursor()
    total = 0

    for table in tables:
        done = load_checkpoint_dict(table)
        if done:
            cur.executemany(
                f"""UPDATE "{table}"
                    SET description_english = ?,
                        needs_description_translation = 0
                    WHERE id = ?
                      AND (description_english IS NULL
                           OR TRIM(description_english) = '')""",
                [(v, k) for k, v in done.items()],
            )
            total += cur.rowcount
        conn.commit()
        print(f"  {table}: {len(done):,} descs applied", flush=True)

    conn.close()
    print(f"\nTotal applied: {total:,}", flush=True)


# ─── Reset descriptions for re-run ──────────────────────────────────────────

def reset_descriptions(tables: list[str]):
    """Reset only rows previously translated by this script (via checkpoints).

    Rows where needs_description_translation was already 0 for other reasons
    (e.g. text was already English) are NOT touched.
    """
    conn = get_conn()
    conn.execute("PRAGMA synchronous = OFF")
    cur = conn.cursor()
    total = 0

    for table in tables:
        p = _ckpt_path(table)
        done = load_checkpoint_dict(table)

        if done:
            cur.executemany(
                f"""UPDATE "{table}"
                    SET description_english = NULL,
                        needs_description_translation = 1
                    WHERE id = ?""",
                [(k,) for k in done.keys()],
            )
            n = cur.rowcount
            total += n
            print(f"  {table}: {n:,} rows reset (from checkpoint)", flush=True)

        if p.exists():
            p.unlink()
            print(f"  deleted {p.name}", flush=True)

    conn.commit()
    conn.close()
    print(f"\nTotal reset: {total:,} rows "
          f"(only previously translated by this script)", flush=True)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply-only", action="store_true",
                        help="Skip inference; write JSONL checkpoints → DB")
    parser.add_argument("--table", default=None,
                        help="Single table (e.g. jobads_jp)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count pending rows and exit")
    parser.add_argument("--model", default=MODEL_ID,
                        help=f"HuggingFace model (default: {MODEL_ID})")
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK,
                        help="Prompts per generate() call")
    parser.add_argument("--max-seqs", type=int, default=256,
                        help="Max concurrent sequences in vLLM (default: 256, "
                             "try 384 if 256 is stable)")
    parser.add_argument("--reset-desc", action="store_true",
                        help="Reset rows translated by this script + clear checkpoints")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete checkpoint files and start inference from scratch")
    args = parser.parse_args()

    chunk_size = args.chunk
    tables = [args.table] if args.table else TABLES

    print("=" * 65)
    print("TranslateGemma 4B  —  descriptions only  (A40)")
    print(f"Model      : {args.model}")
    print(f"DB         : {DB_PATH.resolve()}")
    print(f"Context    : {MAX_MODEL_LEN} tokens  (99.92% descs fit in full)")
    print(f"Chunk size : {chunk_size:,}")
    print(f"Max seqs   : {args.max_seqs}")
    print(f"Time       : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 65)

    # ── Reset / Fresh ────────────────────────────────────────────────────
    if args.reset_desc:
        print("\n[Resetting descriptions translated by this script]")
        reset_descriptions(tables)
        delete_truncation_cache()
    elif args.fresh:
        print("\n[Fresh start — clearing checkpoint files]")
        for table in tables:
            p = _ckpt_path(table)
            if p.exists():
                p.unlink()
                print(f"  deleted {p.name}", flush=True)
        delete_truncation_cache()
        print("  Checkpoints cleared. DB untouched (needs flags preserved).")

    # ── Dry-run ───────────────────────────────────────────────────────────
    if args.dry_run:
        conn = get_conn()
        total = 0
        for t in tables:
            c = count_pending(conn, t)
            print(f"  {t:20s}  desc={c:>8,}")
            total += c
        conn.close()
        print(f"\n  {'TOTAL':20s}  desc={total:>8,}")
        est_hours = total / 100 / 3600  # conservative 100 req/s
        print(f"\n  Est. time at ~100 req/s: {est_hours:.1f}h")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Apply-only ────────────────────────────────────────────────────────
    if args.apply_only:
        print("\n[Apply-only] Writing checkpoints → DB ...")
        apply_checkpoints_to_db(tables)
        return

    # ── Collect jobs ──────────────────────────────────────────────────────
    print("\n[Collecting pending descriptions]")
    raw_jobs = collect_desc_jobs(tables)
    print(f"\n  TOTAL: {len(raw_jobs):,} descriptions\n")

    if not raw_jobs:
        print("Nothing to translate.")
        return

    # ── Load vLLM ─────────────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    from vllm import LLM, SamplingParams

    def _patch_rope(hf_config):
        rs = getattr(hf_config, "rope_scaling", None)
        if isinstance(rs, dict) and "rope_type" not in rs:
            rs["rope_type"] = rs.get("type", "default")
        return hf_config

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=True,
        max_num_seqs=args.max_seqs,
        hf_overrides=_patch_rope,
    )
    tokenizer = llm.get_tokenizer()
    print("Model loaded.\n")

    desc_params = SamplingParams(
        temperature=0.0,
        max_tokens=DESC_MAX_TOKENS,
        stop=["</s>", "<end_of_turn>"],
    )

    # ── Build prompts with tokenizer-aware truncation ─────────────────────
    print("[Building prompts]")
    jobs = build_prompts(raw_jobs, tokenizer, model_id=args.model)
    del raw_jobs

    # ── Time estimate ─────────────────────────────────────────────────────
    est_hours = len(jobs) / 100 / 3600  # conservative
    print(f"\n  Conservative time estimate (~100 req/s): {est_hours:.1f}h")
    if est_hours > 24:
        print(f"  WARNING: may exceed 24h budget")
    print()

    # ── Translate ─────────────────────────────────────────────────────────
    print(f"[Translating {len(jobs):,} descriptions, "
          f"max_tokens={DESC_MAX_TOKENS}]")
    t0 = time.time()
    run_inference(llm, jobs, desc_params, chunk_size)
    elapsed = time.time() - t0
    print(f"\n  Translation done in {elapsed/3600:.1f}h "
          f"({len(jobs)/elapsed:.0f} req/s avg)\n", flush=True)
    del jobs

    # ── Write results → DB ────────────────────────────────────────────────
    print("[Writing results to DB]")
    apply_checkpoints_to_db(tables)
    print(f"\nCompleted: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
