#!/usr/bin/env python3
"""
Filter skill-relevant sentences from translated job descriptions
using BERT-OJA-SkillLess (serino28/BERT-OJA-SkillLess).

Reads *_desc.jsonl from gemma_results/ (or gemma_results_full/):
    {"id": 12345, "value": "English translation of job description"}

Writes *_skillsent.jsonl:
    {"id": 12345, "skill_sentences": ["sent1", …], "n_total": 15, "n_skill": 8}

The BERT model (~110 M params) is tiny — expect 5 000–15 000 sents/s on A40
with batch_size=512 and FP16.

Usage:
    python classify_skill_sents.py                                # all files
    python classify_skill_sents.py --file jobads_jp_desc.jsonl    # single
    python classify_skill_sents.py --input-dir gemma_results_full # alt dir
    python classify_skill_sents.py --batch-size 1024              # tune
    python classify_skill_sents.py --threshold 0.6                # stricter
    python classify_skill_sents.py --dry-run                      # preview
"""

import os
import sys
import re
import json
import argparse
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME     = "serino28/BERT-OJA-SkillLess"
INPUT_DIR      = Path("gemma_results")
BATCH_SIZE     = 2048
MAX_LENGTH     = 128       # job-ad sentences rarely exceed 80 tokens
CHUNK_SIZE     = 10_000
MIN_SENT_CHARS = 10

# ── Sentence splitting ───────────────────────────────────────────────────────
# Section-header keywords (normalized form).
HEADER_PATTERNS: frozenset[str] = frozenset({
    "responsibilities", "major responsibilities", "key responsibilities",
    "job responsibilities", "main responsibilities", "core responsibilities",
    "your responsibilities", "what you will do", "what you'll do",
    "duties", "key duties", "main duties", "role and responsibilities",
    "qualifications", "requirements", "job requirements", "key requirements",
    "preferred qualifications", "minimum qualifications", "basic qualifications",
    "required qualifications", "your qualifications", "your profile",
    "who we are looking for", "what we are looking for", "we are looking for",
    "preferred requirements", "additional requirements",
    "about us", "about the company", "about the role", "about this role",
    "about the position", "about the job", "about", "company overview",
    "position overview", "role overview", "job overview", "overview",
    "the opportunity", "the role", "the job", "the position",
    "what we offer", "we offer", "what we provide", "what you get",
    "benefits", "our benefits", "what's in it for you",
    "perks", "perks and benefits", "compensation and benefits",
    "learning and growth", "impactful work", "entrepreneurial spirit",
    "our values", "our culture", "why join us", "why us", "why work with us",
    "job description", "job details", "job information",
    "job summary", "position details", "job posting",
    "skills", "required skills", "technical skills", "core skills",
    "key skills", "preferred skills", "tools and technologies",
    "education", "educational requirements", "academic requirements",
    "experience", "work experience", "relevant experience",
    "notes", "important", "note", "please note",
    "how to apply", "application process", "next steps", "application",
    "contact", "inquiries", "for inquiries", "to apply",
    "open positions", "available positions", "key selling points",
    "background", "profile", "desired candidate profile",
})

_CONNECTING_WORDS: frozenset[str] = frozenset({
    "and", "or", "in", "with", "for", "to", "of", "at", "by",
    "from", "through", "into", "on", "upon", "between", "within",
    "about", "that", "which", "who", "whose", "whom", "during",
    "before", "after", "the", "a", "an", "as", "but", "nor",
    "so", "yet", "both", "either", "neither", "not", "only", "just",
    "also", "including", "such", "per", "via", "than", "when", "where",
})

_BULLET_RE         = re.compile(r"^[\s*•·▪▸►→●○◦★☆✓✔⁃‣⦿⦾✅\-–—]+\s*")
_BOLD_RE           = re.compile(r"\*{1,2}([^*]+)\*{1,2}")
_TRAILING_STARS_RE = re.compile(r"[\s*]+$")
_HEADER_RE         = re.compile(r"^\[.+\]$")
_LABEL_PREFIX_RE   = re.compile(r"^\[.+\]\s*[:\s]\s*")
_SEPARATOR_RE      = re.compile(r"^[-=_*#]{3,}$")
_MARKDOWN_H_RE     = re.compile(r"^#{1,4}\s+")
_GT_PREFIX_RE      = re.compile(r"^(>\s*)+")
_URL_RE            = re.compile(r"https?://\S+")
_ESCAPED_MD_RE     = re.compile(r"\\([^\s\\])")
_JUNK_RE           = re.compile(r"^\S{40,}$")  # a bit less aggressive
_SENT_SPLIT_RE     = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"])")
_METADATA_RE       = re.compile(
    r"^(?:job type|pay|salary|work location|location|working hours|work hours|"
    r"employment type|job id|reference no?\.?|ref\.?|address|phone|tel(?:ephone)?|"
    r"start date|end date|contract type|duration|work schedule|schedule|shift|"
    r"deadline|closing date|job category|category|industry|sector|function|"
    r"department|team|division|reporting to|reports to)\s*[:\-]",
    re.IGNORECASE,
)

_VERBISH_RE = re.compile(
    r"\b(?:develop|design|build|implement|manage|support|analy[sz]e|create|"
    r"maintain|lead|coordinate|collaborat|document|test|use|work|communicat|"
    r"troubleshoot|debug|improve|deliver|ensure|perform|assist|learn|apply|"
    r"experience|knowledge|understanding|ability|proficiency|familiarity)\b",
    re.IGNORECASE,
)

def _normalize_header(line: str) -> str:
    s = _BOLD_RE.sub(r"\1", line)
    s = re.sub(r"^#+\s*", "", s)
    s = re.sub(r"[\[\]()*:_]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))

def _looks_like_header(line: str) -> bool:
    """
    Conservative header detection:
    only drop if the line is short and strongly header-like.
    """
    norm = _normalize_header(line)
    wc = _word_count(norm)

    if not norm:
        return False

    if _HEADER_RE.match(line):
        return True

    if norm in HEADER_PATTERNS and wc <= 6:
        return True

    # short colon-ended title, e.g. "Qualifications:"
    if line.rstrip().endswith(":") and wc <= 6:
        return True

    # very short title-case-ish lines without verbs or punctuation
    if wc <= 4 and not re.search(r"[.!?]", line) and not _VERBISH_RE.search(line):
        if norm in HEADER_PATTERNS:
            return True

    return False

def _starts_like_continuation(line: str) -> bool:
    """
    Conservative rule: next line is continuation only if it clearly
    does not look like a fresh sentence/header/bullet item.
    """
    if not line:
        return False

    stripped = line.strip()
    if not stripped:
        return False

    if _looks_like_header(stripped):
        return False

    first = stripped[0]

    # lowercase start or opening punctuation often means continuation
    if first.islower() or first in {",", ";", ")", "]"}:
        return True

    # lines starting with connectors can continue previous line
    first_word = re.split(r"\s+", stripped, maxsplit=1)[0].lower().strip(".,;:()[]\"'")
    if first_word in _CONNECTING_WORDS:
        return True

    return False

def _ends_like_continuation(line: str) -> bool:
    """
    Conservative rule: current line invites merge only when it really
    looks unfinished.
    """
    if not line:
        return False

    stripped = line.strip()
    if not stripped:
        return False

    last_char = stripped[-1]

    # closed sentence / header
    if last_char in ".!?:;":
        return False

    # obvious continuation
    if last_char == ",":
        return True

    words = re.findall(r"\b[\w'-]+\b", stripped)
    if not words:
        return False

    last_word = words[-1].lower()
    if last_word in _CONNECTING_WORDS:
        return True

    # very short lines without verb often continue list/header context
    if len(words) <= 3 and not _VERBISH_RE.search(stripped):
        return True

    return False

def _clean_line(line: str) -> str:
    line = _URL_RE.sub("", line)
    line = _GT_PREFIX_RE.sub("", line)
    line = _ESCAPED_MD_RE.sub(r"\1", line)
    line = _SEPARATOR_RE.sub("", line)
    line = _MARKDOWN_H_RE.sub("", line)
    line = _LABEL_PREFIX_RE.sub("", line)
    line = _BULLET_RE.sub("", line)
    line = _BOLD_RE.sub(r"\1", line)
    line = _TRAILING_STARS_RE.sub("", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line

def split_sentences(text: str) -> list[str]:
    """
    More conservative sentence splitter for job ads.

    Principles:
    - clean gently
    - drop only clearly header-like or metadata-like lines
    - merge lines only when both sides strongly suggest continuation
    - then split into sentence units
    """
    raw_lines = [_clean_line(line) for line in text.split("\n")]

    # Drop empty/separator lines early
    raw_lines = [line for line in raw_lines if line]

    joined: list[str] = []
    i = 0
    n = len(raw_lines)

    while i < n:
        line = raw_lines[i]
        i += 1

        if len(line) < 3:
            continue

        # Keep metadata/header filtering conservative
        if _METADATA_RE.match(line):
            continue
        if _JUNK_RE.match(line):
            continue
        if _looks_like_header(line):
            continue

        current = line

        while i < n:
            nxt = raw_lines[i]

            if not nxt:
                i += 1
                continue

            if _METADATA_RE.match(nxt) or _JUNK_RE.match(nxt) or _looks_like_header(nxt):
                break

            if _ends_like_continuation(current) and _starts_like_continuation(nxt):
                current = f"{current} {nxt}"
                i += 1
            else:
                break

        joined.append(current)

    sentences: list[str] = []

    for line in joined:
        parts = _SENT_SPLIT_RE.split(line)
        for sub in parts:
            sub = re.sub(r"\s+", " ", sub).strip()
            if len(sub) < 10:
                continue
            if _word_count(sub) < 2:
                continue
            sentences.append(sub)

    return sentences

# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_desc_file(path: Path) -> list[tuple]:
    jobs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                jid = obj["id"]
                text = (obj.get("value") or "").strip()
                if text:
                    jobs.append((jid, text))
            except (json.JSONDecodeError, KeyError):
                continue
    return jobs


def output_path_for(input_path: Path) -> Path:
    name = input_path.name
    if name.endswith("_desc.jsonl"):
        return input_path.parent / name.replace("_desc.jsonl", "_skillsent.jsonl")
    return input_path.parent / (input_path.stem + "_skillsent.jsonl")


def load_done_ids(path: Path) -> set:
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line.strip())["id"])
                except Exception:
                    pass
    return done


# ── Batched BERT inference ────────────────────────────────────────────────────

def classify_sentences(
    sentences: list[str],
    model,
    tokenizer,
    device: str,
    batch_size: int,
    threshold: float,
    max_length: int = MAX_LENGTH,
    pbar=None,
) -> list[bool]:
    """Return True for each sentence classified as skill-relevant.

    Uses a pipelined approach: batch N+1 is tokenized on a background
    CPU thread while batch N runs on the GPU, eliminating the stall
    that otherwise keeps the GPU idle during tokenization (~70% of
    wall-clock time for short sentences).
    """
    import torch
    from concurrent.futures import ThreadPoolExecutor

    if not sentences:
        return []

    def _tokenize(batch_sents):
        return tokenizer(
            batch_sents,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    batches = [
        (i, min(i + batch_size, len(sentences)))
        for i in range(0, len(sentences), batch_size)
    ]

    results: list[bool] = []

    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(_tokenize, sentences[batches[0][0]:batches[0][1]])

        for idx, (start, end) in enumerate(batches):
            encoded = future.result()

            if idx + 1 < len(batches):
                ns, ne = batches[idx + 1]
                future = pool.submit(_tokenize, sentences[ns:ne])

            inputs = {k: v.to(device, non_blocking=True)
                      for k, v in encoded.items()}

            with torch.inference_mode():
                logits = model(**inputs).logits

            probs = torch.softmax(logits, dim=-1)[:, 1]
            results.extend((probs >= threshold).cpu().tolist())

            if pbar is not None:
                pbar.update(end - start)

    return results


# ── Per-file processing ──────────────────────────────────────────────────────

def process_file(
    input_path: Path,
    model,
    tokenizer,
    device: str,
    batch_size: int,
    threshold: float,
    max_length: int = MAX_LENGTH,
    limit: int | None = None,
):
    out_path = output_path_for(input_path)
    name = input_path.stem.replace("_desc", "")

    all_jobs = load_desc_file(input_path)
    done_ids = load_done_ids(out_path)
    jobs = [(jid, t) for jid, t in all_jobs if jid not in done_ids]
    if limit:
        jobs = jobs[:limit]

    print(f"\n  [{name}]  total={len(all_jobs):,}  "
          f"done={len(done_ids):,}  pending={len(jobs):,}")

    if not jobs:
        return 0, 0

    from tqdm import tqdm

    t0 = time.time()
    total_sents = 0
    total_skill = 0

    jobs_bar = tqdm(
        total=len(jobs),
        desc=f"[{name}] opisy",
        unit="opis",
        dynamic_ncols=True,
        leave=True,
    )

    for chunk_start in range(0, len(jobs), CHUNK_SIZE):
        chunk = jobs[chunk_start : chunk_start + CHUNK_SIZE]

        flat_sents: list[str] = []
        job_boundaries: list[tuple[int, int, int]] = []  # (start, end, n_total)

        for jid, text in chunk:
            sents = split_sentences(text)
            start = len(flat_sents)
            flat_sents.extend(sents)
            job_boundaries.append((start, len(flat_sents), len(sents)))

        total_sents += len(flat_sents)

        if flat_sents:
            with tqdm(
                total=len(flat_sents),
                desc=f"  zdania (batch={batch_size})",
                unit="zdanie",
                dynamic_ncols=True,
                leave=False,
            ) as sent_bar:
                labels = classify_sentences(
                    flat_sents, model, tokenizer, device, batch_size, threshold,
                    max_length=max_length, pbar=sent_bar,
                )
        else:
            labels = []

        with open(out_path, "a", encoding="utf-8") as f:
            for idx, (jid, _) in enumerate(chunk):
                start, end, n_total = job_boundaries[idx]
                skill_sents = [
                    flat_sents[i]
                    for i in range(start, end)
                    if labels[i]
                ]
                total_skill += len(skill_sents)
                f.write(json.dumps({
                    "id": jid,
                    "skill_sentences": skill_sents,
                    "n_total": n_total,
                    "n_skill": len(skill_sents),
                }, ensure_ascii=False) + "\n")

        jobs_bar.update(len(chunk))
        pct_now = 100 * total_skill / total_sents if total_sents else 0
        jobs_bar.set_postfix(
            sents=f"{total_sents:,}",
            skill_pct=f"{pct_now:.1f}%",
        )

    jobs_bar.close()

    elapsed = time.time() - t0
    pct = 100 * total_skill / total_sents if total_sents else 0
    print(f"  [{name}] done — {total_sents:,} sents, "
          f"{total_skill:,} skill ({pct:.1f}%), "
          f"{elapsed:.0f}s ({total_sents/elapsed:,.0f} sents/s)")
    return total_sents, total_skill


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Classify skill-relevant sentences — BERT-OJA-SkillLess")
    parser.add_argument("--file", type=str,
                        help="Single _desc.jsonl file name")
    parser.add_argument("--input-dir", type=str, default=str(INPUT_DIR),
                        help=f"Directory with *_desc.jsonl (default: {INPUT_DIR})")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Sentences per GPU batch (default: {BATCH_SIZE})")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH,
                        help=f"Max token length per sentence (default: {MAX_LENGTH})")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="P(skill) threshold (default: 0.5)")
    parser.add_argument("--limit", type=int,
                        help="Max jobs per file (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview sentence splitting, no model")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if CUDA available")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if args.file:
        files = [input_dir / args.file]
        for f in files:
            if not f.exists():
                raise SystemExit(f"Not found: {f}")
    else:
        files = sorted(input_dir.glob("*_desc.jsonl"))

    if not files:
        raise SystemExit(f"No *_desc.jsonl files in {input_dir}/")

    print("=" * 65)
    print("BERT-OJA-SkillLess  —  skill sentence filter")
    print(f"Model      : {args.model}")
    print(f"Input dir  : {input_dir}")
    print(f"Files      : {len(files)}")
    for f in files:
        print(f"             {f.name}")
    print(f"Batch size : {args.batch_size:,}")
    print(f"Max length : {args.max_length}")
    print(f"Threshold  : {args.threshold}")
    print("=" * 65)

    # ── Dry run ───────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n[Dry run — sentence splitting preview]\n")
        for fp in files:
            all_jobs = load_desc_file(fp)
            name = fp.stem.replace("_desc", "")
            total_sents = sum(len(split_sentences(t)) for _, t in all_jobs)
            avg = total_sents / len(all_jobs) if all_jobs else 0
            print(f"  [{name}]  {len(all_jobs):,} descs → "
                  f"{total_sents:,} sents (avg {avg:.1f}/desc)")
            for jid, text in all_jobs[:3]:
                sents = split_sentences(text)
                print(f"    id={jid}  → {len(sents)} sentences:")
                for s in sents[:5]:
                    print(f"      \"{s[:90]}\"")
                if len(sents) > 5:
                    print(f"      … +{len(sents)-5} more")
            print()
        return

    # ── Load model ────────────────────────────────────────────────────────
    import torch
    from transformers import BertForSequenceClassification, BertTokenizer

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice     : {device}")
    print("Loading model …")

    tokenizer = BertTokenizer.from_pretrained(args.model)
    model = BertForSequenceClassification.from_pretrained(args.model)
    model = model.to(device).eval()
    if device == "cuda":
        model = model.half()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded ({n_params:.0f}M params, "
          f"{'FP16' if device == 'cuda' else 'FP32'})\n")

    # ── Process files ─────────────────────────────────────────────────────
    from tqdm import tqdm

    t0 = time.time()
    grand_sents = 0
    grand_skill = 0

    for fp in tqdm(files, desc="Pliki", unit="plik", dynamic_ncols=True):
        s, k = process_file(
            fp, model, tokenizer, device,
            args.batch_size, args.threshold, args.max_length, args.limit,
        )
        grand_sents += s
        grand_skill += k

    elapsed = time.time() - t0
    pct = 100 * grand_skill / grand_sents if grand_sents else 0
    print(f"\n{'='*65}")
    print(f"ALL DONE — {grand_sents:,} sents total, "
          f"{grand_skill:,} skill ({pct:.1f}%)")
    print(f"Time: {elapsed:.0f}s "
          f"({grand_sents/elapsed:,.0f} sents/s avg)")
    print("=" * 65)


if __name__ == "__main__":
    main()
