"""
Extract skills from job descriptions using vLLM + Gemma 3 12B-IT locally.
Optimized for A40 GPU (48 GB VRAM).

Reads *_desc.jsonl from gemma_results/:
    {"id": "X", "value": "DESCRIPTION"}

Writes *_skills.jsonl to gemma_results/:
    {"id": "X", "skills": ["skill1", "skill2", ...]}

vLLM handles continuous batching and PagedAttention internally —
no manual batch-size tuning or length-based sorting is needed.
Prefix caching reuses KV for the shared system prompt across all requests.

Usage:
    python extract_skills_vllm.py                             # all *_desc.jsonl
    python extract_skills_vllm.py --file kenya_desc.jsonl      # single file
    python extract_skills_vllm.py --limit 50 --dry-run         # preview
    python extract_skills_vllm.py --model google/gemma-3-4b-it # smaller model

Requires:
    pip install vllm
    HF_TOKEN env var or .env file (Gemma 3 is a gated model)
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

# ── Defaults ──────────────────────────────────────────────────────────────────

MODEL_ID       = "google/gemma-3-12b-it"
INPUT_DIR      = Path("gemma_results_full")
MAX_INPUT_CHARS = 6_000       # ~2 500 tokens hard cap per description
MAX_NEW_TOKENS  = 200
TEMPERATURE     = 0
CHUNK_SIZE      = 10_000        # flush to disk every N jobs (crash safety)

SYSTEM_MSG = (
    "Extract individual skill-related text spans from the job description.\n"
    "A skill includes any ability, knowledge, tool, technology, or method.\n"
    "Return verbatim spans exactly as written in the text.\n"
    "Each span must refer to a single skill.\n"
    "Split phrases with multiple skills into separate spans.\n"
    "If no skills are found, return an empty list [].\n"
    "Return ONLY a JSON array of strings.\n"
    'Example: ["Python", "communication skills"]'
)
USER_MSG_TEMPLATE = "Job offer text: {text}\nJSON array of skills:"


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_token():
    tok = os.environ.get("HF_TOKEN", "")
    if tok:
        return tok
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("hf_token_write"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def clean_text(text):
    if not text or not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def parse_skills(text):
    if not text:
        return []
    try:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return [s.strip() for s in arr
                        if isinstance(s, str) and len(s.strip()) >= 2]
    except (json.JSONDecodeError, ValueError):
        pass
    return [m.strip() for m in re.findall(r'"([^"]{2,})"', text)
            if len(m.strip()) >= 2]


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_desc_file(path: Path):
    """Load a _desc.jsonl file → [(id, text), …]"""
    jobs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                jid = str(obj["id"])
                text = clean_text(obj.get("value", ""))
                if not text:
                    continue
                if len(text) > MAX_INPUT_CHARS:
                    text = text[:MAX_INPUT_CHARS]
                jobs.append((jid, text))
            except (json.JSONDecodeError, KeyError):
                continue
    return jobs


def output_path_for(input_path: Path) -> Path:
    """*_desc.jsonl → *_skills.jsonl"""
    name = input_path.name
    if name.endswith("_desc.jsonl"):
        return input_path.parent / name.replace("_desc.jsonl", "_skills.jsonl")
    return input_path.parent / (input_path.stem + "_skills.jsonl")


def load_done_ids(path: Path) -> set:
    """IDs already present in the output file (resume support)."""
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    done.add(str(json.loads(line.strip())["id"]))
                except Exception:
                    pass
    return done


# ── Prompt building ───────────────────────────────────────────────────────────

def _test_system_role(tokenizer) -> bool:
    """Check whether the tokenizer's chat template accepts a system role."""
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "test"},
             {"role": "user",   "content": "test"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return True
    except Exception:
        return False


_prompt_overhead: int | None = None

def build_prompt(tokenizer, text: str, use_system: bool,
                 max_model_len: int = 3072) -> str:
    global _prompt_overhead
    if _prompt_overhead is None:
        dummy = _make_prompt(tokenizer, "X", use_system)
        _prompt_overhead = len(tokenizer.encode(dummy))

    token_budget = max_model_len - MAX_NEW_TOKENS - _prompt_overhead - 8
    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(text_tokens) > token_budget:
        text_tokens = text_tokens[:token_budget]
        text = tokenizer.decode(text_tokens, skip_special_tokens=True)

    return _make_prompt(tokenizer, text, use_system)


def _make_prompt(tokenizer, text: str, use_system: bool) -> str:
    user_content = USER_MSG_TEMPLATE.format(text=text)
    if use_system:
        msgs = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user",   "content": user_content},
        ]
    else:
        msgs = [{"role": "user", "content": f"{SYSTEM_MSG}\n\n{user_content}"}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
    )


# ── Processing ────────────────────────────────────────────────────────────────

class FileQueue:
    """Manages pending jobs for one input file with chunk-based iteration."""

    def __init__(self, input_path: Path, limit: int | None = None):
        self.input_path = input_path
        self.out_path   = output_path_for(input_path)
        self.name       = input_path.stem.replace("_desc", "")

        all_jobs  = load_desc_file(input_path)
        done_ids  = load_done_ids(self.out_path)
        self.jobs = [(jid, t) for jid, t in all_jobs if jid not in done_ids]
        if limit:
            self.jobs = self.jobs[:limit]

        self.total    = len(all_jobs)
        self.resumed  = len(done_ids)
        self.cursor   = 0
        self.skills_n = 0
        self.with_n   = 0

    @property
    def remaining(self) -> int:
        return len(self.jobs) - self.cursor

    @property
    def done(self) -> bool:
        return self.cursor >= len(self.jobs)

    def next_chunk(self, size: int) -> list[tuple[str, str]]:
        end = min(self.cursor + size, len(self.jobs))
        chunk = self.jobs[self.cursor:end]
        self.cursor = end
        return chunk


def process_round_robin(llm, tokenizer, sampling_params,
                        files: list[Path], use_system: bool,
                        *, limit=None, dry_run=False,
                        max_model_len: int = 3072):
    """Process all files in round-robin: one chunk per file, then rotate."""

    queues = [FileQueue(f, limit) for f in files]

    # ── Summary table ──
    print(f"\n{'='*70}")
    print(f"Round-robin mode: {len(queues)} files, chunk={CHUNK_SIZE:,}")
    print(f"{'─'*70}")
    print(f"  {'File':<25s} {'Total':>10s} {'Resumed':>10s} {'Pending':>10s}")
    print(f"  {'─'*55}")
    total_pending = 0
    for q in queues:
        print(f"  {q.name:<25s} {q.total:>10,d} {q.resumed:>10,d} "
              f"{q.remaining:>10,d}")
        total_pending += q.remaining
    print(f"  {'─'*55}")
    print(f"  {'TOTAL':<25s} {'':>10s} {'':>10s} {total_pending:>10,d}")

    if dry_run:
        for q in queues:
            if q.remaining:
                chunk = q.next_chunk(3)
                print(f"\n  [{q.name}] sample prompts:")
                for jid, text in chunk:
                    prompt = build_prompt(tokenizer, text, use_system,
                                          max_model_len)
                    n_tok = len(tokenizer.encode(prompt))
                    print(f"    id={jid}  chars={len(text)}  "
                          f"prompt_tokens={n_tok}")
        return

    # ── Round-robin loop ──
    t0 = time.time()
    processed_total = 0
    rnd = 0

    while any(not q.done for q in queues):
        rnd += 1
        active = [q for q in queues if not q.done]
        print(f"\n{'━'*70}")
        print(f"Round {rnd}  |  {len(active)} active files  |  "
              f"{sum(q.remaining for q in active):,} jobs left")

        for q in active:
            chunk_jobs = q.next_chunk(CHUNK_SIZE)
            if not chunk_jobs:
                continue

            label = q.name
            print(f"\n  [{label}] chunk {len(chunk_jobs):,}  "
                  f"({q.remaining:,} left after this)")

            prompts = [build_prompt(tokenizer, text, use_system, max_model_len)
                       for _, text in chunk_jobs]

            outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

            with open(q.out_path, "a") as f:
                for (jid, _), output in zip(chunk_jobs, outputs):
                    resp   = output.outputs[0].text
                    skills = parse_skills(resp)
                    q.skills_n += len(skills)
                    if skills:
                        q.with_n += 1
                    f.write(json.dumps({"id": jid, "skills": skills},
                                       ensure_ascii=False) + "\n")

            processed_total += len(chunk_jobs)
            elapsed = time.time() - t0
            rate = processed_total / elapsed if elapsed else 0
            remaining_jobs = sum(qq.remaining for qq in queues)
            eta = remaining_jobs / rate if rate else 0
            print(f"  [{label}] flushed {len(chunk_jobs):,}  |  "
                  f"global: {processed_total:,}/{total_pending:,}  "
                  f"{rate:.1f} jobs/s  "
                  f"ETA {eta/3600:.1f}h")

            if q.done:
                print(f"  [{label}] ✓ FINISHED — "
                      f"skills={q.skills_n:,}  "
                      f"avg={q.skills_n/(q.cursor) :.1f}/job")

    # ── Final summary ──
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"ALL DONE — {processed_total:,} jobs in {elapsed:.0f}s "
          f"({processed_total/elapsed:.1f} jobs/s)\n")
    print(f"  {'File':<25s} {'Processed':>10s} {'Skills':>10s} {'Avg':>6s}")
    print(f"  {'─'*55}")
    for q in queues:
        n = q.cursor
        avg = f"{q.skills_n/n:.1f}" if n else "—"
        print(f"  {q.name:<25s} {n:>10,d} {q.skills_n:>10,d} {avg:>6s}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract skills — vLLM + Gemma 3 on local GPU")
    parser.add_argument("--file", type=str,
                        help="Single _desc.jsonl inside gemma_results/")
    parser.add_argument("--model", type=str, default=MODEL_ID,
                        help=f"HF model ID (default: {MODEL_ID})")
    parser.add_argument("--limit", type=int,
                        help="Max jobs per file (for testing)")
    parser.add_argument("--max-model-len", type=int, default=3072,
                        help="Max sequence length (lower = more concurrent seqs, default: 4096)")
    parser.add_argument("--gpu-mem", type=float, default=0.96,
                        help="Fraction of GPU memory for vLLM (default: 0.85)")
    parser.add_argument("--max-num-seqs", type=int, default=196,
                        help="Max concurrent sequences in batch (default: 128)")
    parser.add_argument("--quantization", type=str, default=None,
                        choices=["awq", "gptq", "squeezellm", "bitsandbytes", "fp8"],
                        help="Weight quantization (default: none → bf16; "
                             "fp8 = 8-bit on-the-fly; bitsandbytes = 4-bit on-the-fly)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview prompts and token counts, skip model")
    args = parser.parse_args()

    # ── Resolve input files ───────────────────────────────────────────
    if args.file:
        files = [INPUT_DIR / args.file]
        for f in files:
            if not f.exists():
                raise SystemExit(f"Not found: {f}")
    else:
        files = sorted(INPUT_DIR.glob("*_desc.jsonl"))

    if not files:
        raise SystemExit(f"No *_desc.jsonl files in {INPUT_DIR}/")

    token = get_token()
    if token:
        os.environ["HF_TOKEN"] = token

    print(f"Model:         {args.model}")
    print(f"Files:         {len(files)}")
    for f in files:
        print(f"               {f.name}")
    print(f"Max model len: {args.max_model_len}")
    print(f"GPU memory:    {args.gpu_mem:.0%}")
    print(f"Max num seqs:  {args.max_num_seqs}")
    print(f"Quantization:  {args.quantization or 'none (bf16)'}")

    # ── Dry run (no GPU needed) ───────────────────────────────────────
    if args.dry_run:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, token=token or None)
        use_system = _test_system_role(tokenizer)
        print(f"System role:   {'yes' if use_system else 'no (merged into user)'}")
        process_round_robin(None, tokenizer, None, files,
                            use_system, limit=args.limit, dry_run=True,
                            max_model_len=args.max_model_len)
        return

    # ── Load model via vLLM ───────────────────────────────────────────
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=True,
        quantization=args.quantization,
        enforce_eager=True,
    )

    tokenizer  = llm.get_tokenizer()
    use_system = _test_system_role(tokenizer)
    print(f"System role:   {'yes' if use_system else 'no (merged into user)'}")

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
    )

    process_round_robin(llm, tokenizer, sampling_params, files,
                        use_system, limit=args.limit,
                        max_model_len=args.max_model_len)


if __name__ == "__main__":
    main()
