#!/usr/bin/env python3
"""
Match extracted skill terms to ESCO labels using ConTeXT-Skill-Extraction-base.

1. Load ESCO preferred labels from skills_en.csv → encode with ConTeXT model
2. Load skill terms from *_skills.jsonl → encode each unique term
3. Cosine similarity → best ESCO match per skill term
4. Test mode: first N rows from each country, tqdm progress

Usage:
    python match_skills_context.py                     # test 500 rows / country
    python match_skills_context.py --limit 0           # all rows
    python match_skills_context.py --threshold 0.5     # stricter matching
"""

import csv
import json
import sys
import argparse
import time
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "TechWolf/ConTeXT-Skill-Extraction-base"
ESCO_CSV    = Path(__file__).resolve().parent / "skills_en.csv"
SKILLS_DIR  = Path(__file__).resolve().parent / "gemma_results_full"
OUT_FILE    = Path(__file__).resolve().parent / "skills_esco_agg.json"
BATCH_SIZE  = 256
THRESHOLD   = 0.40

COUNTRY_LABELS = {
    'in': 'India', 'jp': 'Japan', 'kr': 'South Korea',
    'malaysia': 'Malaysia', 'mx': 'Mexico', 'ph': 'Philippines',
    'pl': 'Poland', 'sg': 'Singapore', 'th': 'Thailand',
    'tw': 'Taiwan', 'vn': 'Vietnam',
}

# ── Load ESCO labels ──────────────────────────────────────────────────────────
def load_esco_labels(csv_path: Path) -> tuple[list[str], list[str], list[str]]:
    """Return (labels, uris, skill_types)."""
    labels, uris, skill_types = [], [], []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = (row.get("preferredLabel") or "").strip()
            uri = (row.get("conceptUri") or "").strip()
            stype = (row.get("skillType") or "").strip()
            if label and uri:
                labels.append(label)
                uris.append(uri)
                skill_types.append(stype)
    return labels, uris, skill_types


# ── Collect unique skill terms per country ────────────────────────────────────
def load_skill_terms(skills_dir: Path, limit: int) -> dict[str, Counter]:
    """Return {country_code: Counter(skill_term → count)}."""
    result = {}
    for f in sorted(skills_dir.glob("*_skills.jsonl")):
        country = f.stem.removeprefix("jobads_").removesuffix("_skills")
        if country not in COUNTRY_LABELS:
            continue
        c: Counter = Counter()
        with open(f) as fh:
            for i, line in enumerate(fh):
                if limit and i >= limit:
                    break
                row = json.loads(line)
                for s in row.get("skills", []):
                    s = s.strip()
                    if s:
                        c[s] += 1
        result[country] = c
        print(f"  {COUNTRY_LABELS[country]:12s}: {sum(c.values()):>7,} instances, {len(c):>6,} unique"
              + (f" (limited to {limit} rows)" if limit else ""))
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500,
                        help="Max rows per country file (0 = all)")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"Min cosine similarity (default {THRESHOLD})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    print(f"Loading model: {MODEL_NAME} …")
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # ── Encode ESCO labels ────────────────────────────────────────────────────
    print(f"\nLoading ESCO labels from {ESCO_CSV.name} …")
    esco_labels, esco_uris, esco_skill_types = load_esco_labels(ESCO_CSV)
    print(f"  {len(esco_labels):,} ESCO labels")

    print("Encoding ESCO labels …")
    t1 = time.time()
    esco_embs = model.encode(
        esco_labels,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"  ESCO encoded in {time.time()-t1:.1f}s → shape {esco_embs.shape}")

    # ── Load skill terms ──────────────────────────────────────────────────────
    print(f"\nLoading skill terms (limit={args.limit or 'all'}) …")
    country_skills = load_skill_terms(SKILLS_DIR, args.limit)

    # Collect global unique terms
    global_terms: Counter = Counter()
    for c in country_skills.values():
        global_terms.update(c)
    unique_terms = list(global_terms.keys())
    print(f"\nTotal unique terms to encode: {len(unique_terms):,}")

    # ── Encode skill terms ────────────────────────────────────────────────────
    print("Encoding skill terms …")
    t2 = time.time()
    term_embs = model.encode(
        unique_terms,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"  Terms encoded in {time.time()-t2:.1f}s → shape {term_embs.shape}")

    # ── Match: cosine similarity (dot product since normalized) ───────────────
    print(f"\nMatching (threshold={args.threshold}) …")
    t3 = time.time()

    # term_embs: (T, d), esco_embs: (E, d)
    # Compute in chunks to avoid OOM
    CHUNK = 2000
    term_to_esco: dict[str, tuple[str, str, str, float]] = {}  # term → (label, uri, skill_type, score)

    for start in tqdm(range(0, len(unique_terms), CHUNK), desc="Matching chunks"):
        end = min(start + CHUNK, len(unique_terms))
        chunk_embs = term_embs[start:end]  # (chunk, d)
        sims = chunk_embs @ esco_embs.T     # (chunk, E)
        best_idx = np.argmax(sims, axis=1)
        best_scores = sims[np.arange(len(best_idx)), best_idx]

        for j, (idx, score) in enumerate(zip(best_idx, best_scores)):
            term = unique_terms[start + j]
            term_to_esco[term] = (
                esco_labels[idx],
                esco_uris[idx],
                esco_skill_types[idx],
                float(score),
            )

    above = sum(1 for v in term_to_esco.values() if v[3] >= args.threshold)
    print(f"  All {len(term_to_esco):,} terms scored in {time.time()-t3:.1f}s")
    print(f"  Above threshold ({args.threshold}): {above:,} ({above/len(term_to_esco):.1%})")

    # ── Show sample matches ───────────────────────────────────────────────────
    print("\n── Sample matches (top-30 by frequency) ──")
    for term, freq in global_terms.most_common(30):
        label, uri, stype, score = term_to_esco[term]
        flag = "✓" if score >= args.threshold else "✗"
        print(f"  {flag} {freq:5d}x  {term:40s} → {label:40s} ({stype}, {score:.3f})")

    # ── Score distribution ────────────────────────────────────────────────────
    scores = [v[3] for v in term_to_esco.values()]
    print("\n── Score distribution ──")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        n = sum(1 for s in scores if s >= t)
        print(f"  ≥ {t:.1f}: {n:>7,} terms ({n/len(scores):.1%})")

    # ── Per-country match rates at current threshold ──────────────────────────
    print(f"\n── Per-country match rates (threshold={args.threshold}) ──")
    for country, skill_counter in sorted(country_skills.items()):
        total = sum(skill_counter.values())
        matched = sum(cnt for term, cnt in skill_counter.items()
                      if term_to_esco.get(term, (None,None,None,0))[3] >= args.threshold)
        pct = matched / total * 100 if total else 0
        print(f"  {COUNTRY_LABELS[country]:12s}: {matched:>7,}/{total:>7,} instances matched ({pct:.1f}%)")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    # ── Save match cache for later aggregation ────────────────────────────────
    cache_file = Path(__file__).resolve().parent / "term_esco_matches.json"
    cache = {term: {"label": m[0], "uri": m[1], "skill_type": m[2], "score": m[3]}
             for term, m in term_to_esco.items()}
    cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
    print(f"Match cache saved → {cache_file.name} ({len(cache):,} entries)")


if __name__ == "__main__":
    main()
