#!/usr/bin/env python3
"""
Map skill-relevant sentences to ESCO skills using ConTeXT-match.
(TechWolf/ConTeXT-Skill-Extraction-base, Decorte et al. 2025)

The ConTeXT-match mechanism uses token-level attention between sentence
tokens and ESCO skill embeddings — a significant improvement over plain
cosine similarity of averaged embeddings:

    match(x, s) = Σ_j  α_j · cos(x_j, s)
    α_j = softmax(z_xj · z_s)

After thresholding, redundancy filtering retains only skills that have
the highest dot product with at least one content token, eliminating
near-duplicate predictions (e.g. "machine learning" vs "utilise ML").

Reads *_skillsent.jsonl (from classify_skill_sents.py):
    {"id": X, "skill_sentences": ["sent1", …], "n_total": N, "n_skill": M}

Writes *_esco.jsonl:
    {"id": X, "skills": [{"uri": "…", "label": "…", "score": 0.72}, …]}

Usage:
    python map_esco_context.py                                # all files
    python map_esco_context.py --file jobads_jp_skillsent.jsonl
    python map_esco_context.py --threshold 0.50               # stricter
    python map_esco_context.py --no-redundancy-filter         # skip dedup
    python map_esco_context.py --max-skills 30                # cap output
    python map_esco_context.py --dry-run                      # preview
"""

import os
import sys
import json
import argparse
import time
import csv
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME   = "TechWolf/ConTeXT-Skill-Extraction-base"
ESCO_CSV     = Path(__file__).resolve().parent.parent / "esco.csv"
INPUT_DIR    = Path("gemma_results")
THRESHOLD    = 0.48      # paper calibrated: 0.48 with filtering, 0.53 without
MAX_SKILLS   = 50        # max ESCO skills per job ad
SENT_BATCH   = 64        # sentences per GPU batch (token-level encoding)
CHUNK_SIZE   = 5_000     # job ads per checkpoint flush


# ── ESCO taxonomy loading ────────────────────────────────────────────────────

def load_esco(path: Path) -> tuple[list[str], list[str], list[str]]:
    """Load ESCO skills from CSV → (labels, descriptions, uris)."""
    labels, descs, uris = [], [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = (row.get("preferredLabel") or "").strip()
            uri = (row.get("conceptUri") or "").strip()
            desc = (row.get("description") or "").strip()
            if label and uri:
                labels.append(label)
                descs.append(desc)
                uris.append(uri)
    print(f"  Loaded {len(labels):,} ESCO skills from {path.name}")
    return labels, descs, uris


# ── ConTeXT-match engine ─────────────────────────────────────────────────────

class ConTeXTMatcher:
    """Wraps the ConTeXT-Skill-Extraction-base model for inference."""

    def __init__(self, model_name: str, esco_labels: list[str],
                 esco_uris: list[str], device: str):
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = device
        self.torch = torch

        print(f"  Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        if device == "cuda":
            self.model = self.model.half()

        self.esco_labels = esco_labels
        self.esco_uris = esco_uris

        print(f"  Encoding {len(esco_labels):,} ESCO skill labels …")
        raw = self.model.encode(
            esco_labels, batch_size=256,
            show_progress_bar=True, convert_to_tensor=True,
        )
        self.skill_embs = raw.to(device)  # (S, d)
        if device == "cuda":
            self.skill_embs = self.skill_embs.half()
        self.skill_norms = self.skill_embs.norm(dim=-1)  # (S,)
        print(f"  Skill embeddings ready: {self.skill_embs.shape}")

    # ── Token-level encoding ──────────────────────────────────────────────

    def _encode_tokens(self, sentences: list[str]):
        """Run the Transformer (no pooling) → token embeddings + mask."""
        torch = self.torch
        features = self.model.tokenize(sentences)
        features = {k: v.to(self.device) for k, v in features.items()}
        with torch.no_grad():
            out = self.model[0](features)
        tok_embs = out["token_embeddings"]      # (B, T, d)
        mask = out["attention_mask"]             # (B, T)
        if self.device == "cuda":
            tok_embs = tok_embs.half()
        return tok_embs, mask

    # ── ConTeXT-match scoring (equations 1–4 from the paper) ──────────────

    def context_match(self, tok_embs, mask):
        """Compute match(x, s) for all sentences × all skills.

        Returns: match_scores (B, S)  and  dots (B, T, S) for filtering.
        """
        torch = self.torch
        B, T, d = tok_embs.shape

        # Dot products: z_xj · z_s  → (B, T, S)
        dots = torch.einsum("btd,sd->bts", tok_embs, self.skill_embs)

        # Token norms for cosine similarity
        tok_norms = tok_embs.norm(dim=-1, keepdim=True)           # (B, T, 1)
        skill_norms = self.skill_norms.unsqueeze(0).unsqueeze(0)  # (1, 1, S)
        cos_sim = dots / (tok_norms * skill_norms + 1e-8)         # (B, T, S)

        # Masked softmax over tokens: α_j = softmax(z_xj · z_s)
        neg_inf = torch.finfo(dots.dtype).min
        attn_mask = mask.unsqueeze(-1).bool()                     # (B, T, 1)
        dots_masked = dots.masked_fill(~attn_mask, neg_inf)
        alpha = torch.softmax(dots_masked, dim=1)                 # (B, T, S)

        # Match score: Σ_j α_j · cos(x_j, s)
        match_scores = (alpha * cos_sim).sum(dim=1)               # (B, S)

        return match_scores, dots

    # ── Redundancy filtering (Section III.B of the paper) ─────────────────

    def redundancy_filter(self, candidate_indices, dots_row, mask_row):
        """Among candidates above threshold, keep only skills that have the
        highest dot product with at least one content token.

        Template tokens (BOS=0, EOS=last valid) are excluded per the paper.
        """
        torch = self.torch
        valid_len = int(mask_row.sum().item())
        if valid_len <= 2:
            return candidate_indices

        # Exclude BOS (pos 0) and EOS (pos valid_len-1)
        content_start, content_end = 1, valid_len - 1
        if content_start >= content_end:
            return candidate_indices

        # dots_row: (T, S) — subset to content tokens and candidate skills
        content_dots = dots_row[content_start:content_end, :]  # (T', S_full)
        cand_dots = content_dots[:, candidate_indices]         # (T', |cand|)

        # For each content token, which candidate has the highest dot?
        winners = cand_dots.argmax(dim=1)                      # (T',)
        unique_winners = torch.unique(winners)

        return candidate_indices[unique_winners.cpu()]

    # ── Full pipeline: sentences → ESCO skills ────────────────────────────

    def predict_batch(self, sentences: list[str], threshold: float,
                      do_filter: bool = True, max_skills: int = 50):
        """Predict ESCO skills for a batch of sentences.

        Returns list of lists:
          [[(idx_in_esco, score), …], …]  — one list per sentence.
        """
        torch = self.torch
        if not sentences:
            return []

        tok_embs, mask = self._encode_tokens(sentences)
        match_scores, dots = self.context_match(tok_embs, mask)

        results = []
        for i in range(len(sentences)):
            scores = match_scores[i]                           # (S,)
            above = (scores >= threshold).nonzero(as_tuple=True)[0]  # indices

            if len(above) == 0:
                results.append([])
                continue

            if do_filter and len(above) > 1:
                above = self.redundancy_filter(above, dots[i], mask[i])

            skill_scores = scores[above]
            order = skill_scores.argsort(descending=True)[:max_skills]
            kept = above[order]
            kept_scores = scores[kept]

            results.append([
                (int(idx), float(sc))
                for idx, sc in zip(kept.cpu(), kept_scores.cpu())
            ])

        return results


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_skillsent_file(path: Path) -> list[tuple]:
    """Load _skillsent.jsonl → [(id, [sent, …]), …]"""
    jobs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                jid = obj["id"]
                sents = obj.get("skill_sentences", [])
                if sents:
                    jobs.append((jid, sents))
            except (json.JSONDecodeError, KeyError):
                continue
    return jobs


def output_path_for(input_path: Path) -> Path:
    name = input_path.name
    if name.endswith("_skillsent.jsonl"):
        return input_path.parent / name.replace("_skillsent.jsonl", "_esco.jsonl")
    return input_path.parent / (input_path.stem + "_esco.jsonl")


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


# ── Per-file processing ──────────────────────────────────────────────────────

def process_file(
    input_path: Path,
    matcher: "ConTeXTMatcher",
    threshold: float,
    do_filter: bool,
    max_skills: int,
    sent_batch: int,
    limit: int | None = None,
):
    from tqdm import tqdm

    out_path = output_path_for(input_path)
    name = input_path.stem.replace("_skillsent", "")

    all_jobs = load_skillsent_file(input_path)
    done_ids = load_done_ids(out_path)
    jobs = [(jid, ss) for jid, ss in all_jobs if jid not in done_ids]
    if limit:
        jobs = jobs[:limit]

    print(f"\n  [{name}]  total={len(all_jobs):,}  "
          f"done={len(done_ids):,}  pending={len(jobs):,}")

    if not jobs:
        return 0, 0

    t0 = time.time()
    total_ads = 0
    total_skills_found = 0

    jobs_bar = tqdm(total=len(jobs), desc=f"[{name}]",
                    unit="ad", dynamic_ncols=True, leave=True)

    for chunk_start in range(0, len(jobs), CHUNK_SIZE):
        chunk = jobs[chunk_start : chunk_start + CHUNK_SIZE]

        # Flatten all sentences with ad-to-sentence mapping
        flat_sents: list[str] = []
        ad_ranges: list[tuple[int, int]] = []  # (start, end) in flat_sents

        for jid, sents in chunk:
            start = len(flat_sents)
            flat_sents.extend(sents)
            ad_ranges.append((start, len(flat_sents)))

        # Encode + predict in batches of sent_batch
        all_predictions: list[list[tuple[int, float]]] = [
            [] for _ in range(len(flat_sents))
        ]

        for b_start in range(0, len(flat_sents), sent_batch):
            batch = flat_sents[b_start : b_start + sent_batch]
            preds = matcher.predict_batch(
                batch, threshold, do_filter, max_skills,
            )
            for j, p in enumerate(preds):
                all_predictions[b_start + j] = p

        # Aggregate skills per ad: max score across sentences
        with open(out_path, "a", encoding="utf-8") as f:
            for ad_idx, (jid, _) in enumerate(chunk):
                rng_start, rng_end = ad_ranges[ad_idx]

                skill_best: dict[int, float] = {}
                for sent_idx in range(rng_start, rng_end):
                    for esco_idx, score in all_predictions[sent_idx]:
                        if esco_idx not in skill_best or score > skill_best[esco_idx]:
                            skill_best[esco_idx] = score

                # Sort by score, cap at max_skills
                sorted_skills = sorted(
                    skill_best.items(), key=lambda x: x[1], reverse=True,
                )[:max_skills]

                skills_out = [
                    {
                        "uri": matcher.esco_uris[idx],
                        "label": matcher.esco_labels[idx],
                        "score": round(sc, 4),
                    }
                    for idx, sc in sorted_skills
                ]
                total_skills_found += len(skills_out)
                total_ads += 1

                f.write(json.dumps(
                    {"id": jid, "skills": skills_out},
                    ensure_ascii=False,
                ) + "\n")

        jobs_bar.update(len(chunk))
        avg_sk = total_skills_found / total_ads if total_ads else 0
        jobs_bar.set_postfix(avg_skills=f"{avg_sk:.1f}")

    jobs_bar.close()

    elapsed = time.time() - t0
    avg = total_skills_found / total_ads if total_ads else 0
    print(f"  [{name}] done — {total_ads:,} ads, "
          f"{total_skills_found:,} skills ({avg:.1f}/ad), "
          f"{elapsed:.0f}s")
    return total_ads, total_skills_found


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Map skill sentences to ESCO — ConTeXT-match")
    parser.add_argument("--file", type=str,
                        help="Single _skillsent.jsonl file name")
    parser.add_argument("--input-dir", type=str, default=str(INPUT_DIR))
    parser.add_argument("--esco-csv", type=str, default=str(ESCO_CSV),
                        help=f"ESCO skills CSV (default: {ESCO_CSV})")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"Match score threshold τ (default: {THRESHOLD})")
    parser.add_argument("--max-skills", type=int, default=MAX_SKILLS,
                        help=f"Max ESCO skills per job ad (default: {MAX_SKILLS})")
    parser.add_argument("--sent-batch", type=int, default=SENT_BATCH,
                        help=f"Sentences per GPU batch (default: {SENT_BATCH})")
    parser.add_argument("--no-redundancy-filter", action="store_true",
                        help="Skip redundancy filtering (paper: τ→0.53)")
    parser.add_argument("--limit", type=int,
                        help="Max ads per file (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview input data, no model")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    esco_path = Path(args.esco_csv)
    do_filter = not args.no_redundancy_filter

    if not esco_path.exists():
        raise SystemExit(f"ESCO CSV not found: {esco_path}")

    if args.file:
        files = [input_dir / args.file]
        for f in files:
            if not f.exists():
                raise SystemExit(f"Not found: {f}")
    else:
        files = sorted(input_dir.glob("*_skillsent.jsonl"))

    if not files:
        raise SystemExit(f"No *_skillsent.jsonl files in {input_dir}/")

    eff_threshold = args.threshold
    if not do_filter and args.threshold == THRESHOLD:
        eff_threshold = 0.53
        print(f"  (no redundancy filter → using paper threshold τ=0.53)")

    print("=" * 65)
    print("ConTeXT-match  —  ESCO skill mapping")
    print(f"Model      : {args.model}")
    print(f"ESCO CSV   : {esco_path}")
    print(f"Input dir  : {input_dir}")
    print(f"Files      : {len(files)}")
    for f in files:
        print(f"             {f.name}")
    print(f"Threshold  : {eff_threshold}")
    print(f"Redundancy : {'filter' if do_filter else 'disabled'}")
    print(f"Max skills : {args.max_skills}")
    print(f"Sent batch : {args.sent_batch}")
    print("=" * 65)

    # ── Load ESCO ─────────────────────────────────────────────────────────
    labels, descs, uris = load_esco(esco_path)

    # ── Dry run ───────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n[Dry run — no model loaded]\n")
        for fp in files:
            all_jobs = load_skillsent_file(fp)
            name = fp.stem.replace("_skillsent", "")
            total_sents = sum(len(ss) for _, ss in all_jobs)
            avg = total_sents / len(all_jobs) if all_jobs else 0
            print(f"  [{name}]  {len(all_jobs):,} ads, "
                  f"{total_sents:,} skill sents (avg {avg:.1f}/ad)")
            for jid, sents in all_jobs[:3]:
                print(f"    id={jid}  sents={len(sents)}:")
                for s in sents[:3]:
                    print(f"      \"{s[:90]}\"")
            print()
        return

    # ── Load model + encode ESCO ──────────────────────────────────────────
    import torch
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    matcher = ConTeXTMatcher(args.model, labels, uris, device)

    # ── Process files ─────────────────────────────────────────────────────
    from tqdm import tqdm

    t0 = time.time()
    grand_ads = 0
    grand_skills = 0

    for fp in tqdm(files, desc="Files", unit="file", dynamic_ncols=True):
        a, s = process_file(
            fp, matcher, eff_threshold, do_filter,
            args.max_skills, args.sent_batch, args.limit,
        )
        grand_ads += a
        grand_skills += s

    elapsed = time.time() - t0
    avg = grand_skills / grand_ads if grand_ads else 0
    print(f"\n{'='*65}")
    print(f"ALL DONE — {grand_ads:,} ads, "
          f"{grand_skills:,} skills ({avg:.1f}/ad avg)")
    print(f"Time: {elapsed:.0f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
