#!/usr/bin/env python3
"""
Match skill terms from *_skills.jsonl to ESCO codes and aggregate
to top-level categories:
  - Skills (S-codes):     first 2 chars → S1 … S8 + "T" (transversal) + "L" (language)
  - Knowledge (numeric):  first 3 chars → 061, 052, …

Strategy (applied in order for each unique skill term):
  1. Exact lowercase title match
  2. Longest ESCO title that is a substring of the skill term
  3. rapidfuzz token_set_ratio ≥ FUZZY_CUTOFF (only for top-N by frequency)
  4. Unmatched → "Other"

Outputs:
  skills_esco_agg.json  — per-country shares by ESCO top-level category
"""

import json, sqlite3, re
from pathlib import Path
from collections import Counter, defaultdict

from rapidfuzz import process as rfprocess
from rapidfuzz.fuzz import token_set_ratio

# ── Config ────────────────────────────────────────────────────────────────────
ESCO_DB      = Path('/Users/michalpalinski/Desktop/east_asia/comprehensive_esco.db')
SKILLS_DIR   = Path('/Users/michalpalinski/Desktop/east_asia/east_asia_2026/gemma_results_full')
OUT_FILE     = Path('/Users/michalpalinski/Desktop/east_asia/east_asia_2026/skills_esco_agg.json')

FUZZY_CUTOFF = 82          # token_set_ratio threshold for fuzzy match
TOP_N_FUZZY  = 15_000      # apply fuzzy only to most-frequent unique terms
INCLUDE_LANGUAGE = True    # include language skills (L-codes → "Language")

COUNTRY_LABELS = {
    'in': 'India', 'jp': 'Japan', 'kr': 'South Korea',
    'malaysia': 'Malaysia', 'mx': 'Mexico', 'ph': 'Philippines',
    'pl': 'Poland', 'sg': 'Singapore', 'th': 'Thailand',
    'tw': 'Taiwan', 'vn': 'Vietnam',
}

# ── Load ESCO ─────────────────────────────────────────────────────────────────
def load_esco():
    conn = sqlite3.connect(ESCO_DB)
    cur = conn.cursor()
    cur.execute("""
        SELECT code, title, skill_type, level, parent_code
        FROM esco_concepts
    """)
    rows = cur.fetchall()
    conn.close()

    by_title = {}      # lower(title) → (code, skill_type)
    all_codes = {}     # code → (title, skill_type, level, parent_code)
    all_titles = []    # for rapidfuzz list
    all_title_codes = []

    for code, title, skill_type, level, parent_code in rows:
        lower = title.lower().strip()
        by_title[lower] = (code, skill_type)
        all_codes[code] = (title, skill_type, level, parent_code)
        all_titles.append(lower)
        all_title_codes.append((code, skill_type))

    return by_title, all_codes, all_titles, all_title_codes

# ── Derive top-level category ─────────────────────────────────────────────────
def top_category(code: str, skill_type: str) -> str | None:
    """Return S1-S8, 3-char-knowledge-code, 'Language', 'Transversal', or None."""
    if skill_type == 'language_skills':
        return 'Language' if INCLUDE_LANGUAGE else None
    if skill_type == 'transversal_skills':
        return 'Transversal'
    if skill_type == 'skills':
        # S1.4.2.8 → S1
        m = re.match(r'^(S\d)', code)
        return m.group(1) if m else None
    if skill_type == 'knowledge':
        # 0613.59 → 061 (3 chars)
        digits = re.sub(r'[^0-9]', '', code)
        return digits[:3] if len(digits) >= 3 else (digits.zfill(3) if digits else None)
    return None

# ── Normalize for matching ────────────────────────────────────────────────────
_strip_re = re.compile(r'[^a-z0-9 ]')

def norm(s: str) -> str:
    return _strip_re.sub('', s.lower()).strip()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading ESCO database …")
    by_title, all_codes, all_titles, all_title_codes = load_esco()
    print(f"  {len(by_title):,} ESCO concepts loaded")

    # Normalised lookup (strip punctuation)
    norm_lookup = {norm(t): v for t, v in by_title.items()}

    # Pre-sort titles by length descending for substring match
    titles_sorted_desc = sorted(by_title.items(), key=lambda x: -len(x[0]))

    print("Reading skill files …")
    # Count every (country, skill_term) occurrence
    country_skill_counts: dict[str, Counter] = {}
    global_counter: Counter = Counter()

    for f in sorted(SKILLS_DIR.glob('*_skills.jsonl')):
        stem = f.stem  # e.g. jobads_jp_skills
        country_key = stem.removeprefix('jobads_').removesuffix('_skills')
        if country_key not in COUNTRY_LABELS:
            print(f"  Skipping unknown country file: {f.name}")
            continue
        c: Counter = Counter()
        with open(f) as fh:
            for line in fh:
                row = json.loads(line)
                for s in row.get('skills', []):
                    if s and isinstance(s, str):
                        s = s.strip()
                        c[s] += 1
        country_skill_counts[country_key] = c
        global_counter.update(c)
        print(f"  {COUNTRY_LABELS[country_key]:12s}: {sum(c.values()):>9,} instances, {len(c):>7,} unique")

    print(f"\nTotal: {sum(global_counter.values()):,} instances, {len(global_counter):,} unique terms")

    # ── Build match cache ─────────────────────────────────────────────────────
    print("\nMatching skill terms to ESCO …")
    match_cache: dict[str, tuple[str, str] | None] = {}  # term → (code, skill_type)

    top_terms = [t for t, _ in global_counter.most_common()]

    def find_match(term: str):
        lower = term.lower().strip()
        n = norm(term)

        # 1. Exact
        if lower in by_title:
            return by_title[lower]
        if n in norm_lookup:
            return norm_lookup[n]

        # 2. Longest ESCO title that is a substring of term (lower)
        for esco_title, val in titles_sorted_desc:
            if len(esco_title) >= 4 and esco_title in lower:
                return val

        return None  # no match yet (fuzzy applied later for top-N)

    # Pass 1: exact + substring (fast, all terms)
    unmatched = []
    for term in top_terms:
        result = find_match(term)
        if result:
            match_cache[term] = result
        else:
            unmatched.append(term)

    print(f"  After exact+substring: {len(match_cache):,} matched, {len(unmatched):,} unmatched")

    # Pre-build normalized ESCO titles list once for rapidfuzz
    esco_keys = list(by_title.keys())
    esco_keys_normed = [norm(t) for t in esco_keys]

    # Pass 2: rapidfuzz for top-N unmatched by global frequency
    unmatched_top = [t for t in unmatched if global_counter[t] >= 3][:TOP_N_FUZZY]
    print(f"  Running rapidfuzz on {len(unmatched_top):,} high-frequency unmatched terms …")

    for i, term in enumerate(unmatched_top):
        if i % 1000 == 0:
            print(f"    {i:,}/{len(unmatched_top):,} …", flush=True)
        n = norm(term)
        if not n or len(n) <= 2:
            match_cache[term] = None
            continue
        result = rfprocess.extractOne(
            n,
            esco_keys_normed,
            scorer=token_set_ratio,
            score_cutoff=FUZZY_CUTOFF,
        )
        if result:
            matched_title = esco_keys[result[2]]
            match_cache[term] = by_title[matched_title]
        else:
            match_cache[term] = None

    # remaining unmatched → None
    for term in unmatched:
        if term not in match_cache:
            match_cache[term] = None

    total_matched = sum(1 for v in match_cache.values() if v is not None)
    print(f"  Final: {total_matched:,} / {len(match_cache):,} unique terms matched "
          f"({total_matched/len(match_cache):.1%})")

    # Weighted coverage (by frequency)
    weighted_match = sum(global_counter[t] for t, v in match_cache.items() if v is not None)
    weighted_total = sum(global_counter.values())
    print(f"  Weighted coverage: {weighted_match:,} / {weighted_total:,} instances "
          f"({weighted_match/weighted_total:.1%})")

    # ── Aggregate per country ─────────────────────────────────────────────────
    print("\nAggregating …")

    # ESCO category metadata: code → title
    cat_titles: dict[str, str] = {}
    for code, (title, skill_type, level, parent_code) in all_codes.items():
        cat = top_category(code, skill_type)
        if cat and cat not in cat_titles:
            if skill_type == 'skills' and level == 1:
                cat_titles[cat] = title
            elif skill_type == 'knowledge' and len(code.replace('.', '').replace('-', '')) == 3:
                cat_titles[cat] = title
            elif cat in ('Language', 'Transversal'):
                cat_titles[cat] = cat

    # Manual fallback titles for S-codes and knowledge groups
    s_code_titles = {}
    for code, (title, skill_type, level, _) in all_codes.items():
        if skill_type == 'skills' and level == 1:
            s_code_titles[code] = title

    # 3-digit knowledge titles (level == 2)
    k3_titles = {}
    for code, (title, skill_type, level, _) in all_codes.items():
        if skill_type == 'knowledge' and level == 2:
            k3 = code[:3] if len(code) >= 3 else code
            k3_titles[k3] = title

    result: dict = {}

    for country_key, skill_counter in country_skill_counts.items():
        # Count instances by category
        cat_counts: Counter = Counter()
        total_instances = 0
        matched_instances = 0

        for term, count in skill_counter.items():
            total_instances += count
            match = match_cache.get(term)
            if match:
                code, skill_type = match
                cat = top_category(code, skill_type)
                if cat:
                    cat_counts[cat] += count
                    matched_instances += count
            # unmatched → skip (goes into "Other" implicitly via total)

        cat_counts['Other'] = total_instances - matched_instances

        result[country_key] = {
            'label': COUNTRY_LABELS[country_key],
            'total': total_instances,
            'matched': matched_instances,
            'categories': dict(cat_counts),
        }

    # Collect all categories seen
    all_cats = set()
    for v in result.values():
        all_cats.update(v['categories'].keys())

    # Separate skills, knowledge, language, transversal, other
    skill_cats = sorted([c for c in all_cats if c.startswith('S')])
    knowledge_cats = sorted([c for c in all_cats if re.match(r'^\d', c)])
    other_cats = [c for c in all_cats if c in ('Language', 'Transversal', 'Other')]

    # Add category titles
    category_meta = {}
    for c in skill_cats:
        category_meta[c] = s_code_titles.get(c, c)
    for c in knowledge_cats:
        category_meta[c] = k3_titles.get(c, c)
    category_meta['Language'] = 'Language skills'
    category_meta['Transversal'] = 'Transversal skills'
    category_meta['Other'] = 'Unclassified'

    output = {
        'countries': result,
        'category_meta': category_meta,
        'skill_cats': skill_cats,
        'knowledge_cats': knowledge_cats,
        'other_cats': other_cats,
    }

    OUT_FILE.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_FILE}")
    print("Done.")

if __name__ == '__main__':
    main()
