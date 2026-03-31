#!/usr/bin/env python3
"""
Merge metadata from the CORRECT offers.db (east_asia/offers.db) by URL matching
This offers.db has: job_title_english, description_english, skills, region_iso, skill_match
"""
import sqlite3
from datetime import datetime
from collections import defaultdict

# Correct database paths
OFFERS_DB = '/Users/michalpalinski/Desktop/east_asia/offers.db'
TARGET_DB = '/Users/michalpalinski/Desktop/east_asia/east_asia_2026/east_asia_job_ads.db'

# Columns to merge from offers.db (the ones user specifically requested)
COLUMNS_TO_MERGE = [
    'job_title_english',
    'description_english', 
    'skills',
    'region_iso',
    'skill_match'
]

# Additional useful columns
EXTRA_COLUMNS = [
    'skills_list',
    'isic_section',
    'labels',
    'hierarchies',
    'skill_types'
]

ALL_COLUMNS = COLUMNS_TO_MERGE + EXTRA_COLUMNS

# Target tables mapping
TARGET_TABLES = [
    'jobads_jp', 'jobads_kr', 'jobads_th', 'jobads_malaysia',
    'jobads_mx', 'jobads_pl', 'jobads_tw', 'jobads_ph',
    'jobads_cl', 'jobads_vn', 'jobads_id', 'jobads_sg', 'jobads_in'
]

def add_columns_if_needed(db_path, table_name, columns):
    """Add columns to table if they don't exist"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get existing columns
    cursor.execute(f'PRAGMA table_info("{table_name}")')
    existing = {row[1] for row in cursor.fetchall()}
    
    added = []
    for col in columns:
        if col not in existing:
            try:
                cursor.execute(f'ALTER TABLE "{table_name}" ADD COLUMN "{col}" TEXT')
                added.append(col)
            except sqlite3.OperationalError as e:
                print(f"      Warning: Could not add {col}: {e}")
    
    conn.commit()
    conn.close()
    return added

def create_url_index(db_path):
    """Create index on job_url in offers table for faster lookups"""
    print("\n  Creating URL index on offers table...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_offers_url ON offers(job_url)")
        conn.commit()
        print("  ✓ Index created")
    except Exception as e:
        print(f"  ℹ️  Index note: {e}")
    finally:
        conn.close()

def load_offers_by_url(offers_db, columns):
    """Load offers data indexed by URL for fast lookup"""
    print("\n  Loading offers data into memory...")
    conn = sqlite3.connect(offers_db)
    cursor = conn.cursor()
    
    cols_str = ', '.join(['job_url'] + columns)
    cursor.execute(f"SELECT {cols_str} FROM offers WHERE job_url IS NOT NULL")
    
    offers_dict = {}
    count = 0
    for row in cursor:
        url = row[0]
        values = row[1:]
        offers_dict[url] = values
        count += 1
        if count % 50000 == 0:
            print(f"    Loaded {count:,} offers...")
    
    conn.close()
    print(f"  ✓ Loaded {count:,} offers with URLs")
    return offers_dict

def merge_table_by_url(target_db, table_name, offers_dict, columns):
    """Merge data for one target table using URL matching"""
    conn = sqlite3.connect(target_db)
    cursor = conn.cursor()
    
    # Get all URLs from target table
    cursor.execute(f'SELECT id, job_url FROM "{table_name}" WHERE job_url IS NOT NULL')
    target_rows = cursor.fetchall()
    
    matched = 0
    updated = 0
    
    # Build update query
    set_clause = ', '.join([f'"{col}" = ?' for col in columns])
    update_query = f'UPDATE "{table_name}" SET {set_clause} WHERE id = ?'
    
    updates = []
    for row_id, url in target_rows:
        if url in offers_dict:
            matched += 1
            values = offers_dict[url]
            updates.append(values + (row_id,))
            
            if len(updates) >= 1000:
                cursor.executemany(update_query, updates)
                updated += len(updates)
                updates = []
    
    # Final batch
    if updates:
        cursor.executemany(update_query, updates)
        updated += len(updates)
    
    conn.commit()
    conn.close()
    
    return len(target_rows), matched, updated

def main():
    print("=" * 80)
    print("MERGING METADATA FROM CORRECT OFFERS.DB BY URL")
    print("=" * 80)
    print(f"\nSource: {OFFERS_DB}")
    print(f"Target: {TARGET_DB}")
    print(f"\nColumns to merge:")
    for col in COLUMNS_TO_MERGE:
        print(f"  • {col}")
    print(f"\nExtra columns:")
    for col in EXTRA_COLUMNS:
        print(f"  • {col}")
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create index for faster lookups
    create_url_index(OFFERS_DB)
    
    # Load all offers into memory indexed by URL
    offers_dict = load_offers_by_url(OFFERS_DB, ALL_COLUMNS)
    
    print("\n" + "=" * 80)
    print("PROCESSING TABLES")
    print("=" * 80)
    
    results = []
    
    for table_name in TARGET_TABLES:
        print(f"\n{'─' * 80}")
        print(f"Table: {table_name}")
        print('─' * 80)
        
        try:
            # Add columns if needed
            print(f"  Checking/adding columns...")
            added = add_columns_if_needed(TARGET_DB, table_name, ALL_COLUMNS)
            if added:
                print(f"  ✓ Added: {', '.join(added)}")
            else:
                print(f"  ✓ All columns exist")
            
            # Merge by URL
            print(f"  Matching by URL...")
            total, matched, updated = merge_table_by_url(TARGET_DB, table_name, offers_dict, ALL_COLUMNS)
            
            coverage = (matched / total * 100) if total > 0 else 0
            print(f"  ✓ Total rows: {total:,}")
            print(f"  ✓ Matched: {matched:,} ({coverage:.1f}%)")
            print(f"  ✓ Updated: {updated:,}")
            
            results.append({
                'table': table_name,
                'status': 'SUCCESS',
                'total': total,
                'matched': matched,
                'updated': updated,
                'coverage': coverage
            })
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'table': table_name,
                'status': 'ERROR',
                'total': 0,
                'matched': 0,
                'updated': 0,
                'coverage': 0
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\n{'Table':<20} {'Status':<10} {'Total':<12} {'Matched':<12} {'Updated':<12} {'Coverage'}")
    print("─" * 80)
    
    total_rows = 0
    total_matched = 0
    total_updated = 0
    
    for r in results:
        print(f"{r['table']:<20} {r['status']:<10} {r['total']:>10,}  {r['matched']:>10,}  {r['updated']:>10,}  {r['coverage']:>6.1f}%")
        total_rows += r['total']
        total_matched += r['matched']
        total_updated += r['updated']
    
    print("─" * 80)
    overall_coverage = (total_matched / total_rows * 100) if total_rows > 0 else 0
    print(f"{'TOTAL':<20} {'':<10} {total_rows:>10,}  {total_matched:>10,}  {total_updated:>10,}  {overall_coverage:>6.1f}%")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == '__main__':
    main()
