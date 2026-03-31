#!/usr/bin/env python3
"""
Merge metadata from offers.db into east_asia_job_ads.db based on URL matching
Uses the correct offers.db from east_asia folder with single 'offers' table
"""
import sqlite3
from pathlib import Path
from datetime import datetime

# Database paths - CORRECTED
OFFERS_DB = '/Users/michalpalinski/Desktop/east_asia/offers.db'
TARGET_DB = '/Users/michalpalinski/Desktop/east_asia/east_asia_2026/east_asia_job_ads.db'

# Target tables (separate per country in 2026 db)
TARGET_TABLES = [
    'jobads_jp', 'jobads_kr', 'jobads_th', 'jobads_malaysia',
    'jobads_mx', 'jobads_pl', 'jobads_tw', 'jobads_ph',
    'jobads_cl', 'jobads_vn', 'jobads_id', 'jobads_sg', 'jobads_in'
]

# Columns to copy from offers.db
COLUMNS_TO_COPY = [
    'job_title_english',
    'description_english',
    'skills',
    'region_iso',
    'skill_match'
]

def add_columns_to_target(target_db, table_name, columns):
    """Add new columns to target table if they don't exist"""
    conn = sqlite3.connect(target_db)
    cursor = conn.cursor()
    
    # Get existing columns
    cursor.execute(f'PRAGMA table_info("{table_name}")')
    existing_cols = [row[1] for row in cursor.fetchall()]
    
    added = []
    for col in columns:
        if col not in existing_cols:
            try:
                cursor.execute(f'ALTER TABLE "{table_name}" ADD COLUMN {col} TEXT')
                added.append(col)
            except sqlite3.OperationalError as e:
                print(f"    ⚠️  Could not add column {col}: {e}")
    
    conn.commit()
    conn.close()
    return added

def merge_by_url(offers_db, target_db, target_table, columns):
    """Merge data by matching job_url"""
    conn_offers = sqlite3.connect(offers_db)
    conn_target = sqlite3.connect(target_db)
    
    cursor_offers = conn_offers.cursor()
    cursor_target = conn_target.cursor()
    
    # Build column list for SELECT
    cols_str = ', '.join(['job_url'] + columns)
    
    # Get all URLs from target table
    cursor_target.execute(f'SELECT job_url FROM "{target_table}" WHERE job_url IS NOT NULL')
    target_urls = set(row[0] for row in cursor_target.fetchall())
    
    print(f"  Target URLs: {len(target_urls):,}")
    
    # Get data from offers table
    cursor_offers.execute(f'SELECT {cols_str} FROM offers WHERE job_url IS NOT NULL')
    
    matched = 0
    updated = 0
    batch = []
    batch_size = 1000
    
    for row in cursor_offers:
        url = row[0]
        values = row[1:]
        
        # Check if this URL exists in target
        if url in target_urls:
            matched += 1
            batch.append((values, url))
            
            # Execute batch updates
            if len(batch) >= batch_size:
                set_clause = ', '.join([f"{col} = ?" for col in columns])
                for vals, u in batch:
                    try:
                        cursor_target.execute(
                            f'UPDATE "{target_table}" SET {set_clause} WHERE job_url = ?',
                            vals + (u,)
                        )
                        updated += 1
                    except Exception as e:
                        print(f"    ⚠️  Error updating URL {u}: {e}")
                
                conn_target.commit()
                batch = []
    
    # Execute remaining batch
    if batch:
        set_clause = ', '.join([f"{col} = ?" for col in columns])
        for vals, u in batch:
            try:
                cursor_target.execute(
                    f'UPDATE "{target_table}" SET {set_clause} WHERE job_url = ?',
                    vals + (u,)
                )
                updated += 1
            except Exception as e:
                print(f"    ⚠️  Error updating URL {u}: {e}")
        
        conn_target.commit()
    
    conn_offers.close()
    conn_target.close()
    
    return matched, updated

def get_table_row_count(db_path, table_name):
    """Get row count for a table"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
    count = cursor.fetchone()[0]
    conn.close()
    return count

def main():
    print("=" * 80)
    print("MERGING METADATA FROM OFFERS.DB TO EAST_ASIA_JOB_ADS.DB")
    print("=" * 80)
    print(f"\nSource: {OFFERS_DB}")
    print(f"Target: {TARGET_DB}")
    print(f"\nColumns to merge: {', '.join(COLUMNS_TO_COPY)}")
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if databases exist
    if not Path(OFFERS_DB).exists():
        print(f"\n❌ ERROR: Offers database not found at {OFFERS_DB}")
        return
    
    if not Path(TARGET_DB).exists():
        print(f"\n❌ ERROR: Target database not found at {TARGET_DB}")
        return
    
    # Get total rows from offers table
    offers_count = get_table_row_count(OFFERS_DB, 'offers')
    print(f"\nSource table 'offers': {offers_count:,} rows")
    
    print("\n" + "=" * 80)
    print("PROCESSING TABLES")
    print("=" * 80)
    
    results = []
    
    for target_table in TARGET_TABLES:
        country = target_table.replace('jobads_', '').upper()
        print(f"\n{'─' * 80}")
        print(f"Country: {country} → {target_table}")
        print('─' * 80)
        
        try:
            # Get target table info
            target_count = get_table_row_count(TARGET_DB, target_table)
            print(f"  Target rows: {target_count:,}")
            
            # Add columns to target if needed
            print(f"  Adding columns to target table...")
            added = add_columns_to_target(TARGET_DB, target_table, COLUMNS_TO_COPY)
            if added:
                print(f"  ✓ Added columns: {', '.join(added)}")
            else:
                print(f"  ✓ All columns already exist")
            
            # Merge data by URL
            print(f"  Merging data by URL...")
            matched, updated = merge_by_url(OFFERS_DB, TARGET_DB, target_table, COLUMNS_TO_COPY)
            
            print(f"  ✓ Matched: {matched:,} records")
            print(f"  ✓ Updated: {updated:,} records")
            
            coverage = (matched / target_count * 100) if target_count > 0 else 0
            
            results.append({
                'country': country,
                'status': 'SUCCESS',
                'matched': matched,
                'updated': updated,
                'target_rows': target_count,
                'coverage': coverage
            })
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'country': country,
                'status': 'ERROR',
                'matched': 0,
                'updated': 0,
                'target_rows': 0,
                'coverage': 0
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Country':<12} {'Status':<10} {'Target':<12} {'Matched':<10} {'Updated':<10} {'Coverage':<10}")
    print("─" * 80)
    
    total_matched = 0
    total_updated = 0
    total_target = 0
    
    for r in results:
        print(f"{r['country']:<12} {r['status']:<10} {r['target_rows']:>10,}  {r['matched']:>8,}  {r['updated']:>8,}  {r['coverage']:>8.1f}%")
        total_matched += r['matched']
        total_updated += r['updated']
        total_target += r['target_rows']
    
    print("─" * 80)
    overall_coverage = (total_matched / total_target * 100) if total_target > 0 else 0
    print(f"{'TOTAL':<12} {'':<10} {total_target:>10,}  {total_matched:>8,}  {total_updated:>8,}  {overall_coverage:>8.1f}%")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == '__main__':
    main()
