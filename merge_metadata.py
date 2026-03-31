#!/usr/bin/env python3
"""
Merge metadata from offers.db into east_asia_job_ads.db based on URL matching
"""
import sqlite3
from pathlib import Path
from datetime import datetime

# Database paths
OFFERS_DB = '/Users/michalpalinski/Desktop/east_asia/flask/static/data/offers.db'
TARGET_DB = '/Users/michalpalinski/Desktop/east_asia/east_asia_2026/east_asia_job_ads.db'

# Country mapping: offers.db table -> target db table
COUNTRY_MAPPING = {
    'jp': 'jobads_jp',
    'kr': 'jobads_kr',
    'th': 'jobads_th',
    'my': 'jobads_malaysia',
    'mx': 'jobads_mx',
    'pl': 'jobads_pl',
    'tw': 'jobads_tw',
    'ph': 'jobads_ph',
    'cl': 'jobads_cl',
    'vn': 'jobads_vn',
    'id': 'jobads_id',
    'sg': 'jobads_sg',
    'in': 'jobads_in'
}

# Columns to copy from offers.db (excluding id)
COLUMNS_TO_COPY = [
    'job_title_english',
    'region_iso',
    'esco_codes',
    'esco_occupation_title_semantic',
    'esco_occupation_code_semantic',
    'esco_occupation_similarity_semantic',
    'isic_section'
]

def get_table_info(db_path, table_name):
    """Get basic info about a table"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get column names (use quotes to handle reserved keywords)
    cursor.execute(f"PRAGMA table_info(\"{table_name}\")")
    columns = [row[1] for row in cursor.fetchall()]
    
    # Get row count (use quotes to handle reserved keywords)
    cursor.execute(f"SELECT COUNT(*) FROM \"{table_name}\"")
    count = cursor.fetchone()[0]
    
    conn.close()
    return columns, count

def check_url_column(db_path, table_name):
    """Check if table has a URL-like column"""
    columns, _ = get_table_info(db_path, table_name)
    url_columns = [col for col in columns if 'url' in col.lower()]
    return url_columns

def quote_table(table_name):
    """Quote table name to handle reserved keywords"""
    return f'"{table_name}"'

def add_columns_to_target(target_db, table_name, columns):
    """Add new columns to target table if they don't exist"""
    conn = sqlite3.connect(target_db)
    cursor = conn.cursor()
    
    # Get existing columns (use quotes to handle reserved keywords)
    cursor.execute(f"PRAGMA table_info(\"{table_name}\")")
    existing_cols = [row[1] for row in cursor.fetchall()]
    
    added = []
    for col in columns:
        if col not in existing_cols:
            try:
                cursor.execute(f"ALTER TABLE \"{table_name}\" ADD COLUMN {col} TEXT")
                added.append(col)
            except sqlite3.OperationalError as e:
                print(f"    ⚠️  Could not add column {col}: {e}")
    
    conn.commit()
    conn.close()
    return added

def merge_by_id(offers_db, target_db, offers_table, target_table, columns):
    """Merge data by matching ID (since URL not in offers.db)"""
    conn_offers = sqlite3.connect(offers_db)
    conn_target = sqlite3.connect(target_db)
    
    cursor_offers = conn_offers.cursor()
    cursor_target = conn_target.cursor()
    
    # Get data from offers table (use quotes to handle reserved keywords)
    cols_str = ', '.join(['id'] + columns)
    cursor_offers.execute(f"SELECT {cols_str} FROM \"{offers_table}\"")
    offers_data = cursor_offers.fetchall()
    
    matched = 0
    updated = 0
    
    for row in offers_data:
        offer_id = row[0]
        values = row[1:]
        
        # Check if this ID exists in target (use quotes to handle reserved keywords)
        cursor_target.execute(f"SELECT id FROM \"{target_table}\" WHERE id = ?", (offer_id,))
        if cursor_target.fetchone():
            matched += 1
            
            # Build UPDATE query (use quotes to handle reserved keywords)
            set_clause = ', '.join([f"{col} = ?" for col in columns])
            try:
                cursor_target.execute(
                    f"UPDATE \"{target_table}\" SET {set_clause} WHERE id = ?",
                    values + (offer_id,)
                )
                updated += 1
            except Exception as e:
                print(f"    ⚠️  Error updating ID {offer_id}: {e}")
    
    conn_target.commit()
    conn_offers.close()
    conn_target.close()
    
    return matched, updated

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
    
    print("\n" + "=" * 80)
    print("PROCESSING TABLES")
    print("=" * 80)
    
    results = []
    
    for offers_table, target_table in COUNTRY_MAPPING.items():
        print(f"\n{'─' * 80}")
        print(f"Country: {offers_table.upper()} → {target_table}")
        print('─' * 80)
        
        try:
            # Get source table info
            offers_cols, offers_count = get_table_info(OFFERS_DB, offers_table)
            print(f"  Source: {offers_count:,} rows")
            
            # Get target table info
            target_cols, target_count = get_table_info(TARGET_DB, target_table)
            print(f"  Target: {target_count:,} rows")
            
            # Check which columns exist in source
            available_cols = [col for col in COLUMNS_TO_COPY if col in offers_cols]
            missing_cols = [col for col in COLUMNS_TO_COPY if col not in offers_cols]
            
            if missing_cols:
                print(f"  ℹ️  Columns not in source: {', '.join(missing_cols)}")
            
            if not available_cols:
                print(f"  ⚠️  No columns to merge!")
                results.append({
                    'country': offers_table.upper(),
                    'status': 'SKIPPED',
                    'matched': 0,
                    'updated': 0,
                    'source_rows': offers_count,
                    'target_rows': target_count
                })
                continue
            
            # Add columns to target if needed
            print(f"  Adding columns to target table...")
            added = add_columns_to_target(TARGET_DB, target_table, available_cols)
            if added:
                print(f"  ✓ Added columns: {', '.join(added)}")
            else:
                print(f"  ✓ All columns already exist")
            
            # Merge data by ID
            print(f"  Merging data by ID...")
            matched, updated = merge_by_id(OFFERS_DB, TARGET_DB, offers_table, target_table, available_cols)
            
            print(f"  ✓ Matched: {matched:,} records")
            print(f"  ✓ Updated: {updated:,} records")
            
            results.append({
                'country': offers_table.upper(),
                'status': 'SUCCESS',
                'matched': matched,
                'updated': updated,
                'source_rows': offers_count,
                'target_rows': target_count
            })
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append({
                'country': offers_table.upper(),
                'status': 'ERROR',
                'matched': 0,
                'updated': 0,
                'source_rows': 0,
                'target_rows': 0
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Country':<10} {'Status':<10} {'Source':<12} {'Target':<12} {'Matched':<10} {'Updated':<10}")
    print("─" * 80)
    
    total_matched = 0
    total_updated = 0
    
    for r in results:
        print(f"{r['country']:<10} {r['status']:<10} {r['source_rows']:>10,}  {r['target_rows']:>10,}  {r['matched']:>8,}  {r['updated']:>8,}")
        total_matched += r['matched']
        total_updated += r['updated']
    
    print("─" * 80)
    print(f"{'TOTAL':<10} {'':<10} {'':<12} {'':<12} {total_matched:>8,}  {total_updated:>8,}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == '__main__':
    main()
