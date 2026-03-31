#!/usr/bin/env python3
"""
Delete specified columns from all tables in east_asia_job_ads.db
"""
import sqlite3
from datetime import datetime

TARGET_DB = '/Users/michalpalinski/Desktop/east_asia/east_asia_2026/east_asia_job_ads.db'

# Columns to delete
COLUMNS_TO_DELETE = ['skill_types', 'hierarchies', 'labels', 'skills_list']

# All target tables
TARGET_TABLES = [
    'jobads_jp', 'jobads_kr', 'jobads_th', 'jobads_malaysia',
    'jobads_mx', 'jobads_pl', 'jobads_tw', 'jobads_ph',
    'jobads_cl', 'jobads_vn', 'jobads_id', 'jobads_sg', 'jobads_in'
]

def check_column_exists(db_path, table_name, column_name):
    """Check if a column exists in a table"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f'PRAGMA table_info("{table_name}")')
    columns = {row[1] for row in cursor.fetchall()}
    conn.close()
    return column_name in columns

def delete_column(db_path, table_name, column_name):
    """Delete a column from a table"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute(f'ALTER TABLE "{table_name}" DROP COLUMN "{column_name}"')
        conn.commit()
        success = True
    except sqlite3.OperationalError as e:
        success = False
        error = str(e)
    finally:
        conn.close()
    
    return success

def main():
    print("=" * 80)
    print("DELETING COLUMNS FROM ALL TABLES")
    print("=" * 80)
    print(f"\nDatabase: {TARGET_DB}")
    print(f"\nColumns to delete:")
    for col in COLUMNS_TO_DELETE:
        print(f"  • {col}")
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "=" * 80)
    print("PROCESSING TABLES")
    print("=" * 80)
    
    results = []
    
    for table_name in TARGET_TABLES:
        print(f"\n{'─' * 80}")
        print(f"Table: {table_name}")
        print('─' * 80)
        
        deleted = []
        skipped = []
        
        for column in COLUMNS_TO_DELETE:
            # Check if column exists
            if check_column_exists(TARGET_DB, table_name, column):
                # Try to delete it
                success = delete_column(TARGET_DB, table_name, column)
                if success:
                    deleted.append(column)
                    print(f"  ✓ Deleted: {column}")
                else:
                    print(f"  ⚠️  Failed to delete: {column}")
            else:
                skipped.append(column)
        
        if skipped:
            print(f"  ℹ️  Already absent: {', '.join(skipped)}")
        
        results.append({
            'table': table_name,
            'deleted': len(deleted),
            'skipped': len(skipped),
            'columns_deleted': deleted
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Table':<20} {'Deleted':<10} {'Skipped':<10} {'Columns Removed'}")
    print("─" * 80)
    
    total_deleted = 0
    
    for r in results:
        cols_str = ', '.join(r['columns_deleted']) if r['columns_deleted'] else 'none'
        print(f"{r['table']:<20} {r['deleted']:>8}  {r['skipped']:>8}  {cols_str}")
        total_deleted += r['deleted']
    
    print("─" * 80)
    print(f"{'TOTAL':<20} {total_deleted:>8}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == '__main__':
    main()
