#!/usr/bin/env python3
"""
Script to investigate database schemas and find tables with required columns
"""
import sqlite3
import os
from pathlib import Path

# Columns we're looking for
REQUIRED_COLUMNS = ['job_title_english', 'description_english', 'skills', 'region_iso', 'skill_match']

def get_table_columns(db_path, table_name):
    """Get column names for a table"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        conn.close()
        return columns
    except Exception as e:
        return []

def investigate_database(db_path):
    """Investigate a database and report on its structure"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if not tables:
            return None
        
        results = {}
        for table in tables:
            columns = get_table_columns(db_path, table)
            if columns:
                # Check which required columns are present
                present = [col for col in REQUIRED_COLUMNS if col in columns or 
                          col.replace('_', '') in [c.replace('_', '') for c in columns]]
                results[table] = {
                    'all_columns': columns,
                    'required_present': present,
                    'has_url': any('url' in col.lower() for col in columns),
                    'has_id': 'id' in columns
                }
        
        return results
    except Exception as e:
        print(f"Error investigating {db_path}: {e}")
        return None

# Main investigation
base_dir = Path('/Users/michalpalinski/Desktop/east_asia')
db_files = [
    'offers_with_skill_labels.db',
    'offers_nested.db',
    'flask/static/data/offers.db',
    'east_asia_2026/east_asia_job_ads.db'
]

print("=" * 80)
print("DATABASE INVESTIGATION REPORT")
print("=" * 80)
print(f"\nLooking for columns: {', '.join(REQUIRED_COLUMNS)}\n")

for db_file in db_files:
    db_path = base_dir / db_file
    if not db_path.exists():
        print(f"\n{db_file}: NOT FOUND")
        continue
    
    print(f"\n{'=' * 80}")
    print(f"DATABASE: {db_file}")
    print(f"Size: {db_path.stat().st_size / 1024 / 1024:.2f} MB")
    print('=' * 80)
    
    results = investigate_database(str(db_path))
    if not results:
        print("  No tables found or database is empty")
        continue
    
    for table_name, info in results.items():
        print(f"\nTable: {table_name}")
        print(f"  Has URL column: {info['has_url']}")
        print(f"  Has ID column: {info['has_id']}")
        print(f"  Required columns present: {', '.join(info['required_present']) if info['required_present'] else 'NONE'}")
        print(f"  All columns ({len(info['all_columns'])}): {', '.join(info['all_columns'][:10])}" + 
              ("..." if len(info['all_columns']) > 10 else ""))

print("\n" + "=" * 80)
