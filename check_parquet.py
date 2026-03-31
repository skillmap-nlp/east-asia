#!/usr/bin/env python3
"""
Check parquet files for the columns we need
"""
import pandas as pd
from pathlib import Path

data_dir = Path('/Users/michalpalinski/Desktop/east_asia/data/combined')
parquet_files = list(data_dir.glob('*jp*.parquet')) + list(data_dir.glob('*combined*.parquet'))[:3]

print("=" * 80)
print("CHECKING PARQUET FILES FOR REQUIRED COLUMNS")
print("=" * 80)

required_cols = ['job_title_english', 'description_english', 'skills', 'region_iso', 'skill_match', 'url']

for pq_file in parquet_files[:5]:  # Check first 5
    print(f"\n{pq_file.name}")
    try:
        df = pd.read_parquet(pq_file, engine='pyarrow')
        print(f"  Rows: {len(df):,}")
        print(f"  Columns ({len(df.columns)}): {', '.join(list(df.columns)[:15])}" + 
              ("..." if len(df.columns) > 15 else ""))
        
        # Check for required columns
        found = [col for col in required_cols if col in df.columns or 
                col.replace('_', '') in [c.replace('_', '') for c in df.columns]]
        if found:
            print(f"  ✓ Found: {', '.join(found)}")
        
        # Check for URL columns
        url_cols = [col for col in df.columns if 'url' in col.lower()]
        if url_cols:
            print(f"  URLs: {', '.join(url_cols)}")
            
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "=" * 80)
