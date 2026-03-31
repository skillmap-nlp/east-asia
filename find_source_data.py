#!/usr/bin/env python3
"""
Find source data files with required columns for each country
"""
import pandas as pd
from pathlib import Path

data_dir = Path('/Users/michalpalinski/Desktop/east_asia/data/combined')
countries = ['cl', 'id', 'in', 'jp', 'kr', 'malaysia', 'mx', 'ph', 'pl', 'sg', 'th', 'tw', 'vn']
required_cols = ['job_title_english', 'description_english', 'skills', 'region_iso', 'skill_match', 'job_url']

print("=" * 80)
print("FINDING SOURCE DATA FOR EACH COUNTRY")
print("=" * 80)

results = {}

for country in countries:
    print(f"\nCountry: {country.upper()}")
    
    # Look for files matching this country
    patterns = [
        f"*{country}*.parquet",
        f"{country}_*.parquet"
    ]
    
    found_files = []
    for pattern in patterns:
        found_files.extend(list(data_dir.glob(pattern)))
    
    found_files = list(set(found_files))  # Remove duplicates
    
    if not found_files:
        print("  ❌ No files found")
        continue
    
    # Check each file for required columns
    for file in found_files:
        try:
            df = pd.read_parquet(file, engine='pyarrow')
            cols = set(df.columns)
            
            has_cols = {col: col in cols for col in required_cols}
            missing = [col for col, present in has_cols.items() if not present]
            
            if len(missing) == 0:
                print(f"  ✓ {file.name} ({len(df):,} rows) - ALL COLUMNS PRESENT")
                results[country] = file
            elif len(missing) <= 2:
                print(f"  ~ {file.name} ({len(df):,} rows) - Missing: {', '.join(missing)}")
        except Exception as e:
            pass

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
for country, file in results.items():
    print(f"{country.upper()}: {file.name}")

if not results:
    print("No complete source files found. Checking what's available...")
    # Check the jp_onet file specifically
    jp_file = data_dir / 'jp_onet.parquet'
    if jp_file.exists():
        df = pd.read_parquet(jp_file)
        print(f"\njp_onet.parquet columns: {', '.join(df.columns.tolist())}")
