#!/usr/bin/env python3
"""
Script to rename tables in an existing SQLite database
Renames all tables to: jobads_<isocode>_<original_name>
"""

import sqlite3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
SQLITE_DB = os.getenv('SQLITE_DB', 'eaast_asia_job_ads.db')
ISO_CODE = os.getenv('ISO_CODE', '').lower()


def get_all_tables(cursor):
    """Get all table names from SQLite database"""
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    return [row[0] for row in cursor.fetchall()]


def get_new_table_name(old_name, iso_code):
    """Generate new table name with jobads_ prefix, removing 'indeed' from name"""
    # Skip if already renamed
    if old_name.startswith('jobads_'):
        print(f"  ⊗ Skipping {old_name} (already renamed)")
        return None
    
    # Remove 'indeed' from the name (case-insensitive)
    cleaned_name = old_name.replace('indeed', '').replace('Indeed', '').replace('INDEED', '')
    
    # Clean up any resulting double underscores or leading/trailing underscores
    cleaned_name = cleaned_name.replace('__', '_').strip('_')
    
    # If name is empty after cleaning, use 'table'
    if not cleaned_name:
        cleaned_name = 'table'
    
    if iso_code:
        return f"jobads_{iso_code}_{cleaned_name}"
    else:
        return f"jobads_{cleaned_name}"


def rename_tables():
    """Rename all tables in the SQLite database"""
    
    if not os.path.exists(SQLITE_DB):
        print(f"✗ Error: Database file not found: {SQLITE_DB}")
        print("\nMake sure:")
        print(f"  1. The file exists at: {os.path.abspath(SQLITE_DB)}")
        print(f"  2. Or update SQLITE_DB in your .env file")
        return
    
    print(f"Opening SQLite database: {SQLITE_DB}")
    print(f"Absolute path: {os.path.abspath(SQLITE_DB)}\n")
    
    conn = sqlite3.connect(SQLITE_DB)
    cursor = conn.cursor()
    
    try:
        # Get all existing tables
        tables = get_all_tables(cursor)
        
        if not tables:
            print("✗ No tables found in database")
            return
        
        print(f"Found {len(tables)} tables\n")
        
        if ISO_CODE:
            print(f"Renaming with ISO code: {ISO_CODE.upper()}")
            print(f"Format: jobads_{ISO_CODE}_<table_name>\n")
        else:
            print(f"Renaming with prefix: jobads_<table_name>\n")
        
        print("=" * 70)
        
        renamed_count = 0
        skipped_count = 0
        
        # Rename each table
        for old_name in tables:
            new_name = get_new_table_name(old_name, ISO_CODE)
            
            if new_name is None:
                skipped_count += 1
                continue
            
            print(f"📊 {old_name} → {new_name}")
            
            try:
                cursor.execute(f"ALTER TABLE `{old_name}` RENAME TO `{new_name}`")
                conn.commit()
                print(f"  ✓ Renamed successfully\n")
                renamed_count += 1
            except sqlite3.Error as e:
                print(f"  ✗ Error: {e}\n")
        
        print("=" * 70)
        print(f"\n✓ Renaming complete!")
        print(f"  Renamed: {renamed_count} tables")
        print(f"  Skipped: {skipped_count} tables")
        
        # Show final table list
        print(f"\nFinal tables in database:")
        final_tables = get_all_tables(cursor)
        for table in final_tables:
            print(f"  - {table}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()


if __name__ == '__main__':
    rename_tables()
