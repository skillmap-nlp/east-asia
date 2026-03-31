#!/usr/bin/env python3
"""
Create wide format analysis of job ads data from SQLite database
Generates multiple wide format views for analysis
"""

import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path


# Configuration
DB_PATH = "east_asia_job_ads.db"
OUTPUT_XLSX = "job_ads_wide_format_analysis.xlsx"


def list_tables(conn, prefix="jobads_"):
    """Get all tables with optional prefix filter"""
    query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    tables = [r[0] for r in conn.execute(query).fetchall()]
    if prefix:
        tables = [t for t in tables if t.startswith(prefix)]
    return sorted(tables)


def get_columns(conn, table):
    """Get all column names for a table"""
    info = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    return [row[1] for row in info]


def year_month_counts_wide(conn, tables, date_col="date_posted"):
    """Create wide format of year-month counts by table"""
    print("\n📊 Creating year-month counts (wide format)...")
    
    per_table = {}
    for table in tables:
        try:
            df = pd.read_sql_query(f'SELECT "{date_col}" FROM "{table}"', conn)
            
            if date_col not in df.columns:
                print(f"  ⊗ Skipping {table}: missing column {date_col}")
                continue
            
            s = pd.to_datetime(df[date_col], errors="coerce")
            s = s.dropna()
            
            if len(s) == 0:
                print(f"  ⊗ Skipping {table}: no valid dates")
                continue
            
            # Year-month format (YYYY-MM)
            ym = s.dt.to_period("M").astype(str)
            per_table[table] = ym.value_counts().sort_index()
            print(f"  ✓ {table}: {len(s):,} records")
            
        except Exception as e:
            print(f"  ✗ Error {table}: {e}")
    
    if not per_table:
        print("  ⚠ No data found")
        return None
    
    # Create wide format DataFrame
    monthly_wide = pd.DataFrame(per_table).fillna(0).astype(int)
    monthly_wide.index.name = "year_month"
    monthly_wide["TOTAL"] = monthly_wide.sum(axis=1)
    
    print(f"  ✓ Wide format: {len(monthly_wide)} months x {len(per_table)} tables")
    return monthly_wide


def table_statistics_wide(conn, tables):
    """Create wide format of table statistics"""
    print("\n📊 Creating table statistics (wide format)...")
    
    stats = {}
    for table in tables:
        try:
            # Get row count
            row_count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            
            # Get column count
            col_count = len(get_columns(conn, table))
            
            stats[table] = {
                'total_rows': row_count,
                'total_columns': col_count
            }
            print(f"  ✓ {table}: {row_count:,} rows, {col_count} columns")
            
        except Exception as e:
            print(f"  ✗ Error {table}: {e}")
    
    if not stats:
        return None
    
    df = pd.DataFrame(stats).T
    df.index.name = "table"
    return df


def null_percentage_wide(conn, tables):
    """Create wide format of null percentages for key columns"""
    print("\n📊 Creating null percentages (wide format)...")
    
    # Common columns to check
    common_cols = [
        'title', 'job_title', 'description', 'company', 'company_name',
        'location', 'salary', 'date_posted', 'url', 'job_type'
    ]
    
    null_data = {}
    for table in tables:
        try:
            cols = get_columns(conn, table)
            n = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            
            if n == 0:
                continue
            
            table_nulls = {}
            for col in common_cols:
                if col in cols:
                    nulls = conn.execute(
                        f'SELECT COUNT(*) FROM "{table}" WHERE "{col}" IS NULL'
                    ).fetchone()[0]
                    table_nulls[col] = (nulls / n * 100) if n > 0 else 0
            
            null_data[table] = table_nulls
            print(f"  ✓ {table}: checked {len(table_nulls)} columns")
            
        except Exception as e:
            print(f"  ✗ Error {table}: {e}")
    
    if not null_data:
        return None
    
    df = pd.DataFrame(null_data).T
    df.index.name = "table"
    return df.round(2)


def country_summary_wide(conn, tables):
    """Create wide format summary by country (from table names)"""
    print("\n📊 Creating country summary (wide format)...")
    
    country_stats = {}
    for table in tables:
        try:
            # Extract country code from table name (e.g., jobads_vn_tablename -> VN)
            parts = table.split('_')
            if len(parts) >= 2 and parts[0] == 'jobads':
                country = parts[1].upper()
            else:
                country = 'UNKNOWN'
            
            row_count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            
            if country not in country_stats:
                country_stats[country] = {'tables': 0, 'total_rows': 0}
            
            country_stats[country]['tables'] += 1
            country_stats[country]['total_rows'] += row_count
            
        except Exception as e:
            print(f"  ✗ Error {table}: {e}")
    
    if not country_stats:
        return None
    
    df = pd.DataFrame(country_stats).T
    df.index.name = "country_code"
    df = df.sort_values('total_rows', ascending=False)
    
    for country, row in df.iterrows():
        print(f"  ✓ {country}: {int(row['total_rows']):,} rows across {int(row['tables'])} tables")
    
    return df.astype(int)


def main():
    """Main function to create all wide format analyses"""
    print(f"Opening database: {DB_PATH}\n")
    
    if not Path(DB_PATH).exists():
        print(f"✗ Error: Database not found: {DB_PATH}")
        return
    
    with sqlite3.connect(DB_PATH) as conn:
        # Get all tables
        tables = list_tables(conn, prefix="jobads_")
        
        if not tables:
            print("✗ No tables found with 'jobads_' prefix")
            return
        
        print(f"Found {len(tables)} tables:")
        for table in tables:
            print(f"  - {table}")
        
        # Create all wide format analyses
        year_month_df = year_month_counts_wide(conn, tables)
        stats_df = table_statistics_wide(conn, tables)
        null_df = null_percentage_wide(conn, tables)
        country_df = country_summary_wide(conn, tables)
        
        # Export to Excel with multiple sheets
        print(f"\n📝 Saving to {OUTPUT_XLSX}...")
        
        with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
            if year_month_df is not None:
                year_month_df.to_excel(writer, sheet_name="YearMonth_Wide")
                print(f"  ✓ Sheet: YearMonth_Wide ({year_month_df.shape[0]} x {year_month_df.shape[1]})")
                
                # Also save long format for plotting
                year_month_long = (
                    year_month_df.drop(columns="TOTAL")
                    .reset_index()
                    .melt(id_vars="year_month", var_name="table", value_name="count")
                )
                year_month_long.to_excel(writer, sheet_name="YearMonth_Long", index=False)
                print(f"  ✓ Sheet: YearMonth_Long ({year_month_long.shape[0]} x {year_month_long.shape[1]})")
            
            if stats_df is not None:
                stats_df.to_excel(writer, sheet_name="TableStats_Wide")
                print(f"  ✓ Sheet: TableStats_Wide ({stats_df.shape[0]} x {stats_df.shape[1]})")
            
            if null_df is not None:
                null_df.to_excel(writer, sheet_name="NullPct_Wide")
                print(f"  ✓ Sheet: NullPct_Wide ({null_df.shape[0]} x {null_df.shape[1]})")
            
            if country_df is not None:
                country_df.to_excel(writer, sheet_name="CountrySummary_Wide")
                print(f"  ✓ Sheet: CountrySummary_Wide ({country_df.shape[0]} x {country_df.shape[1]})")
        
        print(f"\n✓ Analysis complete! Saved to: {OUTPUT_XLSX}")
        print(f"  Absolute path: {Path(OUTPUT_XLSX).absolute()}")


if __name__ == '__main__':
    main()
