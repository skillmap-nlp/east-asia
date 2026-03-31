#!/usr/bin/env python3
"""
Script to export all tables from MySQL on PythonAnywhere to SQLite locally
Run this script on your LOCAL machine to connect via SSH tunnel and save SQLite file locally
Uses SSH tunneling as recommended by PythonAnywhere for secure remote MySQL access

Run from this folder so .env is picked up:
    cd east_asia_2026 && python retrieve_ads_pythonanywhere.py
"""

import sqlite3
import mysql.connector
import os
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import sshtunnel

_SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(_SCRIPT_DIR / ".env")

# SSH connection settings
SSH_HOST = os.getenv('SSH_HOST', 'ssh.pythonanywhere.com')  # or ssh.eu.pythonanywhere.com for EU
SSH_USER = os.getenv('SSH_USER', 'mpalinski')
SSH_PASSWORD = os.getenv('SSH_PASSWORD', '')  # Your PythonAnywhere login password

# MySQL connection settings (configure these in .env file)
MYSQL_HOST = os.getenv('MYSQL_HOST', 'mpalinski.mysql.pythonanywhere-services.com')
MYSQL_USER = os.getenv('MYSQL_USER', 'mpalinski')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')  # Your MySQL database password
MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'mpalinski$default')

# SQLite output — always resolved to an absolute path next to this script
_sqlite_raw = os.getenv('SQLITE_DB', 'east_asia_job_ads.db')
SQLITE_DB = str(_SCRIPT_DIR / _sqlite_raw) if not os.path.isabs(_sqlite_raw) else _sqlite_raw

# ISO Code for table naming (e.g., 'vn', 'ph', 'my', 'th', etc.)
ISO_CODE = os.getenv('ISO_CODE', '').lower()

# SSH tunnel timeout settings
sshtunnel.SSH_TIMEOUT = 10.0
sshtunnel.TUNNEL_TIMEOUT = 10.0


def get_mysql_connection(tunnel):
    """Create and return MySQL connection through SSH tunnel."""
    conn = mysql.connector.connect(
        host="127.0.0.1",
        port=tunnel.local_bind_port,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        connection_timeout=15,
        use_pure=True,
        ssl_disabled=True,
    )
    # Give each query up to 10 min to stream data before the server drops us
    cur = conn.cursor()
    cur.execute("SET SESSION net_read_timeout=600")
    cur.execute("SET SESSION net_write_timeout=600")
    cur.execute("SET SESSION wait_timeout=600")
    cur.close()
    return conn


def get_table_names(mysql_cursor):
    """Get all table names from MySQL database"""
    mysql_cursor.execute("SHOW TABLES")
    return [table[0] for table in mysql_cursor.fetchall()]


def get_create_table_statement(mysql_cursor, table_name):
    """Get the CREATE TABLE statement from MySQL"""
    mysql_cursor.execute(f"SHOW CREATE TABLE `{table_name}`")
    return mysql_cursor.fetchone()[1]


def convert_mysql_to_sqlite_type(mysql_type):
    """Convert MySQL data types to SQLite data types"""
    mysql_type = mysql_type.upper()
    
    if 'INT' in mysql_type:
        return 'INTEGER'
    elif 'CHAR' in mysql_type or 'TEXT' in mysql_type or 'BLOB' in mysql_type:
        return 'TEXT'
    elif 'FLOAT' in mysql_type or 'DOUBLE' in mysql_type or 'DECIMAL' in mysql_type:
        return 'REAL'
    elif 'DATE' in mysql_type or 'TIME' in mysql_type:
        return 'TEXT'
    else:
        return 'TEXT'


def get_renamed_table(table_name):
    """Generate new table name with jobads_ prefix and ISO code"""
    if ISO_CODE:
        return f"jobads_{ISO_CODE}_{table_name}"
    else:
        return f"jobads_{table_name}"


def create_sqlite_table(sqlite_cursor, mysql_cursor, table_name, new_table_name):
    """Create table in SQLite with compatible schema"""
    # Get column information
    mysql_cursor.execute(f"DESCRIBE `{table_name}`")
    columns = mysql_cursor.fetchall()
    
    # Build CREATE TABLE statement for SQLite
    column_defs = []
    for col in columns:
        col_name = col[0]
        col_type = convert_mysql_to_sqlite_type(col[1])
        null_constraint = '' if col[2] == 'YES' else 'NOT NULL'
        
        # Handle primary key
        if col[3] == 'PRI':
            column_def = f"`{col_name}` {col_type} PRIMARY KEY {null_constraint}"
        else:
            column_def = f"`{col_name}` {col_type} {null_constraint}"
        
        column_defs.append(column_def)
    
    create_statement = f"CREATE TABLE IF NOT EXISTS `{new_table_name}` ({', '.join(column_defs)})"
    sqlite_cursor.execute(create_statement)


DATE_FILTER_YEAR = os.getenv("DATE_FILTER_YEAR", "2026")
DATE_POSTED_COL = os.getenv("DATE_POSTED_COL", "date_posted")


def has_column(mysql_cursor, table_name, col_name):
    mysql_cursor.execute(f"SHOW COLUMNS FROM `{table_name}` LIKE %s", (col_name,))
    return mysql_cursor.fetchone() is not None


def copy_table_data(sqlite_cursor, mysql_conn, table_name, new_table_name):
    """Copy rows from MySQL to SQLite in one shot per table.

    Uses a single SELECT with fetchall() — filtered 2026 data is small enough
    to fit in memory, and avoids the catastrophic LIMIT/OFFSET rescans.
    """
    cur = mysql_conn.cursor(buffered=True)

    if has_column(cur, table_name, DATE_POSTED_COL):
        where = (
            f"WHERE `{DATE_POSTED_COL}` >= '{DATE_FILTER_YEAR}-01-01' "
            f"AND `{DATE_POSTED_COL}` < '{int(DATE_FILTER_YEAR)+1}-01-01'"
        )
        print(f"  └─ Filter: {DATE_POSTED_COL} in {DATE_FILTER_YEAR}")
    else:
        where = ""
        print(f"  └─ ⚠ Column '{DATE_POSTED_COL}' not found — copying all rows")

    cur.execute(f"SELECT COUNT(*) FROM `{table_name}` {where}")
    total_rows = cur.fetchone()[0]

    if total_rows == 0:
        print(f"  └─ No matching rows")
        cur.close()
        return

    print(f"  └─ Fetching {total_rows:,} rows from MySQL...")
    cur.execute(f"SELECT * FROM `{table_name}` {where}")
    rows = cur.fetchall()
    cur.close()

    if not rows:
        print(f"  └─ No matching rows")
        return

    placeholders = ','.join(['?' for _ in range(len(rows[0]))])
    print(f"  └─ Writing to SQLite...")
    sqlite_cursor.executemany(
        f"INSERT INTO `{new_table_name}` VALUES ({placeholders})",
        rows
    )
    print(f"  └─ ✓ Copied {len(rows):,} rows")


def export_mysql_to_sqlite():
    """Main function to export all MySQL tables to SQLite via SSH tunnel"""
    print(f"Setting up SSH tunnel to PythonAnywhere...")
    print(f"SSH Host: {SSH_HOST}")
    print(f"SSH User: {SSH_USER}")
    print(f"MySQL Host: {MYSQL_HOST}")
    print(f"Database: {MYSQL_DATABASE}\n")
    
    # Create SSH tunnel
    try:
        print("Establishing SSH tunnel...")
        tunnel = sshtunnel.SSHTunnelForwarder(
            SSH_HOST,
            ssh_username=SSH_USER,
            ssh_password=SSH_PASSWORD,
            remote_bind_address=(MYSQL_HOST, 3306)
        )
        tunnel.start()
        print(f"✓ SSH tunnel established on local port {tunnel.local_bind_port}\n")
    except Exception as e:
        print(f"✗ Failed to establish SSH tunnel")
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your SSH_PASSWORD in .env (this is your PythonAnywhere login password)")
        print("2. Verify your SSH_USER is correct (case-sensitive!)")
        print("3. Ensure you have a paid PythonAnywhere account")
        print("4. Check if SSH_HOST is correct (ssh.pythonanywhere.com or ssh.eu.pythonanywhere.com)")
        return
    
    mysql_conn = None
    mysql_cursor = None
    sqlite_conn = None
    sqlite_cursor = None

    # Connect to MySQL through tunnel
    try:
        print("Connecting to MySQL database through tunnel...")
        print(f"  (user={MYSQL_USER}, db={MYSQL_DATABASE}, port={tunnel.local_bind_port})")
        mysql_conn = get_mysql_connection(tunnel)
        mysql_cursor = mysql_conn.cursor()
        print("✓ Successfully connected to MySQL database\n")
    except mysql.connector.Error as e:
        print(f"✗ Failed to connect to MySQL database")
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check MYSQL_PASSWORD in .env (this is your MySQL database password)")
        print("2. Verify MYSQL_USER matches your PythonAnywhere username")
        print("3. Check MYSQL_DATABASE format: username$databasename")
        print("4. Run from east_asia_2026 so .env loads: cd east_asia_2026 && python retrieve_ads_pythonanywhere.py")
        if tunnel.is_active:
            tunnel.stop()
        return
    
    # Remove existing SQLite database and any leftover WAL/SHM files
    for suffix in ("", "-wal", "-shm"):
        p = SQLITE_DB + suffix
        if os.path.exists(p):
            os.remove(p)
    print(f"Writing to: {SQLITE_DB}\n")

    sqlite_conn = sqlite3.connect(SQLITE_DB)
    sqlite_conn.execute("PRAGMA journal_mode=WAL")
    sqlite_conn.execute("PRAGMA synchronous=OFF")
    sqlite_conn.execute("PRAGMA cache_size=-64000")  # 64 MB cache
    sqlite_cursor = sqlite_conn.cursor()
    
    try:
        # Get all table names
        table_names = get_table_names(mysql_cursor)
        print(f"Found {len(table_names)} tables: {', '.join(table_names)}\n")
        
        if ISO_CODE:
            print(f"Tables will be renamed with ISO code: {ISO_CODE.upper()}")
            print(f"Format: jobads_{ISO_CODE}_<table_name>\n")
        else:
            print(f"Tables will be renamed with prefix: jobads_<table_name>\n")
        
        print("=" * 70)
        
        # Export each table with progress bar
        for table_name in tqdm(table_names, desc="Exporting tables", unit="table"):
            new_table_name = get_renamed_table(table_name)
            tqdm.write(f"\n📊 Table: {table_name} → {new_table_name}")
            
            # Create table in SQLite
            tqdm.write(f"  ├─ Creating table structure...")
            create_sqlite_table(sqlite_cursor, mysql_cursor, table_name, new_table_name)
            
            # Copy data
            tqdm.write(f"  ├─ Copying data...")
            copy_table_data(sqlite_cursor, mysql_conn, table_name, new_table_name)
            
            sqlite_conn.commit()
        
        print("\n" + "=" * 70)
        
        # Get absolute path for display
        abs_path = os.path.abspath(SQLITE_DB)
        
        print(f"\n{'='*60}")
        print(f"✓ Export completed successfully!")
        print(f"{'='*60}")
        print(f"SQLite database saved locally to:")
        print(f"  {abs_path}")
        print(f"\nTotal tables exported: {len(table_names)}")
        
    except Exception as e:
        print(f"\n✗ Error during export: {e}")
        raise
    
    finally:
        if mysql_cursor is not None:
            mysql_cursor.close()
        if mysql_conn is not None:
            mysql_conn.close()
        if sqlite_cursor is not None:
            sqlite_cursor.close()
        if sqlite_conn is not None:
            sqlite_conn.close()
        if tunnel.is_active:
            tunnel.stop()
        print("SSH tunnel closed.")


if __name__ == '__main__':
    export_mysql_to_sqlite()
