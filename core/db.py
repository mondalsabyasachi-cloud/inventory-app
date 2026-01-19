
import os
import sqlite3
from contextlib import contextmanager

DB_PATH = os.getenv("INVENTORY_DB", "inventory.db")

@contextmanager
def get_conn():
    conn = sqlite3.connect(
        DB_PATH,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
    )
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.commit()
        conn.close()
