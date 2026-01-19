
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
# --------------------------------------------------
# Utility: Resequence Serial Numbers (ERP safe)
# --------------------------------------------------

def resequence_paper_reels(conn):
    """
    Ensures SL No is continuous (1,2,3...)
    after insert or delete.
    """
    cur = conn.cursor()

    cur.execute("""
        SELECT id
        FROM paper_reels
        ORDER BY material_rcv_date, id
    """)

    rows = cur.fetchall()
    seq = 1

    for r in rows:
        cur.execute(
            "UPDATE paper_reels SET sl_no = ? WHERE id = ?",
            (seq, r["id"])
        )
        seq += 1
