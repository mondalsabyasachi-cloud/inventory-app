import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, date

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Packsmart Inventory – Paper Reels",
    page_icon="📜",
    layout="wide"
)

# =====================================================
# CSS (ERP LOOK)
# =====================================================
st.markdown("""
<style>
body { background-color: #f4f6fb; }

.header {
    font-size: 26px;
    font-weight: 700;
}

.box {
    background-color: white;
    padding: 16px;
    border-radius: 14px;
    box-shadow: 0 8px 18px rgba(0,0,0,0.06);
}

.small {
    font-size: 13px;
    color: #666;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# DATABASE
# =====================================================
conn = sqlite3.connect("inventory.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS paper_reels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reel_no TEXT,
    supplier TEXT,
    maker TEXT,
    material_rcv_date TEXT,
    supplier_invoice_date TEXT,
    reel_no_internal TEXT,
    deckle_cm REAL,
    deckle_inch REAL,
    gsm
