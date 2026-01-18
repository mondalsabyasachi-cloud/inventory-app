import streamlit as st
import sqlite3
from datetime import datetime

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="Inventory System", layout="centered")
st.title("Simple Inventory System")

# -----------------------------
# DATABASE SETUP
# -----------------------------
conn = sqlite3.connect("inventory.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS inventory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_name TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    location TEXT,
    last_updated TEXT
)
""")
conn.commit()

# -----------------------------
# SIDEBAR MENU
# -----------------------------
menu = st.sidebar.radio("Menu", ["Add Item", "View Inventory"])

# -----------------------------
# ADD ITEM PAGE
# -----------------------------
if menu == "Add Item":
    st.subheader("Add / Update Item")

    item_name = st.text_input("Item Name")
    quantity = st.number_input("Quantity", min_value=0, step=1)
    location = st.text_input("Warehouse Location")

    if st.button("Save Item"):
        cursor.execute("""
        INSERT INTO inventory (item_name, quantity, location, last_updated)
        VALUES (?, ?, ?, ?)
        """, (item_name, quantity, location, datetime.now().strftime("%Y-%m-%d %H:%M")))
        conn.commit()
        st.success("Item saved successfully")

# -----------------------------
# VIEW INVENTORY PAGE
# -----------------------------
if menu == "View Inventory":
    st.subheader("Current Inventory")

    data = cursor.execute("""
    SELECT item_name, quantity, location, last_updated
    FROM inventory
    ORDER BY item_name
    """).fetchall()

    if data:
        st.table(data)
    else:
        st.info("No inventory data available")
