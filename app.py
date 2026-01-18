import streamlit as st
import sqlite3
from datetime import datetime

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Packsmart Inventory",
    page_icon="📦",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
body { background-color: #f4f6fb; }
.card {
    padding: 20px;
    border-radius: 16px;
    color: white;
    font-weight: 600;
}
.rm { background: linear-gradient(135deg, #ff5f6d, #ffc371); }
.wip { background: linear-gradient(135deg, #f7971e, #ffd200); }
.fg { background: linear-gradient(135deg, #56ab2f, #a8e063); }
.box {
    background-color: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.05);
}
.title {
    font-size: 28px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# DATABASE SETUP (RESET SAFE)
# -----------------------------
conn = sqlite3.connect("inventory.db", check_same_thread=False)
cursor = conn.cursor()

# SAFE RESET FOR EARLY STAGE
cursor.execute("DROP TABLE IF EXISTS inventory")

cursor.execute("""
CREATE TABLE inventory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inventory_type TEXT NOT NULL,
    item_name TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    unit TEXT NOT NULL,
    location TEXT,
    last_updated TEXT
)
""")
conn.commit()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📦 Packsmart Inventory")
menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Add Inventory", "View Inventory"]
)

inventory_types = ["Raw Material", "WIP", "Finished Goods"]

# -----------------------------
# DASHBOARD
# -----------------------------
if menu == "Dashboard":
    st.markdown('<div class="title">Packsmart India Pvt Ltd – Dashboard</div>', unsafe_allow_html=True)
    st.write("")

    def get_summary(inv_type):
        result = cursor.execute(
            "SELECT COALESCE(SUM(quantity),0), COUNT(*) FROM inventory WHERE inventory_type = ?",
            (inv_type,)
        ).fetchone()
        return result[0], result[1]

    rm_qty, rm_count = get_summary("Raw Material")
    wip_qty, wip_count = get_summary("WIP")
    fg_qty, fg_count = get_summary("Finished Goods")

    c1, c2, c3 = st.columns(3)

    c1.markdown(f"""
    <div class="card rm">
        <h3>Raw Materials</h3>
        <h1>{rm_qty}</h1>
        <p>{rm_count} items</p>
    </div>
    """, unsafe_allow_html=True)

    c2.markdown(f"""
    <div class="card wip">
        <h3>WIP Items</h3>
        <h1>{wip_qty}</h1>
        <p>{wip_count} items</p>
    </div>
    """, unsafe_allow_html=True)

    c3.markdown(f"""
    <div class="card fg">
        <h3>Finished Goods</h3>
        <h1>{fg_qty}</h1>
        <p>{fg_count} items</p>
    </div>
