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
    border-radius: 18px;
    color: white;
    font-weight: 600;
}

.rm { background: linear-gradient(135deg, #ff6a6a, #ffb199); }
.wip { background: linear-gradient(135deg, #f6a100, #ffd36a); }
.fg { background: linear-gradient(135deg, #6bbf59, #b7e4a1); }

.box {
    background-color: white;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.06);
}

.title {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# DATABASE (PERSISTENT – NO RESET)
# -----------------------------
conn = sqlite3.connect("inventory.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS inventory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inventory_type TEXT,
    item_name TEXT,
    quantity INTEGER,
    unit TEXT,
    location TEXT,
    last_updated TEXT
)
""")
conn.commit()

inventory_types = ["Raw Material", "WIP", "Finished Goods"]

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📦 Packsmart Inventory")
menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Add Inventory", "View Inventory"]
)

# -----------------------------
# COMMON KPI FUNCTION
# -----------------------------
def get_summary(inv_type):
    row = cursor.execute(
        "SELECT COALESCE(SUM(quantity),0), COUNT(*) FROM inventory WHERE inventory_type = ?",
        (inv_type,)
    ).fetchone()
    return row[0], row[1]

# =====================================================
# DASHBOARD (FULL FEATURE PAGE)
# =====================================================
if menu == "Dashboard":
    st.markdown('<div class="title">Packsmart India Pvt Ltd – Dashboard</div>', unsafe_allow_html=True)

    # KPI ROW
    rm_qty, rm_count = get_summary("Raw Material")
    wip_qty, wip_count = get_summary("WIP")
    fg_qty, fg_count = get_summary("Finished Goods")

    c1, c2, c3 = st.columns(3)

    c1.markdown(
        '<div class="card rm"><h3>Raw Materials</h3><h1>{}</h1><p>{} items</p></div>'.format(rm_qty, rm_count),
        unsafe_allow_html=True
    )

    c2.markdown(
        '<div class="card wip"><h3>WIP Items</h3><h1>{}</h1><p>{} items</p></div>'.format(wip_qty, wip_count),
        unsafe_allow_html=True
    )

    c3.markdown(
        '<div class="card fg"><h3>Finished Goods</h3><h1>{}</h1><p>{} items</p></div>'.format(fg_qty, fg_count),
        unsafe_allow_html=True
    )

    st.write("")

    # SECOND ROW – ADD & VIEW
    left, right = st.columns(2)

    # ADD INVENTORY (LEFT)
    with left:
        st.markdown('<div class="box">', unsafe_allow_html=True)
        st.subheader("➕ Add Inventory")

        inv_type = st.selectbox("Inventory Type", inventory_types, key="dash_type")
        item_name = st.text_input("Item Name", key="dash_item")
        quantity = st.number_input("Quantity", min_value=0, step=1, key="dash_qty")
        unit = st.selectbox("Unit", ["Kg", "Nos"], key="dash_unit")
        location = st.text_input("Location / Godown", key="dash_loc")

        if st.button("Add Item", key="dash_add"):
            cursor.execute("""
                INSERT INTO inventory
                (inventory_type, item_name, quantity, unit, location, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                inv_type, item_name, quantity, unit, location,
                datetime.now().strftime("%Y-%m-%d %H:%M")
            ))
            conn.commit()
            st.success("Item added")

        st.markdown('</div>', unsafe_allow_html=True)

    # VIEW INVENTORY (RIGHT)
    with right:
        st.markdown('<div class="box">', unsafe_allow_html=True)
        st.subheader("📋 View Inventory")

        filter_type = st.selectbox("Filter by Type", ["All"] + inventory_types, key="dash_filter")

        if filter_type == "All":
            rows = cursor.execute("""
                SELECT inventory_type, item_name, quantity, unit, location
                FROM inventory
                ORDER BY inventory_type, item_name
            """).fetchall()
        else:
            rows = cursor.execute("""
                SELECT inventory_type, item_name, quantity, unit, location
                FROM inventory
                WHERE inventory_type = ?
                ORDER BY item_name
            """, (filter_type,)).fetchall()

        st.dataframe(rows, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# ADD INVENTORY – FULL PAGE
# =====================================================
elif menu == "Add Inventory":
    st.markdown('<div class="title">Add Inventory</div>', unsafe_allow_html=True)

    inv_type = st.selectbox("Inventory Type", inventory_types)
    item_name = st.text_input("Item Name")
    quantity = st.number_input("Quantity", min_value=0, step=1)
    unit = st.selectbox("Unit", ["Kg", "Nos"])
    location = st.text_input("Location / Godown")

    if st.button("Add Item"):
        cursor.execute("""
            INSERT INTO inventory
            (inventory_type, item_name, quantity, unit, location, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            inv_type, item_name, quantity, unit, location,
            datetime.now().strftime("%Y-%m-%d %H:%M")
        ))
        conn.commit()
        st.success("Inventory added successfully")

# =====================================================
# VIEW INVENTORY – FULL PAGE
# =====================================================
elif menu == "View Inventory":
    st.markdown('<div class="title">View Inventory</div>', unsafe_allow_html=True)

    filter_type = st.selectbox("Filter by Type", ["All"] + inventory_types)

    if filter_type == "All":
        rows = cursor.execute("""
            SELECT inventory_type, item_name, quantity, unit, location, last_updated
            FROM inventory
            ORDER BY inventory_type, item_name
        """).fetchall()
    else:
        rows = cursor.execute("""
            SELECT inventory_type, item_name, quantity, unit, location, last_updated
            FROM inventory
            WHERE inventory_type = ?
            ORDER BY item_name
        """, (filter_type,)).fetchall()

    st.dataframe(rows, use_container_width=True)
