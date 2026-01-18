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
body {
    background-color: #f4f6fb;
}
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
# DB SETUP
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
        data = cursor.execute(
            "SELECT SUM(quantity), COUNT(*) FROM inventory WHERE inventory_type = ?",
            (inv_type,)
        ).fetchone()
        return data[0] or 0, data[1] or 0

    rm_qty, rm_count = get_summary("Raw Material")
    wip_qty, wip_count = get_summary("WIP")
    fg_qty, fg_count = get_summary("Finished Goods")

    col1, col2, col3 = st.columns(3)

    col1.markdown(f"""
    <div class="card rm">
        <h3>Raw Materials</h3>
        <h1>{rm_qty}</h1>
        <p>{rm_count} items</p>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="card wip">
        <h3>WIP Items</h3>
        <h1>{wip_qty}</h1>
        <p>{wip_count} items</p>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div class="card fg">
        <h3>Finished Goods</h3>
        <h1>{fg_qty}</h1>
        <p>{fg_count} items</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# ADD INVENTORY
# -----------------------------
elif menu == "Add Inventory":
    st.markdown('<div class="title">Add Inventory</div>', unsafe_allow_html=True)
    st.write("")

    with st.container():
        st.markdown('<div class="box">', unsafe_allow_html=True)

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
                inv_type,
                item_name,
                quantity,
                unit,
                location,
                datetime.now().strftime("%Y-%m-%d %H:%M")
            ))
            conn.commit()
            st.success("Inventory added successfully")

        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# VIEW INVENTORY
# -----------------------------
elif menu == "View Inventory":
    st.markdown('<div class="title">View Inventory</div>', unsafe_allow_html=True)
    st.write("")

    filter_type = st.selectbox("Filter by Type", ["All"] + inventory_types)

    if filter_type == "All":
        data = cursor.execute("""
            SELECT inventory_type, item_name, quantity, unit, location, last_updated
            FROM inventory
            ORDER BY inventory_type, item_name
        """).fetchall()
    else:
        data = cursor.execute("""
            SELECT inventory_type, item_name, quantity, unit, location, last_updated
            FROM inventory
            WHERE inventory_type = ?
            ORDER BY item_name
        """, (filter_type,)).fetchall()

    st.dataframe(
        data,
        use_container_width=True
    )
