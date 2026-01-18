import streamlit as st
import sqlite3
import pandas as pd
from datetime import date

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Packsmart Inventory System",
    page_icon="📦",
    layout="wide"
)

# =====================================================
# CSS (ERP LOOK)
# =====================================================
st.markdown(
    "<style>"
    "body{background-color:#f4f6fb;}"
    ".title{font-size:26px;font-weight:700;}"
    ".small{font-size:13px;color:#666;}"
    ".box{background:#ffffff;padding:16px;border-radius:14px;"
    "box-shadow:0 8px 18px rgba(0,0,0,0.06);}"
    "</style>",
    unsafe_allow_html=True
)

# =====================================================
# DATABASE
# =====================================================
conn = sqlite3.connect("inventory.db", check_same_thread=False)
cursor = conn.cursor()

# =====================================================
# TABLE: PAPER REELS
# =====================================================
cursor.execute(
    "CREATE TABLE IF NOT EXISTS paper_reels ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT,"
    "reel_no TEXT,"
    "supplier TEXT,"
    "maker TEXT,"
    "material_rcv_date TEXT,"
    "supplier_invoice_date TEXT,"
    "reel_no_internal TEXT,"
    "deckle_cm REAL,"
    "deckle_inch REAL,"
    "gsm INTEGER,"
    "bf INTEGER,"
    "paper_shade TEXT,"
    "weight_kg REAL,"
    "consumed_wt REAL,"
    "consume_date TEXT,"
    "consumption_entry_date TEXT,"
    "closing_stock REAL,"
    "reel_location TEXT,"
    "target_sku TEXT,"
    "target_customer TEXT,"
    "reel_shift_date TEXT,"
    "delivery_challan TEXT,"
    "reorder_level REAL,"
    "paper_rate REAL,"
    "transport_rate REAL,"
    "landed_cost REAL,"
    "current_stock_value REAL,"
    "holding_days INTEGER,"
    "remarks TEXT)"
)
conn.commit()

# =====================================================
# DEMO DATA LOADER (MANUAL)
# =====================================================
def load_demo_paper_reels():
    cursor.execute(
        "INSERT INTO paper_reels "
        "(reel_no,supplier,maker,material_rcv_date,gsm,bf,paper_shade,"
        "weight_kg,closing_stock,reel_location,reorder_level,remarks) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("R-12001","XYZ Traders","ABC Paper Mills","2026-08-01",
         120,18,"Brown",820,410,"Godown A",300,"Moisture sensitive")
    )
    cursor.execute(
        "INSERT INTO paper_reels "
        "(reel_no,supplier,maker,material_rcv_date,gsm,bf,paper_shade,"
        "weight_kg,closing_stock,reel_location,reorder_level,remarks) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("R-12002","PQR Suppliers","LMN Mills","2026-08-05",
         150,20,"White",900,600,"Godown B",400,"High BF reel")
    )
    conn.commit()

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
st.sidebar.title("📦 Packsmart Inventory")
menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📥 Raw Material → Paper Reels",
        "⚙️ WIP (Coming Soon)",
        "📦 Finished Goods (Coming Soon)"
    ]
)

# =====================================================
# DASHBOARD (SUMMARY ONLY)
# =====================================================
if menu == "🏠 Dashboard":

    st.markdown('<div class="title">Packsmart India Pvt Ltd – Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="small">Inventory overview (summary)</div>', unsafe_allow_html=True)
    st.write("")

    rm_qty, rm_count = cursor.execute(
        "SELECT COALESCE(SUM(closing_stock),0), COUNT(*) FROM paper_reels"
    ).fetchone()

    c1, c2, c3 = st.columns(3)

    c1.metric("Raw Materials (Paper Reels)", f"{rm_qty} Kg", f"{rm_count} reels")
    c2.metric("WIP Items", "—", "Coming soon")
    c3.metric("Finished Goods", "—", "Coming soon")

    st.write("")

    if st.button("Load Demo Paper Reel Data"):
        load_demo_paper_reels()
        st.success("Demo paper reel data loaded. Refresh page.")

# =====================================================
# RAW MATERIAL → PAPER REELS
# =====================================================
elif menu == "📥 Raw Material → Paper Reels":

    st.markdown('<div class="title">Raw Material → Paper Reels</div>', unsafe_allow_html=True)
    st.markdown('<div class="small">ERP-style reel tracking with full traceability</div>', unsafe_allow_html=True)
    st.write("")

    st.markdown('<div class="box">', unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])

    # -------- ADD PAPER REEL --------
    with col1:
        st.subheader("➕ Add Paper Reel")

        reel_no = st.text_input("Reel No")
        supplier = st.text_input("Supplier")
        maker = st.text_input("Maker")
        material_rcv_date = st.date_input("Material Received Date", value=date.today())
        gsm = st.number_input("GSM", min_value=80, step=10)
        bf = st.number_input("BF", min_value=10, step=2)
        paper_shade = st.text_input("Paper Shade")
        weight_kg = st.number_input("Weight (Kg)", min_value=0.0, step=10.0)
        reel_location = st.text_input("Reel Location")
        reorder_level = st.number_input("Reorder Level (Kg)", min_value=0.0, step=50.0)
        remarks = st.text_input("Remarks")

        if st.button("Save Paper Reel"):
            cursor.execute(
                "INSERT INTO paper_reels "
                "(reel_no,supplier,maker,material_rcv_date,gsm,bf,paper_shade,"
                "weight_kg,closing_stock,reel_location,reorder_level,remarks) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    reel_no, supplier, maker, str(material_rcv_date),
                    gsm, bf, paper_shade,
                    weight_kg, weight_kg,
                    reel_location, reorder_level, remarks
                )
            )
            conn.commit()
            st.success("Paper reel added successfully")

    # -------- EXCEL UPLOAD --------
    with col2:
        st.subheader("⬆ Upload Excel")
        excel_file = st.file_uploader("Upload Paper Reel Excel", type=["xlsx"])
        if excel_file:
            df_excel = pd.read_excel(excel_file)
            df_excel.to_sql("paper_reels", conn, if_exists="append", index=False)
            st.success("Excel data uploaded")

    st.markdown('</div>', unsafe_allow_html=True)
    st.write("")

    # -------- TABLE VIEW --------
    st.subheader("📜 Paper Reel Master (Scrollable)")

    df = pd.read_sql(
        "SELECT * FROM paper_reels ORDER BY material_rcv_date DESC",
        conn
    )

    st.dataframe(df, use_container_width=True, height=450)

# =====================================================
# PLACEHOLDERS
# =====================================================
else:
    st.markdown('<div class="title">Module coming soon</div>', unsafe_allow_html=True)
    st.info("This module will be enabled in the next phase.")
