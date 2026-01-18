import streamlit as st
import sqlite3
import pandas as pd
from datetime import date

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Packsmart Inventory – Paper Reels",
    page_icon="📜",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown(
    "<style>"
    "body{background-color:#f4f6fb;}"
    ".title{font-size:26px;font-weight:700;}"
    ".box{background:#ffffff;padding:16px;border-radius:14px;"
    "box-shadow:0 8px 18px rgba(0,0,0,0.06);}"
    "</style>",
    unsafe_allow_html=True
)

# ---------------- DATABASE ----------------
conn = sqlite3.connect("inventory.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute(
    "CREATE TABLE IF NOT EXISTS paper_reels ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT,"
    "reel_no TEXT,"
    "supplier TEXT,"
    "maker TEXT,"
    "material_rcv_date TEXT,"
    "gsm INTEGER,"
    "bf INTEGER,"
    "paper_shade TEXT,"
    "weight_kg REAL,"
    "closing_stock REAL,"
    "reel_location TEXT,"
    "reorder_level REAL,"
    "remarks TEXT)"
)
conn.commit()

# ---------------- SAMPLE DATA (ONLY ONCE) ----------------
row_count = cursor.execute("SELECT COUNT(*) FROM paper_reels").fetchone()[0]

if row_count == 0:
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

# ---------------- SIDEBAR ----------------
st.sidebar.title("📦 Packsmart Inventory")
st.sidebar.markdown("**Raw Materials**")
st.sidebar.markdown("▶ Paper Reels")

# ---------------- HEADER ----------------
st.markdown('<div class="title">Raw Material → Paper Reels</div>', unsafe_allow_html=True)
st.write("")

# ---------------- ADD + UPLOAD ----------------
with st.container():
    st.markdown('<div class="box">', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])

    # ----- MANUAL ENTRY -----
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

    # ----- EXCEL UPLOAD -----
    with col2:
        st.subheader("⬆ Upload Excel")
        excel_file = st.file_uploader("Upload Paper Reel Excel", type=["xlsx"])

        if excel_file:
            df_excel = pd.read_excel(excel_file)
            df_excel.to_sql("paper_reels", conn, if_exists="append", index=False)
            st.success("Excel data uploaded successfully")

    st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# ---------------- TABLE VIEW ----------------
st.subheader("📜 Paper Reel Master (Scrollable)")

df = pd.read_sql("SELECT * FROM paper_reels ORDER BY material_rcv_date DESC", conn)

st.dataframe(
    df,
    use_container_width=True,
    height=450
)
