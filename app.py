
# -------------------------------------------------------------
# Packsmart Inventory App (RM / WIP / FG) - Streamlit (single file)
# Prepared for: Saby Mondal | Packsmart India Pvt Ltd
# -------------------------------------------------------------

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from typing import Optional, Tuple, Dict

import pandas as pd
import streamlit as st

# -------------------------
# Page config & theme
# -------------------------
st.set_page_config(
    page_title="Packsmart India Pvt Ltd | Inventory System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

PRIMARY = "#2b6cb0"
ACCENT_RM = "#e76f51"
ACCENT_WIP = "#f4a261"
ACCENT_FG = "#2a9d8f"
CARD_BG = "#ffffff"

# Raw Material type master list (controls the drop-down in Raw Materials page)
RM_TYPES = [
    "Paper Reel",
    "GUM / Adhesives",
    "Stitching Wire",
    "Strapping Wire",
    "Board / Sheet",
    "Ink / Chemicals",
    "Packaging Accessories",
    "Others"
]

st.markdown(f"""
<style>
    .metric-card {{
        background: {CARD_BG};
        padding: 16px 18px;
        border-radius: 12px;
        border: 1px solid #e6e9ef;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        margin-bottom: 8px;
    }}
    .metric-title {{ font-size: 0.9rem; color: #6b7280; margin-bottom: 6px; }}
    .metric-value {{ font-size: 1.6rem; font-weight: 700; }}
    .metric-sub {{ font-size: 0.8rem; color: #6b7280; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{ padding-top: 8px; padding-bottom: 8px; }}
</style>
""", unsafe_allow_html=True)

# -------------------------
# NAVIGATION: process any pending nav BEFORE rendering sidebar
# -------------------------
PAGES = ["Dashboard", "Raw Materials", "WIP Items", "Finished Goods", "Settings"]

# Initialize left_nav once
if "left_nav" not in st.session_state:
    st.session_state.left_nav = "Dashboard"

# If a button set a pending navigation in a previous run, apply it now
if "_pending_nav" in st.session_state:
    target = st.session_state.pop("_pending_nav")
    if target in PAGES:
        st.session_state.left_nav = target
    st.rerun()

# -------------------------
# DB helpers
# -------------------------
DB_PATH = os.getenv("INVENTORY_DB", "inventory.db")

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.commit()
        conn.close()

def init_db():
    with get_conn() as conn:
        cur = conn.cursor()

        # Masters
        cur.execute("""
            CREATE TABLE IF NOT EXISTS Customer(
              CustomerId INTEGER PRIMARY KEY AUTOINCREMENT,
              Name TEXT UNIQUE NOT NULL,
              GSTIN TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS SKU(
              SKUId INTEGER PRIMARY KEY AUTOINCREMENT,
              SKUCode TEXT UNIQUE NOT NULL,
              Description TEXT,
              UoM TEXT DEFAULT 'Nos'
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS Warehouse(
              WarehouseId INTEGER PRIMARY KEY AUTOINCREMENT,
              Name TEXT UNIQUE NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS Bin(
              BinId INTEGER PRIMARY KEY AUTOINCREMENT,
              WarehouseId INTEGER NOT NULL,
              Aisle TEXT, Rack TEXT, Bin TEXT,
              UNIQUE(WarehouseId, Aisle, Rack, Bin),
              FOREIGN KEY (WarehouseId) REFERENCES Warehouse(WarehouseId)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS Supplier(
              SupplierId INTEGER PRIMARY KEY AUTOINCREMENT,
              Name TEXT UNIQUE NOT NULL,
              GSTIN TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS Maker(
              MakerId INTEGER PRIMARY KEY AUTOINCREMENT,
              Name TEXT UNIQUE NOT NULL
            )
        """)

        # Raw Material: Paper Reels (full field set)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS PaperReel(
              ReelId INTEGER PRIMARY KEY AUTOINCREMENT,
              SLNo TEXT,
              ReelNo TEXT UNIQUE NOT NULL,
              SupplierId INTEGER,
              MakerId INTEGER,
              ReceiveDate TEXT,
              SupplierInvDate TEXT,
              DeckleCm REAL,
              GSM INTEGER,
              BF INTEGER,
              Shade TEXT,
              OpeningKg REAL DEFAULT 0,
              WeightKg REAL DEFAULT 0,
              LastConsumeDate TEXT,
              ConsumptionEntryDate TEXT,
              ReelLocationBinId INTEGER,
              TargetSKUId INTEGER,
              TargetCustomerId INTEGER,
              ReelShiftDate TEXT,
              DeliveryChallanNo TEXT,
              ReorderLevelKg REAL DEFAULT 0,
              PaperRatePerKg REAL DEFAULT 0,
              TransportRatePerKg REAL DEFAULT 0,
              BasicLandedCostPerKg REAL DEFAULT 0,
              Remarks TEXT,
              FOREIGN KEY (SupplierId) REFERENCES Supplier(SupplierId),
              FOREIGN KEY (MakerId) REFERENCES Maker(MakerId),
              FOREIGN KEY (ReelLocationBinId) REFERENCES Bin(BinId),
              FOREIGN KEY (TargetSKUId) REFERENCES SKU(SKUId),
              FOREIGN KEY (TargetCustomerId) REFERENCES Customer(CustomerId)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS RM_Movement(
              MoveId INTEGER PRIMARY KEY AUTOINCREMENT,
              ReelId INTEGER NOT NULL,
              DateTime TEXT NOT NULL,
              Type TEXT NOT NULL,  -- Receive/Issue/Return/TransferIn/TransferOut/Adjust/Hold/Release/Scrap
              QtyKg REAL DEFAULT 0,
              FromBinId INTEGER,
              ToBinId INTEGER,
              RefDocType TEXT,
              RefDocNo TEXT,
              User TEXT,
              FOREIGN KEY (ReelId) REFERENCES PaperReel(ReelId)
            )
        """)

        # Work In Process
        cur.execute("""
            CREATE TABLE IF NOT EXISTS WorkOrder(
              WOId INTEGER PRIMARY KEY AUTOINCREMENT,
              SKUId INTEGER NOT NULL,
              CustomerId INTEGER NOT NULL,
              QtyPlanned REAL NOT NULL,
              StartPlan TEXT, EndPlan TEXT,
              Status TEXT DEFAULT 'Open',
              FOREIGN KEY (SKUId) REFERENCES SKU(SKUId),
              FOREIGN KEY (CustomerId) REFERENCES Customer(CustomerId)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS WIP_Unit(
              UnitId INTEGER PRIMARY KEY AUTOINCREMENT,
              WOId INTEGER NOT NULL,
              Step TEXT NOT NULL,  -- Corrugation/PrinterSlotter/DieCutter/FolderGluer/Stitcher/Bundling/QA
              Workcenter TEXT,
              PalletId TEXT,
              Qty REAL NOT NULL,
              Status TEXT DEFAULT 'In-Process',
              InTime TEXT, OutTime TEXT,
              FOREIGN KEY (WOId) REFERENCES WorkOrder(WOId)
            )
        """)

        # Finished Goods
        cur.execute("""
            CREATE TABLE IF NOT EXISTS FG_Pallet(
              PalletId TEXT PRIMARY KEY,
              SKUId INTEGER NOT NULL,
              CustomerId INTEGER NOT NULL,
              Batch TEXT,
              PackDate TEXT,
              WarehouseId INTEGER,
              BinId INTEGER,
              OnHandQty REAL NOT NULL,
              ReservedQty REAL DEFAULT 0,
              HoldQty REAL DEFAULT 0,
              FOREIGN KEY (SKUId) REFERENCES SKU(SKUId),
              FOREIGN KEY (CustomerId) REFERENCES Customer(CustomerId),
              FOREIGN KEY (WarehouseId) REFERENCES Warehouse(WarehouseId),
              FOREIGN KEY (BinId) REFERENCES Bin(BinId)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS FG_Movement(
              MoveId INTEGER PRIMARY KEY AUTOINCREMENT,
              PalletId TEXT NOT NULL,
              DateTime TEXT NOT NULL,
              Type TEXT NOT NULL,  -- Pack/Putaway/Reserve/Unreserve/Pick/Dispatch/Hold/Adjust
              Qty REAL NOT NULL,
              FromBinId INTEGER,
              ToBinId INTEGER,
              RefDoc TEXT,
              FOREIGN KEY (PalletId) REFERENCES FG_Pallet(PalletId)
            )
        """)

init_db()

# -------------------------
# Utilities
# -------------------------
def cm_to_inch(cm: Optional[float]) -> Optional[float]:
    """Convert centimeters to inches (3 decimal places)."""
    if cm is None:
        return None
    try:
        return round(float(cm) / 2.54, 3)
    except Exception:
        return None

def today_str() -> str:
    return date.today().isoformat()

def compute_reel_closing(conn, reel_id: int) -> Tuple[float, float]:
    """
    Returns: (ConsumedKg, ClosingKg)
    Closing = Opening + Receipts - (Issues + Scrap + TransfersOut) + TransfersIn + Adjustments
    """
    df = pd.read_sql_query("""
        SELECT Type, COALESCE(SUM(QtyKg),0) AS Qty
        FROM RM_Movement WHERE ReelId = ?
        GROUP BY Type
    """, conn, params=[reel_id])
    q: Dict[str, float] = df.set_index("Type")["Qty"].to_dict() if len(df) else {}
    opening_row = pd.read_sql_query("SELECT OpeningKg FROM PaperReel WHERE ReelId=?",
                                    conn, params=[reel_id]).iloc[0]
    opening = float(opening_row["OpeningKg"] or 0)

    receipts = q.get("Receive", 0.0)
    issues = q.get("Issue", 0.0)
    scrap = q.get("Scrap", 0.0)
    trans_out = q.get("TransferOut", 0.0)
    trans_in = q.get("TransferIn", 0.0)
    adjust = q.get("Adjust", 0.0)

    consumed = issues + scrap
    closing = opening + receipts - (issues + scrap + trans_out) + trans_in + adjust
    return (round(consumed, 3), round(closing, 3))

# -------------------------
# Demo Data seeding
# -------------------------
def seed_demo_data():
    with get_conn() as conn:
        cur = conn.cursor()

        # Masters
        for c in ["Customer X", "Customer Y"]:
            cur.execute("INSERT OR IGNORE INTO Customer(Name) VALUES(?)", (c,))
        cur.execute("INSERT OR IGNORE INTO SKU(SKUCode, Description) VALUES(?,?)",
                    ("Product-1", "Printed RSC 5Ply"))
        cur.execute("INSERT OR IGNORE INTO SKU(SKUCode, Description) VALUES(?,?)",
                    ("Product-2", "Die-cut Auto Bottom"))

        for w in ["Main WH", "FG WH"]:
            cur.execute("INSERT OR IGNORE INTO Warehouse(Name) VALUES(?)", (w,))

        wh_map = {r["Name"]: r["WarehouseId"]
                  for r in cur.execute("SELECT WarehouseId, Name FROM Warehouse")}
        bins = [
            (wh_map["Main WH"], "A", "1", "01"),
            (wh_map["Main WH"], "A", "1", "02"),
            (wh_map["FG WH"], "F", "1", "01"),
            (wh_map["FG WH"], "F", "1", "02"),
        ]
        for b in bins:
            cur.execute("INSERT OR IGNORE INTO Bin(WarehouseId, Aisle, Rack, Bin) VALUES(?,?,?,?)", b)

        for s in ["KraftCo", "PaperWorld"]:
            cur.execute("INSERT OR IGNORE INTO Supplier(Name) VALUES(?)", (s,))
        for m in ["JK Papers", "WestRock", "Local Mill A"]:
            cur.execute("INSERT OR IGNORE INTO Maker(Name) VALUES(?)", (m,))

        # Sample reels
        cur.execute("SELECT BinId FROM Bin LIMIT 1"); sample_bin = cur.fetchone()[0]
        cur.execute("SELECT SupplierId FROM Supplier WHERE Name='KraftCo'"); sup = cur.fetchone()[0]
        cur.execute("SELECT MakerId FROM Maker WHERE Name='JK Papers'"); mk = cur.fetchone()[0]

        cur.execute("""
            INSERT OR IGNORE INTO PaperReel(
                SLNo, ReelNo, SupplierId, MakerId, ReceiveDate, SupplierInvDate,
                DeckleCm, GSM, BF, Shade, OpeningKg, WeightKg, ReelLocationBinId,
                TargetSKUId, TargetCustomerId, DeliveryChallanNo, ReorderLevelKg,
                PaperRatePerKg, TransportRatePerKg, BasicLandedCostPerKg, Remarks
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, ("1", "Reel-1001", sup, mk, (date.today()-relativedelta(days=5)).isoformat(),
              (date.today()-relativedelta(days=5)).isoformat(), 180.0, 150, 22, "Natural",
              0.0, 1200.0, sample_bin, 1, 1, "DC-8891", 300.0, 45.0, 2.5, 47.5, "Demo reel"))
        cur.execute("SELECT ReelId FROM PaperReel WHERE ReelNo='Reel-1001'"); rid1 = cur.fetchone()[0]
        cur.execute("INSERT OR IGNORE INTO RM_Movement(ReelId, DateTime, Type, QtyKg) VALUES(?,?,?,?)",
                    (rid1, datetime.now().isoformat(), "Receive", 1200.0))

        cur.execute("""
            INSERT OR IGNORE INTO PaperReel(
                SLNo, ReelNo, SupplierId, MakerId, ReceiveDate, SupplierInvDate,
                DeckleCm, GSM, BF, Shade, OpeningKg, WeightKg, ReelLocationBinId,
                TargetSKUId, TargetCustomerId, DeliveryChallanNo, ReorderLevelKg,
                PaperRatePerKg, TransportRatePerKg, BasicLandedCostPerKg, Remarks
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, ("2", "Reel-1002", sup, mk, (date.today()-relativedelta(days=2)).isoformat(),
              (date.today()-relativedelta(days=2)).isoformat(), 160.0, 120, 18, "Brown",
              0.0, 1000.0, sample_bin, 2, 2, "DC-8892", 250.0, 42.0, 2.0, 44.0, "Demo reel 2"))
        cur.execute("SELECT ReelId FROM PaperReel WHERE ReelNo='Reel-1002'"); rid2 = cur.fetchone()[0]
        cur.execute("INSERT OR IGNORE INTO RM_Movement(ReelId, DateTime, Type, QtyKg) VALUES(?,?,?,?)",
                    (rid2, datetime.now().isoformat(), "Receive", 1000.0))

        # FG demo pallets
        cur.execute("SELECT SKUId FROM SKU WHERE SKUCode='Product-1'"); sku1 = cur.fetchone()[0]
        cur.execute("SELECT CustomerId FROM Customer WHERE Name='Customer X'"); cx = cur.fetchone()[0]
        cur.execute("SELECT WarehouseId FROM Warehouse WHERE Name='FG WH'"); fgwh = cur.fetchone()[0]
        cur.execute("SELECT BinId FROM Bin WHERE WarehouseId=? LIMIT 1", (fgwh,)); fgbin = cur.fetchone()[0]
        for pid, qty in [("PAL-001", 4000), ("PAL-002", 8480)]:
            cur.execute("""
                INSERT OR IGNORE INTO FG_Pallet(PalletId, SKUId, CustomerId, Batch, PackDate,
                                                WarehouseId, BinId, OnHandQty, ReservedQty, HoldQty)
                VALUES(?,?,?,?,?,?,?,?,?,?)
            """, (pid, sku1, cx, "BATCH-A", today_str(), fgwh, fgbin, qty, 0, 0))

    st.success("Demo data seeded.")

# -------------------------
# Sidebar (radio is the single controller)
# -------------------------
with st.sidebar:
    st.markdown("### 🧃 Packsmart India Pvt Ltd\n**Inventory System**")
    st.radio(
        "Go to",
        PAGES,
        index=PAGES.index(st.session_state.left_nav),
        key="left_nav",
        label_visibility="collapsed",
        captions=[
            "Overview KPIs", "Paper, Boards, GUM etc.", "In-process tracking",
            "FG stock & dispatch", "Masters & demo"
        ]
    )
    st.markdown("---")
    st.caption("Logged in as: Stores/Planning")

# Always read the selected page from the radio key
page = st.session_state.left_nav

# -------------------------
# Dashboard
# -------------------------
def show_dashboard():
    col1, col2, col3 = st.columns(3)
    with get_conn() as conn:
        # RM summary
        rids = pd.read_sql_query("SELECT ReelId FROM PaperReel", conn)
        total_rm_kg = 0.0
        for _, r in rids.iterrows():
            _, closing = compute_reel_closing(conn, int(r["ReelId"]))
            total_rm_kg += closing
        total_reels = len(rids)

        # WIP summary
        wip_qty = pd.read_sql_query(
            "SELECT COALESCE(SUM(Qty),0) AS Qty FROM WIP_Unit WHERE Status='In-Process'", conn
        ).iloc[0]["Qty"]
        wip_units = pd.read_sql_query(
            "SELECT COUNT(*) AS Cnt FROM WIP_Unit WHERE Status='In-Process'", conn
        ).iloc[0]["Cnt"]

        # FG summary (ATP)
        fg = pd.read_sql_query("""
            SELECT COALESCE(SUM(OnHandQty - ReservedQty - HoldQty),0) AS ATP,
                   COUNT(*) AS Pallets
            FROM FG_Pallet
        """, conn).iloc[0]
        fg_atp, fg_pallets = int(fg["ATP"]), int(fg["Pallets"])

    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left:6px solid {ACCENT_RM}">
            <div class="metric-title">Raw Materials</div>
            <div class="metric-value">{total_rm_kg:,.2f} Kg</div>
            <div class="metric-sub">in {total_reels} reels</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left:6px solid {ACCENT_WIP}">
            <div class="metric-title">WIP Items</div>
            <div class="metric-value">{int(wip_qty):,} Nos</div>
            <div class="metric-sub">in {int(wip_units)} units</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left:6px solid {ACCENT_FG}">
            <div class="metric-title">Finished Goods</div>
            <div class="metric-value">{fg_atp:,} Nos</div>
            <div class="metric-sub">available to promise • {fg_pallets} pallets</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("➕ Add Inventory")
        st.caption("Use **Raw Materials** or **Finished Goods** pages for detailed operations.")
        if st.button("Go to Raw Materials", use_container_width=True, type="primary"):
            # Set pending nav and rerun; on next run the radio starts on the target page
            st.session_state["_pending_nav"] = "Raw Materials"
            st.rerun()
    with c2:
        st.subheader("🔎 View Inventory")
        st.caption("Filter and drill down in each module (RM / WIP / FG).")
        if st.button("Go to Finished Goods", use_container_width=True):
            st.session_state["_pending_nav"] = "Finished Goods"
            st.rerun()

# -------------------------
# Raw Materials (Paper Reels)
# -------------------------
def fetch_reel_grid() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query("""
           SELECT
             pr.ReelId,
             pr.SLNo AS "SL No.",
             pr.ReelNo AS "Reel No",
             COALESCE(s.Name,'') AS "Reel Supplier",
             COALESCE(m.Name,'') AS "Reel Maker",
             pr.ReceiveDate AS "Material Rcv Dt.",
             pr.SupplierInvDate AS "Maker's/Supplier's Inv Dt.",
             pr.DeckleCm AS "Deckle in cm",
             ROUND(pr.DeckleCm / 2.54, 3) AS "Deckle in Inch",
             pr.GSM, pr.BF,
             pr.Shade AS "Paper Shade",
             pr.OpeningKg AS "Opening Stk Till Date",
             pr.WeightKg AS "Weight (Kg)",
             COALESCE((SELECT SUM(CASE WHEN Type IN ('Issue','Scrap') THEN QtyKg ELSE 0 END)
                       FROM RM_Movement WHERE ReelId = pr.ReelId),0) AS "Consumed Wt",
             pr.LastConsumeDate AS "Consume Dt",
             pr.ConsumptionEntryDate AS "Consumption Entry Date",
             '' AS "Closing Stock till date",
             COALESCE(w.Name || '/' || b.Aisle || '-' || b.Rack || '-' || b.Bin, '') AS "Reel Location",
             COALESCE(sku.SKUCode,'') AS "Target SKU",
             COALESCE(cu.Name,'') AS "Target Customer",
             pr.ReelShiftDate AS "Reel Shifting Date",
             pr.DeliveryChallanNo AS "Delivery Challan No.",
             pr.ReorderLevelKg AS "Reorder Level",
             pr.PaperRatePerKg AS "Paper Rate/Kg",
             pr.TransportRatePerKg AS "Transport Rate/Kg",
             pr.BasicLandedCostPerKg AS "Basic Landed Cost/Kg",
             '' AS "Current Stock Value(INR)",
             CAST((julianday('now') - julianday(pr.ReceiveDate)) AS INT) AS "Reel Holding Time (Days)",
             pr.Remarks AS "Remarks"
           FROM PaperReel pr
           LEFT JOIN Supplier s ON pr.SupplierId = s.SupplierId
           LEFT JOIN Maker m ON pr.MakerId = m.MakerId
           LEFT JOIN Bin b ON pr.ReelLocationBinId = b.BinId
           LEFT JOIN Warehouse w ON b.WarehouseId = w.WarehouseId
           LEFT JOIN SKU sku ON pr.TargetSKUId = sku.SKUId
           LEFT JOIN Customer cu ON pr.TargetCustomerId = cu.CustomerId
           ORDER BY pr.ReceiveDate DESC, pr.ReelNo
        """, conn)

        closings, values = [], []
        for _, row in df.iterrows():
            _, closing = compute_reel_closing(conn, int(row["ReelId"]))
            closings.append(closing)
            cost = pd.read_sql_query("""
                 SELECT PaperRatePerKg, TransportRatePerKg, BasicLandedCostPerKg
                 FROM PaperReel WHERE ReelId=?
            """, conn, params=[int(row["ReelId"])]).iloc[0]
            perkg = float(cost["BasicLandedCostPerKg"] or 0.0)
            if perkg <= 0:
                perkg = float(cost["PaperRatePerKg"] or 0.0) + float(cost["TransportRatePerKg"] or 0.0)
            values.append(round(closing * perkg, 2))

        if len(df) > 0:
            df.loc[:, "Closing Stock till date"] = closings
            df.loc[:, "Current Stock Value(INR)"] = values

        return df.drop(columns=["ReelId"])

def rm_receive_form():
    st.subheader("📥 Receive Paper Reel")
    with get_conn() as conn:
        suppliers = [r["Name"] for r in conn.execute("SELECT Name FROM Supplier ORDER BY Name")]
        makers = [r["Name"] for r in conn.execute("SELECT Name FROM Maker ORDER BY Name")]
        bins = pd.read_sql_query("""
            SELECT BinId, (w.Name || '/' || b.Aisle || '-' || b.Rack || '-' || b.Bin) AS Label
            FROM Bin b JOIN Warehouse w ON b.WarehouseId=w.WarehouseId
            ORDER BY w.Name, b.Aisle, b.Rack, b.Bin
        """, conn)

    c1, c2, c3 = st.columns(3)
    with c1:
        sl = st.text_input("SL No.")
        reelno = st.text_input("Reel No*", placeholder="e.g., Reel-1050")
        supplier = st.selectbox("Reel Supplier*", options=suppliers)
        maker = st.selectbox("Reel Maker*", options=makers)
        rcv_dt = st.date_input("Material Rcv Dt.", value=date.today())
        inv_dt = st.date_input("Maker's/Supplier's Inv Dt.", value=date.today())
    with c2:
        deckle_cm = st.number_input("Deckle in cm*", min_value=1.0, value=160.0, step=0.1)
        st.text_input("Deckle in Inch (auto)", value=str(cm_to_inch(deckle_cm)), disabled=True)
        gsm = st.number_input("GSM*", min_value=60, value=150, step=1)
        bf = st.number_input("BF*", min_value=12, value=22, step=1)
        shade = st.text_input("Paper Shade", value="Natural")
    with c3:
        opening = st.number_input("Opening Stk Till Date (Kg)", min_value=0.0, value=0.0, step=1.0)
        weight = st.number_input("Weight (Kg)*", min_value=1.0, value=1200.0, step=1.0)
        bin_label = st.selectbox("Reel Location*", options=bins["Label"].tolist())
        dc_no = st.text_input("Delivery Challan No.", value="")
        reorder = st.number_input("Reorder Level (Kg)", min_value=0.0, value=300.0, step=1.0)

    c4, c5, c6 = st.columns(3)
    with c4:
        paper_rate = st.number_input("Paper Rate/Kg (INR)", min_value=0.0, value=45.0, step=0.1)
    with c5:
        transport_rate = st.number_input("Transport Rate/Kg (INR)", min_value=0.0, value=2.5, step=0.1)
    with c6:
        landed = st.number_input("Basic Landed Cost/Kg (INR)", min_value=0.0, value=47.5, step=0.1)

    remarks = st.text_area("Remarks")

    if st.button("Receive Reel", type="primary", use_container_width=True):
        with get_conn() as conn:
            cur = conn.cursor()
            # FKs
            cur.execute("SELECT SupplierId FROM Supplier WHERE Name=?", (supplier,)); supplier_id = cur.fetchone()[0]
            cur.execute("SELECT MakerId FROM Maker WHERE Name=?", (maker,)); maker_id = cur.fetchone()[0]
            sel_bin = bins[bins["Label"] == bin_label]["BinId"].iloc[0]

            cur.execute("""
                INSERT INTO PaperReel(
                    SLNo, ReelNo, SupplierId, MakerId, ReceiveDate, SupplierInvDate,
                    DeckleCm, GSM, BF, Shade, OpeningKg, WeightKg, ReelLocationBinId,
                    DeliveryChallanNo, ReorderLevelKg, PaperRatePerKg, TransportRatePerKg,
                    BasicLandedCostPerKg, Remarks
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (sl, reelno, supplier_id, maker_id, rcv_dt.isoformat(), inv_dt.isoformat(),
                  deckle_cm, gsm, bf, shade, opening, weight, int(sel_bin),
                  dc_no, reorder, paper_rate, transport_rate, landed, remarks))
            cur.execute("SELECT ReelId FROM PaperReel WHERE ReelNo=?", (reelno,))
            new_id = cur.fetchone()[0]
            cur.execute("""
                INSERT INTO RM_Movement(ReelId, DateTime, Type, QtyKg, ToBinId, RefDocType, RefDocNo)
                VALUES(?,?,?,?,?,?,?)
            """, (new_id, datetime.now().isoformat(), "Receive", weight, int(sel_bin), "DC", dc_no))
        st.success(f"Reel **{reelno}** received and stored.")

def rm_issue_form():
    st.subheader("📤 Issue to Corrugation / Production")
    with get_conn() as conn:
        reels = pd.read_sql_query("SELECT ReelId, ReelNo FROM PaperReel ORDER BY ReceiveDate DESC", conn)
    if len(reels) == 0:
        st.info("No reels found. Receive a reel first.")
        return
    rmap = {row["ReelNo"]: int(row["ReelId"]) for _, row in reels.iterrows()}
    chosen = st.selectbox("Select Reel", options=list(rmap.keys()))
    qty = st.number_input("Issue Qty (Kg)", min_value=1.0, value=100.0, step=1.0)
    consume_dt = st.date_input("Consume Dt", value=date.today())
    if st.button("Post Issue", type="primary"):
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO RM_Movement(ReelId, DateTime, Type, QtyKg) VALUES(?,?,?,?)",
                        (rmap[chosen], datetime.now().isoformat(), "Issue", qty))
            cur.execute("UPDATE PaperReel SET LastConsumeDate=?, ConsumptionEntryDate=? WHERE ReelId=?",
                        (consume_dt.isoformat(), datetime.now().isoformat(), rmap[chosen]))
        st.success(f"Issued {qty} Kg from **{chosen}**.")

def rm_transfer_adjust_form():
    st.subheader("🔁 Transfer / Adjust / Hold")
    action = st.selectbox("Action", ["Transfer", "Adjust (+/-)", "Mark Hold", "Release Hold"])
    with get_conn() as conn:
        reels = pd.read_sql_query("SELECT ReelId, ReelNo FROM PaperReel ORDER BY ReelNo", conn)
        bins = pd.read_sql_query("""
            SELECT BinId, (w.Name || '/' || b.Aisle || '-' || b.Rack || '-' || b.Bin) AS Label
            FROM Bin b JOIN Warehouse w ON b.WarehouseId=w.WarehouseId
            ORDER BY w.Name, b.Aisle, b.Rack, b.Bin
        """, conn)
    if len(reels) == 0:
        st.info("No reels available.")
        return

    rmap = {row["ReelNo"]: int(row["ReelId"]) for _, row in reels.iterrows()}
    chosen = st.selectbox("Reel", options=list(rmap.keys()))
    qty = st.number_input("Qty (Kg) (for Adjust)", value=0.0, step=0.5)
    bin_label = st.selectbox("To Bin (for Transfer)", options=bins["Label"].tolist())
    refdoc = st.text_input("Reference Doc No.")
    if st.button("Execute", type="secondary"):
        with get_conn() as conn:
            cur = conn.cursor()
            rid = rmap[chosen]
            if action == "Transfer":
                to_bin = int(bins[bins["Label"] == bin_label]["BinId"].iloc[0])
                cur.execute("INSERT INTO RM_Movement(ReelId, DateTime, Type, QtyKg, ToBinId, RefDocNo) VALUES(?,?,?,?,?,?)",
                            (rid, datetime.now().isoformat(), "TransferIn", 0.0, to_bin, refdoc))
                cur.execute("UPDATE PaperReel SET ReelLocationBinId=?, ReelShiftDate=? WHERE ReelId=?",
                            (to_bin, datetime.now().isoformat(), rid))
            elif action == "Adjust (+/-)":
                cur.execute("INSERT INTO RM_Movement(ReelId, DateTime, Type, QtyKg, RefDocNo) VALUES(?,?,?,?,?)",
                            (rid, datetime.now().isoformat(), "Adjust", qty, refdoc))
            elif action == "Mark Hold":
                cur.execute("INSERT INTO RM_Movement(ReelId, DateTime, Type, QtyKg, RefDocNo) VALUES(?,?,?,?,?)",
                            (rid, datetime.now().isoformat(), "Hold", 0.0, refdoc))
            elif action == "Release Hold":
                cur.execute("INSERT INTO RM_Movement(ReelId, DateTime, Type, QtyKg, RefDocNo) VALUES(?,?,?,?,?)",
                            (rid, datetime.now().isoformat(), "Release", 0.0, refdoc))
        st.success(f"{action} recorded for **{chosen}**.")

def show_raw_materials():
    tabs = st.tabs(["📃 Paper Reels (Grid)", "📥 Receive", "📤 Issue", "🔁 Transfer/Adjust"])
    t1, t2, t3, t4 = tabs

    with t1:
        st.caption("Tip: Use column filters and the inbuilt download to export.")
        df = fetch_reel_grid()

        def highlight_reorder(row):
            try:
                if float(row["Closing Stock till date"]) <= float(row["Reorder Level"]):
                    return ["background-color: #fff4f2"] * len(row)
            except Exception:
                pass
            return [""] * len(row)

        st.dataframe(
            df.style.apply(highlight_reorder, axis=1) if len(df) else df,
            use_container_width=True,
            hide_index=True
        )

    with t2:
        rm_receive_form()
    with t3:
        rm_issue_form()
    with t4:
        rm_transfer_adjust_form()

# -------------------------
# WIP
# -------------------------
def show_wip():
    st.subheader("🛠️ WIP Job Board")
    with get_conn() as conn:
        wos = pd.read_sql_query("""
            SELECT wo.WOId, sku.SKUCode, cu.Name AS Customer, wo.QtyPlanned, wo.Status
            FROM WorkOrder wo
            JOIN SKU sku ON wo.SKUId=sku.SKUId
            JOIN Customer cu ON wo.CustomerId=cu.CustomerId
            ORDER BY WOId DESC
        """, conn)
        wip_grid = pd.read_sql_query("""
            SELECT wu.UnitId, wu.WOId, wu.Step, wu.Workcenter, wu.PalletId, wu.Qty, wu.Status, wu.InTime, wu.OutTime
            FROM WIP_Unit wu ORDER BY wu.UnitId DESC
        """, conn)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Open Work Orders**")
        st.dataframe(wos, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**WIP Units**")
        st.dataframe(wip_grid, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("➕ Create Work Order")
    with get_conn() as conn:
        sku_opts = [r["SKUCode"] for r in conn.execute("SELECT SKUCode FROM SKU")]
        cust_opts = [r["Name"] for r in conn.execute("SELECT Name FROM Customer")]
    c3, c4, c5 = st.columns(3)
    with c3:
        sku_ = st.selectbox("SKU", options=sku_opts)
    with c4:
        cust_ = st.selectbox("Customer", options=cust_opts)
    with c5:
        qty_ = st.number_input("Planned Qty", min_value=1.0, value=1000.0, step=1.0)
    if st.button("Create WO", type="primary"):
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT SKUId FROM SKU WHERE SKUCode=?", (sku_,)); sku_id = cur.fetchone()[0]
            cur.execute("SELECT CustomerId FROM Customer WHERE Name=?", (cust_,)); cust_id = cur.fetchone()[0]
            cur.execute("INSERT INTO WorkOrder(SKUId, CustomerId, QtyPlanned, StartPlan, Status) VALUES(?,?,?,?,?)",
                        (sku_id, cust_id, qty_, date.today().isoformat(), "Open"))
        st.success("Work Order created.")

    st.markdown("---")
    st.subheader("↔️ Move WIP")
    steps = ["Corrugation", "PrinterSlotter", "DieCutter", "FolderGluer", "Stitcher", "Bundling", "QA"]
    with get_conn() as conn:
        wo_ids = [r["WOId"] for r in conn.execute("SELECT WOId FROM WorkOrder ORDER BY WOId DESC")]
    c6, c7, c8, c9 = st.columns(4)
    with c6:
        wo_sel = st.selectbox("WO", options=wo_ids)
    with c7:
        step_sel = st.selectbox("Current Step", options=steps, index=0)
    with c8:
        qty_wip = st.number_input("Qty", min_value=1.0, value=500.0, step=1.0)
    with c9:
        pallet = st.text_input("WIP Pallet Id (optional)", value="")
    if st.button("Scan-In at Step", type="secondary"):
        with get_conn() as conn:
            conn.execute("""
                INSERT INTO WIP_Unit(WOId, Step, Workcenter, PalletId, Qty, Status, InTime)
                VALUES(?,?,?,?,?,?,?)
            """, (wo_sel, step_sel, step_sel, pallet, qty_wip, "In-Process", datetime.now().isoformat()))
        st.success("WIP scanned-in.")

    c10, c11 = st.columns(2)
    with c10:
        unit_id = st.number_input("WIP UnitId to Scan-Out", min_value=1, value=1, step=1)
    with c11:
        status_out = st.selectbox("Set Status", ["Good", "Rework", "Hold"])
    if st.button("Scan-Out from Step"):
        with get_conn() as conn:
            conn.execute("UPDATE WIP_Unit SET Status=?, OutTime=? WHERE UnitId=?",
                         (status_out, datetime.now().isoformat(), int(unit_id)))
        st.success("WIP scanned-out.")

# -------------------------
# Finished Goods
# -------------------------
def fg_instant_answer(customer_name: str, sku_code: str) -> Tuple[int, pd.DataFrame]:
    with get_conn() as conn:
        q = pd.read_sql_query("""
            SELECT p.PalletId, s.SKUCode, c.Name AS Customer, w.Name AS Warehouse, 
                   (b.Aisle || '-' || b.Rack || '-' || b.Bin) AS Bin,
                   (p.OnHandQty - p.ReservedQty - p.HoldQty) AS AvailableQty,
                   p.OnHandQty, p.ReservedQty, p.HoldQty
            FROM FG_Pallet p
            JOIN SKU s ON p.SKUId = s.SKUId
            JOIN Customer c ON p.CustomerId = c.CustomerId
            LEFT JOIN Warehouse w ON p.WarehouseId = w.WarehouseId
            LEFT JOIN Bin b ON p.BinId = b.BinId
            WHERE c.Name=? AND s.SKUCode=?
            ORDER BY w.Name, Bin
        """, conn, params=[customer_name, sku_code])
        total = int(q["AvailableQty"].sum()) if len(q) else 0
        return total, q

def fg_pack_form():
    st.subheader("📦 Pack / Putaway")
    with get_conn() as conn:
        sku_opts = [r["SKUCode"] for r in conn.execute("SELECT SKUCode FROM SKU")]
        cust_opts = [r["Name"] for r in conn.execute("SELECT Name FROM Customer")]
        bins = pd.read_sql_query("""
            SELECT BinId, (w.Name || '/' || b.Aisle || '-' || b.Rack || '-' || b.Bin) AS Label
            FROM Bin b JOIN Warehouse w ON b.WarehouseId=w.WarehouseId
            ORDER BY w.Name, b.Aisle, b.Rack, b.Bin
        """, conn)
    c1, c2, c3 = st.columns(3)
    with c1:
        pallet_id = st.text_input("Pallet Id*", value=f"PAL-{int(datetime.now().timestamp())}")
        sku = st.selectbox("SKU*", options=sku_opts)
    with c2:
        customer = st.selectbox("Customer*", options=cust_opts)
        qty = st.number_input("Pack Qty (Nos)*", min_value=1.0, value=2000.0, step=1.0)
    with c3:
        bin_label = st.selectbox("Putaway Bin*", options=bins["Label"].tolist())
        batch = st.text_input("Batch", value="BATCH-NEW")

    if st.button("Create Pallet", type="primary"):
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT SKUId FROM SKU WHERE SKUCode=?", (sku,)); sku_id = cur.fetchone()[0]
            cur.execute("SELECT CustomerId FROM Customer WHERE Name=?", (customer,)); cust_id = cur.fetchone()[0]
            to_bin = int(bins[bins["Label"] == bin_label]["BinId"].iloc[0])
            cur.execute("SELECT WarehouseId FROM Bin WHERE BinId=?", (to_bin,)); wh_id = cur.fetchone()[0]
            cur.execute("""
                INSERT INTO FG_Pallet(PalletId, SKUId, CustomerId, Batch, PackDate, WarehouseId, BinId, OnHandQty)
                VALUES(?,?,?,?,?,?,?,?)
            """, (pallet_id, sku_id, cust_id, batch, today_str(), wh_id, to_bin, qty))
            cur.execute("""
                INSERT INTO FG_Movement(PalletId, DateTime, Type, Qty, ToBinId, RefDoc)
                VALUES(?,?,?,?,?,?)
            """, (pallet_id, datetime.now().isoformat(), "Pack", qty, to_bin, "PACK"))
        st.success(f"Pallet **{pallet_id}** created with {int(qty)} Nos.")

def fg_reserve_dispatch():
    st.subheader("🧾 Reserve & Dispatch")
    with get_conn() as conn:
        pallets = pd.read_sql_query("""
            SELECT PalletId, OnHandQty, ReservedQty, HoldQty FROM FG_Pallet ORDER BY PalletId DESC
        """, conn)
    st.dataframe(pallets, use_container_width=True, hide_index=True)

    if len(pallets) == 0:
        st.info("No pallets found. Use Pack/Putaway first.")
        return

    c1, c2, c3 = st.columns(3)
    pallet = st.selectbox("Pallet Id", options=pallets["PalletId"].tolist())
    qty = st.number_input("Qty", min_value=1.0, value=500.0, step=1.0)
    action = st.selectbox("Action", ["Reserve", "Unreserve", "Dispatch"])
    if st.button("Apply"):
        with get_conn() as conn:
            cur = conn.cursor()
            if action == "Reserve":
                cur.execute("UPDATE FG_Pallet SET ReservedQty = ReservedQty + ? WHERE PalletId=?", (qty, pallet))
                cur.execute("INSERT INTO FG_Movement(PalletId, DateTime, Type, Qty, RefDoc) VALUES(?,?,?,?,?)",
                            (pallet, datetime.now().isoformat(), "Reserve", qty, "SO"))
            elif action == "Unreserve":
                cur.execute("UPDATE FG_Pallet SET ReservedQty = MAX(ReservedQty - ?, 0) WHERE PalletId=?", (qty, pallet))
                cur.execute("INSERT INTO FG_Movement(PalletId, DateTime, Type, Qty, RefDoc) VALUES(?,?,?,?,?)",
                            (pallet, datetime.now().isoformat(), "Unreserve", qty, "SO"))
            else:  # Dispatch
                cur.execute("""
                    UPDATE FG_Pallet
                    SET OnHandQty = MAX(OnHandQty - ?, 0),
                        ReservedQty = MAX(ReservedQty - ?, 0)
                    WHERE PalletId=?
                """, (qty, qty, pallet))
                cur.execute("INSERT INTO FG_Movement(PalletId, DateTime, Type, Qty, RefDoc) VALUES(?,?,?,?,?)",
                            (pallet, datetime.now().isoformat(), "Dispatch", qty, "DC"))
        st.success(f"{action} complete for {pallet}.")

def show_fg():
    st.subheader("🔎 Instant Answer")
    with get_conn() as conn:
        cust_opts = [r["Name"] for r in conn.execute("SELECT Name FROM Customer")]
        sku_opts = [r["SKUCode"] for r in conn.execute("SELECT SKUCode FROM SKU")]
    c1, c2, c3 = st.columns([2,2,1])
    with c1:
        cust = st.selectbox("Customer", options=cust_opts)
    with c2:
        sku = st.selectbox("SKU", options=sku_opts)
    with c3:
        st.write(""); st.write("")
        btn = st.button("View Stock", type="primary")

    if btn:
        total, df = fg_instant_answer(cust, sku)
        st.markdown(f"""
        <div class="metric-card" style="border-left:6px solid {ACCENT_FG}">
            <div class="metric-title">Available to Promise</div>
            <div class="metric-value">{total:,} Nos</div>
            <div class="metric-sub">for {cust} • {sku}</div>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    tabs = st.tabs(["📦 Pack/Putaway", "🧾 Reserve & Dispatch", "📋 FG Inventory"])
    t1, t2, t3 = tabs
    with t1:
        fg_pack_form()
    with t2:
        fg_reserve_dispatch()
    with t3:
        with get_conn() as conn:
            inv = pd.read_sql_query("""
                SELECT p.PalletId, s.SKUCode, c.Name AS Customer, p.Batch, p.PackDate,
                       (p.OnHandQty - p.ReservedQty - p.HoldQty) AS Available,
                       p.OnHandQty, p.ReservedQty, p.HoldQty
                FROM FG_Pallet p
                JOIN SKU s ON p.SKUId=s.SKUId
                JOIN Customer c ON p.CustomerId=c.CustomerId
                ORDER BY p.PackDate DESC
            """, conn)
        st.dataframe(inv, use_container_width=True, hide_index=True)

# -------------------------
# Settings (masters + demo)
# -------------------------
def show_settings():
    st.subheader("⚙️ Masters & Demo")
    st.caption("Create/update masters or seed demo data to explore the app quickly.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Customer**")
        cust = st.text_input("Name")
        gstin = st.text_input("GSTIN", value="")
        if st.button("Add Customer"):
            if cust.strip():
                with get_conn() as conn:
                    conn.execute("INSERT OR IGNORE INTO Customer(Name, GSTIN) VALUES(?,?)",
                                 (cust.strip(), gstin.strip()))
                st.success("Customer saved.")
            else:
                st.warning("Customer name required.")

    with c2:
        st.markdown("**SKU**")
        sku = st.text_input("SKU Code")
        sdesc = st.text_input("Description", value="")
        if st.button("Add SKU"):
            if sku.strip():
                with get_conn() as conn:
                    conn.execute("INSERT OR IGNORE INTO SKU(SKUCode, Description) VALUES(?,?)",
                                 (sku.strip(), sdesc.strip()))
                st.success("SKU saved.")
            else:
                st.warning("SKU Code required.")

    with c3:
        st.markdown("**Warehouse & Bin**")
        wh = st.text_input("Warehouse")
        if st.button("Add Warehouse"):
            if wh.strip():
                with get_conn() as conn:
                    conn.execute("INSERT OR IGNORE INTO Warehouse(Name) VALUES(?)", (wh.strip(),))
                st.success("Warehouse saved.")
            else:
                st.warning("Warehouse name required.")
        st.markdown("Add Bin")
        aisle = st.text_input("Aisle", value="A")
        rack = st.text_input("Rack", value="1")
        bin_ = st.text_input("Bin", value="01")
        if st.button("Add Bin"):
            if not wh.strip():
                st.warning("Enter/select Warehouse name above, then click Add Bin.")
            else:
                with get_conn() as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT WarehouseId FROM Warehouse WHERE Name=?", (wh.strip(),))
                    r = cur.fetchone()
                    if r:
                        cur.execute("INSERT OR IGNORE INTO Bin(WarehouseId, Aisle, Rack, Bin) VALUES(?,?,?,?)",
                                    (int(r[0]), aisle.strip(), rack.strip(), bin_.strip()))
                        st.success("Bin saved.")
                    else:
                        st.warning("Create the Warehouse first.")

    st.markdown("---")
    st.warning("**Demo Mode**: Inserts example customers, SKUs, bins, reels, and pallets.")
    if st.button("Seed Demo Data", type="primary"):
        seed_demo_data()

# -------------------------
# Router
# -------------------------
if page == "Dashboard":
    show_dashboard()
elif page == "Raw Materials":
    show_raw_materials()
elif page == "WIP Items":
    show_wip()
elif page == "Finished Goods":
    show_fg()
else:
    show_settings()
