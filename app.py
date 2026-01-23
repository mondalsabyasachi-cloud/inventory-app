
# --------------------------------------------------------------
# Packsmart Inventory App (RM / WIP / FG) - Streamlit (single file)
# Prepared for: Saby Mondal | Packsmart India Pvt Ltd
# --------------------------------------------------------------
from core.paper_reel_repo import list_paper_reels
from core.db import get_conn as shared_get_conn
from io import BytesIO
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from typing import Optional, Tuple, Dict

import pandas as pd
import streamlit as st

# -------------------------
# Paper Reel Column Canon (GLOBAL)
# -------------------------
PR_COL = {
    "SL": "SL No",
    "OPENING": "Opening Stk Till Date(Kg)",
    "DECKLE_CM": "Deckle(CM)",
    "DECKLE_IN": "Deckle(Inch)",
    "REEL_WT": "Reel Original Weight(Kg)",
    "CONS_WT": "Consumed Wt(Kg)",
    "CLOSE": "Closing Stock Till Date(Kg)",
    "REORDER": "Reorder Level(Kg)",
    "P_RATE": "Paper Rate/Kg(₹)",
    "T_RATE": "Transport Rate/Kg(₹)",
    "L_RATE": "Basic Landed Cost/Kg(₹)",
    "STOCK_VAL": "Current Stock Value(₹)"
}

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

# Raw Material type master list (controls the dropdown on Raw Materials page)
RM_TYPES = [
    "Paper Reel",
    "GUM / Adhesives",
    "Stitching Wire",
    "Strapping Wire",
    "Board / Sheet",
    "Ink / Chemicals",
    "Packaging Accessories",
    "Others",
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

    /* -------- ERP-style inline KPIs -------- */
    .erp-kpi {{
        background: #ffffff;
        border: 1px solid #e6e9ef;
        border-radius: 10px;
        padding: 8px 12px;
        white-space: nowrap;
    }}
    .erp-kpi-label {{
        font-size: 0.75rem;
        color: #6b7280;
    }}
    .erp-kpi-value {{
        font-size: 1.15rem;        
        font-weight: 600;
        color: #111827;
        overflow: visible;          
        text-overflow: clip;        
    }}
</style>
""", unsafe_allow_html=True)


# -------------------------
# NAVIGATION: process any pending nav BEFORE rendering sidebar
# -------------------------
PAGES = ["Dashboard", "Raw Materials", "WIP Items", "Finished Goods", "Settings"]

if "left_nav" not in st.session_state:
    st.session_state.left_nav = "Dashboard"

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
        # -----------------------------------
        # Paper Reel Movement Ledger (PHASE-1)
        # -----------------------------------
        cur.execute("""
        CREATE TABLE IF NOT EXISTS PaperReelMovement(
            MovementId INTEGER PRIMARY KEY AUTOINCREMENT,
            ReelId INTEGER NOT NULL,
            EventType TEXT NOT NULL,     -- RECEIVE / CONSUME / RETURN / ADJUST
            QtyKg REAL NOT NULL,
            EventDate TEXT NOT NULL,
            RefDoc TEXT,
            CreatedBy TEXT,
            CreatedAt TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Raw Material: Paper Reels
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

        # WIP
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
              Step TEXT NOT NULL,
              Workcenter TEXT,
              PalletId TEXT,
              Qty REAL NOT NULL,
              Status TEXT DEFAULT 'In-Process',
              InTime TEXT, OutTime TEXT,
              FOREIGN KEY (WOId) REFERENCES WorkOrder(WOId)
            )
        """)

        # FG
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
# SL No resequencing (Paper Reel)
# -------------------------
def resequence_slno(conn):
    """
    Reassigns SLNo as clean numeric sequence (1,2,3...)
    ordered by ReceiveDate then ReelNo.
    """
    cur = conn.cursor()

    cur.execute("""
        WITH ordered AS (
            SELECT ReelId,
                   ROW_NUMBER() OVER (
                       ORDER BY
                           date(ReceiveDate) ASC,
                           ReelNo ASC
                   ) AS new_sl
            FROM PaperReel
        )
        UPDATE PaperReel
        SET SLNo = (
            SELECT new_sl
            FROM ordered
            WHERE ordered.ReelId = PaperReel.ReelId
        )
    """)

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
    Returns: (ConsumedKg, ClosingKg).
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

# =========================
# Paper Reels: helpers
# =========================

# Display column grouping (two-row headers)
COLUMN_GROUPS = {
    # Identity & Status
    "SL No.": "Identity & Status",
    "Reel No": "Identity & Status",
    "Reel Location": "Identity & Status",
    "Reel Holding Time (Days)": "Identity & Status",
    "Remarks": "Identity & Status",

    # Commercials & Links
    "Reel Supplier": "Commercials & Links",
    "Reel Maker": "Commercials & Links",
    "Material Rcv Dt.": "Commercials & Links",
    "Maker's/Supplier's Inv Dt.": "Commercials & Links",
    "Delivery Challan No.": "Commercials & Links",
    "Paper Rate/Kg": "Commercials & Links",
    "Transport Rate/Kg": "Commercials & Links",
    "Basic Landed Cost/Kg": "Commercials & Links",
    "Current Stock Value(INR)": "Commercials & Links",

    # Technical Specs
    "Deckle in cm": "Technical Specs",
    "Deckle in Inch": "Technical Specs",
    "GSM": "Technical Specs",
    "BF": "Technical Specs",
    "Paper Shade": "Technical Specs",

    # Stock & Consumption
    "Opening Stk Till Date": "Stock & Consumption",
    "Weight (Kg)": "Stock & Consumption",
    "Consumed Wt": "Stock & Consumption",
    "Consume Dt": "Stock & Consumption",
    "Consumption Entry Date": "Stock & Consumption",
    "Closing Stock till date": "Stock & Consumption",
    "Reorder Level": "Stock & Consumption",

    # Planning Hooks
    "Target SKU": "Planning Hooks",
    "Target Customer": "Planning Hooks",
    "Reel Shifting Date": "Planning Hooks",
}

def group_columns_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """Apply MultiIndex columns using COLUMN_GROUPS as the top level."""
    tuples = []
    for col in df.columns:
        tuples.append((COLUMN_GROUPS.get(col, ""), col))
    df = df.copy()
    df.columns = pd.MultiIndex.from_tuples(tuples)
    return df

def paper_cost_per_kg(row: pd.Series) -> float:
    """Prefer Basic Landed Cost; else Paper + Transport rate."""
    try:
        basic = float(row.get("Basic Landed Cost/Kg", 0) or 0)
        if basic > 0:
            return basic
        pr = float(row.get("Paper Rate/Kg", 0) or 0)
        tr = float(row.get("Transport Rate/Kg", 0) or 0)
        return pr + tr
    except Exception:
        return 0.0

# -------------------------
# Opening Stock Carry-Forward Logic
# -------------------------
def apply_opening_stock_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Opening Stk Till Date (Kg) logic:
    - First (Reel Maker + Reel No) => 0
    - Next occurrences => previous Closing Stock Till Date(Kg)
    """
    if df.empty:
        return df

    df = df.copy()
    df["_key"] = df["Reel Maker"].astype(str) + "||" + df["Reel No"].astype(str)

    df["Opening Stk Till Date"] = 0.0
    last_close = {}

    for idx, row in df.iterrows():
        key = row["_key"]
        if key in last_close:
            df.at[idx, "Opening Stk Till Date"] = last_close[key]
        else:
            df.at[idx, "Opening Stk Till Date"] = 0.0

        closing = row.get("Closing Stock till date", 0)
        last_close[key] = closing

    df.drop(columns=["_key"], inplace=True)
    return df

def build_paper_grid_with_calcs(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Takes the flat DataFrame from fetch_reel_grid() and fills computed columns."""
    if raw_df.empty:
        return raw_df

    df = raw_df.copy()
    # Apply Opening Stock carry-forward logic
    df = apply_opening_stock_logic(df)
    # -------------------------
    # STEP-4A: Closing Stock Till Date (Kg)
    # -------------------------
    df["Closing Stock till date"] = 0.0

    for idx, row in df.iterrows():
        opening = float(row.get("Opening Stk Till Date", 0) or 0)
        consumed = float(row.get("Consumed Wt", 0) or 0)

        # Repeat usage case → Opening exists
        if opening > 0:
            closing = opening - consumed
        else:
            # First receipt case
            weight = float(row.get("Weight (Kg)", 0) or 0)
            closing = weight - consumed

        df.at[idx, "Closing Stock till date"] = round(closing, 3)
    
    
    # Reel Holding Time (Days): ensure present
    if "Material Rcv Dt." in df.columns and "Reel Holding Time (Days)" not in df.columns:
        rcvs = pd.to_datetime(df["Material Rcv Dt."], errors="coerce")
        df["Reel Holding Time (Days)"] = (pd.Timestamp.today().normalize() - rcvs).dt.days

    # Deckle in Inch mirrors Deckle in cm
    if "Deckle in cm" in df.columns:
        df["Deckle in Inch"] = (pd.to_numeric(df["Deckle in cm"], errors="coerce") / 2.54).round(3)

    # Current Stock Value (INR)
    if "Closing Stock till date" in df.columns:
        close_numeric = pd.to_numeric(df["Closing Stock till date"], errors="coerce").fillna(0.0)
        perkg = df.apply(paper_cost_per_kg, axis=1)
        df["Current Stock Value(INR)"] = (close_numeric * perkg).round(2)

    return df

# -------- Excel Upload (bulk import) ----------
PAPER_EXCEL_COLUMNS = [
    "SL No.",
    "Reel No",
    "Reel Supplier",
    "Reel Maker",
    "Material Rcv Dt.",
    "Maker's/Supplier's Inv Dt.",
    "Deckle in cm",
    "GSM",
    "BF",
    "Paper Shade",
    "Opening Stk Till Date (Kg)",
    "Weight (Kg)",
    "Reel Location",          # e.g., "Main WH/A-1-01"
    "Delivery Challan No.",
    "Reorder Level (Kg)",
    "Paper Rate/Kg",
    "Transport Rate/Kg",
    "Basic Landed Cost/Kg",
    "Remarks",
]

def get_or_create_id(conn, table: str, key_field: str, key_value: str, id_field: str):
    cur = conn.cursor()
    cur.execute(f"SELECT {id_field} FROM {table} WHERE {key_field}=?", (key_value,))
    r = cur.fetchone()
    if r:
        return r[0]
    cur.execute(f"INSERT INTO {table}({key_field}) VALUES(?)", (key_value,))
    return cur.lastrowid

def parse_bin_label(conn, label: str) -> Optional[int]:
    """Convert a label like 'Main WH/A-1-01' into BinId. If not found, create Warehouse/Bin."""
    try:
        wh_name, rest = label.split("/", 1)
        aisle, rack, bin_ = rest.split("-")
    except Exception:
        return None
    cur = conn.cursor()
    # Warehouse
    cur.execute("SELECT WarehouseId FROM Warehouse WHERE Name=?", (wh_name.strip(),))
    r = cur.fetchone()
    if r:
        wh_id = int(r[0])
    else:
        cur.execute("INSERT INTO Warehouse(Name) VALUES(?)", (wh_name.strip(),))
        wh_id = cur.lastrowid
    # Bin
    cur.execute("""SELECT BinId FROM Bin
                   WHERE WarehouseId=? AND Aisle=? AND Rack=? AND Bin=?""",
                (wh_id, aisle.strip(), rack.strip(), bin_.strip()))
    r = cur.fetchone()
    if r:
        return int(r[0])
    cur.execute("INSERT INTO Bin(WarehouseId, Aisle, Rack, Bin) VALUES(?,?,?,?)",
                (wh_id, aisle.strip(), rack.strip(), bin_.strip()))
    return cur.lastrowid


def make_paper_template() -> tuple[bytes, str, str]:
    """
    Builds a template for Paper Reels.
    - Tries to produce XLSX using xlsxwriter if it's installed.
    - Falls back to CSV (UTF-8 with BOM) when xlsxwriter is not present.
    Returns: (data_bytes, file_name, mime_type)
    """
    df = pd.DataFrame(columns=PAPER_EXCEL_COLUMNS)

    # Try XLSX first (optional dependency)
    try:
        import xlsxwriter  # noqa: F401
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="PaperReels")
        buf.seek(0)
        return (
            buf.getvalue(),
            "paper_reels_template.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception:
        # Fallback to CSV (no extra packages required)
        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        return (csv_bytes, "paper_reels_template.csv", "text/csv")



def rm_upload_excel_ui():
    st.subheader("⬆ Upload Paper Reels (Excel / CSV)")
    st.caption("You can upload either a .xlsx (requires openpyxl on the server) or a .csv (no extra packages).")

    # Offer a template (XLSX if possible, else CSV)
    data, fname, mime = make_paper_template()
    st.download_button(
        label=f"Download Template ({fname.split('.')[-1].upper()})",
        data=data,
        file_name=fname,
        mime=mime,
        use_container_width=False
    )

    uploaded = st.file_uploader(
        "Upload completed template (.xlsx or .csv)",
        type=["xlsx", "csv"],
        key="paper_excel_uploader"
    )
    if uploaded is None:
        return

    # Read the uploaded file
    try:
        if uploaded.name.lower().endswith(".csv"):
            xdf = pd.read_csv(uploaded)
        else:
            # Pandas needs openpyxl to read .xlsx; handle missing engine cleanly
            try:
                xdf = pd.read_excel(uploaded)
            except ModuleNotFoundError:
                st.error(
                    "Reading .xlsx requires the 'openpyxl' package on the server. "
                    "Please either add 'openpyxl' to your environment (requirements.txt) "
                    "or upload the CSV template instead."
                )
                return
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return

    # Column validation
    missing = [c for c in PAPER_EXCEL_COLUMNS if c not in xdf.columns]
    if missing:
        st.error(f"Missing columns in the uploaded file: {missing}")
        st.info("Download the template above and paste your data into it, then upload again.")
        return

    # Insert rows
    inserted = 0
    with get_conn() as conn:
        cur = conn.cursor()
        for _, r in xdf.iterrows():
            try:
                supplier_id = get_or_create_id(conn, "Supplier", "Name", str(r["Reel Supplier"]).strip(), "SupplierId")
                maker_id    = get_or_create_id(conn, "Maker", "Name", str(r["Reel Maker"]).strip(), "MakerId")
                bin_id      = parse_bin_label(conn, str(r["Reel Location"]).strip()) if pd.notna(r["Reel Location"]) else None

                cur.execute("""
                    INSERT OR IGNORE INTO PaperReel(
                        SLNo, ReelNo, SupplierId, MakerId, ReceiveDate, SupplierInvDate,
                        DeckleCm, GSM, BF, Shade, OpeningKg, WeightKg, ReelLocationBinId,
                        DeliveryChallanNo, ReorderLevelKg, PaperRatePerKg, TransportRatePerKg,
                        BasicLandedCostPerKg, Remarks
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    str(r["SL No."]) if pd.notna(r["SL No."]) else None,
                    str(r["Reel No"]).strip(),
                    supplier_id, maker_id,
                    pd.to_datetime(r["Material Rcv Dt."], errors="coerce").date().isoformat()
                        if pd.notna(r["Material Rcv Dt."]) else None,
                    pd.to_datetime(r["Maker's/Supplier's Inv Dt."], errors="coerce").date().isoformat()
                        if pd.notna(r["Maker's/Supplier's Inv Dt."]) else None,
                    float(r["Deckle in cm"]) if pd.notna(r["Deckle in cm"]) else None,
                    int(r["GSM"]) if pd.notna(r["GSM"]) else None,
                    int(r["BF"]) if pd.notna(r["BF"]) else None,
                    str(r["Paper Shade"]).strip() if pd.notna(r["Paper Shade"]) else None,
                    float(r["Opening Stk Till Date (Kg)"]) if pd.notna(r["Opening Stk Till Date (Kg)"]) else 0.0,
                    float(r["Weight (Kg)"]) if pd.notna(r["Weight (Kg)"]) else 0.0,
                    bin_id,
                    str(r["Delivery Challan No."]).strip() if pd.notna(r["Delivery Challan No."]) else None,
                    float(r["Reorder Level (Kg)"]) if pd.notna(r["Reorder Level (Kg)"]) else 0.0,
                    float(r["Paper Rate/Kg"]) if pd.notna(r["Paper Rate/Kg"]) else 0.0,
                    float(r["Transport Rate/Kg"]) if pd.notna(r["Transport Rate/Kg"]) else 0.0,
                    float(r["Basic Landed Cost/Kg"]) if pd.notna(r["Basic Landed Cost/Kg"]) else 0.0,
                    str(r["Remarks"]).strip() if pd.notna(r["Remarks"]) else None
                ))

                # Fetch ReelId & add a Receive movement so closing stock computes
                cur.execute("SELECT ReelId, WeightKg FROM PaperReel WHERE ReelNo=?", (str(r["Reel No"]).strip(),))
                row = cur.fetchone()
                if not row:
                    continue
                reel_id, wt = int(row[0]), float(row[1] or 0.0)
                cur.execute("""
                    INSERT INTO RM_Movement(ReelId, DateTime, Type, QtyKg, ToBinId, RefDocType, RefDocNo)
                    VALUES(?,?,?,?,?,?,?)
                """, (reel_id, datetime.now().isoformat(), "Receive", wt, bin_id, uploaded.name.upper(), "UPLOAD"))
                inserted += 1

            except Exception as e:
                st.warning(f"Row skipped due to error: {e}")

    st.success(f"Imported {inserted} paper reels from file: **{uploaded.name}**")
    st.rerun()

def insert_two_sample_reels():
    """Adds two sample reels with correct sequential SL numbers and business logic."""
    with get_conn() as conn:
        cur = conn.cursor()

        # -------- Determine next SL No (numeric) --------
        cur.execute("""
            SELECT MAX(CAST(SLNo AS INTEGER))
            FROM PaperReel
            WHERE SLNo GLOB '[0-9]*'
        """)
        r = cur.fetchone()
        next_sl = int(r[0]) + 1 if r and r[0] is not None else 1
                # -------- Determine next SAMPLE ReelNo (monotonic, never reused) --------
        cur.execute("""
            SELECT MAX(
                CAST(SUBSTR(ReelNo, LENGTH('SAMPLE-PR-') + 1) AS INTEGER)
            )
            FROM PaperReel
            WHERE ReelNo LIKE 'SAMPLE-PR-%'
        """)
        r = cur.fetchone()
        next_sample_no = int(r[0]) + 1 if r and r[0] is not None else 1
     
        
        # -------- Ensure a bin exists --------
        cur.execute("SELECT BinId FROM Bin ORDER BY BinId LIMIT 1")
        rb = cur.fetchone()
        if rb:
            sample_bin = int(rb[0])
        else:
            cur.execute("INSERT INTO Warehouse(Name) VALUES('Main WH')")
            wh_id = cur.lastrowid
            cur.execute(
                "INSERT INTO Bin(WarehouseId, Aisle, Rack, Bin) VALUES(?,?,?,?)",
                (wh_id, "A", "1", "01")
            )
            sample_bin = cur.lastrowid

        # -------- Masters --------
        sup = get_or_create_id(conn, "Supplier", "Name", "Demo Supplier", "SupplierId")
        mk  = get_or_create_id(conn, "Maker", "Name", "Demo Maker", "MakerId")

        # -------- Insert 2 sample reels --------
        for i in range(2):
            sl_no = str(next_sl + i)
            reel_no = f"SAMPLE-PR-{next_sample_no + i}"

            cur.execute("""
                from core.paper_reel_repo import insert_paper_reel

            insert_paper_reel(conn, {
                "sl_no": sl_no,
                "reel_no": reel_no,
                "supplier_id": sup,
                "maker_id": mk,
                "receive_date": today_str(),
                "supplier_inv_date": today_str(),
                "deckle_cm": 180.0,
                "gsm": 150,
                "bf": 22,
                "shade": "Natural",
                "opening_kg": 0.0,
                "received_kg": 1000.0,
                "bin_id": sample_bin,
                "delivery_challan_no": f"DC-SAMPLE-{sl_no}",
                "reorder_level_kg": 300.0,
                "paper_rate_per_kg": 45.0,
                "transport_rate_per_kg": 2.5,
                "basic_landed_cost_per_kg": 47.5,
                "remarks": "Auto-generated sample reel"
            })


            # Movement entry
            cur.execute("SELECT ReelId FROM PaperReel WHERE ReelNo=?", (reel_no,))
            rid = cur.fetchone()[0]

            cur.execute("""
                INSERT INTO RM_Movement(
                    ReelId, DateTime, Type, QtyKg, ToBinId, RefDocType, RefDocNo
                ) VALUES(?,?,?,?,?,?,?)
            """, (
                rid,
                datetime.now().isoformat(),
                "Receive",
                1000.0,
                sample_bin,
                "SAMPLE",
                sl_no
            ))

# =========================
# Edit/Delete helpers (Simple & Stable)
# =========================

PAPER_EDITABLE_MAP = {
    "SL No.": "SLNo",
    "Material Rcv Dt.": "ReceiveDate",
    "Maker's/Supplier's Inv Dt.": "SupplierInvDate",
    "Deckle in cm": "DeckleCm",
    "GSM": "GSM",
    "BF": "BF",
    "Paper Shade": "Shade",
    "Opening Stk Till Date": "OpeningKg",
    "Weight (Kg)": "WeightKg",
    "Consume Dt": "LastConsumeDate",
    "Consumption Entry Date": "ConsumptionEntryDate",
    "Reel Shifting Date": "ReelShiftingDate",
    "Delivery Challan No.": "DeliveryChallanNo",
    "Reorder Level": "ReorderLevelKg",
    "Paper Rate/Kg": "PaperRatePerKg",
    "Transport Rate/Kg": "TransportRatePerKg",
    "Basic Landed Cost/Kg": "BasicLandedCostPerKg",
    "Remarks": "Remarks",
}

PAPER_EDIT_COLUMNS_ORDER = [
    "Select",
    "Reel No",
    "SL No.",
    "Material Rcv Dt.",
    "Maker's/Supplier's Inv Dt.",
    "Deckle in cm",
    "Deckle in Inch",
    "GSM",
    "BF",
    "Paper Shade",
    "Opening Stk Till Date",
    "Weight (Kg)",
    "Consume Dt",
    "Consumption Entry Date",
    "Reel Shifting Date",
    "Delivery Challan No.",
    "Reorder Level",
    "Paper Rate/Kg",
    "Transport Rate/Kg",
    "Basic Landed Cost/Kg",
    "Remarks",
]

def _norm_date(val):
    """Normalize value into ISO 'YYYY-MM-DD' or None."""
    if val is None or val == "" or (isinstance(val, float) and pd.isna(val)):
        return None
    ts = pd.to_datetime(val, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date().isoformat()

def build_paper_edit_df(calc_df: pd.DataFrame) -> pd.DataFrame:
    """Return a smaller, ordered dataframe for editing (with Select bool column)."""
    if calc_df.empty:
        return pd.DataFrame(columns=PAPER_EDIT_COLUMNS_ORDER)
    df = calc_df.copy()
    if "Deckle in cm" in df.columns:
        df["Deckle in Inch"] = (pd.to_numeric(df["Deckle in cm"], errors="coerce") / 2.54).round(3)
    if "Opening Stk Till Date (Kg)" in df.columns and "Opening Stk Till Date" not in df.columns:
        df = df.rename(columns={"Opening Stk Till Date (Kg)": "Opening Stk Till Date"})
    edit_df = pd.DataFrame()
    edit_df["Select"] = False  # ensure boolean
    for col in PAPER_EDIT_COLUMNS_ORDER:
        if col == "Select":
            continue
        edit_df[col] = df[col] if col in df.columns else None
    return edit_df[PAPER_EDIT_COLUMNS_ORDER]

def _coerce_value(db_field: str, ui_value):
    """Type-safe coercion before DB UPDATE."""
    numeric_fields = {"DeckleCm","GSM","BF","OpeningKg","WeightKg","ReorderLevelKg","PaperRatePerKg","TransportRatePerKg","BasicLandedCostPerKg"}
    date_fields    = {"ReceiveDate","SupplierInvDate","LastConsumeDate","ConsumptionEntryDate","ReelShiftingDate"}
    if db_field in numeric_fields:
        try:
            return float(ui_value) if ui_value not in (None, "") else None
        except Exception:
            return None
    if db_field in date_fields:
        return _norm_date(ui_value)
    return None if ui_value in (None, "") else str(ui_value)

def save_paper_edits(orig_df: pd.DataFrame, edited_df: pd.DataFrame) -> int:
    """Diff original vs edited, and persist updates to DB."""
    if edited_df is None or edited_df.empty:
        return 0
    o = orig_df.set_index("Reel No", drop=False)
    e = edited_df.set_index("Reel No", drop=False)
    changed_rows = 0
    with get_conn() as conn:
        cur = conn.cursor()
        for reel_no, row in e.iterrows():
            if reel_no not in o.index:
                continue
            updates, params = [], []
            for ui_col, db_field in PAPER_EDITABLE_MAP.items():
                if ui_col not in e.columns or ui_col not in o.columns:
                    continue
                old_val = o.loc[reel_no, ui_col]
                new_val = row[ui_col]
                if db_field in {"ReceiveDate","SupplierInvDate","LastConsumeDate","ConsumptionEntryDate","ReelShiftingDate"}:
                    old_norm = _norm_date(old_val)
                    new_norm = _norm_date(new_val)
                    if old_norm != new_norm:
                        updates.append(f"{db_field}=?")
                        params.append(new_norm)
                else:
                    if str(old_val) != str(new_val):
                        updates.append(f"{db_field}=?")
                        params.append(_coerce_value(db_field, new_val))
            if updates:
                params.append(reel_no)
                cur.execute(f"UPDATE PaperReel SET {', '.join(updates)} WHERE ReelNo=?", params)
                changed_rows += 1
    return changed_rows

def delete_paper_rows_by_reel_nos(reel_nos) -> int:
    """Delete rows by Reel No and resequence SLNo."""
    if not reel_nos:
        return 0

    with get_conn() as conn:
        cur = conn.cursor()

        for rn in reel_nos:
            cur.execute("SELECT ReelId FROM PaperReel WHERE ReelNo=?", (rn,))
            r = cur.fetchone()
            if not r:
                continue
            rid = int(r[0])

            cur.execute("DELETE FROM RM_Movement WHERE ReelId=?", (rid,))
            cur.execute("DELETE FROM PaperReel WHERE ReelId=?", (rid,))

        # ✅ MUST happen after all deletes, inside same connection
        resequence_slno(conn)

        return len(reel_nos)



def resequence_slno(conn):
    """
    Reassigns SLNo as continuous numeric values (1,2,3...)
    ordered by ReceiveDate, ReelNo.
    """
    cur = conn.cursor()

    cur.execute("""
        SELECT ReelId
        FROM PaperReel
        ORDER BY
            date(ReceiveDate) ASC,
            ReelNo ASC
    """)
    rows = cur.fetchall()

    for idx, (reel_id,) in enumerate(rows, start=1):
        cur.execute(
            "UPDATE PaperReel SET SLNo=? WHERE ReelId=?",
            (str(idx), reel_id)
        )



# -------------------------
# Demo Data seeding
# -------------------------
def seed_demo_data():
    with get_conn() as conn:
        cur = conn.cursor()
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

        # one demo receive
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

page = st.session_state.left_nav

# -------------------------
# Dashboard
# -------------------------
def show_dashboard():
    col1, col2, col3 = st.columns(3)
    with shared_get_conn() as conn:
        rids = pd.read_sql_query("SELECT ReelId FROM PaperReel", conn)
        total_rm_kg, total_reels = 0.0, len(rids)
        for _, r in rids.iterrows():
            _, closing = compute_reel_closing(conn, int(r["ReelId"]))
            total_rm_kg += closing
        wip_qty = pd.read_sql_query(
            "SELECT COALESCE(SUM(Qty),0) AS Qty FROM WIP_Unit WHERE Status='In-Process'", conn
        ).iloc[0]["Qty"]
        wip_units = pd.read_sql_query(
            "SELECT COUNT(*) AS Cnt FROM WIP_Unit WHERE Status='In-Process'", conn
        ).iloc[0]["Cnt"]
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
            st.markdown(
                f'<div class="metric-value">{total_rm_kg:,.2f} Kg</div>',
                unsafe_allow_html=True
            )

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
            st.markdown(
                f'<div class="metric-sub">available to promise • {fg_pallets} pallets</div>',
                unsafe_allow_html=True
            )

        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("➕ Add Inventory")
        st.caption("Use **Raw Materials** or **Finished Goods** pages for detailed operations.")
        if st.button("Go to Raw Materials", use_container_width=True, type="primary"):
            st.session_state["_pending_nav"] = "Raw Materials"; st.rerun()
    with c2:
        st.subheader("🔎 View Inventory")
        st.caption("Filter and drill down in each module (RM / WIP / FG).")
        if st.button("Go to Finished Goods", use_container_width=True):
            st.session_state["_pending_nav"] = "Finished Goods"; st.rerun()

# -------------------------
# Raw Materials (Paper Reels and others)
# -------------------------
def fetch_reel_grid() -> pd.DataFrame:
    """
    Phase-2: Read-side authority moved to repository.
    UI structure and columns remain unchanged.
    """
    with get_conn() as conn:
        return list_paper_reels(conn)
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
        supplier = st.selectbox("Reel Supplier*", options=suppliers, help="Select from Supplier master. Add new supplier via Settings.")
        maker = st.selectbox("Reel Maker*", options=makers, help="Select from Maker master. Add new maker via Settings.")
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
            cur.execute("SELECT SupplierId FROM Supplier WHERE Name=?", (supplier,)); supplier_id = cur.fetchone()[0]
            cur.execute("SELECT MakerId FROM Maker WHERE Name=?", (maker,)); maker_id = cur.fetchone()[0]
            sel_bin = int(bins[bins["Label"] == bin_label]["BinId"].iloc[0])
            cur.execute("""
                INSERT INTO PaperReel(
                    SLNo, ReelNo, SupplierId, MakerId, ReceiveDate, SupplierInvDate,
                    DeckleCm, GSM, BF, Shade, OpeningKg, WeightKg, ReelLocationBinId,
                    DeliveryChallanNo, ReorderLevelKg, PaperRatePerKg, TransportRatePerKg,
                    BasicLandedCostPerKg, Remarks
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (sl, reelno, supplier_id, maker_id, rcv_dt.isoformat(), inv_dt.isoformat(),
                  deckle_cm, gsm, bf, shade, opening, weight, sel_bin,
                  dc_no, reorder, paper_rate, transport_rate, landed, remarks))
            cur.execute("SELECT ReelId FROM PaperReel WHERE ReelNo=?", (reelno,))
            new_id = cur.fetchone()[0]
            # -------------------------------------------------
            # PHASE-1: RECEIVE → PaperReelMovement (AUDIT LEDGER)
            # -------------------------------------------------
            cur.execute("""
            INSERT INTO PaperReelMovement(
                ReelId,
                EventType,
                QtyKg,
                EventDate,
                RefDoc,
                CreatedBy
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                new_id,
                "RECEIVE",
                weight,                     # Reel Original Weight(Kg)
                rcv_dt.strftime("%Y-%m-%d"),
                dc_no,
                "Stores"
            ))

            cur.execute("""
                INSERT INTO RM_Movement(ReelId, DateTime, Type, QtyKg, ToBinId, RefDocType, RefDocNo)
                VALUES(?,?,?,?,?,?,?)
            """, (new_id, datetime.now().isoformat(), "Receive", weight, sel_bin, "DC", dc_no))
        st.success(f"Reel **{reelno}** received and stored.")

def rm_issue_form():
    st.subheader("📤 Issue to Corrugation / Production")
    with get_conn() as conn:
        reels = pd.read_sql_query("SELECT ReelId, ReelNo FROM PaperReel ORDER BY ReceiveDate DESC", conn)
    if len(reels) == 0:
        st.info("No reels found. Receive a reel first."); return
    rmap = {row["ReelNo"]: int(row["ReelId"]) for _, row in reels.iterrows()}
    chosen = st.selectbox("Select Reel", options=list(rmap.keys()))
    qty = st.number_input("Issue Qty (Kg)", min_value=1.0, value=100.0, step=1.0)
    
    if st.button("Post Issue", type="primary"):
        with get_conn() as conn:
            cur = conn.cursor()
            # -----------------------------------
            # PHASE-1: CONSUMPTION → AUDIT LEDGER
            # -----------------------------------
            cur.execute("""
            INSERT INTO PaperReelMovement(
                ReelId,
                EventType,
                QtyKg,
                EventDate,
                RefDoc,
                CreatedBy
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                rmap[chosen],                     # ReelId
                "ISSUE",                          # EventType
                qty,                              # Qty issued (NOT consumed)
                date.today().strftime("%Y-%m-%d"),# Issue date (known today)
                "OUTSOURCE",                      # Reason / RefDoc
                "Stores"                          # Who entered
            ))

            
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
        st.info("No reels available."); return
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
                cur.execute("""
                    INSERT INTO RM_Movement(ReelId, DateTime, Type, QtyKg, ToBinId, RefDocNo)
                    VALUES(?,?,?,?,?,?)
                """, (rid, datetime.now().isoformat(), "TransferIn", 0.0, to_bin, refdoc))
                cur.execute("UPDATE PaperReel SET ReelLocationBinId=?, ReelShiftDate=? WHERE ReelId=?",
                            (to_bin, datetime.now().isoformat(), rid))
            elif action == "Adjust (+/-)":
                cur.execute("""
                    INSERT INTO RM_Movement(ReelId, DateTime, Type, QtyKg, RefDocNo)
                    VALUES(?,?,?,?,?)
                """, (rid, datetime.now().isoformat(), "Adjust", qty, refdoc))
            elif action == "Mark Hold":
                cur.execute("""
                    INSERT INTO RM_Movement(ReelId, DateTime, Type, QtyKg, RefDocNo)
                    VALUES(?,?,?,?,?)
                """, (rid, datetime.now().isoformat(), "Hold", 0.0, refdoc))
            elif action == "Release Hold":
                cur.execute("""
                    INSERT INTO RM_Movement(ReelId, DateTime, Type, QtyKg, RefDocNo)
                    VALUES(?,?,?,?,?)
                """, (rid, datetime.now().isoformat(), "Release", 0.0, refdoc))
        st.success(f"{action} recorded for **{chosen}**.")


def show_raw_materials():
    st.subheader("Raw Materials")

    rm_type = st.selectbox(
        "Raw Material Type",
        options=RM_TYPES,
        index=0,
        key="rm_type_selector"
    )

    if rm_type != "Paper Reel":
        st.info("Only Paper Reel is enabled currently.")
        return

    # -------------------------------------------------
    # Fetch & compute data ONCE
    # -------------------------------------------------
    base_df = fetch_reel_grid()
    calc_df = build_paper_grid_with_calcs(base_df)

    # -------------------------------------------------
    # Header + actions + KPIs
    # -------------------------------------------------
    st.markdown("### 📜 Paper Reels")

    
    h1, h2, h3, h4 = st.columns([1.8, 1.2, 1.5, 1.5])

    with h1:
        ca, cb = st.columns(2)
        with ca:
            if st.button("➕ Add 2 Sample Reels", use_container_width=True):
                insert_two_sample_reels()
                st.rerun()
        with cb:
            grouped = st.toggle(
                "Group columns",
                value=True,
                help="Show grouped headers for readability."
            )

    # KPI calculations (GLOBAL, before filters)
    total_stock_kg = (
        pd.to_numeric(calc_df["Closing Stock till date"], errors="coerce")
        .fillna(0)
        .sum()
    ) if not calc_df.empty else 0.0

    total_stock_value = (
        pd.to_numeric(calc_df["Current Stock Value(INR)"], errors="coerce")
        .fillna(0)
        .sum()
    ) if not calc_df.empty else 0.0

    # -------------------------------------------------
    # ERP KPI ROW (Total Reel Weight & Cost)
    # -------------------------------------------------
  
    with h2:
        st.markdown(
            f"""
            <div class="erp-kpi">
                <div class="erp-kpi-label">Total Stock (Kg)</div>
                st.markdown(
                    f'<div class="erp-kpi-value">{total_stock_kg:,.2f}</div>',
                    unsafe_allow_html=True
                )

            </div>
            """,
            unsafe_allow_html=True
        )

    with h3:
        st.markdown(
            f"""
            <div class="erp-kpi">
                <div class="erp-kpi-label">Total Stock Value (₹)</div>
                <div class="erp-kpi-value">₹ {total_stock_value:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )    
    

    # -------------------------------------------------
    # Filters (Reel No, BF, GSM, Deckle)
    # -------------------------------------------------
    st.markdown("#### Filters")

    f1, f2, f3, f4 = st.columns(4)

    with f1:
        q_reel = st.text_input("Reel No").strip()
    with f2:
        q_bf = st.number_input("BF", min_value=0, value=0, step=1)
    with f3:
        q_gsm = st.number_input("GSM", min_value=0, value=0, step=1)
    with f4:
        q_deckle = st.number_input("Deckle (cm)", min_value=0.0, value=0.0, step=0.1)

    df_f = calc_df.copy()

    if q_reel:
        df_f = df_f[df_f["Reel No"].str.contains(q_reel, case=False, na=False)]
    if q_bf > 0:
        df_f = df_f[pd.to_numeric(df_f["BF"], errors="coerce") == q_bf]
    if q_gsm > 0:
        df_f = df_f[pd.to_numeric(df_f["GSM"], errors="coerce") == q_gsm]
    if q_deckle > 0:
        df_f = df_f[pd.to_numeric(df_f["Deckle in cm"], errors="coerce").round(1) == round(q_deckle, 1)]

    # -------------------------------------------------
    # Display table
    # -------------------------------------------------
    def highlight_reorder(row):
        try:
            # STEP-3C: Highlight negative Opening Stock
            if isinstance(row.index, pd.MultiIndex):
                opening = row[("Stock & Consumption", "Opening Stk Till Date")]
            else:
                opening = row["Opening Stk Till Date"]

            if float(opening) < 0:
                return ["background-color: #ffe4e6"] * len(row)
            # Existing reorder logic
            if isinstance(row.index, pd.MultiIndex):
                closing = row[("Stock & Consumption", "Closing Stock till date")]
                reorder = row[("Stock & Consumption", "Reorder Level")]
            else:
                closing = row["Closing Stock till date"]
                reorder = row["Reorder Level"]
            if float(closing) <= float(reorder):
                return ["background-color: #fff4f2"] * len(row)
        except Exception:
            pass
        return [""] * len(row)

   
    with st.container(border=True):
        st.caption("Tip: Use column filters and the inbuilt download to export.")

        display_df = group_columns_multiindex(df_f) if grouped else df_f

        st.dataframe(
            display_df.style.apply(highlight_reorder, axis=1)
            if len(display_df)
            else display_df,
            use_container_width=True,
            hide_index=True
        )
    # -------------------------------------------------
    # BULK EDIT / DELETE (Paper Reels)
    # -------------------------------------------------
    st.markdown("## ✏️ Bulk Edit / Delete (Paper Reels)")
    st.caption("✔ Edit cells to update values • ✔ Tick checkbox to delete entire reel(s)")
    
    edit_df = build_paper_edit_df(calc_df)

    if edit_df.empty:
        st.info("No records available for bulk edit.")
    else:
        edited = st.data_editor(
            edit_df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            key="paper_bulk_editor",
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    "Delete",
                    help="Tick to delete entire reel row"
                )
            },
            disabled=[
                "Reel No",                      # primary identifier
                "Deckle in Inch",               # computed
                "Opening Stk Till Date",        # system derived
                "Closing Stock till date",      # system derived
                "Reel Supplier",                # master driven (STEP-5)
                "Reel Maker"                    # master driven (STEP-5)
            ]
        )

        b1, b2 = st.columns(2)

        with b1:
            if st.button("💾 Save Selected Changes", type="primary"):
                changed = save_paper_edits(edit_df, edited)
                if changed:
                    st.success(f"{changed} reel(s) updated successfully.")
                    st.rerun()
                else:
                    st.info("No changes detected.")

        # ----------------------------
        # DELETE ROWS
        # ----------------------------

        with b2:
            to_delete = (edited.loc[edited["Select"] == True, "Reel No"]
                         .dropna()
                         .tolist()
                        )

            if st.button("🗑 Delete Selected Reels", type="secondary"):
                if not to_delete:
                    st.warning("Select at least one reel to delete.")
                else:
                    deleted = delete_paper_rows_by_reel_nos(to_delete)
                    st.success(f"{deleted} reel(s) deleted.")
                    st.rerun()

    st.markdown("---")

    # -------------------------------------------------
    # RECEIVE / ISSUE / TRANSFER (GOES LAST)
    # -------------------------------------------------
    tabs = st.tabs(["📥 Receive", "📤 Issue", "🔁 Transfer / Adjust"])

    with tabs[0]:
        rm_receive_form()

    with tabs[1]:
        rm_issue_form()

    with tabs[2]:
        rm_transfer_adjust_form()

    st.markdown("---")
    st.markdown("## ⬆ Bulk Import / Export")
    # -------------------------------------------------
    # EXCEL UPLOAD (RESTORED – WITH VALIDATIONS)
    # -------------------------------------------------
    rm_upload_excel_ui()


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
        st.info("No pallets found. Use Pack/Putaway first."); return
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
            else:
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
    with tabs[0]:
        fg_pack_form()
    with tabs[1]:
        fg_reserve_dispatch()
    with tabs[2]:
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
        cust = st.text_input("Name"); gstin = st.text_input("GSTIN", value="")
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
        sku = st.text_input("SKU Code"); sdesc = st.text_input("Description", value="")
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

    # =========================
    # Reel Supplier Master
    # =========================
    st.subheader("🧾 Reel Supplier (Master)")

    new_supplier = st.text_input(
        "Supplier Name",
        key="settings_new_supplier"
    )

    if st.button("Add Supplier", key="btn_add_supplier"):
        if new_supplier.strip():
            with get_conn() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO Supplier(Name) VALUES(?)",
                    (new_supplier.strip(),)
                )
            st.success("Supplier added successfully.")
        else:
            st.warning("Supplier name cannot be empty.")

    st.markdown("---")

    # =========================
    # Reel Maker Master
    # =========================
    st.subheader("🏭 Reel Maker (Master)")

    new_maker = st.text_input(
        "Maker Name",
        key="settings_new_maker"
    )

    if st.button("Add Maker", key="btn_add_maker"):
        if new_maker.strip():
            with get_conn() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO Maker(Name) VALUES(?)",
                    (new_maker.strip(),)
                )
            st.success("Maker added successfully.")
        else:
            st.warning("Maker name cannot be empty.")

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
