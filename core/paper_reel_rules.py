# --------------------------------------------------
# Paper Reel – Business Rules & Column Definitions
# Packsmart India Pvt Ltd
# --------------------------------------------------

# ==================================================
# MASTER DROPDOWNS (TEMPORARY – HARD CODED)
# ==================================================

# -----------------------------
# Reel Supplier (TEMP)
# Later to be populated from:
# Vendor SKU Mapping Table
# -----------------------------
REEL_SUPPLIER_OPTIONS = [
    "Aries Enterprise",
    "LAXMI BOARD & PAPERS",
    "Pranshu Paper Pvt Ltd",
    "KCT Trading Pvt Ltd",
    "PIPL",
    "B. J. Bhandari Papers",
    "Tuljabhawani Paper mills pvt ltd",
    "Om shree Hari Paper Industries",
    "RUDRA PAPERS",
    "UNISOURCE"
]

# -----------------------------
# Reel Maker (TEMP)
# -----------------------------
REEL_MAKER_OPTIONS = [
    "PADMAVATI PULP & PAPER",
    "LAXMI BOARD & PAPERS",
    "Om shree Hari Paper Industries",
    "TULJABHAVANI",
    "WESTROCK",
    "SUKRAFT PAPERS",
    "DBS",
    "KAYGAON PAPER MILLS PVT LTD",
    "Aries Enterprises",
    "SHRIPAPER PRODUCT",
    "ACHIEVE PAPER",
    "Pranshu Paper Pvt Ltd",
    "PIPL",
    "RUDRA PAPERS",
    "AADITYA PAPTECH PVT LTD",
    "Ganga Papers india ltd",
    "Foreign Reel Maker",
    "Sunshine Paper-Tech Pvt Ltd",
    "TULJABHAVANI"
]

# -----------------------------
# Paper Shade Master
# -----------------------------
PAPER_SHADE_OPTIONS = [
    "Duplex",
    "Golden",
    "Imported",
    "Natural",
    "RGS",
    "VTL"
]

# -----------------------------
# Reel Location Master
# -----------------------------
REEL_LOCATION_OPTIONS = [
    "DBS",
    "Atharva",
    "PIPL",
    "Avadhoot"
]

# ==================================================
# COLUMN DISPLAY NAMES
# (Single Source of Truth)
# ==================================================

PAPER_REEL_COLUMNS = {
    "sl_no": "SL No",
    "reel_no": "Reel No",
    "supplier": "Reel Supplier",
    "maker": "Reel Maker",
    "material_rcv_dt": "Material Rcv Dt.",
    "supplier_inv_dt": "Maker's/Supplier's Inv Dt.",
    "deckle_cm": "Deckle(CM)",
    "deckle_inch": "Deckle(Inch)",
    "gsm": "GSM",
    "bf": "BF",
    "paper_shade": "Paper Shade",
    "opening_kg": "Opening Stk Till Date(Kg)",
    "original_weight": "Reel Original Weight(Kg)",
    "consumed_kg": "Consumed Wt(Kg)",
    "consume_dt": "Consume Date",
    "consume_entry_dt": "Consumption Entry Date",
    "closing_kg": "Closing Stock Till Date(Kg)",
    "reel_location": "Reel Location",
    "target_sku": "Target SKU",
    "target_customer": "Target Customer",
    "reel_shift_dt": "Reel Shifting Date",
    "dc_no": "Delivery Challan No.",
    "reorder_level": "Reorder Level(Kg)",
    "paper_rate": "Paper Rate/Kg(₹)",
    "transport_rate": "Transport Rate/Kg(₹)",
    "landed_cost": "Basic Landed Cost/Kg(₹)",
    "stock_value": "Current Stock Value(₹)",
    "holding_days": "Reel Holding Time (Days)",
    "remarks": "Remarks",
}

# ==================================================
# CALCULATION RULES (BUSINESS LOGIC)
# ==================================================

def calc_deckle_inch(deckle_cm: float | None) -> float | None:
    if deckle_cm is None:
        return None
    return round(deckle_cm / 2.54, 3)


def calc_landed_cost(paper_rate: float, transport_rate: float) -> float:
    return round((paper_rate or 0) + (transport_rate or 0), 2)


def calc_closing_stock(opening: float, consumed: float) -> float:
    return round((opening or 0) - (consumed or 0), 2)


def calc_stock_value(closing_kg: float, landed_cost: float) -> float:
    return round((closing_kg or 0) * (landed_cost or 0), 2)

# ==================================================
# FUTURE PLACEHOLDER – VENDOR SKU MAPPING TABLE
# ==================================================

"""
FUTURE TABLE: Vendor_SKU_Mapping

Columns:
- Serial No
- Vendor Name
- Product Description
- Product Name
- Product Business Name
- Part Code
- Old Rate
- Existing Rate

Purpose:
- Populate Reel Supplier dropdown dynamically
- Map supplier → SKU → rates

Example SQL (to be implemented later):

CREATE TABLE Vendor_SKU_Mapping (
    serial_no INTEGER PRIMARY KEY,
    vendor_name TEXT,
    product_description TEXT,
    product_name TEXT,
    product_business_name TEXT,
    part_code TEXT,
    old_rate REAL,
    existing_rate REAL
);

Future Fetch Logic (example):

def fetch_reel_suppliers_from_vendor_table(conn):
    query = '''
        SELECT DISTINCT vendor_name
        FROM Vendor_SKU_Mapping
        ORDER BY vendor_name
    '''
    return [row["vendor_name"] for row in conn.execute(query)]

Once enabled:
- Replace REEL_SUPPLIER_OPTIONS
- Use DB-driven dropdown instead of hard-coded list
"""
