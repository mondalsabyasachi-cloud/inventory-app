# --------------------------------------------------
# Paper Reel Repository – DB Write Logic
# --------------------------------------------------

from datetime import date
from core.db import resequence_paper_reels
from core.paper_reel_rules import (
    calc_deckle_inch,
    calc_landed_cost,
    calc_closing_stock,
    calc_stock_value
)

def insert_paper_reel(conn, data: dict):
    """
    Single source of truth for inserting paper reels.
    Used by:
    - UI form
    - Excel upload
    """

    # Phase-1 rule:
    # At RECEIVE time, opening stock = original reel weight
    # No consumption is applied here
    opening_kg = data.get("original_weight")

    if opening_kg is None or opening_kg <= 0:
        raise ValueError("Original Reel Weight must be greater than zero")

    consumed_kg = 0  # consumption will be recorded via RM_Movement only

    deckle_inch = calc_deckle_inch(data.get("deckle_cm"))
    landed_cost = calc_landed_cost(
        data.get("paper_rate", 0),
        data.get("transport_rate", 0)
    )
    closing_kg = opening_kg
    stock_value = calc_stock_value(closing_kg, landed_cost)

    cur = conn.cursor()

    cur.execute("""
        INSERT INTO paper_reels (
            sl_no,
            reel_no,
            supplier,
            maker,
            material_rcv_date,
            supplier_invoice_date,
            deckle_cm,
            deckle_inch,
            gsm,
            bf,
            paper_shade,
            original_weight,
            consumed_wt,
            consume_date,
            consumption_entry_date,
            closing_stock,
            reel_location,
            target_sku,
            target_customer,
            reel_shift_date,
            delivery_challan,
            reorder_level,
            paper_rate,
            transport_rate,
            landed_cost,
            current_stock_value,
            holding_days,
            remarks
        )
        VALUES (
            NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
        )
    """, (
        data.get("reel_no"),
        data.get("supplier"),
        data.get("maker"),
        data.get("material_rcv_date"),
        data.get("supplier_invoice_date"),
        data.get("deckle_cm"),
        deckle_inch,
        data.get("gsm"),
        data.get("bf"),
        data.get("paper_shade"),
        opening_kg,
        consumed_kg,
        data.get("consume_date"),
        date.today(),
        closing_kg,
        data.get("reel_location"),
        data.get("target_sku"),
        data.get("target_customer"),
        data.get("reel_shift_date"),
        data.get("delivery_challan"),
        data.get("reorder_level"),
        data.get("paper_rate"),
        data.get("transport_rate"),
        landed_cost,
        stock_value,
        data.get("holding_days"),
        data.get("remarks")
    ))

    resequence_paper_reels(conn)
