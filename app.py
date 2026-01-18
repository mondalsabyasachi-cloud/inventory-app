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
