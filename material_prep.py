# material_prep.py
from typing import List, Dict, Any
from decimal import Decimal
from sqlalchemy import text
from datetime import datetime

# ----------------------------
# 取得看板所需物料備料進度列表（從資料庫查）
# ----------------------------
def get_material_production_progress_list_of_kanban(conn, kanbanId: str) -> List[Dict[str, Any]]:
    """
    依據 to_kanban_produce = :kanbanId 從資料表 `kanban_material_prep_status` 撈取備料進度。

    參數：
        conn: SQLAlchemy Connection（指向 database_leanplay_supp）
        kanbanId: 生產看板 ID（對應欄位 to_kanban_produce）

    回傳：
        list[dict]，欄位包含：
        material_kanban_id, part_no, part_name, finish_qty(float), request_qty(float),
        supplier_name, to_kanban_produce, load_dts(ISO 字串)
    """
    if not kanbanId:
        print(kanbanId)
        return []

    sql = text("""
        SELECT
            material_kanban_id,
            part_no,
            part_name,
            finish_qty,
            request_qty,
            supplier_name,
            to_kanban_produce,
            load_dts
        FROM kanban_material_prep_status
        WHERE to_kanban_produce = :kanbanId
        ORDER BY part_no, material_kanban_id
    """)

    try:
        result = conn.execute(sql, {"kanbanId": kanbanId})
        rows = result.fetchall()
    except Exception:
        # 靜默失敗：回空陣列避免中斷畫面（可視需求加 log）
        return []

    items: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r._mapping) if hasattr(r, "_mapping") else dict(r)
        # Decimal -> float
        for k in ("finish_qty", "request_qty"):
            v = d.get(k)
            if isinstance(v, Decimal):
                d[k] = float(v)
        # datetime -> 字串（保留 UTC；若要轉台北，請在呼叫端統一處理）
        ts = d.get("load_dts")
        if isinstance(ts, datetime):
            d["load_dts"] = ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        items.append(d)

    return items