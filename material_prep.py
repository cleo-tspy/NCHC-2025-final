# material_prep.py
from typing import List, Dict, Any
from decimal import Decimal
from sqlalchemy import text

def get_material_production_progress_list_of_kanban(conn, kanbanId: str) -> List[Dict[str, Any]]:
    """
    依據 kanbanId 從資料表 `kanban_material_prep_status` 撈取備料進度
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
            to_kanban_produce
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
        items.append(d)

    return items