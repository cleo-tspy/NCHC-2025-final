# prev_process.py
from typing import Tuple, Dict, Any
from sqlalchemy import text

def fetch_prev_process_and_upstream(conn, row) -> Dict[str, Any]:
    """
    依據目前被點擊的列（row）查詢前序製程與其上游在 fact 的狀態。

    輸入：
      - conn: SQLAlchemy Connection（主資料庫連線）
      - row: 來自明細表的一筆資料（dict-like），預期包含：
        `kanban_id`, `work_order_id`, `part_no`, `work_center_id`,
        `process_id`, `process_seq`, `expected_start_time`, `actual_start_time`, `produce_status`

    輸出：
      - upstream: dict，包含上一道製程（前序）對應的最新狀態；查無資料時欄位為 None。

    查詢策略：
      1) 先查 `fact_kanban_prev_process` 對應的前序製程 ID/SEQ；若無，回傳空 prev。
      2) 依 prev 於 `fact_production_job` 查上一道製程的最新狀態（同一張工單、同一製程與序號）。
    """
    # --- 1) 從 row 取必要鍵，並做基本保護 ---
    def _get(d, k, default=None):
        try:
            return d.get(k, default)
        except Exception:
            return default

    work_order_id   = _get(row, "work_order_id")
    kanban_id       = _get(row, "kanban_id")
    cur_process_id  = _get(row, "process_id")
    cur_process_seq = _get(row, "process_seq")

    try:
        if cur_process_seq is not None:
            cur_process_seq = int(cur_process_seq)
    except Exception:
        cur_process_seq = None

    # 預設回傳骨架
    prev = {"prev_process_id": None, "prev_process_seq": None}
    upstream = {
        "work_order_id": work_order_id,
        "kanban_id": kanban_id,
        "process_id": None,
        "process_seq": None,
        "part_no": None,
        "work_center_id": None,
        "actual_start_time": None,
        "actual_end_time": None,
        "produce_status": None,
        "request_qty": None,
        "finish_qty": None,
    }

    if not work_order_id or cur_process_id is None or cur_process_seq is None:
        # 缺關鍵欄位，直接回傳預設
        return upstream

    # --- 2) 查前序製程：fact_kanban_prev_process ---
    try:
        sql_prev = text(
            """
            SELECT prev_process_id, prev_process_seq
            FROM fact_kanban_prev_process
            WHERE work_order_id = :wo
              AND process_seq   = :pseq
            LIMIT 1
            """
        )
        res_prev = conn.execute(sql_prev, {"wo": work_order_id, "pseq": cur_process_seq}).fetchone()
        # 把查到的一筆資料，安全地轉成可用的 dict，然後取出兩個欄位
        if res_prev:
            m = res_prev._mapping if hasattr(res_prev, "_mapping") else res_prev
            prev = {"prev_process_id": m["prev_process_id"], "prev_process_seq": m["prev_process_seq"]}
    except Exception:
        # 查表失敗時，維持預設 prev
        pass

    # 若查不到前序，直接回傳（upstream 保持 None）
    if not prev["prev_process_id"] or prev["prev_process_seq"] is None:
        return upstream

    # --- 3) 查上游在 fact_production_job 的狀態（同工單、前序製程/序號）---
    try:
        sql_up = text(
            """
            SELECT work_order_id,
                   kanban_id,
                   process_id,
                   process_seq,
                   part_no,
                   work_center_id,
                   actual_start_time,
                   actual_end_time,
                   produce_status,
                   COALESCE(request_qty, 0) AS request_qty,
                   COALESCE(finish_qty, 0) AS finish_qty
            FROM fact_production_job
            WHERE work_order_id = :wo
              AND process_id    = :pid
              AND process_seq   = :pseq
            ORDER BY COALESCE(actual_end_time, actual_start_time) DESC, actual_start_time DESC
            LIMIT 1
            """
        )
        res_up = conn.execute(sql_up, {"wo": work_order_id, "pid": prev["prev_process_id"], "pseq": prev["prev_process_seq"]}).fetchone()
        if res_up:
            m = res_up._mapping if hasattr(res_up, "_mapping") else res_up
            upstream = {
                "work_order_id": m.get("work_order_id"),
                "kanban_id":     m.get("kanban_id"),
                "process_id":    m.get("process_id"),
                "process_seq":   m.get("process_seq"),
                "part_no":      m.get("part_no"),
                "work_center_id":m.get("work_center_id"),
                "actual_start_time": m.get("actual_start_time"),
                "actual_end_time":   m.get("actual_end_time"),
                "produce_status":    m.get("produce_status"),
                "request_qty":        m.get("request_qty"),
                "finish_qty":        m.get("finish_qty"),
            }
        else:
            # 保持預設，但把 prev 寫進去，利於前端顯示上下文
            upstream.update({
                "process_id":  prev["prev_process_id"],
                "process_seq": prev["prev_process_seq"],
            })
    except Exception:
        # 若查詢錯誤，保留預設 upstream 並帶入 prev 的 process 資訊
        upstream.update({
            "process_id":  prev["prev_process_id"],
            "process_seq": prev["prev_process_seq"],
        })

    return upstream