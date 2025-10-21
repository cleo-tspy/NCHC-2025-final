#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日 20:30（Asia/Taipei）抓取今日「逾期未開工」看板，
對每筆建立 payload，呼叫 LLM 與 RAG，將結果寫入 JSONL。

依賴：
- SQLAlchemy + PyMySQL
- pandas
- 專案內的 prev_process.py / material_prep.py / call_llm_api.py / call_rag_api.py
- config.ini 需含 [database], [database_leanplay_supp], [rag], [llm] (跟現有一致)

執行：
  $ python scripts/save_daily_overdue_ai.py
"""
from __future__ import annotations
import os, json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List

import pandas as pd
from sqlalchemy import create_engine

# 專案內模組
from prev_process import fetch_prev_process_and_upstream
from material_prep import get_material_production_progress_list_of_kanban
from call_llm_api import analyze_overdue_with_llm
from call_rag_api import get_long_term_improvements

import configparser
from urllib.parse import quote_plus

# ---------- 基礎：讀設定 / 建連線字串 ----------
def load_config():
    cfg_path = os.getenv("CONFIG_PATH")
    cfg = Path(cfg_path) if cfg_path else Path(__file__).resolve().parents[1] / "config.ini"
    if not cfg.exists():
        raise FileNotFoundError(f"找不到設定檔：{cfg}")
    cp = configparser.ConfigParser()
    cp.read(cfg, encoding="utf-8")
    return cp, cfg

def build_db_url(db_conf: configparser.SectionProxy) -> str:
    driver   = db_conf.get("driver", "mysql+pymysql")
    host     = db_conf.get("host", "127.0.0.1").strip()
    port     = db_conf.get("port", "3306").strip()
    user     = quote_plus(db_conf.get("user", ""))
    password = quote_plus(db_conf.get("password", ""))
    database = db_conf.get("database", "")
    charset  = db_conf.get("charset", "utf8mb4")
    query    = db_conf.get("query", "").lstrip("?&")
    base = f"{driver}://{user}:{password}@{host}:{port}/{database}?charset={charset}"
    if query:
        base = f"{base}&{query}"
    return base

# ---------- JSON 安全序列化 ----------
from decimal import Decimal
import numpy as np
from datetime import date, time

def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ _json_safe(v) for v in obj ]
    if isinstance(obj, set):
        return [ _json_safe(v) for v in obj ]
    if isinstance(obj, Decimal):
        try: return float(obj)
        except: return str(obj)
    if isinstance(obj, pd.Timestamp):
        try: return obj.isoformat()
        except: return str(obj)
    if isinstance(obj, (datetime, date, time)):
        try: return obj.isoformat()
        except: return str(obj)
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    return obj

# ---------- 主流程 ----------
def main():
    tz = ZoneInfo("Asia/Taipei")
    now_local = datetime.now(tz)
    ymd = now_local.strftime("%Y%m%d")

    # 準備輸出資料夾
    out_dir = Path(__file__).resolve().parents[1] / "data" / "daily" / ymd
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"overdue_ai_{ymd}.jsonl"

    cp, _ = load_config()

    # DB: 主資料庫（讀 v_today_kanban_start_detail）
    db_conf = cp["database"]
    engine = create_engine(build_db_url(db_conf), pool_pre_ping=True)

    # DB: 物料備料（supp）
    supp_conf = cp["database_leanplay_supp"]
    supp_engine = create_engine(build_db_url(supp_conf), pool_pre_ping=True)

    # 拉「今日逾期未開工」：直接用 view 的旗標
    sql = """
    SELECT *
    FROM v_today_kanban_start_detail
    WHERE planned_today_flag = 1
      AND overdue_not_started_so_far_flag = 1
    ORDER BY work_center_id, process_id, process_seq, expected_start_time
    """
    df = pd.read_sql(sql, engine)

    n = 0
    with open(out_jsonl, "a", encoding="utf-8") as fw:
        for _, row in df.iterrows():
            try:
                # 取得上游與物料
                with engine.connect() as conn:
                    upstream = fetch_prev_process_and_upstream(conn, row)
                mats = []
                kanban_id = row.get("kanban_id")
                if kanban_id:
                    with supp_engine.connect() as sconn:
                        mats = get_material_production_progress_list_of_kanban(sconn, kanban_id)

                # 時間顯示轉台北（保留字串）
                for tcol in ("actual_start_time","actual_end_time"):
                    ts = (upstream or {}).get(tcol)
                    if ts:
                        s = pd.to_datetime(ts, errors="coerce")
                        if s is not pd.NaT:
                            if getattr(s, "tzinfo", None) is None:
                                s = s.tz_localize("UTC").tz_convert(tz)
                            else:
                                s = s.tz_convert(tz)
                            upstream[tcol] = s.strftime("%Y-%m-%d %H:%M:%S")

                for it in mats or []:
                    ts = it.get("load_dts")
                    if ts:
                        s = pd.to_datetime(ts, errors="coerce")
                        if s is not pd.NaT:
                            if getattr(s, "tzinfo", None) is None:
                                s = s.tz_localize("UTC").tz_convert(tz)
                            else:
                                s = s.tz_convert(tz)
                            it["load_dts"] = s.strftime("%Y-%m-%d %H:%M:%S")

                # 組 payload
                payload = {
                    "as_of": now_local.strftime("%Y-%m-%d %H:%M:%S (Asia/Taipei)"),
                    "day_start": f"{now_local:%Y-%m-%d} 00:00:00 (Asia/Taipei)",
                    "day_end":   f"{now_local:%Y-%m-%d} 24:00:00 (Asia/Taipei)",
                    "task": {
                        "kanban_id": row.get("kanban_id"),
                        "work_order_id": row.get("work_order_id"),
                        "part_no": row.get("part_no"),
                        "work_center_id": row.get("work_center_id"),
                        "process_id": row.get("process_id"),
                        "process_seq": int(row.get("process_seq") or 0),
                        "expected_start_time": str(row.get("expected_start_time")),
                        "actual_start_time": str(row.get("actual_start_time")),
                        "produce_status": int(row.get("produce_status") or 0) if pd.notna(row.get("produce_status")) else None,
                        "flags": {
                            "planned_today_flag": int(row.get("planned_today_flag") or 0),
                            "overdue_not_started_so_far_flag": int(row.get("overdue_not_started_so_far_flag") or 0),
                            "started_on_time_today_flag": int(row.get("started_on_time_today_flag") or 0),
                            "started_late_today_flag": int(row.get("started_late_today_flag") or 0),
                        }
                    },
                    "context": {
                        "upstream_status": dict(upstream) if upstream else None,
                        "materials_prep_status": mats or [],
                    }
                }

                # 呼叫 LLM 與 RAG
                llm_res = analyze_overdue_with_llm(payload, stream=False)
                rag_res = get_long_term_improvements(llm_res.get("analysis") or {}, payload)

                record = {
                    "generated_at": now_local.strftime("%Y-%m-%d %H:%M:%S"),
                    "kanban_id": row.get("kanban_id"),
                    "work_order_id": row.get("work_order_id"),
                    "payload": _json_safe(payload),
                    "llm": _json_safe(llm_res),
                    "rag": _json_safe(rag_res),
                }
                fw.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1

            except Exception as e:
                err = {
                    "generated_at": now_local.strftime("%Y-%m-%d %H:%M:%S"),
                    "kanban_id": row.get("kanban_id"),
                    "work_order_id": row.get("work_order_id"),
                    "error": f"{type(e).__name__}: {e}",
                }
                fw.write(json.dumps(err, ensure_ascii=False) + "\n")

    print(f"[OK] 已寫入：{out_jsonl} ，筆數={n}")

if __name__ == "__main__":
    main()