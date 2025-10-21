import os
import sys
from pathlib import Path
import configparser
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

from urllib.parse import quote_plus

import json
from sqlalchemy import text
from decimal import Decimal
import numpy as np
from datetime import date

# 前序查詢邏輯模組
from prev_process import fetch_prev_process_and_upstream
from material_prep import get_material_production_progress_list_of_kanban

# from call_llm_api import analyze_overdue_with_llm
from call_llm_api import analyze_overdue_with_llm

# RAG（長期改善）模組：若尚未建立檔案，提供安全降級函式
try:
    from call_rag_api import get_long_term_improvements
except Exception:
    def get_long_term_improvements(*args, **kwargs):
        return {"long_term_actions": [], "raw": "", "error": "call_rag_api module not found"}
# --- JSON sanitization helper ---
def _json_safe(obj):
    """Recursively convert objects to JSON-serializable types (Decimal, numpy, pandas Timestamps, sets, etc.)."""
    try:
        # pandas/numpy aware NaN
        import pandas as pd  # local reference
    except Exception:
        pd = None

    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_json_safe(v) for v in obj)
    if isinstance(obj, set):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Decimal):
        try:
            return float(obj)
        except Exception:
            return str(obj)
    if pd is not None and isinstance(obj, pd.Timestamp):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    if isinstance(obj, (datetime, date)):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    # map pandas NA/NaT to None
    try:
        import math
        if obj is None:
            return None
        # handle pandas NA without importing pandas types explicitly
        if hasattr(obj, "_isnan") and obj._isnan():
            return None
        if isinstance(obj, float) and math.isnan(obj):
            return None
    except Exception:
        pass
    return obj

# ----------------------------
# 1) 讀取 config.ini
# ----------------------------
def load_config():
    """
    讀取 config.ini，優先順序：
    1) 環境變數 CONFIG_PATH 指定的路徑
    2) 與此檔同層的 ./config.ini
    3) 專案根目錄或習慣的 fallback（可自行加）
    """
    cfg_path = os.getenv("CONFIG_PATH")
    if cfg_path:
        cfg = Path(cfg_path)
    else:
        # 預設找跟此檔同一層的 config.ini
        cfg = Path(__file__).resolve().parent / "config.ini"

    if not cfg.exists():
        st.error(f"找不到設定檔：{cfg}")
        st.stop()

    cp = configparser.ConfigParser()
    return cp, cfg

cp, cfg = load_config()

def build_db_url(db_conf: configparser.SectionProxy) -> str:
    """
    由 config.ini 生成 SQLAlchemy 連線字串
    只針對 MySQL/MariaDB + PyMySQL（driver=mysql+pymysql）
    """
    driver   = db_conf.get("driver", "mysql+pymysql")
    host     = db_conf.get("host", "127.0.0.1").strip()
    port_raw = db_conf.get("port", "3306")
    user     = quote_plus(db_conf.get("user", ""))       # 使用者名稱做 URL 編碼
    password = quote_plus(db_conf.get("password", ""))   # 密碼做 URL 編碼（處理特殊字元）
    database = db_conf.get("database", "")
    charset  = db_conf.get("charset", "utf8mb4")
    query    = db_conf.get("query", "")  # 可填 ssl=..., ssl_disabled=... 等

    # host 不可包含 SSH 用法的 '@'（防止把 'user@host' 誤放在 host 欄位）
    if "@" in host:
        st.error("config.ini 的 host 不可包含 '@'。請把 SSH 帳號放在 SSH 指令，不要放在 host（例：ssh -L... your_ssh_user@host）。")
        st.stop()

    # 轉型 port（若填了非數字，交給 SQLAlchemy 也能處理；這裡盡量轉 int）
    try:
        port = int(port_raw)
    except Exception:
        port = port_raw  # 保留原字串

    base = f"{driver}://{user}:{password}@{host}:{port}/{database}?charset={charset}"
    if query:
        # 移除前導 ? 或 &，避免重覆
        query = query.lstrip("?&")
        base = f"{base}&{query}"
    return base

def get_db_config(cp, cfg):
    """
    """
    cp.read(cfg, encoding="utf-8")
    if "database" not in cp:
        st.error("config.ini 缺少 [database] 區段")
        st.stop()
    return cp["database"]

def get_db_leanplay_supp_config(cp, cfg):
    """
    """
    cp.read(cfg, encoding="utf-8")
    if "database_leanplay_supp" not in cp:
        st.error("config.ini 缺少 [database_leanplay_supp] 區段")
        st.stop()
    return cp["database_leanplay_supp"]
# ----------------------------
# 2) Streamlit 頁面設定
# ----------------------------
st.set_page_config(page_title="今日生產看板監控", layout="wide")
st.title("今日生產看板監控")

db_conf = get_db_config(cp, cfg)
DB_URL = build_db_url(db_conf)
timezone_name = db_conf.get("timezone", "DB預設")

# 使用者可於 config.ini 設定 timezone，如未設定則預設使用 Asia/Taipei 作為顯示時區
display_tz = timezone_name if (timezone_name and timezone_name not in ("DB預設", "")) else "Asia/Taipei"

# 建立 Engine（使用連線存活檢查）
engine = create_engine(DB_URL, pool_pre_ping=True)

# 顯示時間（UI 顯示用途）
timezone_name = db_conf.get("timezone", "DB預設")

try:
    if display_tz:
        now_local = datetime.now(ZoneInfo(display_tz))
    else:
        now_local = datetime.now()
except Exception:
    # 固定偏移（無 DST 規則）
    if display_tz == "Asia/Taipei":
        now_local = datetime.now(timezone(timedelta(hours=8)))
    else:
        now_local = datetime.now()

# ----------------------------
# 4) 資料讀取（使用 View）
# ----------------------------
@st.cache_data(ttl=60)
def load_today_views():
    with engine.connect() as conn:
        summary = pd.read_sql("SELECT * FROM v_today_kanban_planned_vs_actual", conn)
        detail  = pd.read_sql("SELECT * FROM v_today_kanban_start_detail", conn)
    return summary, detail


try:
    summary_df, detail_df = load_today_views()
except Exception as e:
    st.error(f"讀取資料失敗：{e}")
    st.stop()

# ----------------------------
# 將明細的時間欄位轉為顯示時區（假設 DB 儲存為 UTC）
# ----------------------------
time_cols = [
    "expected_start_time",
    "actual_start_time",
    "actual_end_time",
]

for col in time_cols:
    if col in detail_df.columns:
        s = pd.to_datetime(detail_df[col], errors="coerce")
        try:
            # 若為 naive 時間，先當作 UTC，再轉成 display_tz
            if s.dt.tz is None:
                s = s.dt.tz_localize("UTC").dt.tz_convert(display_tz)
            else:
                s = s.dt.tz_convert(display_tz)
            detail_df[col] = s.dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            # 轉換失敗時，保留為字串以避免中斷
            detail_df[col] = s.astype(str)

# ----------------------------
# 資料更新時間（以 fact_production_job 的 MAX(load_dts) 為準）
# ----------------------------
asof_str = None
try:
    with engine.connect() as conn:
        asof_df = pd.read_sql("SELECT MAX(load_dts) AS asof FROM fact_production_job", conn)
        if not asof_df.empty and pd.notna(asof_df.loc[0, "asof"]):
            asof_val = pd.to_datetime(asof_df.loc[0, "asof"])
            # load_dts 為 UTC，再轉 display_tz（避免 DB/連線時區不一致）
            try:
                if getattr(asof_val, "tzinfo", None) is None:
                    asof_val = asof_val.tz_localize("UTC").tz_convert(display_tz)
                else:
                    asof_val = asof_val.tz_convert(display_tz)
            except Exception:
                # 退回：直接字串化（仍附上統一的時區標籤）
                asof_str = f"{asof_val.strftime('%Y-%m-%d %H:%M:%S') if hasattr(asof_val, 'strftime') else str(asof_val)} ({display_tz})"
            else:
                # 統一用文字標示 "(display_tz)"，避免顯示為 CST
                asof_str = f"{asof_val.strftime('%Y-%m-%d %H:%M:%S')} ({display_tz})"
except Exception:
    asof_str = None

if asof_str:
    st.caption(f"資料更新時間：{asof_str}；頁面查詢 {now_local:%Y-%m-%d %H:%M:%S} ({display_tz})")
else:
    st.caption(f"資料更新時間：頁面查詢 {now_local:%Y-%m-%d %H:%M:%S} ({display_tz})")

# ----------------------------
# 3) 篩選器（前端過濾）
# ----------------------------

# ----------------------------
# 3.1) 根據明細建立選項並渲染一次（加入工單篩選）
# ----------------------------
wc_opts   = sorted(detail_df["work_center_id"].dropna().unique().tolist()) if "work_center_id" in detail_df else []
proc_opts = sorted(detail_df["process_id"].dropna().unique().tolist()) if "process_id" in detail_df else []
part_opts = sorted(detail_df["part_no"].dropna().unique().tolist()) if "part_no" in detail_df else []
wo_opts   = sorted(detail_df["work_order_id"].dropna().unique().tolist()) if "work_order_id" in detail_df else []

frow = st.columns([1,1,1,1,1])
with frow[0]:
    filter_wc = st.multiselect("工作站 Work Center", options=wc_opts, default=[])
with frow[1]:
    filter_proc = st.multiselect("製程 Process", options=proc_opts, default=[])
with frow[2]:
    filter_part = st.multiselect("料號 Part", options=part_opts, default=[])
with frow[3]:
    filter_wo = st.multiselect("工單 Work Order", options=wo_opts, default=[])
with frow[4]:
    only_overdue = st.toggle("只看逾期未開工", value=False)

# ----------------------------
# 5) KPI 卡片
# ----------------------------
if not summary_df.empty:
    s = summary_df.iloc[0]
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    with kpi1:
        st.metric(
            label="今日應開工數 (看板)",
            value=int(s.get("planned_kanbans_today", 0)),
            help="今日計畫要開工的看板數量（以 expected_start_time 落在今日 00:00–24:00 為準）。"
        )
    with kpi2:
        st.metric(
            label="今日實際已開工 (看板)",
            value=int(s.get("actual_started_kanbans_today", 0)),
            help="今日實際發生開工的看板數量（以 actual_start_time 落在今日 00:00–24:00 為準，不論原計畫日）。"
        )
    with kpi3:
        st.metric(
            label="準時開工",
            value=int(s.get("on_time_starts_today", 0)),
            help="同時滿足：計畫今日開工（expected_start_time 在今日）且今日實際開工，並且 actual_start_time ≤ expected_start_time。"
        )
    with kpi4:
        st.metric(
            label="延遲開工",
            value=int(s.get("late_starts_today", 0)),
            help="同時滿足：計畫今日開工（expected_start_time 在今日）且今日實際開工，但 actual_start_time ＞ expected_start_time。"
        )
    with kpi5:
        st.metric(
            label="逾期未開工",
            value=int(s.get("overdue_not_started_so_far", 0)),
            help="截至目前時間，計畫時間已到（expected_start_time ≤ 現在）但尚未實際開工（actual_start_time 為空）的看板數。"
        )
    with kpi6:
        st.metric(
            label="提前完工（計畫今日）",
            value=int(s.get("planned_today_but_finished_before_today", 0)),
            help="計畫今日才開工，但實際完工時間早於今日（actual_end_time ＜ 今日 00:00），代表任務已在先前日子完成。"
        )

    planned = int(s.get("planned_kanbans_today", 0))
    actual  = int(s.get("actual_started_kanbans_today", 0))
    rate = (actual/planned) if planned else 0
    st.progress(min(max(rate, 0), 1.0), text=f"今日開工達成率：{rate:.0%}")
    st.caption("**達成率計算方式**：今日開工達成率 = 今日實際已開工(看板) ÷ 今日應開工數(看板)。當分母為 0 時，顯示為 0%。")

st.divider()

# ----------------------------
# 6) 明細表 + 高亮 + 下載（分頁 + 全量匯出）
# ----------------------------
df = detail_df.copy()

# 前端篩選
if filter_wc:
    df = df[df["work_center_id"].isin(filter_wc)]
if filter_proc:
    df = df[df["process_id"].isin(filter_proc)]
if filter_part:
    df = df[df["part_no"].isin(filter_part)]
if filter_wo:
    df = df[df["work_order_id"].isin(filter_wo)]
if only_overdue and "overdue_not_started_so_far_flag" in df:
    df = df[df["overdue_not_started_so_far_flag"] == 1]

# 欄位改為繁體中文（僅對已存在欄位進行重命名）
zh_map = {
    "kanban_id": "看板ID",
    "work_order_id": "工單號",
    "part_no": "料號",
    "work_center_id": "工作站",
    "process_id": "製程",
    "process_seq": "製程序號",
    "expected_start_time": "計畫開工時間",
    "actual_start_time": "實際開工時間",
    "produce_status": "生產狀態",
    "planned_today_flag": "今日計畫",
    "started_today_flag": "今日實際開工",
    "overdue_not_started_so_far_flag": "逾期未開工",
    "started_on_time_today_flag": "準時開工",
    "started_late_today_flag": "延遲開工"
}
display_cols_order = [
    "kanban_id","work_order_id","part_no","work_center_id",
    "process_id","process_seq","expected_start_time","actual_start_time",
    "planned_today_flag","started_today_flag","started_on_time_today_flag",
    "started_late_today_flag","overdue_not_started_so_far_flag","produce_status"
]

# 排序欄位存在才排序（穩定且一致）
sort_cols = [c for c in ["work_center_id", "process_id", "process_seq", "expected_start_time"] if c in df.columns]
df_sorted = df.sort_values(sort_cols, na_position="last") if sort_cols else df

st.subheader("今日看板明細")
st.caption("顏色：紅=逾期未開工；綠=準時；黃=延遲")


# ---------------- 分頁狀態（移到表格上方計算，控制在表格下方）----------------
# 以 session_state 記住頁碼與每頁筆數（避免換頁回跳）
if "detail_page_size" not in st.session_state:
    st.session_state.detail_page_size = 50
if "detail_page" not in st.session_state:
    st.session_state.detail_page = 1

page_size = int(st.session_state.detail_page_size)

total_rows  = len(df_sorted)
total_pages = max((total_rows - 1) // page_size + 1, 1)

# 保險：頁碼落在有效範圍
page = int(st.session_state.detail_page)
if page < 1:
    page = 1
elif page > total_pages:
    page = total_pages

start = (page - 1) * page_size
end   = min(start + page_size, total_rows)

# 取本頁資料
display_cols = [c for c in display_cols_order if c in df_sorted.columns] or list(df_sorted.columns)
df_page = df_sorted.iloc[start:end][display_cols]
df_display = df_page.rename(columns={k: v for k, v in zh_map.items() if k in display_cols})

# 高亮規則（沿用原邏輯）
def highlight_row(row):
    """
    根據旗標欄位上色。支援英/中文欄位名稱：
      - overdue_not_started_so_far_flag / 逾期未開工  -> 紅
      - started_on_time_today_flag       / 準時開工    -> 綠
      - started_late_today_flag          / 延遲開工    -> 黃
    """
    def get_flag(r, en, zh):
        val = r.get(en, r.get(zh, 0))
        try:
            return int(val) if val is not None else 0
        except Exception:
            return 1 if str(val).strip() in ("1", "True", "true", "Y") else 0
    
    overdue = get_flag(row, "overdue_not_started_so_far_flag", "逾期未開工")
    ontime  = get_flag(row, "started_on_time_today_flag", "準時開工")
    late    = get_flag(row, "started_late_today_flag", "延遲開工")

    if overdue == 1:
        return ["background-color: #ffe5e5"] * len(row)   # 逾期：紅
    if ontime == 1:
        return ["background-color: #eaffe5"] * len(row)   # 準時：綠
    if late == 1:
        return ["background-color: #fff6e0"] * len(row)   # 延遲：黃
    return [""] * len(row)

# 表格內單列選取觸發（single-row）。選到後會 rerun，並在下方產生分析 payload。
selection = st.dataframe(
    df_display.style.apply(highlight_row, axis=1),
    width='stretch',
    height=520,
    on_select="rerun",
    selection_mode="single-row",
    key="detail_table"
)

# 使用說明
st.caption("💡 在表格內點選任一列即可分析；目前僅針對『逾期未開工』的列產生分析。")

# 取得選取結果：支援 st.dataframe 直接回傳或經由 session_state 取得
sel_rows = []
try:
    # 1) 直接回傳（某些版本直接回 dict）
    if isinstance(selection, dict):
        if "rows" in selection and isinstance(selection["rows"], list):
            sel_rows = selection["rows"]
        elif "selection" in selection and isinstance(selection["selection"], dict):
            sel_rows = selection["selection"].get("rows", []) or []
    # 2) 由 session_state 取得（較新版本會把 selection 放在 key 下）
    if not sel_rows:
        tbl_state = st.session_state.get("detail_table", {})
        if isinstance(tbl_state, dict):
            if "rows" in tbl_state and isinstance(tbl_state["rows"], list):
                sel_rows = tbl_state["rows"]
            elif "selection" in tbl_state and isinstance(tbl_state["selection"], dict):
                sel_rows = tbl_state["selection"].get("rows", []) or []
except Exception:
    sel_rows = []

# 若本次沒有選取任何列，清空暫存 payload（避免誤用舊資料）
if not sel_rows:
    st.session_state.pop("current_payload", None)
    st.session_state.pop("current_row_key", None)

# ---------------- 分頁控制（搬到表格下方）----------------
ctrl = st.container()
with ctrl:
    c = st.columns([1.6, 2.4, 3.0])
    with c[0]:
        new_size = st.selectbox("每頁筆數", options=[50, 100, 200], index=[50,100,200].index(page_size), help="調整每頁顯示的列數。")
        if new_size != page_size:
            st.session_state.detail_page_size = int(new_size)
            st.session_state.detail_page = 1  # 換每頁數回到第一頁
            st.rerun()
    with c[1]:
        n1, n2, n3, n4 = st.columns([1,1,1,1])
        if n1.button("⏮ 回到第一頁", disabled=(page <= 1)):
            st.session_state.detail_page = 1
            st.rerun()
        if n2.button("« 上一頁", disabled=(page <= 1)):
            st.session_state.detail_page = max(1, page - 1)
            st.rerun()
        if n3.button("下一頁 »", disabled=(page >= total_pages)):
            st.session_state.detail_page = min(total_pages, page + 1)
            st.rerun()
        if n4.button("跳到最後頁 ⏭", disabled=(page >= total_pages)):
            st.session_state.detail_page = total_pages
            st.rerun()
    with c[2]:
        st.markdown(f"**第 {page} / {total_pages} 頁**　顯示第 **{start+1}–{end}** 筆（共 **{total_rows}** 筆）")

# 全量 CSV 匯出（不受分頁影響）
st.download_button(
    label="下載明細 CSV（全量）",
    data=df_sorted.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"today_kanban_detail_{now_local:%Y%m%d_%H%M}.csv",
    mime="text/csv",
    help="匯出當前篩選後的所有列（不受分頁影響）。"
)

if sel_rows:
    sel_idx = sel_rows[0]
    # df_display 與 df_page 的順序一致，sel_idx 對應 df_page 的相對位置
    if 0 <= sel_idx < len(df_page):
        row = df_page.iloc[sel_idx]
        # 若選取了與上次不同的列，清除上一筆 AI 分析結果，避免殘留
        new_key = f"{row.get('work_order_id','')}-{row.get('kanban_id','')}"
        prev_key = st.session_state.get("current_row_key")
        if new_key != prev_key:
            st.session_state.pop("last_ai_result", None)
            st.session_state.pop("last_ai_key", None)
        # 僅針對逾期列
        try:
            overdue_flag = int(row.get("overdue_not_started_so_far_flag", 0))
        except Exception:
            overdue_flag = 1 if str(row.get("overdue_not_started_so_far_flag", "")).strip() in ("1","True","true","Y") else 0

        if overdue_flag != 1:
            st.info("僅針對『逾期未開工』的列提供分析。請選取紅色高亮列。")
        else:
            with engine.connect() as conn:
                upstream = fetch_prev_process_and_upstream(conn, row)

            # 將 upstream 的時間欄位（若為 UTC 或 naive）轉為顯示時區（例如 Asia/Taipei）
            try:
                for tcol in ("actual_start_time", "actual_end_time"):
                    ts = upstream.get(tcol)
                    if ts is None or ts == "":
                        continue
                    s = pd.to_datetime(ts, errors="coerce")
                    if pd.isna(s):
                        continue
                    # 若為 naive，視為 UTC；若已有 tz，直接轉顯示時區
                    if getattr(s, "tzinfo", None) is None:
                        s = s.tz_localize("UTC").tz_convert(display_tz)
                    else:
                        s = s.tz_convert(display_tz)
                    upstream[tcol] = s.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass

            # 連線資料庫（supp）取得物料備料進度列表
            material_prep_list = []
            kanban_id_val = row.get("kanban_id")
            if kanban_id_val:
                db_lps_conf = get_db_leanplay_supp_config(cp, cfg)
                lps_DB_URL = build_db_url(db_lps_conf)
                lps_engine = create_engine(lps_DB_URL, pool_pre_ping=True)
                with lps_engine.connect() as supp_conn:
                    material_prep_list = get_material_production_progress_list_of_kanban(supp_conn, kanban_id_val)

            # 將物料清單的 load_dts 視為 UTC，轉為顯示時區（例如 Asia/Taipei）以便前端顯示
            if material_prep_list:
                for _item in material_prep_list:
                    ts = _item.get("load_dts")
                    if ts is None or ts == "":
                        continue
                    try:
                        s = pd.to_datetime(ts, errors="coerce")
                        if pd.isna(s):
                            continue
                        # 若為 naive，先視為 UTC，再轉 display_tz；若已有 tz，直接轉 display_tz
                        if getattr(s, "tzinfo", None) is None:
                            s = s.tz_localize("UTC").tz_convert(display_tz)
                        else:
                            s = s.tz_convert(display_tz)
                        _item["load_dts"] = s.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        # 任何轉換失敗則維持原值
                        pass
            
            payload = {
                "as_of": asof_str,
                "day_start": f"{now_local.strftime('%Y-%m-%d')} 00:00:00 ({display_tz})",
                "day_end":   f"{now_local.strftime('%Y-%m-%d')} 24:00:00 ({display_tz})",
                "task": {
                    "kanban_id": row.get("kanban_id"),
                    "work_order_id": row.get("work_order_id"),
                    "part_no": row.get("part_no"),
                    "work_center_id": row.get("work_center_id"),
                    "process_id": row.get("process_id"),
                    "process_seq": int(row.get("process_seq",0)),
                    "expected_start_time": str(row.get("expected_start_time")),
                    "actual_start_time": str(row.get("actual_start_time")),
                    "produce_status": int(row.get("produce_status",0)) if pd.notna(row.get("produce_status")) else None,
                    "flags": {
                        "planned_today_flag": int(row.get("planned_today_flag",0)),
                        "overdue_not_started_so_far_flag": int(row.get("overdue_not_started_so_far_flag",0)),
                        "started_on_time_today_flag": int(row.get("started_on_time_today_flag",0)),
                        "started_late_today_flag": int(row.get("started_late_today_flag",0))
                    }
                },
                "context": {
                    "upstream_status": dict(upstream) if upstream else None,
                    "materials_prep_status": material_prep_list
                }
            }
            st.success(f"已組成 AI 分析 payload：{row.get('work_order_id','')}-{row.get('kanban_id','')}，請至下方 **Payload** 分頁查看或下載。")
            # 將本次選取的 payload/row key 存入 session，供下方 Tabs 使用
            st.session_state["current_payload"] = payload
            st.session_state["current_row_key"] = f"{row.get('work_order_id','')}-{row.get('kanban_id','')}"


# --- 送給 AI（示例骨架）---

# 建立結果容器（就在你顯示 payload 的附近）
tabs = st.tabs(["AI 分析", "Payload"])

# 取出暫存 payload/key，供各 Tab 使用
current_payload = st.session_state.get("current_payload")
current_row_key = st.session_state.get("current_row_key", "")

with tabs[0]:
    # 一鍵送出
    send_col1, send_col2 = st.columns([1,4])
    with send_col1:
        run = st.button("🚀 送 AI 分析", type="primary", width='stretch', key="run_ai")
    with send_col2:
        st.caption("點選表格一列後，可在此送出並查看結論。")

    if run and current_payload:
        with st.spinner("AI 分析中…"):
            result = analyze_overdue_with_llm(current_payload, stream=False)
        st.session_state["last_ai_result"] = result
        # 以 session 中的 key（若無，從 payload.task 組）
        key_from_payload = ""
        try:
            t = current_payload.get("task", {}) if isinstance(current_payload, dict) else {}
            key_from_payload = f"{t.get('work_order_id','')}-{t.get('process_id','')}-{t.get('process_seq','')}"
        except Exception:
            key_from_payload = ""
        st.session_state["last_ai_key"] = current_row_key or key_from_payload
    elif run and not current_payload:
        st.warning("請先在表格中選取一列（逾期未開工）以產生分析 Payload。")

    result = st.session_state.get("last_ai_result")
    if result:
        # 如果 LLM 呼叫層回傳了錯誤，先顯示提醒
        if result.get("error"):
            st.warning(f"LLM 呼叫失敗：{result['error']}")

        st.success(f"分析完成（{st.session_state.get('last_ai_key', current_row_key)}）")
        st.markdown(f"### 結論\n{result.get('summary','')}")

        analysis = result.get("analysis", {}) if isinstance(result, dict) else {}
        root_items = analysis.get("root_causes", []) or []

        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown("**疑似原因**")
            bullets = []
            for it in root_items:
                if not isinstance(it, dict):
                    bullets.append(f"* {str(it)}")
                    continue
                title = it.get("title") or "(未命名原因)"
                conf = it.get("confidence")
                if isinstance(conf, (int, float)):
                    title = f"{title}（信心 {float(conf):.0%}）"
                bullets.append(f"* {title}")
                for s in (it.get("signals") or []):
                    bullets.append(f"  * {s}")
            if not bullets:
                bullets = ["（模型未提供原因摘要）"]
            st.write("\n".join(bullets))

        with k2:
            st.markdown("**建議行動**")

            # ---- 短期改善：沿用 LLM analysis 的 follow_up_queries ----
            st.markdown("**短期改善**")
            short_actions = analysis.get("follow_up_queries") or []
            if short_actions:
                st.write("\n".join([f"* {a}" for a in short_actions]))
            else:
                st.caption("（尚無短期建議）")

            # ---- 長期改善：RAG 從精實生產理論補充 ----
            st.markdown("**長期改善（精實生產知識庫）**")
            long_actions: list[str] = []
            rag_err = None

            # 以選取列的唯一鍵當作快取 key，避免每次重算
            rag_key = st.session_state.get("last_ai_key") or st.session_state.get("current_row_key") or ""
            cached_key = st.session_state.get("last_rag_key")
            cached_res = st.session_state.get("last_rag_result")

            if rag_key and cached_key == rag_key and cached_res is not None:
                long_actions = cached_res.get("long_term_actions") or []
                rag_err = cached_res.get("error")
            else:
                # 即時呼叫 RAG（以 LLM analysis 為輸入，可附帶 payload 作為背景）
                with st.spinner("查詢長期改善建議（RAG）…"):
                    rag_res = get_long_term_improvements(analysis, current_payload)
                st.session_state["last_rag_result"] = rag_res
                st.session_state["last_rag_key"] = rag_key
                long_actions = rag_res.get("long_term_actions") or []
                rag_err = rag_res.get("error")

            if rag_err:
                st.warning(f"RAG 取得建議時發生問題：{rag_err}")
            if long_actions:
                st.write("\n".join([f"* {x}" for x in long_actions]))
            else:
                st.caption("（暫無長期建議）")

        with k3:
            st.markdown("**關鍵依據**")
            ev = {
                "upstream_status": current_payload.get("context", {}).get("upstream_status") if current_payload else None,
                "materials_prep_status": current_payload.get("context", {}).get("materials_prep_status") if current_payload else None,
            }
            st.json(ev, expanded=False)

        # 下載鈕（含完整 result）
        st.download_button(
            "下載分析結果（JSON）",
            data=json.dumps(_json_safe(result), ensure_ascii=False, indent=2),
            file_name=f"ai_analysis_{now_local:%Y%m%d_%H%M}.json",
            mime="application/json"
        )


with tabs[1]:
    if current_payload:
        st.code(json.dumps(_json_safe(current_payload), ensure_ascii=False, indent=2), language="json")
    else:
        st.info("尚未產生 Payload。請在上方表格選取一列（逾期未開工）以建立分析資料。")

# # ----------------------------
# # 7) 側邊工具
# # ----------------------------
# with st.sidebar:
#     st.markdown("### 工具")
#     if st.button("重新整理"):
#         st.cache_data.clear()
#         st.rerun()