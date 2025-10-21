from __future__ import annotations
import os
import re
import json
import configparser
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from openai import OpenAI
from decimal import Decimal
from datetime import datetime, date, time

# ============================================================
# LLM 呼叫工具
# - 讀取 config.ini 或環境變數
# - 建立可重用的 Client
# - 產出穩健的 Prompt（System/User）
# - 解析模型輸出：摘要 + 嚴格 JSON（帶修復）
# ============================================================

# -------- Default Prompts --------
DEFAULT_SYSTEM_PROMPT = (
    "你是生產製造現場的「運作分析助理」。請嚴格遵守：\n"
    "1) 僅根據「提供的 payload」分析，不得臆測未提供的事實；必要時標註資料不足。\n"
    "2) 採「證據 → 推論 → 結論」邏輯，重點精煉，避免贅述。\n"
    "3) 請輸出兩段：第一段僅有摘要文字（**不要加任何標題或( A )( B )等編號**）；第二段僅輸出以 ```json ... ``` 包住的 analysis JSON。\n"
    "4) analysis JSON 必須可被 json.loads() 解析：使用雙引號；不得包含註解、尾逗號或多餘文字；欄位皆須出現（即便為空陣列）。\n"
    "5) 不要透露你的思考鏈；只輸出結論、證據與可執行建議。\n"
    "6) analysis JSON 的「原因物件」欄位固定為：title（短述）、signals（原因佐證點陣列）、evidence（含 upstream_status / materials_prep_status）、confidence（0~1）。\n"
    "7) 用語規範：以「看板」稱呼任務（例如：看板 <kanban_id> 逾期未開工），不要稱「任務」。在文字描述中，將 finish_qty 稱為「已完成數量」、request_qty 稱為「需求數量」；但在 evidence 中維持原始鍵名不變。\n"
    "8) 站別規則：若 task.process_seq==10 且 context.upstream_status 的製程資訊（process_id/process_seq）為空，表示此看板是製程第一道工序，沒有前工序；此時不要以「上游狀態不明」作為原因，請改以「製程第一道工序，沒有前工序」描述。\n"
    "9) 摘要不得重申「逾期未開工/尚未啟動」等已知事實，直接切入最可能成因與優先動作。\n"
    "10) 判斷規則：若 (製程第一道工序/沒有前工序) 或 (上游已完工) 且 (物料已齊全 或 本站無備料指示 materials_prep_status==[])，才視為「需另查其他因素」；請在摘要與 follow_up_queries 指出應優先確認：設備/治具/換線與保養、產能與人力調度、排程派工、品質封鎖與放行、5S/目視化管理等。嚴禁臆測其為真正原因，僅以「請確認…」方式提示。\n"
)

DEFAULT_USER_PROMPT = (
    "這是一張「逾期開工」的看板，請分析原因與對策。\n\n"
    "【資料】\n"
    "- 報表基準時間（as_of）：{as_of}，日界：{day_start} ~ {day_end}\n"
    "- 任務（看板）資訊（task）：\n{task_json}\n"
    "- 補充上下文（context）：\n{context_json}\n"
    "- 派生判斷（meta）：\n{meta_json}\n\n"
    "【請輸出】\n"
    "第一段：直接輸出 120 字內的中文「摘要 summary」，說明最可能原因與優先動作（不要加標題/編號）。\n"
    "第二段：**僅**輸出一段名為「analysis」的 JSON（analysis JSON），結構如下：\n"
    "```json\n"
    "{{\n"
    "  \"root_causes\": [\n"
    "    {{\n"
    "      \"title\": \"短述主要原因\",\n"
    "      \"signals\": [\"原因細節 1\", \"原因細節 2\"],\n"
    "      \"evidence\": {{\n"
    "        \"upstream_status\": CONTEXT.upstream_status,\n"
    "        \"materials_prep_status\": CONTEXT.materials_prep_status\n"
    "      }},\n"
    "      \"confidence\": 0.0\n"
    "    }}\n"
    "  ],\n"
    "  \"follow_up_queries\": [\"建議追加查詢（SQL/來源）的簡述\"],\n"
    "  \"notes\": \"口徑或判斷界線補充（若有）\"\n"
    "}}\n"
    "```\n\n"
    "注意：\n"
    "- 第一段不要加任何標題或 (A)(B) 等編號；第二段只包含 JSON。\n"
    "- 在文字描述中，將 finish_qty 稱為「已完成數量」、request_qty 稱為「需求數量」；但 evidence 的鍵名不要改。\n"
    "- 若 task.process_seq==10 且 upstream 製程資訊為空，視為無前工序，請依此描述。\n"
)

# -------- Config & Client --------

def _load_llm_conf() -> Dict[str, Any]:
    cfg_path = os.getenv("CONFIG_PATH")
    cfg = Path(cfg_path) if cfg_path else Path(__file__).resolve().parent / "config.ini"
    cp = configparser.ConfigParser()
    cp.read(cfg, encoding="utf-8")

    sec = cp["llm"] if cp.has_section("llm") else {}
    api_key = os.getenv("OPENAI_API_KEY") or sec.get("api_key") or ""
    base_url = os.getenv("OPENAI_BASE_URL") or sec.get("base_url") or "http://000.00.00.00:8083/v1"
    model = os.getenv("LLM_MODEL") or sec.get("model") or "google/gemma-3-4b-it"
    timeout = int(os.getenv("LLM_TIMEOUT") or sec.get("timeout", "60"))
    return {"api_key": api_key, "base_url": base_url, "model": model, "timeout": timeout}


def _get_client(conf: Optional[Dict[str, Any]] = None) -> OpenAI:
    conf = conf or _load_llm_conf()
    return OpenAI(api_key=conf.get("api_key"), base_url=conf.get("base_url"), timeout=conf.get("timeout", 60))


#
# -------- JSON Helper --------

# 將 Decimal / datetime 等轉為可序列化型別
def _json_default(o):
    if isinstance(o, Decimal):
        return float(o)
    if isinstance(o, (datetime, date, time)):
        try:
            return o.isoformat()
        except Exception:
            return str(o)
    if isinstance(o, bytes):
        try:
            return o.decode("utf-8", errors="ignore")
        except Exception:
            return str(o)
    if isinstance(o, set):
        return list(o)
    # 最後手段：可讀字串
    try:
        return str(o)
    except Exception:
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _compute_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
    task = (payload or {}).get("task", {}) or {}
    ctx = (payload or {}).get("context", {}) or {}
    up = ctx.get("upstream_status") or {}
    mats = ctx.get("materials_prep_status")

    # 第一站：process_seq==10 且上游製程資訊為空
    first_station = (task.get("process_seq") == 10 and (up.get("process_id") in (None, "")) and (up.get("process_seq") in (None, "")))

    # 上游已完工：以生產狀態==2 或 有 actual_end_time 判定
    upstream_done = False
    try:
        upstream_done = (up.get("produce_status") == 2) or bool(up.get("actual_end_time"))
    except Exception:
        upstream_done = False

    # 本站是否無備料指示：materials_prep_status == []
    materials_na = isinstance(mats, list) and len(mats) == 0

    # 物料是否齊全：所有項目的 finish_qty >= request_qty
    materials_ready = False
    if isinstance(mats, list) and len(mats) > 0:
        ok_all = True
        for m in mats:
            if not isinstance(m, dict):
                ok_all = False; break
            rq = _safe_float(m.get("request_qty"))
            fq = _safe_float(m.get("finish_qty"))
            if rq is None or fq is None:
                ok_all = False; break
            if fq + 1e-9 < rq:  # 容忍極小浮點誤差
                ok_all = False; break
        materials_ready = ok_all

    return {
        "first_station": bool(first_station),
        "upstream_done": bool(upstream_done),
        "materials_na": bool(materials_na),
        "materials_ready": bool(materials_ready),
    }

# -------- Prompt builders --------

def build_system_prompt(custom: Optional[str] = None) -> str:
    return custom.strip() if custom else DEFAULT_SYSTEM_PROMPT


def build_user_prompt(payload: Dict[str, Any], custom: Optional[str] = None) -> str:
    if custom:
        return custom
    meta = _compute_meta(payload)
    # 格式化資料（縮排、確保 ASCII 以外字元可讀）
    task_json = json.dumps(payload.get("task", {}), ensure_ascii=False, indent=2, default=_json_default)
    context_json = json.dumps(payload.get("context", {}), ensure_ascii=False, indent=2, default=_json_default)
    meta_json = json.dumps(meta, ensure_ascii=False, indent=2, default=_json_default)
    return DEFAULT_USER_PROMPT.format(
        as_of=payload.get("as_of", ""),
        day_start=payload.get("day_start", ""),
        day_end=payload.get("day_end", ""),
        task_json=task_json,
        context_json=context_json,
        meta_json=meta_json,
    )


# -------- Output parsing --------

def _extract_json_block(text: str) -> Optional[str]:
    """優先擷取 ```json ... ``` 中的 JSON；若找不到，退而求其次抓第一個成對的大括號。"""
    fence = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if fence:
        return fence.group(1)
    # 寬鬆匹配：自第一個 { 到最後一個 }
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first:last + 1]
    return None


def _normalize_analysis(payload: Dict[str, Any], obj: Dict[str, Any]) -> Dict[str, Any]:
    """填補缺欄位 + 修正 evidence（硬性灌入 payload 的 context）。"""
    obj = obj or {}
    obj.setdefault("root_causes", [])
    obj.setdefault("data_gaps", [])
    obj.setdefault("follow_up_queries", [])
    obj.setdefault("notes", "")

    ctx = payload.get("context", {}) if isinstance(payload, dict) else {}

    fixed_causes: List[Dict[str, Any]] = []
    for item in obj.get("root_causes", []) or []:
        if not isinstance(item, dict):
            fixed_causes.append({
                "title": str(item),
                "signals": [],
                "evidence": {
                    "upstream_status": ctx.get("upstream_status"),
                    "materials_prep_status": ctx.get("materials_prep_status"),
                },
                "confidence": 0.0,
            })
            continue

        # 僅支援新欄位名稱：title / signals / evidence / confidence
        title = item.get("title", "")
        signals = item.get("signals") or []
        evidence = item.get("evidence") or {}
        confidence = item.get("confidence", 0.0)

        # 型別與範圍保護
        if not isinstance(signals, list):
            signals = [str(signals)]
        else:
            signals = [str(s) for s in signals]
        try:
            c = float(confidence)
            confidence = max(0.0, min(1.0, c))
        except Exception:
            confidence = 0.0

        # 證據強制覆寫為本次 payload 內容，避免模型遺漏/捏造
        ev = evidence if isinstance(evidence, dict) else {}
        ev["upstream_status"] = ctx.get("upstream_status")
        ev["materials_prep_status"] = ctx.get("materials_prep_status")

        fixed_causes.append({
            "title": title,
            "signals": signals,
            "evidence": ev,
            "confidence": confidence,
        })

    # 若為第一站（無前序製程），提供保底原因/訊號
    try:
        task = payload.get("task", {}) if isinstance(payload, dict) else {}
        is_first_station = (
            task.get("process_seq") == 10 and
            (ctx.get("upstream_status") or {}).get("process_id") in (None, "") and
            (ctx.get("upstream_status") or {}).get("process_seq") in (None, "")
        )
    except Exception:
        is_first_station = False

    if is_first_station:
        if not fixed_causes:
            fixed_causes.append({
                "title": "製程第一道工序，沒有前工序",
                "signals": ["process_seq=10 且上游製程資訊為空，視為第一道工序"],
                "evidence": {
                    "upstream_status": ctx.get("upstream_status"),
                    "materials_prep_status": ctx.get("materials_prep_status"),
                },
                "confidence": 1.0,
            })
        else:
            fixed_causes[0].setdefault("signals", []).append("製程第一道工序（沒有前工序）：process_seq=10 且上游製程為空")

    obj["root_causes"] = fixed_causes
    return obj


def parse_llm_output(payload: Dict[str, Any], text: str) -> Tuple[str, Dict[str, Any]]:
    """回傳 (summary, analysis_json)。若無法解析 JSON，給出保底結構。"""
    # 摘要：取第一個 ```json 區塊之前的文字，或全文（trim 後限 200 字）
    summary_part = text
    fence_pos = re.search(r"```json", text, flags=re.IGNORECASE)
    if fence_pos:
        summary_part = text[: fence_pos.start()].strip()
    summary = summary_part.strip()
    # 清理可能殘留的段落標記或標題
    summary = re.sub(r"^\s*\(?[ABab]\)\s*", "", summary)
    summary = re.sub(r"^(摘要|Summary)\s*[:：]?\s*", "", summary, flags=re.IGNORECASE)
    # 避免太長
    summary = summary[:400]

    # 將「任務 <kanban_id>」正名為「看板 <kanban_id>」
    try:
        kanban_id = (payload or {}).get("task", {}).get("kanban_id")
        if kanban_id:
            summary = re.sub(rf"任務\s*{re.escape(kanban_id)}", f"看板 {kanban_id}", summary)
    except Exception:
        pass

    # 移除摘要中重覆的「逾期未開工/尚未啟動」敘述，直接切入成因
    cleaned = re.sub(r"(?:看板\s+\S+\s+)?逾期未開工[，,]\s*", "", summary)
    cleaned = re.sub(r"但此看板尚未啟動[。．.]*", "", cleaned)
    if cleaned.strip():
        summary = cleaned.strip()

    # JSON：嘗試擷取並解析
    analysis = {
        "root_causes": [],
        "follow_up_queries": [],
        "notes": ""
    }
    raw_json_str = _extract_json_block(text)
    if raw_json_str:
        try:
            obj = json.loads(raw_json_str)
            analysis = _normalize_analysis(payload, obj)
        except Exception:
            # 保持預設 analysis
            pass

    return summary, analysis


# -------- Core API --------

def analyze_overdue_with_llm(
    payload: Dict[str, Any],
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    stream: bool = False,
    client: Optional[OpenAI] = None,
) -> Dict[str, Any]:
    """
    封裝呼叫 LLM 的主函式。

    回傳：{
      "summary": str,
      "analysis": dict,   # 正規化後的 JSON 結果
      "raw": str          # 原始文字輸出（保留以便除錯）
    }
    """
    conf = _load_llm_conf()
    client = client or _get_client(conf)
    model = model or conf.get("model")

    sys_prompt = build_system_prompt(system_prompt)
    usr_prompt = build_user_prompt(payload, user_prompt)

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": usr_prompt},
    ]

    raw_text = ""
    try:
        if stream:
            result = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            for chunk in result:
                part = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
                if part:
                    raw_text += part
        else:
            result = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=False,
            )
            raw_text = result.choices[0].message.content if result.choices else ""

        summary, analysis = parse_llm_output(payload, raw_text or "")
        return {"summary": summary, "analysis": analysis, "raw": raw_text}

    except Exception as e:
        # 保底回傳，避免前端掛掉
        err_msg = f"{type(e).__name__}: {e}"
        fallback_summary = "⚠️ LLM 連線或服務異常，已回傳保底分析骨架。"
        fallback_analysis = {
            "root_causes": [],
            "follow_up_queries": [
                "重新送出分析請求（待服務恢復）",
                "以規則先行：檢查上游 actual_end_time 與物料「已完成數量/需求數量」",
            ],
            "notes": "離線保底回傳，僅保留 payload context 並未進行模型推論。",
        }
        return {"summary": fallback_summary, "analysis": fallback_analysis, "raw": err_msg, "error": err_msg}


# -------- Optional: CLI quick test --------
if __name__ == "__main__":
    # 最小可執行測試（以環境/設定檔提供的 API 資訊）
    sample_payload = {
        "as_of": "2025-10-21 13:02:20 (Asia/Taipei)",
        "day_start": "2025-10-21 00:00:00 (Asia/Taipei)",
        "day_end": "2025-10-21 24:00:00 (Asia/Taipei)",
        "task": {
            "kanban_id": "S-W-20250916195625108-0031",
            "work_order_id": "T20-000963754",
            "part_no": "32600585",
            "work_center_id": "1902",
            "process_id": "S1",
            "process_seq": 10,
            "expected_start_time": "2025-10-21 08:27:49",
            "actual_start_time": "nan",
            "produce_status": 0,
            "flags": {
                "planned_today_flag": 1,
                "overdue_not_started_so_far_flag": 1,
                "started_on_time_today_flag": 0,
                "started_late_today_flag": 0
            }
        },
        "context": {
            "upstream_status": {
                "work_order_id": "T20-000963754",
                "kanban_id": "S-W-20250916195625108-0031",
                "process_id": None,
                "process_seq": None,
                "part_no": None,
                "work_center_id": None,
                "actual_start_time": None,
                "actual_end_time": None,
                "produce_status": None,
                "request_qty": None,
                "finish_qty": None
            },
            "materials_prep_status": [
                {
                    "material_kanban_id": "S-M-20250916195625108-0151",
                    "part_no": "41200535",
                    "part_name": "車輪本體",
                    "finish_qty": 0.0,
                    "request_qty": 16.0,
                    "supplier_name": "鐘昱",
                    "to_kanban_produce": "S-W-20250916195625108-0031",
                    "load_dts": "2025-10-21 13:04:10"
                },
                {
                    "material_kanban_id": "S-M-20250916195625108-0152",
                    "part_no": "42500156",
                    "part_name": "驅動輪圈(4\")",
                    "finish_qty": 0.0,
                    "request_qty": 16.0,
                    "supplier_name": "蕎鋒",
                    "to_kanban_produce": "S-W-20250916195625108-0031",
                    "load_dts": "2025-10-21 13:04:10"
                },
                {
                    "material_kanban_id": "S-M-20250916195625108-0150",
                    "part_no": "43690143",
                    "part_name": "PU填充胎(含外胎)(4PR)",
                    "finish_qty": 0.0,
                    "request_qty": 16.0,
                    "supplier_name": "偉鈺",
                    "to_kanban_produce": "S-W-20250916195625108-0031",
                    "load_dts": "2025-10-21 13:04:09"
                },
                {
                    "material_kanban_id": "S-M-20250916195625108-0149",
                    "part_no": "51008007",
                    "part_name": "六角頭螺栓 (合金鋼)",
                    "finish_qty": 0.0,
                    "request_qty": 80.0,
                    "supplier_name": "錞慶",
                    "to_kanban_produce": "S-W-20250916195625108-0031",
                    "load_dts": "2025-10-21 13:04:09"
                },
                {
                    "material_kanban_id": "S-M-20250916195625108-0148",
                    "part_no": "53208010",
                    "part_name": "尼龍螺帽",
                    "finish_qty": 0.0,
                    "request_qty": 80.0,
                    "supplier_name": "錞慶",
                    "to_kanban_produce": "S-W-20250916195625108-0031",
                    "load_dts": "2025-10-21 13:04:09"
                }
            ]
        }
    }

    out = analyze_overdue_with_llm(sample_payload, stream=False)
    print("\n===== SUMMARY =====\n", out["summary"]) 
    print("\n===== JSON =====\n", json.dumps(out["analysis"], ensure_ascii=False, indent=2))