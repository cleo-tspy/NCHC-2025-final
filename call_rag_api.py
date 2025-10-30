# /Users/peiyutsai/programming/NCHC-2025-final/call_rag_api.py
from __future__ import annotations
import os
import json
import re
import requests
from typing import Any, Dict, List, Optional
from pathlib import Path
import configparser

from decimal import Decimal
from datetime import datetime, date, time


def _json_default(o):
    """
    JSON 序列化助手
    """
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
    try:
        return str(o)
    except Exception:
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

def _to_json_text(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default)


def _load_rag_conf() -> Dict[str, Any]:
    """
    # 設定來源：環境變數優先、再讀 config.ini 的 [rag]
    """
    cfg_path = os.getenv("CONFIG_PATH")
    cfg = Path(cfg_path) if cfg_path else Path(__file__).resolve().parent / "config.ini"
    cp = configparser.ConfigParser()
    if cfg.exists():
        cp.read(cfg, encoding="utf-8")
    sec = cp["rag"] if cp.has_section("rag") else {}

    return {
        "api": os.getenv("RAG_SUMMARY_API") or sec.get("api", "http://000.00.00.0:8060/rag_summary"),
        "timeout": int(os.getenv("RAG_TIMEOUT") or sec.get("timeout", "90")),
    }

# ----------------------------
# Prompt 建構
# ----------------------------
SYSTEM_PROMPT = (
    "您是專業的精實生產管理顧問，所有回覆使用繁體中文；"
    "請專注於長期性的流程/體制/佈局改善，而非當日派遣或補料等短期行動。"
)

LONG_TERM_GUIDE = (
    "請依據提供的『逾期原因分析（analysis JSON）』，從精實生產原則出發，"
    "提出 3–7 條『長期改善建議』，每條為一句簡短可執行的策略，避免與短期追料/加派人工等重複。"
    "可參考的面向（不限於）：價值流程（VSM）、JIT/拉式補貨、Heijunka 均衡化、SMED 換線縮短、"
    "Jidoka 自働化、標準作業/作業設計、目視化/安燈、工序與瓶頸再配置、供應商開發與交期可靠度、"
    "看板政策（最小批量/安全庫存/補貨規則）等。"
    "輸出格式：每條前綴「• 」，純文字，不要附上條號、不要重覆 LLM 的短期建議。"
    "限制：不得討論設計/研發/LPPD/總工程師等研發治理內容，僅針對製造現場（工作中心/工序/排程/人力/設備/品質放行/拉式補貨）提出長期改善。"
)

def _build_user_prompt(analysis: Dict[str, Any], payload: Optional[Dict[str, Any]] = None) -> str:
    """
    根據 analysis 與（可選）payload 組裝 user prompt，嵌入 kanban_id 與 analysis JSON。
    """
    kanban_id = ""
    if payload and isinstance(payload, dict):
        t = payload.get("task", {}) or {}
        kanban_id = t.get("kanban_id") or ""

    body = {
        "kanban_id": kanban_id,
        "analysis": analysis,  # 包含 root_causes / follow_up_queries / notes
    }
    return (
        "【背景資料】\n"
        + _to_json_text(body)
        + "\n\n【任務】\n"
        + LONG_TERM_GUIDE
    )

def _post_rag(api: str, system_prompt: str, user_prompt: str, timeout: int = 90) -> Dict[str, Any]:
    """
    呼叫 RAG API
    """
    payload = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
    resp = requests.post(api, json=payload, timeout=timeout)
    try:
        data = resp.json()
    except Exception:
        data = {}

    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")

    # 預期回傳格式：{"success": true, "summary": "..."}
    return {
        "success": bool(data.get("success")),
        "summary": data.get("summary", "") or "",
        "raw": data,
    }

def _split_bullets(text: str) -> List[str]:
    """
    將含條列符號的文字切成去重後的建議清單
    """
    # 取每行前綴為「• 」或「- 」或「* 」的條列；並做簡單清洗
    items: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(("• ", "- ", "* ")):
            s = s[2:].strip()
            # 去除收尾標點與重覆空白
            s = re.sub(r"[ \t]+", " ", s).strip(" 　；;，,")
            if s:
                items.append(s)
    # 若完全沒有條列，就把整段當作一條
    if not items and text.strip():
        items = [re.sub(r"[\\s]+", " ", text.strip())]
    # 去重
    seen = set()
    uniq = []
    for it in items:
        if it not in seen:
            uniq.append(it); seen.add(it)
    return uniq[:10]

def get_long_term_improvements(analysis: Dict[str, Any], payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    主流程：組 Prompt → 呼叫 RAG → 解析為長期改善建議清單
    輸入：
      - analysis：call_llm_api.analyze_overdue_with_llm 的 analysis JSON
      - payload：可選，原始 payload（為了給出看板ID等脈絡）

    回傳：
      {
        "long_term_actions": [str, ...],   # 清單
        "raw": str,                        # 伺服器回傳的 summary 原文
        "error": str | None
      }
    """
    conf = _load_rag_conf()
    user_prompt = _build_user_prompt(analysis, payload)
    try:
        res = _post_rag(conf["api"], SYSTEM_PROMPT, user_prompt, timeout=conf["timeout"])
        bullets = _split_bullets(res.get("summary", "") or "")
        return {
            "long_term_actions": bullets,
            "raw": res.get("summary", ""),
            "error": None if res.get("success") else "RAG 回傳 success=false",
        }
    except Exception as e:
        return {
            "long_term_actions": [],
            "raw": "",
            "error": f"{type(e).__name__}: {e}",
        }

# 直接測試
if __name__ == "__main__":
    demo_analysis = {
        "root_causes": [
            {
                "title": "物料準備不足",
                "signals": ["多項物料已完成數量為 0", "供應商交期不穩定"],
                "evidence": {"upstream_status": None, "materials_prep_status": []},
                "confidence": 0.8,
            }
        ],
        "follow_up_queries": ["即時確認供應商實際交期與可替代料", "盤點看板政策與提醒點"],
        "notes": "",
    }
    out = get_long_term_improvements(demo_analysis, payload={"task": {"kanban_id": "DEMO-001"}})
    print(json.dumps(out, ensure_ascii=False, indent=2))