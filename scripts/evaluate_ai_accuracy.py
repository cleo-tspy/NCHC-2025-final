#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
讀入 JSONL（save_daily_overdue_ai.py 產出的）與人工標註 CSV，
輸出混淆矩陣與基本指標。

人工標註 CSV 欄位（自行建立/維護）：
  - kanban_id
  - true_cause   （單一分類；或多標籤以 ; 分隔）
  - true_rag_cat （可選；RAG 的長期改善主類別）

建議原因分類（可依你習慣微調）：
  Material shortage / Upstream not finished / First station (no upstream) /
  Scheduling-Dispatch / Equipment issue / Workforce-Capacity / Quality hold / Other

RAG 類別建議：
  JIT-Kanban / Heijunka / SMED / Jidoka-Andon / Standard Work /
  Supplier Reliability / Layout-Flow / Visual Mgmt / Other
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

CAUSE_LABELS = [
    "Material shortage",
    "Upstream not finished",
    "First station (no upstream)",
    "Scheduling-Dispatch",
    "Equipment issue",
    "Workforce-Capacity",
    "Quality hold",
    "Other",
]

RAG_LABELS = [
    "JIT-Kanban",
    "Heijunka",
    "SMED",
    "Jidoka-Andon",
    "Standard Work",
    "Supplier Reliability",
    "Layout-Flow",
    "Visual Mgmt",
    "Other",
]

def map_llm_to_cause(row: dict) -> str:
    """簡單規則：從 evidence 與 title/ signals 取最可能主因（top-1）。"""
    analysis = (row.get("llm") or {}).get("analysis") or {}
    root = (analysis.get("root_causes") or [])
    payload = row.get("payload") or {}
    ctx = (payload.get("context") or {})
    up  = (ctx.get("upstream_status") or {})
    mats = ctx.get("materials_prep_status") or []

    # 1) 物料不足（有清單且任一項 finish_qty < request_qty）
    def _flt(x):
        try: return float(x)
        except: return None
    for m in mats:
        rq, fq = _flt(m.get("request_qty")), _flt(m.get("finish_qty"))
        if rq is not None and fq is not None and fq + 1e-9 < rq:
            return "Material shortage"

    # 2) 第一站（無前序）
    if (payload.get("task", {}).get("process_seq") == 10) and (not up.get("process_id")):
        # 若非物料問題，視為「第一站（其他因素）」→ 讓人評回填更細類別
        return "First station (no upstream)"

    # 3) 上游未完工
    if not up or (up.get("produce_status") not in (2,) and not up.get("actual_end_time")):
        return "Upstream not finished"

    # 4) 從文字線索補捉（可再擴充字典）
    text = " ".join([
        " ".join([c.get("title","") for c in root if isinstance(c, dict)]),
        " ".join(sum([c.get("signals") or [] for c in root if isinstance(c, dict)], [])),
    ])
    text = text.lower()
    if any(k in text for k in ["機台", "設備", "維修", "保養", "治具", "換線", "setup", "smed"]):
        return "Equipment issue"
    if any(k in text for k in ["人力", "加班", "產能", "瓶頸", "超負荷"]):
        return "Workforce-Capacity"
    if any(k in text for k in ["排程", "派工", "順序", "dispatch", "schedule"]):
        return "Scheduling-Dispatch"
    if any(k in text for k in ["品質", "放行", "封鎖", "oqc", "iqc", "不良"]):
        return "Quality hold"

    return "Other"

def map_rag_to_cat(row: dict) -> str:
    """從長期改善條列做簡單關鍵字歸類（可再擴充）。"""
    items = (row.get("rag") or {}).get("long_term_actions") or []
    s = " ".join(items).lower()
    if any(k in s for k in ["kanban", "jit", "補貨", "拉式", "安全庫存"]): return "JIT-Kanban"
    if any(k in s for k in ["均衡", "heijunka"]): return "Heijunka"
    if any(k in s for k in ["smed", "換線", "setup"]): return "SMED"
    if any(k in s for k in ["jidoka", "安燈", "andon", "自働化"]): return "Jidoka-Andon"
    if any(k in s for k in ["標準作業", "standard work", "sop"]): return "Standard Work"
    if any(k in s for k in ["供應商", "交期", "可靠度", "supplier"]): return "Supplier Reliability"
    if any(k in s for k in ["動線", "cell", "佈置", "layout", "流動", "瓶頸重排"]): return "Layout-Flow"
    if any(k in s for k in ["目視化", "看板板面", "5s", "visual"]): return "Visual Mgmt"
    return "Other"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="保存的 JSONL 檔路徑（或資料夾）")
    ap.add_argument("--truth", required=True, help="人工標註 CSV 路徑")
    ap.add_argument("--out", required=False, default="eval_out", help="輸出資料夾")
    args = ap.parse_args()

    p = Path(args.jsonl)
    if p.is_dir():
        files = sorted(p.glob("*.jsonl"))
    else:
        files = [p]

    rows = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    df = pd.DataFrame(rows)

    # 預測類別
    df["pred_cause"] = df.apply(map_llm_to_cause, axis=1)
    df["pred_rag_cat"] = df.apply(map_rag_to_cat, axis=1)

    # 載入真值
    gt = pd.read_csv(args.truth)  # 欄：kanban_id,true_cause,true_rag_cat
    merged = pd.merge(df, gt, on="kanban_id", how="inner")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # 混淆矩陣（原因）
    cm_cause = pd.crosstab(merged["true_cause"], merged["pred_cause"]).reindex(index=CAUSE_LABELS, columns=CAUSE_LABELS, fill_value=0)
    cm_cause.to_csv(out_dir / "cm_cause.csv", encoding="utf-8-sig")

    # 混淆矩陣（RAG）
    if "true_rag_cat" in merged.columns:
        cm_rag = pd.crosstab(merged["true_rag_cat"], merged["pred_rag_cat"]).reindex(index=RAG_LABELS, columns=RAG_LABELS, fill_value=0)
        cm_rag.to_csv(out_dir / "cm_rag.csv", encoding="utf-8-sig")

    # 基礎指標（單標籤 top-1 accuracy）
    acc_cause = (merged["true_cause"] == merged["pred_cause"]).mean()
    acc_rag = (merged["true_rag_cat"] == merged["pred_rag_cat"]).mean() if "true_rag_cat" in merged.columns else None

    merged[["kanban_id","true_cause","pred_cause","true_rag_cat","pred_rag_cat"]].to_csv(out_dir / "pairs.csv", index=False, encoding="utf-8-sig")

    print(f"Cause Accuracy: {acc_cause:.1%}")
    if acc_rag is not None:
        print(f"RAG Accuracy:   {acc_rag:.1%}")
    print(f"Saved: {out_dir}")

if __name__ == "__main__":
    main()