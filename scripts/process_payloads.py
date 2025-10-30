#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
讀取 data/daily/YYYYMMDD/raw_payload_YYYYMMDD.jsonl，
逐筆呼叫 LLM（原因/短期）與 RAG（長期），輸出 overdue_ai_YYYYMMDD.jsonl。

使用方式：
  $ python scripts/process_payloads.py --date {ymd}
  # 或
  $ python scripts/process_payloads.py --raw data/daily/{ymd}/raw_payload_{ymd}.jsonl

  $ python scripts/process_payloads.py --date {ymd} --sample 300

選項：
  --date YYYYMMDD     指定日期（預設為今天，Asia/Taipei）
  --raw  PATH         指定原始 payload 檔（覆蓋 --date）
  --out  PATH         指定輸出檔名（預設 data/daily/<date>/overdue_ai_<date>.jsonl）
  --limit N           只處理前 N 筆（預設不限）
  --min-interval-ms MSEC  兩次 LLM 呼叫的最小間隔（毫秒，預設 0）
  --sleep-after-ms  MSEC  每筆完成後固定暫停（毫秒，預設 10）

旗標（執行變數）：
  --resume / --no-resume   是否續跑（預設：--resume）
  --retry-errors           續跑時，對先前錯誤的項目重試（預設關）
  --errors-only            只處理「先前錯誤」的項目（隱含啟用續跑）
  --only-new               只處理「未曾出現在 overdue_ai 檔」的項目（隱含啟用續跑）

優先順序（互斥邏輯）：
  --only-new > --errors-only > --resume

  --sample N [--seed SEED] [--sample-out PATH] 抽樣模式（避免一次跑完整檔）
  從 raw_payload 隨機抽取 N 筆，先另存為 sample 檔，再以 sample 檔進行分析。
  預設 sample 會輸出到 `data/daily/<date>/raw_payload_sample_<date>_<N>.jsonl`；
  若未指定 --out，分析輸出預設為 `overdue_ai_sample_<date>_<N>.jsonl`。


"""
from __future__ import annotations
import os, json, argparse, time, random, re
from pathlib import Path
from datetime import datetime, date, time as dtime
from zoneinfo import ZoneInfo
from typing import Any, Dict

# --- 專案根路徑納入 sys.path，確保可匯入專案模組 ---
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 專案內模組
from call_llm_api import analyze_overdue_with_llm
from call_rag_api import get_long_term_improvements

# ---------- JSON 安全序列化 ----------
from decimal import Decimal
import numpy as np
try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, set):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Decimal):
        try:
            return float(obj)
        except Exception:
            return str(obj)
    if _HAS_PANDAS and isinstance(obj, pd.Timestamp):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    if isinstance(obj, (datetime, date, dtime)):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj

def _parse_args() -> argparse.Namespace:
    """
    參數處理
    """
    ap = argparse.ArgumentParser(description="Process raw payloads to AI analysis JSONL")
    ap.add_argument("--date", dest="ymd", default=None, help="YYYYMMDD (Asia/Taipei)")
    ap.add_argument("--raw", dest="raw_path", default=None, help="path to raw_payload_YYYYMMDD.jsonl")
    ap.add_argument("--out", dest="out_path", default=None, help="output JSONL path")
    ap.add_argument("--limit", dest="limit", type=int, default=None, help="only process first N records")
    ap.add_argument("--min-interval-ms", dest="min_interval_ms", type=int, default=0,
                    help="minimum interval between consecutive LLM calls in milliseconds (default: 0)")
    ap.add_argument("--sleep-after-ms", dest="sleep_after_ms", type=int, default=10,
                    help="fixed sleep after each record in milliseconds (default: 10)")
    ap.add_argument("--resume", dest="resume", action="store_true", default=True, help="resume from existing overdue_ai if present (default: on)")
    ap.add_argument("--no-resume", dest="resume", action="store_false", help="disable resume; process all raw payloads")
    ap.add_argument("--retry-errors", dest="retry_errors", action="store_true", default=False, help="re-process entries that previously had error in overdue_ai")
    ap.add_argument("--errors-only", dest="errors_only", action="store_true", default=False, help="only process entries that previously had error (implies resume)")
    ap.add_argument("--only-new", dest="only_new", action="store_true", default=False,
                    help="only process entries not present in overdue_ai (if file exists)")    
    ap.add_argument("--sample", dest="sample", type=int, default=None,
                    help="randomly sample N records from raw payload, write to a sample file, then process the sample")
    ap.add_argument("--seed", dest="seed", type=int, default=None,
                    help="random seed for reproducible sampling")
    ap.add_argument("--sample-out", dest="sample_out", default=None,
                    help="path to write sampled raw_payload jsonl; default under daily dir")
    return ap.parse_args()

# ---------- 續跑用 helper ----------

def _make_key(kanban_id: str | None, work_order_id: str | None) -> str:
    return f"{kanban_id or ''}||{work_order_id or ''}"


def _record_has_error(rec: Dict[str, Any]) -> bool:
    # 可能的錯誤位置：頂層 error、llm.error、rag.error
    if rec.get("error"):
        return True
    for k in ("llm", "rag"):
        v = rec.get(k)
        if isinstance(v, dict) and v.get("error"):
            return True
    return False

def _sample_raw_to_file(src_path: Path, dst_path: Path, k: int, seed: int | None = None) -> tuple[int, int]:
    """Reservoir sampling from src_path, write k lines to dst_path. Returns (selected, total)."""
    if k <= 0:
        return 0, 0
    if seed is not None:
        random.seed(seed)
    reservoir: list[str] = []
    total = 0
    with open(src_path, "r", encoding="utf-8") as fr:
        for line in fr:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            total += 1
            if len(reservoir) < k:
                reservoir.append(line_stripped)
            else:
                j = random.randrange(total)
                if j < k:
                    reservoir[j] = line_stripped
    with open(dst_path, "w", encoding="utf-8") as fw:
        for s in reservoir:
            fw.write(s + "\n")
    return len(reservoir), total

# ---------- 主流程 ----------
def main():
    args = _parse_args()
    tz = ZoneInfo("Asia/Taipei")
    now_local = datetime.now(tz)

    # 解析輸入/輸出路徑
    if args.raw_path:
        raw_file = Path(args.raw_path)
        if args.ymd is None:
            # 從檔名/父資料夾猜日期
            try:
                args.ymd = raw_file.stem.split("_")[-1]
            except Exception:
                args.ymd = now_local.strftime("%Y%m%d")
    else:
        ymd = args.ymd or now_local.strftime("%Y%m%d")
        raw_file = PROJECT_ROOT / "data" / "daily" / ymd / f"raw_payload_{ymd}.jsonl"

    # 檔案存在性檢查：在進入抽樣/處理前先確認原始 raw 檔存在
    if not raw_file.exists():
        raise FileNotFoundError(f"找不到 raw payload 檔案：{raw_file}")

    if args.out_path:
        out_file = Path(args.out_path)
    else:
        ymd = args.ymd or now_local.strftime("%Y%m%d")
        out_file = PROJECT_ROOT / "data" / "daily" / ymd / f"overdue_ai_{ymd}.jsonl"
        out_file.parent.mkdir(parents=True, exist_ok=True)

    # 抽樣模式：先從 raw 建立 sample，再改以 sample 檔作為輸入
    sample_info = None
    if args.sample is not None and args.sample > 0:
        ymd = args.ymd or now_local.strftime("%Y%m%d")
        if args.sample_out:
            sample_file = Path(args.sample_out)
            sample_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            sample_file = PROJECT_ROOT / "data" / "daily" / ymd / f"raw_payload_sample_{ymd}_{args.sample}.jsonl"
            sample_file.parent.mkdir(parents=True, exist_ok=True)
        sel, total = _sample_raw_to_file(raw_file, sample_file, args.sample, args.seed)
        print(f"[SAMPLE] 從 {raw_file} 隨機抽取 {sel}/{total} 筆 → {sample_file}")
        raw_file = sample_file  # 之後的處理都以 sample 為準
        # 若未指定 out，預設也改為 sample 版本，避免覆蓋原本全量輸出
        if args.out_path is None:
            out_file = PROJECT_ROOT / "data" / "daily" / ymd / f"overdue_ai_sample_{ymd}_{sel}.jsonl"
            out_file.parent.mkdir(parents=True, exist_ok=True)

    processed_ok: set[str] = set()
    processed_err: set[str] = set()
    if (args.resume or args.only_new) and out_file.exists():
        with open(out_file, "r", encoding="utf-8") as frx:
            for line in frx:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                key = _make_key(rec.get("kanban_id"), rec.get("work_order_id"))
                if not key.strip("|"):
                    continue
                if _record_has_error(rec):
                    processed_err.add(key)
                else:
                    processed_ok.add(key)

    # 追蹤上一筆 LLM 呼叫時間
    last_llm_ts = None

    processed = 0
    done_count = 0  # 成功或錯誤皆計為已執行一筆
    with open(out_file, "a", encoding="utf-8") as fw:
        with open(raw_file, "r", encoding="utf-8") as fr:
            for idx, line in enumerate(fr):
                if args.limit is not None and idx >= args.limit:
                    break
                line = line.strip()
                if not line:
                    continue

                # 一筆 payload
                try:
                    rowj: Dict[str, Any] = json.loads(line)
                except Exception as e:
                    err = {
                        "generated_at": now_local.strftime("%Y-%m-%d %H:%M:%S"),
                        "kanban_id": None,
                        "work_order_id": None,
                        "error": f"JSONDecodeError: {e}",
                    }
                    fw.write(json.dumps(err, ensure_ascii=False) + "\n")
                    done_count += 1
                    if done_count % 100 == 0:
                        print(f"[PROGRESS] 已執行 {done_count} 筆...", flush=True)
                    # 例外情況也做固定暫停，避免持續快速重試
                    if args.sleep_after_ms > 0:
                        time.sleep(args.sleep_after_ms / 1000.0)
                    continue
                payload = rowj.get("payload") or {}
                kanban_id = rowj.get("kanban_id")
                work_order_id = rowj.get("work_order_id")

                key = _make_key(kanban_id, work_order_id)
                if args.only_new:
                    # 僅處理從未出現在 overdue_ai 的鍵（無論先前成功或錯誤都跳過）
                    if key in processed_ok or key in processed_err:
                        continue
                elif args.errors_only:
                    # 只處理先前有錯誤的資料
                    if key not in processed_err:
                        continue
                elif args.resume:
                    # 續跑：已成功處理過則跳過；是否重試錯誤依參數而定
                    if key in processed_ok:
                        continue
                    if (not args.retry_errors) and key in processed_err:
                        continue

                try:
                    # (Optional) 節流：控制連續 LLM 呼叫間隔
                    if last_llm_ts is not None and args.min_interval_ms > 0:
                        elapsed_ms = int((time.perf_counter() - last_llm_ts) * 1000)
                        if elapsed_ms < args.min_interval_ms:
                            wait_ms = args.min_interval_ms - elapsed_ms + random.randint(0, 150)  # 加少量抖動
                            time.sleep(max(0, wait_ms) / 1000.0)

                    # LLM（含延遲量測）
                    t0 = time.perf_counter()
                    llm_res = analyze_overdue_with_llm(payload, stream=False)
                    llm_latency_ms = int((time.perf_counter() - t0) * 1000)
                    last_llm_ts = time.perf_counter()

                    # RAG：僅在 LLM 成功時呼叫
                    if llm_res.get("error"):
                        rag_res = {"long_term_actions": [], "raw": "", "error": "skipped_due_to_llm_error"}
                        rag_latency_ms = None
                    else:
                        t1 = time.perf_counter()
                        rag_res = get_long_term_improvements(llm_res.get("analysis") or {}, payload)
                        rag_latency_ms = int((time.perf_counter() - t1) * 1000)

                    record = {
                        "generated_at": now_local.strftime("%Y-%m-%d %H:%M:%S"),
                        "kanban_id": kanban_id,
                        "work_order_id": work_order_id,
                        "llm": _json_safe(llm_res),
                        "rag": _json_safe(rag_res),
                        "llm_latency_ms": llm_latency_ms,
                        "rag_latency_ms": rag_latency_ms,
                    }
                    fw.write(json.dumps(record, ensure_ascii=False) + "\n")
                    processed += 1
                    done_count += 1
                    if done_count % 100 == 0:
                        print(f"[PROGRESS] 已執行 {done_count} 筆...", flush=True)

                    # (Optional) 每筆完成後固定暫停，避免連續壓力
                    if args.sleep_after_ms > 0:
                        time.sleep(args.sleep_after_ms / 1000.0)

                except Exception as e:
                    err = {
                        "generated_at": now_local.strftime("%Y-%m-%d %H:%M:%S"),
                        "kanban_id": kanban_id,
                        "work_order_id": work_order_id,
                        "error": f"{type(e).__name__}: {e}",
                    }
                    fw.write(json.dumps(err, ensure_ascii=False) + "\n")
                    # 例外情況也做固定暫停，避免持續快速重試
                    if args.sleep_after_ms > 0:
                        time.sleep(args.sleep_after_ms / 1000.0)
                    done_count += 1
                    if done_count % 100 == 0:
                        print(f"[PROGRESS] 已執行 {done_count} 筆...", flush=True)

    print(f"[OK] 已處理 {processed} 筆，輸出：{out_file}")
    if args.resume or args.only_new:
        print(f"[RESUME] 跳過已成功：{len(processed_ok)} 筆；先前錯誤：{len(processed_err)} 筆（retry_errors={args.retry_errors}, errors_only={args.errors_only}, only_new={args.only_new}）")

if __name__ == "__main__":
    main()