# scripts/ 使用說明

> output data raw_20251021.csv 使用欄位是初版的逾期未開工，不是逾期四小時未開工 

這個目錄放三支批次腳本，協助**每日擷取逾期看板**、**呼叫 AI 分析**，以及**每週評估模型準確度**。

## 內容
- `save_daily_overdue_raw_data.py`（**Phase 1：只組資料，不跑 AI**）
  - 讀取資料庫（`v_today_kanban_start_detail`）取得**今日逾期未開工**看板。
  - 製作每筆 **payload**（含上游製程、物料備料狀態），並輸出：
    - `data/daily/YYYYMMDD/raw_YYYYMMDD.csv`（方便人工快看）
    - `data/daily/YYYYMMDD/raw_payload_YYYYMMDD.jsonl`（供 Phase 2 使用）

- `process_payloads.py`（**Phase 2：讀 payload，呼叫 AI**）
  - 讀取 `raw_payload_YYYYMMDD.jsonl`，逐筆：
    - 呼叫 `call_llm_api.py` → 逾期原因分析（短期改善）
    - 呼叫 `call_rag_api.py` → 精實生產理論建議（長期改善）
  - 以 **JSON Lines** 存檔：`data/daily/YYYYMMDD/overdue_ai_YYYYMMDD.jsonl`
  - 內建功能：
    - **續跑 / 篩選**旗標（互斥優先序：`--only-new` > `--errors-only` > `--resume`）：
      - `--resume` / `--no-resume`：是否續跑（預設：`--resume`）。
      - `--retry-errors`：續跑時，對**先前錯誤**的鍵重試。
      - `--errors-only`：只處理先前錯誤（隱含啟用續跑）。
      - `--only-new`：只處理**未曾出現在 overdue_ai** 的鍵（隱含啟用續跑）。
      - `--limit N`：只處理前 N 筆。
    - **節流 / 間隔**（環境變數，可選）：
      - `LLM_MIN_INTERVAL_MS`：兩次 LLM 呼叫的最小間隔（毫秒）。
      - `SLEEP_AFTER_RAG_MS`：每筆完成後固定暫停（毫秒）。
    - **暫時性錯誤重試**（環境變數，可選）：
      - `LLM_EXTRA_RETRIES`（預設 1）、`LLM_RETRY_BACKOFF_MS`（預設 2000）。
      - `RAG_EXTRA_RETRIES`（預設 1）、`RAG_RETRY_BACKOFF_MS`（預設 1500）。
    - **落盤安全**（環境變數，可選）：
      - `FSYNC_EVERY`：每寫入 N 筆做一次 `fsync`（0 表示不 fsync）。
    - **進度提示**：每處理 100 筆會 `print` 一次進度。
  - **注意**：`overdue_ai_YYYYMMDD.jsonl` **不再保存 payload**（payload 只在 Phase 1 的 `raw_payload_*.jsonl` 中）。

- `evaluate_ai_accuracy.py`
  - 讀取 `overdue_ai_*.jsonl` 與**人工標註真值**（CSV），
  - 產出**混淆矩陣**與**準確度**等指標。

---

## 先決條件
- Python 3.10+，已安裝專案相同依賴（`pip install -r requirements.txt`）。
- 專案根目錄須有 `config.ini`（或以 `CONFIG_PATH` 指向其他路徑）。

### `config.ini` 需求段落（摘要）
```ini
[database]                 ; 主資料庫（含 v_today_kanban_start_detail）
# driver = mysql+pymysql
# host = ...
# port = ...
# user = ...
# password = ...
# database = ...
# charset = utf8mb4

[database_leanplay_supp]   ; 物料備料資料庫（kanban_material_prep_status）
# driver = mysql+pymysql
# host = ...
# port = ...
# user = ...
# password = ...
# database = ...
# charset = utf8mb4

[llm]
# base_url = http://<LLM_HOST>:<PORT>/v1
# api_key  = <KEY>
# model    = google/gemma-3-4b-it

[rag]
# api     = http://<RAG_HOST>:<PORT>/rag_summary   ; 不要加引號
# timeout = 60
```
> 也可用環境變數覆寫：`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `LLM_MODEL`, `RAG_SUMMARY_API`, `RAG_TIMEOUT`。

---

## 手動執行
```bash
# 1) Phase 1：組資料（輸出 raw_* 檔）
python scripts/save_daily_overdue_ai.py

# 2) Phase 2：讀 raw_payload，呼叫 LLM + RAG（輸出 overdue_ai_*）
python scripts/process_payloads.py --date 20251021           # 續跑（預設）
python scripts/process_payloads.py --date 20251021 --only-new # 只跑新長出的
python scripts/process_payloads.py --date 20251021 --errors-only --retry-errors  # 只重跑錯誤

# （可選）節流與重試（環境變數）
LLM_MIN_INTERVAL_MS=800 SLEEP_AFTER_RAG_MS=300 \
python scripts/process_payloads.py --date 20251021
```

### JSONL 格式
**Phase 1：raw_payload_YYYYMMDD.jsonl**（每行一筆）
```json
{
  "generated_at": "2025-10-21 20:30:05",
  "kanban_id": "S-W-...",
  "work_order_id": "T20-...",
  "payload": { "as_of": "...", "day_start": "...", "day_end": "...", "task": { ... }, "context": { ... } }
}
```

**Phase 2：overdue_ai_YYYYMMDD.jsonl**（每行一筆；⚠️ 不含 payload）
```json
{
  "generated_at": "2025-10-21 20:52:37",
  "kanban_id": "S-W-...",
  "work_order_id": "T20-...",
  "llm": { "summary": "...", "analysis": { ... }, "raw": "...", "error": null },
  "rag": { "long_term_actions": ["..."], "raw": "...", "error": null },
  "llm_latency_ms": 12706,
  "rag_latency_ms": 271564
}
```

---

## 排程（cron）
> 主機時區若非 Asia/Taipei，請換算或將主機時區設為 Asia/Taipei。

**每日 20:45** Phase 1（只組資料）：
```cron
45 20 * * * /path/to/.venv/bin/python \
  /path/to/NCHC-2025-final/scripts/save_daily_overdue_ai.py \
  >> /path/to/NCHC-2025-final/logs/daily_ai_raw.log 2>&1
```

**每日 20:50** Phase 2（AI 分析；含節流與重試範例）：
```cron
50 20 * * * LLM_MIN_INTERVAL_MS=800 SLEEP_AFTER_RAG_MS=300 LLM_EXTRA_RETRIES=2 LLM_RETRY_BACKOFF_MS=3000 \
  /path/to/.venv/bin/python \
  /path/to/NCHC-2025-final/scripts/process_payloads.py --date $(date +\%Y\%m\%d) --resume \
  >> /path/to/NCHC-2025-final/logs/daily_ai_process.log 2>&1
```

**每週一 20:55** 跑評估（以累積資料 + 人工作答）：
```cron
55 20 * * 1 /path/to/.venv/bin/python \
  /path/to/NCHC-2025-final/scripts/evaluate_ai_accuracy.py \
  --jsonl /path/to/NCHC-2025-final/data/daily/ \
  --truth /path/to/NCHC-2025-final/data/labels/ground_truth.csv \
  --out   /path/to/NCHC-2025-final/data/eval \
  >> /path/to/NCHC-2025-final/logs/eval.log 2>&1
```

---

## 故障排查
- **InvalidSchema: No connection adapters...**  
  `rag.api` URL 不要加引號，或確認 `RAG_SUMMARY_API` 是否有多餘的 `"`。
- **TypeError: Decimal is not JSON serializable**  
  程式已內建轉換；若仍發生，檢查自訂欄位是否含 `Decimal` 且未經轉換。
- **LLM/RAG 連線失敗或逾時**  
  腳本會回傳保底結構（寫在 `llm.error` / `rag.error`），可調高 `LLM_MIN_INTERVAL_MS`、`SLEEP_AFTER_RAG_MS` 或重試參數觀察。
- **進度不明**  
  `process_payloads.py` 每處理 100 筆會列印 `[PROGRESS] 已執行 N 筆...`。

---

## 評估結果（輸出）
- `data/eval/cm_cause.csv`、`data/eval/cm_rag.csv`：混淆矩陣
- `data/eval/pairs.csv`：逐筆對照（便於抽查）

> 先以 Top-1 Accuracy 作為目標（≥ 90%），若需更嚴謹，可延伸到 Precision/Recall/F1 與多標籤評估。