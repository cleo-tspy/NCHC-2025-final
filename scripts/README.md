# scripts/ 使用說明

這個目錄放兩支離線批次腳本，協助**每日擷取逾期看板**並**產出 AI 分析**，以及**每週評估模型準確度**。

## 內容
- `save_daily_overdue_ai.py`：
  - 讀取資料庫（`v_today_kanban_start_detail`）取得**今日逾期未開工**看板。
  - 組成 **payload**（含上游製程與物料備料狀態），呼叫：
    - `call_llm_api.py` → 逾期原因分析（短期改善）
    - `call_rag_api.py` → 精實生產理論建議（長期改善）
  - 以 **JSON Lines** 存檔：`data/daily/YYYYMMDD/overdue_ai_YYYYMMDD.jsonl`

- `evaluate_ai_accuracy.py`：
  - 讀取前者的 JSONL 與**人工標註真值**（CSV），
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
# 1) 產出當日逾期 AI 分析（會建立 data/daily/YYYYMMDD/overdue_ai_YYYYMMDD.jsonl）
python scripts/save_daily_overdue_ai.py

# 2) 以一週資料 + 人工標註 CSV 進行評估
python scripts/evaluate_ai_accuracy.py \
  --jsonl data/daily/202510*/ \
  --truth data/labels/ground_truth.csv \
  --out   data/eval
```

### JSONL 內容格式（每行一筆）
```json
{
  "generated_at": "2025-10-21 20:30:05",
  "kanban_id": "S-W-...",
  "work_order_id": "T20-...",
  "payload": { ... },         // 送入 LLM 的 payload
  "llm": { "summary": "...", "analysis": { ... }, "raw": "..." },
  "rag": { "long_term_actions": ["..."], "raw": "..." }
}
```

---

## 人工標註（Ground Truth）格式建議
CSV 檔（UTF-8）欄位：
- `kanban_id`
- `true_cause`   ：LLM 主因分類（單一標籤）
- `true_rag_cat` ：RAG 長期改善主類別（可選）

**原因分類建議**（可依實務微調）：
```
Material shortage | Upstream not finished | First station (no upstream) |
Scheduling-Dispatch | Equipment issue | Workforce-Capacity | Quality hold | Other
```
**RAG 類別建議**：
```
JIT-Kanban | Heijunka | SMED | Jidoka-Andon | Standard Work |
Supplier Reliability | Layout-Flow | Visual Mgmt | Other
```

**範例**：
```csv
kanban_id,true_cause,true_rag_cat
S-W-20250916-0032,Material shortage,JIT-Kanban
S-W-20250925-0010-2,Upstream not finished,Heijunka
```

---

## 排程（cron）
> 主機時區若非 Asia/Taipei，請換算或將主機時區設為 Asia/Taipei。

**每日 20:45** 擷取與分析：
```cron
45 20 * * * /path/to/.venv/bin/python \
  /path/to/NCHC-2025-final/scripts/save_daily_overdue_ai.py \
  >> /path/to/NCHC-2025-final/logs/daily_ai.log 2>&1
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
  `api` URL 不要加引號，或確認 `RAG_SUMMARY_API` 是否有多餘的 `"`。
- **TypeError: Decimal is not JSON serializable**  
  程式已內建轉換；若仍發生，檢查自訂欄位是否含 `Decimal` 且未經轉換。
- **LLM/RAG 連線失敗**  
  腳本會回傳保底結構，記錄在 JSONL 的 `llm.error` / `rag.error`；請檢查 `config.ini` 或網路。

---

## 評估結果（輸出）
- `data/eval/cm_cause.csv`、`data/eval/cm_rag.csv`：混淆矩陣
- `data/eval/pairs.csv`：逐筆對照（便於抽查）

> 先以 Top-1 Accuracy 作為目標（≥ 90%），若需更嚴謹，可延伸到 Precision/Recall/F1 與多標籤評估。