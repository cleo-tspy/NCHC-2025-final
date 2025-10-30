

## Setup env

### Venv
* Linux/mac

    ```
    python3 -m venv .venv
    source .venv/bin/activate
    ```

* Install packages
    ```
    python -m pip install -U pip setuptools wheel
    pip install streamlit sqlalchemy pymysql pandas openai
    ```
### 前端套件 Streamlit 參數建議
* ./streamlit/config.toml
    ```
    [theme]
    base="light"
    primaryColor="#4b8bff"

    [server]
    port = 8081
    ```

## 網頁

* 啟動網頁
    ```
    # 參考 config_example.ini 設定 config.ini 
    streamlit run app_today.py
    ```

### UI 說明

* 今日生產明細，表格呈現排序用意
    * 預設排序是依序用這四欄升冪：work_center_id → process_id → process_seq → expected_start_time
    	1.	work_center_id：先把同一工作站的看板聚在一起——方便班組長/機台看自己那一區。
        2.	process_id：在同一工作站內，再按製程分組，避免不同製程交錯。
        3.	process_seq：同一製程裡，依製程序號由小到大，符合工藝路徑的先後。
        4.	expected_start_time：最後用今天的計畫開工時間排序，讓越早該開工的越前面。

---

### 驗證異常偵測結果 見 scripts/README.md 說明
