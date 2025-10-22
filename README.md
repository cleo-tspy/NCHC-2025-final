

### Venv
* Linux/mac

    ```
    python3 -m venv .venv
    source .venv/bin/activate
    ```

* install packages
```
python -m pip install -U pip setuptools wheel
pip install streamlit sqlalchemy pymysql pandas openai
```

* 表格呈現排序用意
    * 預設排序是依序用這四欄升冪：work_center_id → process_id → process_seq → expected_start_time
    	1.	work_center_id：先把同一工作站的看板聚在一起——方便班組長/機台看自己那一區。
        2.	process_id：在同一工作站內，再按製程分組，避免不同製程交錯。
        3.	process_seq：同一製程裡，依製程序號由小到大，符合工藝路徑的先後。
        4.	expected_start_time：最後用今天的計畫開工時間排序，讓越早該開工的越前面。

### Run
```
streamlit run app_today.py
```