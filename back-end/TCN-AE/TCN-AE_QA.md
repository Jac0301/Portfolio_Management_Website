# TCN-AE 模型問答集

## 關於 TCN-AE 模型的基本資訊

### Q1: TCN-AE 是在做什麼的？
TCN-AE（Temporal Convolutional Network - AutoEncoder）是一個時間序列異常檢測模型，主要用於金融時間序列數據的異常檢測和特徵壓縮。主要功能包括：
1. 資料壓縮：將 117 維特徵壓縮成 20 維
2. 特徵提取：提取時間序列數據中的重要特徵
3. 異常檢測：可以檢測出異常的交易模式
4. 資料標準化：確保所有特徵在相同的數值範圍內

### Q2: 資料夾內每個檔案的功能是什麼？

1. **tcnae.py**：
   - 主要模型架構檔案
   - 輸入：時間序列數據（維度為 117）
   - 輸出：重建的時間序列數據和壓縮後的特徵
   - 主要功能：
     - 使用 TCN 進行特徵提取
     - 將每 20 天的數據壓縮成 1 天
     - 將 117 維特徵壓縮到 10 維

2. **utilities.py**：
   - 工具函數集合
   - 主要功能：
     - GPU 設定
     - 滑動窗口處理
     - 馬氏距離計算
     - 異常檢測結果視覺化

3. **train TCN-AE model.ipynb**：
   - 模型訓練腳本
   - 輸入：`output_original_data.csv`（或自定義的新資料檔案）
   - 輸出：
     - `tcn_20_model.h5`（訓練好的模型）
     - `tcnae_minmax_scaler.pkl`（數據標準化工具）
     - `tcn_daily_trade_info.csv`（壓縮後的特徵）

4. **TCN-AE predict data.ipynb**：
   - 預測腳本
   - 輸入：
     - `output_original_data.csv`（或自定義的新資料檔案）
     - `tcn_20_model.h5`
     - `tcnae_minmax_scaler.pkl`
   - 輸出：`tcnae_predict_daily_trade_info.csv`

5. **tcn_20_model.h5**：
   - 預訓練模型檔案
   - 可直接用於預測

6. **tcnae_minmax_scaler.pkl**：
   - 數據標準化的 scaler
   - 用於預處理輸入數據

7. **output_original_data.csv** 或 **new_output_original_data.csv**：
   - 原始訓練數據
   - 包含 117 個特徵的時間序列數據

### Q3: tcnae_minmax_scaler.pkl 是如何產生的？
這個檔案是在 `train TCN-AE model.ipynb` 訓練過程中產生的數據標準化工具。主要用於：
1. 訓練時：對輸入數據進行標準化處理
2. 預測時：確保新數據的標準化方式與訓練數據一致
3. 保證模型預測的準確性
4. 使用相同的數值範圍進行特徵縮放（通常是 0-1 之間）

### Q4: 如何執行整個流程？

1. **執行順序**：
   a. 首先執行 `train TCN-AE model.ipynb`
   b. 然後執行 `TCN-AE predict data.ipynb`

2. **不需要執行的檔案**：
   - `utilities.py`：這是輔助函數庫
   - `tcnae.py`：這是模型定義檔
   - `README.md`：說明文件

### Q5: 如何使用新資料重新訓練？

1. **準備新資料**：
   - 確保新資料格式與 `output_original_data.csv` 相同，需要包含以下欄位：
     - date：日期
     - stock_id：股票代碼
     - CLOSE_O：收盤價
     - 其他技術指標（共 117 個特徵）
   - 資料需要按照 stock_id 和 date 排序

2. **修改訓練程式碼（train TCN-AE model.ipynb）**：
   - **需要修改的部分**：
     - 第 38 行：修改檔案路徑以指向新資料
       ```python
       # 原始代碼
       csv_file_path = 'output_original_data.csv'
       # 修改為新檔案名稱，例如：
       csv_file_path = 'new_output_original_data.csv'
       ```
     - 第 178-179 行：修改輸出檔案名稱，避免覆蓋原始輸出
       ```python
       # 原始代碼
       coding_tcn_20_df.to_csv('tcn_daily_trade_info.csv', index=False)
       # 修改為新檔案名稱，例如：
       coding_tcn_20_df.to_csv('new_tcn_daily_trade_info.csv', index=False)
       ```
     - 模型檔案名稱在 TCNAE 類中是自動生成的，預設為 `tcn_20_model.h5`，如果要避免覆蓋，可以修改 tcnae.py 中的以下部分：
       ```python
       # tcnae.py 文件中的第 176-177 行
       # 原始代碼
       model_filename = 'tcn_{}_model.h5'.format(self.filters_conv1d)
       # 修改為（例如添加時間戳或版本號）
       model_filename = 'tcn_{}_model_v2.h5'.format(self.filters_conv1d)
       ```
     - 同樣地，scaler 檔案也可以修改名稱：
       ```python
       # train TCN-AE model.ipynb 中的第 152 行
       # 原始代碼
       joblib.dump(ss, 'tcnae_minmax_scaler.pkl')
       # 修改為
       joblib.dump(ss, 'tcnae_minmax_scaler_v2.pkl')
       ```

3. **修改預測程式碼（TCN-AE predict data.ipynb）**：
   - **需要修改的部分**：
     - 第 8-9 行：修改輸入檔案路徑
       ```python
       # 原始代碼
       csv_file_path = 'output_original_data.csv'
       # 修改為新檔案路徑
       csv_file_path = 'new_output_original_data.csv'
       ```
     - 第 27-28 行：如果訓練時修改了 scaler 名稱，這裡也要相應修改
       ```python
       # 原始代碼
       ss_loaded = joblib.load('tcnae_minmax_scaler.pkl')
       # 修改為新檔案名稱
       ss_loaded = joblib.load('tcnae_minmax_scaler_v2.pkl')
       ```
     - 第 40-41 行：如果訓練時修改了模型名稱，這裡也要相應修改
       ```python
       # 原始代碼
       tcn_ae_20.model.load_weights("tcn_20_model.h5")
       # 修改為新檔案名稱
       tcn_ae_20.model.load_weights("tcn_20_model_v2.h5")
       ```
     - 第 81-82 行：修改輸出檔案名稱
       ```python
       # 原始代碼
       coding_tcn_20_df.to_csv('tcnae_predict_daily_trade_info.csv', index=False)
       # 修改為新檔案名稱
       coding_tcn_20_df.to_csv('new_tcnae_predict_daily_trade_info.csv', index=False)
       ```

4. **注意事項**：
   - 訓練資料的列數（row 數）可以改變，程式會根據資料自動處理
   - 但特徵的數量和順序（column 數）必須與原始資料完全相同（共 117 個特徵）
   - 如果要訓練多個版本，建議：
     1. 建立不同的工作目錄
     2. 複製整個 TCN-AE 資料夾到新目錄
     3. 在新目錄中修改程式碼和檔案名稱
   - 如果數據結構（特徵數量）變化，需要修改 tcnae.py 中的 `ts_dimension` 參數

### Q6: 程式間的依賴關係是什麼？

1. **TCN-AE 資料夾內的依賴**：
   - `tcnae.py` 引用了 `utilities.py` 中的函數
   - `train TCN-AE model.ipynb` 引用了 `tcnae.py` 中的 TCNAE 類別
   - `TCN-AE predict data.ipynb` 引用了 `tcnae.py` 中的 TCNAE 類別

2. **檔案輸出與使用關係**：
   - `train TCN-AE model.ipynb` 產生：
     - `tcn_20_model.h5`（或自定義名稱）：被 `TCN-AE predict data.ipynb` 使用
     - `tcnae_minmax_scaler.pkl`（或自定義名稱）：被 `TCN-AE predict data.ipynb` 使用
     - `tcn_daily_trade_info.csv`（或自定義名稱）：被 Trading Agent 資料夾中的 `train trade agent.ipynb` 使用

   - `TCN-AE predict data.ipynb` 產生：
     - `tcnae_predict_daily_trade_info.csv`（或自定義名稱）：可用於預測新數據

3. **跨資料夾的依賴**：
   - Trading Agent 資料夾中的 `train trade agent.ipynb` 使用了 TCN-AE 的輸出：
     - 讀取 `tcn_daily_trade_info.csv` 進行後續的交易代理訓練
     - 該檔案也被壓縮為 `tcn_daily_trade_info.7z` 以節省空間

### 注意事項：
1. 確保新資料的格式（特徵數量和順序）完全符合原始資料格式
2. 使用不同的檔案名稱來避免覆蓋原始訓練結果
3. 在訓練新模型時，建議保留原始模型作為備份
4. 使用有意義的後綴或版本號來區分不同版本的模型和輸出
5. 注意跨資料夾的檔案依賴關係，確保檔案路徑正確 