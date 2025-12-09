# tpbl_2425_rapm
本專案主要目的為：
利用台灣職籃 TPBL 逐球（play-by-play）資料，轉換為 stint 等級資料後，建立 Regularized Adjusted Plus-Minus（RAPM）模型，以估計球員對淨分差的真實貢獻。

專案流程包含三部分程式碼與兩份資料集：
	1.	tpbl_pbp2stints.py — TPBL 逐球資料轉換為 Stints
	2.	stints_TPBL_2024_9-134.csv — 處理後的整季 stints 資料
	3.	rapm.py — 使用 Ridge Regression與回歸分析 計算球員 RAPM
	4.	rapm_TPBL_24-25.csv — 計算出的 RAPM 結果

⸻

# 專案整體流程

本專案的分析步驟如下：

1. 取得 TPBL 官方逐球資料 PBP 並轉換為 Stints（tpbl_pbp2stints.py）

資料來源為 TPBL 官方網站。
逐球事件包含：進球、犯規、換人、暫停等。
為進行 RAPM 模型，需要將比賽切割為「所有球員組合保持不變的時間段」（stint）。

Stint 是 APM 的基本觀察單位，每一筆資料包含：
	•	場上 10 位球員（各 5 位）
	•	該 stint 的淨分差（dependent variable）
	•	stints 的起訖時間與秒數
	•	球隊資訊與比賽 ID

程式功能包含：
	•	從官網api抓取每一場比賽逐球資料
	•   用逐球資料計算box score
	•   從官網api抓取每一場比賽的box score
	•   檢查用PBP計算的box score是否與官方一致（檢查pbp是否有誤用）
	•   切割逐球資料成回合
	•	追蹤場上球員名單的變化
	•	每次有人進出場就切割新的 stint
	•	計算進攻方與防守方的淨分差



輸出檔案：stints_TPBL_2024_9-134.csv (9-134為官方api的比賽id)

⸻

3. 建立 RAPM 模型（rapm.py）

RAPM 使用 ridge regression（L2 regularization）來解決 APM 中的多重共線性問題。

模型形式：

Y = X\beta + \epsilon

其中：
	•	Y = stint 的淨分差
	•	X = 球員出場矩陣（上場為 +1、對手為 −1）
	•	β = 球員的影響值（RAPM）

程式功能包含：

  建立球員向量 Encoding
	•	每位球員分配一個向量位置
	•	上場球員 = +1
	•	對手球員 = −1
	•	未上場 = 0

  執行 Ridge Regression
	•	λ值可自行調整（避免模型不穩定）
	•	計算RAPM / O-RAPM / D-RAPM

  數據清理與維度檢查
	•	確認所有球員都在矩陣中
	•	處理稀疏矩陣

  輸出結果 CSV
	•	RAPM（影響每 100 poss 的貢獻）
	•	CV結果

輸出檔案：rapm_TPBL_24-25.csv, rapm_TPBL_24-25_cv_reault.csv

⸻


