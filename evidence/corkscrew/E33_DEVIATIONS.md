# E33 執行偏離紀錄

`prereg_e33_7lane_v2.json` @ceff4ff 凍結的是**配方與判準**。以下記錄執行
期間對**治具**(冒煙門、接駁腳本)的改動。每一項都不觸及訓練超參、語料、
判準 Z1–Z6,也不改變任何被裁決的產物。

---

## D1 — 冒煙門 sweep2 路徑不可運行(2026-08-11 13:26,已修)

**症狀**:鏈在 `E33_FAIL:smoke_sweep2` 整條退出。

```
AssertionError: qs_st9smoke 缺 ['...layers.3.self_attn.q_proj', ...]
```

**根因**:治具自相矛盾,非配方問題。`train_st --smoke` 只跑 2 個窗
(`train_st.py:110-111`),因此 `qs_st9smoke` 只落盤 layer00–02;而
sweep2 的 `load_state(state_tag, targets)` 要求 200 個 target linear
全數在場(`state.py:26`)。**用冒煙訓出的殘缺 state 去冒煙「吃完整已訓
state」的路徑,在設計上就不可能成立。**

**修法**:造混合 state `qs_st9hyb` —— layer00–02 指向 `qs_st9smoke`
的訓練輸出,其餘 29 層指向 `qs_g9`(皆用 symlink,不佔額外磁碟)。
state 因此完整,且真的餵進了 `train_st` 產出的 joint α 與已學 T,
冒煙門的原意(驗證 sweep2 新用法接得起來)完整保留。冒煙產物用畢即刪。

**驗收**:`st9smoke` CE 2.024320 flip 0.2174 → `st9s2smoke` CE 2.042851
flip 0.0012 → `E33_SMOKE_PRE_OK`。

**代價**:鏈損失 11 分鐘(13:17 首發 → 13:28 重發)。

**順帶加固**(同次改動,冪等化以支援重發):`calib_selfgen_512.pt` 已存在
則跳過生成;`data/qs_g9` 已有 32 層則跳過量化。重發因此不重跑已完成段。

### 附帶觀察(不裁決)

`st9s2smoke` 翻轉率 0.0012,遠低於 `st9smoke` 的 0.2174。這看似是 Z3
(sweep2 有無增益)的不利先驗,但**冒煙不可用於裁決**,且此處有兩個具體
的失效理由:

1. 冒煙的兩個窗正是 layer00–02,即已訓過的那三層——同一批層再訓一次,
   近零翻轉是必然。
2. 冒煙未帶 `--calib`,用預設語料;而正式 sweep2 換的是 cal2(五源檔案級
   不相交)。**換語料才是 sweep2 的作用機制**,冒煙把這個唯一變因洗掉了。

Z3 一律等正式的逐窗末 loss 中位數比較。

---

## D2 — 四科鏈 lp300 臂裁撤與接駁清場失手(2026-08-11 13:16-13:18)

接駁器在 `FOURSUBJ_ARM mmlu/e32_e2e rc=0` 觸發後,`pkill -f "vllm serve"`
打在一個剛起 7 秒、仍在 CUDA/引擎初始化的 vLLM 上,SIGTERM 被吞,該進程
續活並佔用 83 GiB;接駁器只等 5 分鐘 GPU 就會硬發射,將撞港口與顯存。
以 `kill -TERM` 對 pid 補刀後 GPU 歸零,鏈正常發射。

lp300 臂的四科本就非必要(該臂 IFEval 已有),裁撤以釋放 GPU 給 E33 主線。

**教訓**:清場的驗收條件是資源真的空了(`nvidia-smi --query-compute-apps`),
不是訊號送出去了。`pkill` 應放進等待迴圈反覆送。

---

## D3 — MAIN 7元 臂裁撤(2026-08-11 17:0x,用戶裁定)

**性質**:範圍裁減,非治具修理。用戶裁定 E33 專心做 9元 地基
(Corkscrew 表示的 9元 起點 + 新配方 vs E28 的 Z1 檢定);7元 全部
改由 E34 形變降元執行,從頭 GPTQ 7元 連對照臂都不留——對照意義由
E34 判準重排時另定。若 Z1 敗(新配方 < E28 88.54),先調配方再降元。

**方法**:鏈正在跑(PRE sweep1 26/31),對運行中 bash 腳本做
**等長替換**(7745 bytes 不變):MAIN 段整段換成
`echo E33_MAIN_CANCELLED → echo E33_CHAIN_DONE_9ONLY → exit 0` +
註解填充。因腳本 <8KB 可能已被 bash 整檔緩衝(改檔無效),另佈
detached 看門狗(`e33_main_watchdog.sh`,pid 623070):PRE-2 標記
出現 90s 後若鏈仍活且無 CANCELLED 標記,殺進程群組並清 g7 殘檔。
原腳本備份 `e33_chain.sh.bak_mainfull`。

**對判準的影響**:Z2/Z5/Z4(st7 系)與 second_criterion_humaneval
將讀 N/A——prereg 判準文字不改,裁決時連同本偏離揭露。Z1/Z3/Z4
(st9 系)/Z6 不受影響。

**新終態**:鏈收於 `E33_CHAIN_DONE_9ONLY`(或看門狗標記),預計
~00:45,較原計畫提前 ~8h。
