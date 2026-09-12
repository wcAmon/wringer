# 水庫機制文獻對勘與優化路線(2026-08-13)

背景:E37/E38 已實測「旁路蓄水(fill)→ 逐窗排水(drain)→ e2e 拋光」機制。
E37 封冠(ρ=0.969)、E38 排水稅現形(ρ₂=0.485,垃圾水+轉寫稅淨破壞 −2.78pp)。
本文件把六條文獻錨點對到機制的六個部位,產出優化排序。
用戶定調:**資料與模型所在位置的關聯性是一級設計變數;排水=強力全面對齊動作;
teacher 生成強拉學生向老師,老師飄移則學生受害。**

## 一、文獻對勘

### 1. LLM-QAT(arXiv 2305.17888)——「準備問題不準備課文」的直接先例
老師自生資料做 QAT 蒸餾:「data generated from the pretrained model…better
preserves the original output distribution」。與 cal4 設計逐字同構(prompt bank
+老師軌跡+全詞表 KD)。差異:LLM-QAT 從零 prompt 自由生成,我們用三域
prompt bank 對位考科行為——E38 分科解耦定律(語料價值=行為分佈距離)是對
LLM-QAT 的行為級精化。

### 2. GKD(arXiv 2306.13649)——divergence 選擇與 student-policy
兩個可用件:
- **train-inference mismatch**:teacher-forcing 訓練 vs 學生自迴歸推理的
  誤差級聯;解法=學生自生軌跡+老師在其上給校正(已列 E40 GKD 升級檔,
  快照重生成落地)。
- **divergence 彈性**:「alternative loss functions…useful when the student
  lacks the expressivity to mimic the teacher」——3元 學生嚴重欠容量,正是
  這個 régime。前向 KL(現行 kd_loss)是 mode-covering:逼欠容量學生攤薄
  機率去罩住老師整個分佈=浪費容量在學不會的模式上。**反向 KL/JSD(mode-
  seeking)讓學生集中容量學老師的主模式**——低元恢復的正確損失形狀,
  值得單變因臂(O4)。

### 3. Teacher Hacking(arXiv 2502.02671,ICML 2025)——用戶「老師飄移」顧慮的正式名字
固定離線資料集蒸餾會讓學生**利用老師的缺陷**(老師只是真分佈的不完美代理);
偵測訊號=優化偏離多項式收斂律;解法=線上生成或高多樣性離線資料。
對勘我們:E38-A1 的「垃圾水」正是 teacher-hacking 的權重空間版——榨乾語料上
的殘餘 teacher-student 分歧=老師的噪聲成分,學生盡職對齊之,官方能力零增益。
E39 已內建三個緩解(同 prompt 多採樣、bank 超量低重用、每輪可重生成);
可加:**KD 曲線對多項式律的偏離監測**(monitor-only)作為 hacking 早警(O5)。

### 4. 模型崩塌/Accumulate-not-Replace(arXiv 2404.01413)——迭代自生的長期風險
遞迴訓練在自生資料上造成分佈窄化(早期:誤差累積;晚期:低頻事件永久消失);
**accumulate(真實資料+歷代合成資料並存)可證避免崩塌,replace 必崩**。
對勘:水庫路線天然是迭代式(E37→E38→E39…),若每輪語料全換老師自生,
就是 replace 模式。處置:cal4 保留 ~10% 人寫種子 token;**後續輪次語料混入
自然語料列(accumulate),不整體替換**(O5)。

### 5. QA-LoRA(arXiv 2309.14717)——轉寫稅歸零的結構解
「fine-tuning 後 LLM 與輔助權重自然合併入量化模型,**無精度損失**」——靠
group-wise 算子:量化自由度升(逐組)、適配自由度降(組共享),使 adapter
恰好落在量化可表示空間,合併=精確重參數化,無需 PTQ。
對勘:E38 的 ε≈3pp 排水稅正是 QA-LoRA 消滅的那種合併損失。我們的排水是
「任意稠密 ΔW → 離散碼+載體」的有損轉寫;若把水庫結構約束到「逐組尺度
空間」(水只能沿 α/組尺度軸流動),排水變成**建構上精確**的重參數化,
稅=0。代價:蓄水表達力下降(A 天花板可能變低)。這是水庫 v2 的核心
候選重構(O3),值得一條 A/B 鏈裁「表達力損失 vs 轉寫稅歸零」的淨值。

### 6. DARE/TIES(arXiv 2311.03099 / 2306.01708)——排水前的垃圾水過濾
Delta 參數(=我們的水 BA)**可隨機丟棄高達 90% 並重縮放而不損性能**;
TIES 用幅度 trim+符號投票消除合併干擾。
對勘:排水前對水庫 ΔW 做 DARE drop+rescale 或 TIES trim(CPU 上幾秒),
把「能力訊號(少數大成分)」與「teacher 噪聲(大量小成分)」分離,
訊號/稅比直接上升。**最便宜的排水稅緩解**,一個 flag 即可 A/B(O2)。

## 二、優化排序(對應實驗階梯)

| # | 優化 | 成本 | 裁什麼 | 排程 |
|---|------|------|--------|------|
| O1 | 排空對照:B=0 水庫全鏈 | ~3.5h | 轉寫稅是否與水內容無關(切開排水器損耗 vs 垃圾水毒性) | E40a |
| O2 | DARE/TIES 水過濾後排水 | CPU 秒級+一條鏈 | 訊號/稅比提升幅度 | E40b |
| O3 | QA-LoRA 式組尺度水庫(精確合併) | 重構 stq/res_fill | 表達力損失 vs 稅歸零淨值 | E41 候選 |
| O4 | 反向 KL/JSD fill | 改 kd_loss 一行+一臂 | 欠容量學生的 divergence 形狀 | E40/41 插臂 |
| O5 | teacher-hacking 偵測(KD 收斂律偏離,monitor-only)+ accumulate 混料紀律 | 零 GPU | 迭代輪次的長期健康 | 即刻入紀律 |
| O6 | GKD student-policy(快照重生成) | 每輪 +40m 生成 | train-inference mismatch | E40+ |
| O7 | 排水損失加基座錨定項(防冠軍態磨損) | 改 train_res 損失 | 轉寫稅落點控制 | 若 O1 證稅在排水器 |

## 三、E39 已內建的對應(2026-08-13 FROZEN @086b21e)
on-manifold 三域 prompt bank(#1)、同 prompt 多採樣+bank 超量(#3 緩解)、
~10% 人寫種子(#4 部分)、排水 calib 上流形(#1 延伸)、閘門+冠軍資產隔離
(排水稅風險止損)。E39 的分科 ρ 剖面是 O1–O3 排序的決策輸入。

## Sources
- [QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2309.14717)
- [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes(GKD)](https://arxiv.org/abs/2306.13649)
- [On Teacher Hacking in Language Model Distillation](https://arxiv.org/abs/2502.02671)
- [Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data](https://arxiv.org/abs/2404.01413)
- [LLM-QAT: Data-Free Quantization Aware Training for Large Language Models](https://arxiv.org/pdf/2305.17888)
- [TRL GKDTrainer(GKD 實作參考)](https://github.com/huggingface/trl/blob/main/docs/source/gkd_trainer.md)
- [Model Merging: TIES/DARE 綜述](https://arxiv.org/html/2503.08998v1)
