# QAT 排水實驗全數據考古與 η 假說裁決

## 摘要

- E42b-r128 與 E43-w 不是同一種崩潰：前者是 rank 增加造成的路徑過擬；後者具有明顯的離散碼 churn 與測量算子衝突。
- r128 臂雖比健康臂低 8.50pp IFEval，出口 gdisc 反而低 10.1%、最終碼翻轉率也低 10.8%；一般 proxy 看不見它。
- E43-w 的最終碼翻轉率、每列末事件 level、末事件最大 absorb 中位數分別為健康臂的 3.78×、3.12×、3.09×。
- E43-w 並非形式上「全乾淨」：gdisc 最大 0.173146，微幅超過預註冊門檻 0.1724；不足以解釋 14.42pp，但應誠實記為 R6 邊界失敗。
- 512 列同時把每窗 LoRA optimizer steps 從 2,580 增至 5,246（2.03×）；目前 E43 並非純粹的 calib-size 實驗。
- η 的分母漏掉 code logits、θ 等可訓自由度，而且 LoRA 定案後被刪除、不進 export；它既不是總訓練自由度比，也不是部署容量比。
- 裁決：η 可作實驗標籤，但不是統一因果框架；較合適的是「任務對齊有效資訊／有效自由度＋離散盆地穩定性」。
- 明早若無加權臂健康，直接做 r128@512；若同崩，先做同一 E43 資產的 r64@256，而不是再加 rank、W_max 或權重。

## 歸因分析

### 1. 統計口徑

以下數字由 31 個滑窗聚合：

- `level/row`：每個事件 `level` 除以實際 calib 列數，消除 E43 列數翻倍的直接尺度效應。
- 「末 absorb」：每窗最後事件中各 target 的 `absorb` 最大值，再取 31 窗中位數。
- `event flips` 是 solver 累積翻動量，可能來回翻；`t_flip_rate_vs_p0` 才是定案後與起點的淨差異。
- loss 表中的「末值」包含 72 個 drain epoch 加 10 soft-tail、4 STE-tail。

| 指標 | 健康：r64@256 | 崩潰一：r128@256 | 崩潰二：r64@512、ifshape×2 |
|---|---:|---:|---:|
| 官方 IFEval / HE / GSM8K | 68.02 / 44.51 / 77.48 | 59.52 / 32.32 / 74.22 | 53.60 / 33.54 / 66.11 |
| gdisc，31 窗平均 / 最大 | 0.09323 / 0.15199 | 0.08386 / 0.15310 | 0.10967 / **0.17315** |
| 定案淨翻碼率 | 0.6064% | 0.5407% | **2.2948%** |
| event flips 累計 | 8.55M | 11.43M | **62.91M** |
| 末事件 level/row，平均 | 0.788M | 1.515M | **2.457M** |
| 末事件最大 absorb，中位數 | 0.1070 | 0.1336 | **0.3303** |
| tail loss，跨窗平均 | 1.162e-4 | 1.351e-4 | **2.338e-4** |
| holdout 末端 gap，最大 | 1.2e-5 | 6.5e-5 | 9.2e-5 |
| holdout 從窗內最低點回升 >2× | 20/31 | 21/31 | **31/31** |
| e2e 後 val CE | 1.6232 | **1.6152** | 1.6473 |
| e2e 定案碼翻轉率 | 8e-6 | 8e-6 | 7e-6 |

逐窗原始來源：[健康 r64]($REPO/evidence/p1_grouping/corkscrew/train_st42r.json)、[r128]($REPO/evidence/p1_grouping/corkscrew/train_st42c.json)、[E43-w]($REPO/evidence/p1_grouping/corkscrew/train_st43w.json)；e2e 來源：[r64]($REPO/evidence/p1_grouping/corkscrew/train_st42r_e2e.json)、[r128]($REPO/evidence/p1_grouping/corkscrew/train_st42c_e2e.json)、[E43-w]($REPO/evidence/p1_grouping/corkscrew/train_st43w_e2e.json)。官方分數來自各 `a1eval/{ifeval,humaneval,gsm8k}_e42b_{r1,r2}/summary.json` 與 `*_e43_w/summary.json`，例如 [IFEval-r1]($REPO/evidence/p1_grouping/a1eval/ifeval_e42b_r1/summary.json)、[IFEval-r2]($REPO/evidence/p1_grouping/a1eval/ifeval_e42b_r2/summary.json)、[IFEval-E43w]($REPO/evidence/p1_grouping/a1eval/ifeval_e43_w/summary.json)。

### 2. 崩潰一：r128@256 是「路徑容量反噬」，不是資料品質改變

這是目前最乾淨的因果對照：r64 與 r128 使用相同起點、res42、同一 `calib_drain_e42b.pt`、相同 W_max/gsteps；唯一主變因是 drain LoRA rank。[預註冊]($REPO/p2_anchor/wringer/prereg_e42b.json) 與 [既有裁決]($REPO/p2_anchor/wringer/verdict_e42b.json) 均如此定義。

支持「有限測量下路徑過擬」的證據：

1. 第 0 窗第一個 comp event 兩臂逐值相同；第一次 optimizer epoch 後才在 event 1 分岔。亦即差異不是水庫、calib 或初始 H，而是 rank 改變後的更新軌跡。
2. r128 的官方三科全跌，但 gdisc 平均反而低 10.1%，淨翻碼率低 10.8%，e2e 後 val CE 還優於健康臂 0.00795。這排除「普通未收斂」或「碼翻太多」作為共同解釋。
3. r128 的末事件 level 平均卻是健康臂 1.92×，末 absorb 最大值中位數高 24.9%。局部軟硬出口看似更近，未吸收的 H-幾何殘量卻更大。
4. e2e 階段三臂實際都回到預設 r64，並且只翻約 7–8e-6 的碼；r128 排水造成的差異已被固化進 `T1/T2/a0`，1200-step e2e 無法改寫。

機制上，LoRA、code logit 與 θ 共同參與 optimizer，但定案只保存 `T1/T2/a0`、隨後刪除 LoRA parametrization：[訓練參數]($REPO/p2_anchor/wringer/train_res.py:252)、[定案與刪除]($REPO/p2_anchor/wringer/train_res.py:403)。因此 r128 提供的是「搜尋路徑自由度」，不是最終模型多 1.22 億參數。較大的搜尋空間能沿 calib 的弱約束方向移動，最後再被不連續的碼／scale extraction 投影到較差部署盆地。

裁決：對第一次崩潰，**「欠定／弱識別反問題 → calib 特異路徑過擬」高度相容**；calib 品質變化與 domain weighting 均可排除。

### 3. 崩潰二：E43-w 是測量算子扭曲，加上優化劑量翻倍

E42b 與 E43 的 quota 完全相同，皆為 ifshape/code/math/agentic = 37.5/31.3/18.8/12.5%；但 E43 使用另一隨機種子與一份新 512-row 資產，因此「size」與「sample identity」尚未拆開。[E42b manifest]($REPO/evidence/p1_grouping/E42B_DRAIN_MANIFEST.json)、[E43 manifest]($REPO/evidence/p1_grouping/E43_DRAIN_MANIFEST.json)。

更關鍵的是，E43 的兩條優化路徑使用不同度量：

- LoRA/code/θ 梯度使用 ifshape×2 的加權 Huber loss：[train_res.py]($REPO/p2_anchor/wringer/train_res.py:268)。
- `HAccum → greedy_comp_grid` 的離散 solver 仍由未加權 calib 激活建立 H：[train_res.py]($REPO/p2_anchor/wringer/train_res.py:311)。
- `dom-exit` 最終透過 `wait >= W_max` 的 OR 條件被強制放行；實際 31 窗都打滿 72 epoch，所以它沒有改變排程長度：[退出條件]($REPO/p2_anchor/wringer/train_res.py:348)。

分域 loss 證明加權確實生效：每窗第一事件 ifshape loss 相對其他三域平均為 1.049×；最後事件降至 0.895×，31 窗中 27 窗較低。這不是「權重沒有接上」，而是**成功解了被扭曲的局部問題，卻傷害真正任務**。[E43-w 日誌]($REPO/evidence/p1_grouping/corkscrew/train_st43w.json)

同時，512 列還改變了優化劑量。程式每個 epoch 對所有 train batches 各做一次 Adam step；256 列為 30 steps/epoch，512 列為 61：

\[
30\times(72+10+4)=2{,}580,\qquad
61\times86=5{,}246
\]

亦即每窗 2.03× optimizer steps，而 LR 不變。[batch 與 holdout 切法]($REPO/p2_anchor/wringer/train_res.py:179)、[每 batch 一次 step]($REPO/p2_anchor/wringer/train_res.py:275)

這與軌跡吻合：第 0 窗首事件按列正規化後，E43 與健康臂僅差約 1.7%；到同窗末事件卻差 17.8×。跨全部窗，末事件 level/row 為 3.12×、event flips 7.36×、淨翻碼率 3.78×。差異是在反覆更新中生成，不是 512 列使 obj 單純乘二。

此外，預註冊 R6 要求 gdisc≤0.1724，但 E43-w window 0 為 0.173146，超出 0.000746。[E43 prereg]($REPO/p2_anchor/wringer/prereg_e43.json)、[E43-w 日誌]($REPO/evidence/p1_grouping/corkscrew/train_st43w.json)。這只是邊界違規，不可能獨自解釋 −14.42pp；但「所有過程指標完全乾淨」並不精確。

裁決：第二次崩潰目前最相容於 **「加權造成 gradient/solver model mismatch」＋「固定 epoch 導致更新劑量翻倍」**。新 calib seed 的品質仍是未排除共犯，必須等無加權臂或同資產 256-row 對照。

### 4. 為何 holdout 沒報警

`loss_holdout` 不是獨立資料集，而是同一捕獲激活 list 的最後幾個 batch；沒有先 shuffle，也沒有 source-disjoint：

- E42b 實際 holdout 16 rows，域組成 code 8 / ifshape 5 / math 3，完全沒有 agentic。
- E43 holdout 24 rows，ifshape 14，占 58.3%，高於全體 37.5%。

程式證據見 [train_res.py]($REPO/p2_anchor/wringer/train_res.py:179)，域序列見 [E42b domains]($REPO/evidence/p1_grouping/calib_drain_e42b.domains.json) 與 [E43 domains]($REPO/evidence/p1_grouping/calib_drain_e43.domains.json)。因此小 holdout gap 只能證明同資產內插，不能測出 task shift、seed shift 或 calib-specific null directions。

### 5. 候選機制總裁決

| 機制 | r128@256 | E43-w | 裁決 |
|---|---|---|---|
| 欠定／弱識別反問題 | 強支持 | 部分支持 | 是共同上位框架，但「原始 activation 數量不足」說法太粗 |
| 過擬 calib 特異結構 | 強支持 | 中等支持 | r128 的最佳近因；E43 需無加權臂確認 |
| 加權＝測量算子扭曲 | 不適用 | 強支持 | E43-w 的首要特異近因 |
| calib 成分／品質變化 | 排除 | 未排除 | 不可能統一解釋兩次崩潰 |
| 普通未收斂或 gdisc 過大 | 反證 | 弱支持 | 無法解釋 r128；E43 僅有小幅 R6 越界 |
| W_max 不足 | 證據混合 | 證據混合 | 全窗 hit cap，但更多更新也可能加重過擬，不能據此直接加碼 |

## 文獻對勘

### Gardner α 與統計力學學習曲線

Gardner 的容量比是 \(\alpha=P/N\)，其中 \(P\) 是隨機 patterns、\(N\) 是獨立可調權重；臨界容量依 pattern correlation 改變，並非只由 scalar count 決定。[Gardner, 1988](https://doi.org/10.1088/0305-4470/21/1/030)

Seung、Sompolinsky、Tishby 更直接指出：平滑、可實現系統常有漸進學習曲線；非平滑系統可有不連續相變，但在 unrealizable rules 下，簡單近似可能失效。[Statistical Mechanics of Learning from Examples, 1992](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.45.6056)

本案同時違反幾個關鍵映射：

- 2048×2560 個 activation scalars 高度相關，不能當作 \(P\) 個獨立 patterns。
- 最終三元模型未必能實現 reservoir teacher，屬 unrealizable／misspecified regime。
- 31 個窗的輸入分佈會隨先前窗定案而變，不是固定 i.i.d. disorder。
- extraction 是不連續操作，且 LoRA 搜尋自由度最後被刪除。

因此文獻容許「懸崖」，但不支持把本案的原始 η 數值直接當成 Gardner α。

### Double descent

Double descent 指模型容量穿越 interpolation threshold 時風險先升，進入高度過參數化區後再下降。[Belkin et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6689936/)

它不直接預測本案：

1. 按原始 η，三臂皆大於 1，若每個 activation value 真是獨立限制，根本不在 interpolation threshold 附近。
2. LoRA 的過參數化解不被保留；定案時投影到三元 code/scale，可能摧毀「benign interpolation」選出的連續解。
3. r64→r128 只看到第一段惡化，沒有更大 rank 的第二次下降證據。

所以目前沒有 double descent 證據。更合理的形狀是受碼盆地邊界控制的分段平滑曲線與離散懸崖，而非可平移的 universal double-descent curve。

### 壓縮感知

\(m=O(k\log(n/k))\) 類結果要求已知稀疏／可壓縮結構，以及滿足 RIP、incoherence 或類似性質的近線性測量算子；噪聲條件也必須受控。[Candès, Romberg & Tao, 2006](https://arxiv.org/abs/math/0503066)

本案的 reservoir delta 雖是低 rank，卻不是在固定已知 basis 下的 \(k\)-sparse signal；測量又會隨模型、窗與離散碼變動。η 沒有 \(k\)、H 的有效秩、最小特徵值或 condition number，因此不能由壓縮感知推出「η≈11 足夠」。

### PTQ/QAT 校準集

- GPTQ 使用 128 個隨機 C4、每段 2048 tokens；但其目標是二階 one-shot/block-wise quantization，而非訓練 1.2 億個 ephemeral LoRA 參數。[GPTQ](https://ar5iv.labs.arxiv.org/html/2210.17323)
- AdaRound 使用少量無標籤 calib，典型為 1024 images，以逐層局部 reconstruction 避免全模型自由度。[AdaRound](https://proceedings.mlr.press/v119/nagel20a.html)
- BRECQ 明確報告 stage/net-wise reconstruction 的 validation generalization 較差，並把「全網在 1024 calib samples 上容易過擬」列為採用 block-wise 重建的理由。[BRECQ](https://openreview.net/pdf?id=POWv6hDd9XH)
- OmniQuant 的 LLaMA-7B 消融顯示 16–256 段迅速飽和：W3A16 幾乎固定於 6.46–6.48 PPL；W4A4 在 128 samples 為 11.23，增加至 256 反而成為 11.41。作者也直接檢查 calibration-distribution overfit，而不是假定越多越好。[OmniQuant Table A10–A11](https://ar5iv.labs.arxiv.org/html/2308.13137)
- LLM-QAT 需要約 100k 自生成 samples，並警告狹窄或偏離 pretraining distribution 的 QAT data 會傷害 zero-shot generalization；資料形狀而不只是數量是一級變數。[LLM-QAT](https://arxiv.org/abs/2305.17888)

共同訊息是：calib size 的效果依 reconstruction granularity、可訓自由度、bit regime 與分佈而異；不存在可直接移植的 samples/parameters 常數。

### KD 資料規模

GKD 把自迴歸 KD 視為 imitation learning，指出固定 teacher-forced 資料會產生 train–inference distribution mismatch；學生表達力不足時，loss 與資料採樣策略都會改變結果。[GKD](https://arxiv.org/abs/2306.13649)

更一般的 KD 研究也顯示，teacher/student predictive-distribution 的匹配高度依賴蒸餾資料集與 temperature；學生即使理論容量足夠，也不保證真正匹配老師。[Stanton et al., 2021](https://research.google/pubs/does-knowledge-distillation-really-work/) 最新 distillation scaling law 則把 student size、teacher loss、資料／compute 配置共同納入，並觀察 capacity gap，而不是只用資料點除參數量。[Busbridge et al., 2025](https://machinelearning.apple.com/research/distillation-scaling-laws)

這與 repo 自身歷史一致：E39→E42 的主要增益來自資料行為分佈與 teacher trajectory，而不只是 rank；既有回顧見 [LIT_reservoir_review.md]($REPO/p2_anchor/wringer/LIT_reservoir_review.md)。但該回顧當時偏好的 RKL 已被 E40 官方三科反證，因此文獻機制仍須以 repo 官方裁判覆核。[E40 verdict]($REPO/p2_anchor/wringer/verdict_e40.json)

## η 框架裁決

精確算術為：

\[
\eta(64,256)=11.0345,\quad
\eta(128,256)=5.5172,\quad
\eta(64,512)=22.0690,\quad
\eta(128,512)=11.0345
\]

但 η 有五個結構缺陷：

1. **分子不是獨立證據數**：同一 row 內 token、hidden channel、相鄰層激活高度相關。
2. **分母不是總 trainable DOF**：optimizer 還訓練 code logits 與 \(d\theta\)，η 卻只數 LoRA。
3. **分母也不是部署容量**：LoRA 定案即刪除，export 只保留離散碼與 scale。
4. **忽略測量幾何**：相同 η 可由完全不同 domain、H spectrum、權重與 teacher mismatch 組成。
5. **忽略優化劑量**：512 rows 在固定 epoch 下同時令 optimizer steps 約翻倍。

若硬把三個已出分點畫成曲線，它看似在 η≈11 達峰的倒 U；但 η=22 點同時改了 sample identity、weighting 與 update count，這條倒 U 沒有因果資格。

### 替代框架：任務對齊有效資訊／自由度＋離散盆地穩定性

建議把主軸改成：

\[
\eta_{\mathrm{eff}}
=
\frac{
A_{\mathrm{task}}\cdot r_{\mathrm{eff}}(H_{\mathrm{calib}})
}{
df_{\mathrm{trajectory}}
}
\]

其中：

- \(r_{\mathrm{eff}}(H)\)：H/Fisher 的 stable rank、log-det 或有效特徵值數，而非 activation scalar 數。
- \(A_{\mathrm{task}}\)：calib covariance/Fisher 與獨立、task-like holdout 的對齊度。
- \(df_{\mathrm{trajectory}}\)：LoRA、code logits、θ 及其 implicit regularization 共同形成的有效搜尋自由度。
- 另記 optimizer steps per independent row、不同 calib bootstrap 間的 code agreement、level/absorb 增長率，作為離散盆地穩定性。

此框架預測的是：

- 有效秩不足時可能出現 Gardner 式懸崖；
- weighting 或 domain shift 可在 rows 增加時仍讓 \(A_{\mathrm{task}}\) 下降；
- code agreement 跨過盆地邊界時出現非平滑官方分數跳變；
- 不預期單靠提高原始 η 產生單調改善，也不預期標準 double descent。

η 因此可保留為 prereg 的粗 ledger，但不應再作 root-cause 名稱或下一臂的充分設計依據。

## 實驗路徑排序表

成本均含完整 e2e 3.6h 與官方三科約 2h；res42 蓄水實測約 9.6h，排水時間依 [E42/E42b 裁決]($REPO/p2_anchor/wringer/verdict_e42b.json) 與 [E43 prereg]($REPO/p2_anchor/wringer/prereg_e43.json) 估算。

| 排名 | 路徑 | 假說與理論分岔 | GPU 時數 | 決策價值／立場 |
|---:|---|---|---:|---|
| 1 | **Calib 品質審計＋同 E43 資產 r64@256 無加權** | 先 CPU 查 exact/near duplicates、source/prompt 重用、token entropy、teacher loss、H stable rank、域內 leverage；再從 E43 512 固定取分層 256。若 256 健康、512 崩：size/update-dose；若兩者皆崩：E43 sample identity/品質；若皆健康：weighting 定罪。η 預測 256 應接近 E42b；有效資訊框架允許同列數因 seed 品質而異。 | 審計 0；正式臂約 **12.6h** | **最高**。以一臂拆開目前最大的三重混淆。 |
| 2 | **r128@512 無加權，η≈11 復刻** | scalar-η 預測應回到 r64@256 附近；若仍比同資料 r64@512 低 >1pp，η 被直接否證、rank 路徑反噬成立；若回復，支持有效證據不足確實是 r128 首崩原因。 | **18.6h** | **有條件支持**：只在明早 arm2≥66 時立即做；arm2≤60 時延後。 |
| 3 | **自提：cross-fit code consensus drain** | 以兩個獨立 128-row split 各自求碼，只接受方向／碼位一致的 flips，scale 再於聯合集上擬合。欠定過擬理論預測跨 split 不穩定碼會被濾除、官方分升；純容量不足理論預測共識過強、反而欠擬合。 | 兩次 256/2 drain 約14h＋e2e/eval，合計 **約19.6h** | **高**。直接把「可識別性」變成干預，而不只是量測。 |
| 4 | **蓄排一體化／邊蓄邊排** | 讓 reservoir 只沿當前三元碼＋scale 可吸收方向生長，避免先學任意 BA 再有損轉寫。model-mismatch 理論預測 A 天花板可能略低但 A→B retention 大升；純 η 理論若資料比不變則不預期大改。 | res42 fill 9.6＋drain 7–13＋e2e/eval，約 **22–29h/臂** | **中高但昂貴**。是架構性修正，應在辨識當前崩潰後做。 |
| 5 | **W_max 8→16** | 吞吐不足說預測末 level/absorb 下降且官方升；過擬／更新劑量說預測 flips 與官方惡化。E41 strain 曾無可判增益，E42b 加力又是 bundled PASS，證據混合。 | 256 列 drain 約13.2h，加 e2e/eval共 **約18.8h**；512 約30h | **低、反對盲加**。除非改成以獨立 holdout/code-consensus early-stop。 |
| 6 | **ifshape 權重掃描** | 若 arm2 健康，2.0 已造成至少 12.4pp 稅，3.0/4.0 不應再跑；最多測 1.1–1.25。若 arm2同崩，weighting 效應仍未辨識，也應先拆 size/seed。 | 512 每臂 **18.6h**；兩點37.2h | **目前反對**。成功降低 dom_loss 已證明「方向符合設計」不等於官方收益。 |

若唯一目標是盡快超過 72.27，排序應服從明早結果：健康分支先做 r128@512；非健康分支先做 E43 同資產 256，避免再燒一條錯誤 512/128 鏈。

## 臂2 雙分支預案

令明早 r64@512 無加權的 IFEval 為 \(S_2\)。

| 結果 | 意義 | 下一步 |
|---|---|---|
| **健康：\(S_2\ge66\)** | 512 rows 本身沒有造成 E43-w 等級的崩潰；weighting＋域閘的處置效應為 \(53.60-S_2\le-12.40\)pp，測量算子扭曲定罪。若 \(66\le S_2<68.02\)，size 已近飽和而非明顯增益；若 \(S_2>72.27\)，直接交接冠軍並跑 guard。 | 停止所有較大權重掃描；立即做**同一 E43 calib 的 r128@512 無加權**，用 η≈11 對 r64@512 做 rank 單變因裁決。 |
| **同崩：\(S_2\le60\)** | 無加權、η≈22 仍比 r64@256 至少低 8.02pp，簡單「η 越高越安全」被否證。共同嫌疑縮成 E43 sample identity/品質、512 size、以及 2.03× optimizer dose；weighting 是否另有稅須看 \(53.60-S_2\)。 | **不要做 r128@512**。先跑同一 E43 資產的 r64@256 無加權；若健康，接著做 step-matched 512；若也崩，重建／審計 calib seed。 |
| **居中：\(60<S_2<66\)** | 至少兩個機制同時存在：相對 E42b-r64 的共同 E43 損失為 \(S_2-68.02=-8.02\) 至 −2.02pp；weighting 額外損失為 \(53.60-S_2=-12.40\) 至 −6.40pp。weighting 是較大因素，但 size/seed 亦非零。 | 同樣先做**E43 同資產 r64@256 無加權**；不做權重掃描。256 健康則修正 update dose，256 仍低則處理 calib 品質，再決定是否值得 r128@512。 |

最終裁決是：**η 捕捉到 r128@256「資料相對搜尋路徑太弱」的一部分直覺，但它無法統一 E43-w，更不能直接預測官方分數。下一輪應操縱有效資訊、sample identity 與更新劑量，而不是繼續只操縱 rows/rank 的商。**