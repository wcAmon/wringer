# Roadmap:E57 收卷後研究路線(2026-08-29 用戶裁定版)

**戰略裁定(08-29)**:專注三元蓄水線(A₃ 已達 bf16 的 45.6%);冠軍線(st51c_e2e 0.7388)移植撤下不入路徑。產線教師維持 bf16 A1-4B 自蒸餾(on-manifold 前提 E39 + logit-KD tokenizer 一致性);升級走出題槽與終止軸,不碰示範/標註槽。

## 0. 進行中:E57 16k 全鏈(FROZEN @f7dca92 + 兩筆修理 @2da6154/@bd4f2b0)

- 已裁:A₃ 0.4558(gate PASS,+0.1001 vs A₂ 本帶史上最大天花板增量;he +7.32pp=內容軸定罪,H_window PASS,rank 非瓶頸)。
- 待收:B₃(drain b4 運行中)→ e2e → B₄ 雙跑(~05:30-06:00 08-30)。
- **B₃ ρ 是路線分流器**:ρ≈E54 帶(~0.91)→ 瓶頸=天花板 → 主線 1→2→3;ρ 崩塌(E38 型)→ 碼軸 contingency(§5)升首選。

## 1. 收卷即做:逐位置 KL 判別(用戶已核准;零訓練)

16k holdout 上量 st54r(8k 教)vs st57r(16k 教)的逐位置 KL 曲線。
- 判讀:8k 教在 ~8k 位置後上翹、16k 教持平 → **長窗適配軸**(用戶假說:短窗碼分配傾向眼前學習,FLA 遞歸狀態長程轉換失準)獨立成立;全位置均勻 → 內容軸(閉合示範)已足以解釋。
- 產出:兩曲線 + 上翹位置/斜率量化 → 決定 E58 語料的窗長配比(16k 純投 vs 混窗)。

## 2. 產線升級 v3(bleachers;三槽位風險分級)

| 槽位 | 動作 | 風險 |
|------|------|------|
| 出題槽 | **Qwen3.8-27B-NVFP4 本機 vLLM 出題**(取代雲端 Max;agentic/code 題池擴充優先)。27B 只產題目文字,4B 自解不變 → tokenizer/manifold 閘自動免除 | 低(已核准,模型下載中) |
| 終止軸 | ①產線加 stop 字串(EOS 饑荒根治,cal4 以來 76.7% 頂格通性)②既有語料閉列「截答補 EOS」手術(零 GPU) | 低 |
| 示範/標註槽 | **不動**:bf16 A1-4B 自生軌跡 + 自供 KD logits(E39 on-manifold 封冠、H_richcap FAIL) | — |

- 前置閘(收卷後做,不與訓練搶 GPU):27B NVFP4 vLLM serve 煙測(吞吐/顯存/思考解析);出題 prompt 模板沿 bleachers 綱領(出題自解+程序驗證)。
- 產能參考:16k 產線 ~5900 列/8.5h(cal10 實測);27B 出題為 CPU 輕 GPU 重,與 4B 自解分時段跑。

### 2b. 「還原考場行為」四落實點(08-29 勘查:gen_cal4.sft() 偽模板定罪候選)

發現:產線 prompt 包裝 = 手寫 `<|user|>\n…\n<|assistant|>\n`,**非** Qwen3.5 真 chat 模板;cal4→cal10 全系譜受累。考場(a1eval)= chat completions + apply_chat_template + temp1.0/top_p.95/top_k20/min_p0/presence1.5 + `<|im_end|>` 自動停。

1. **模板還原**:`tok.apply_chat_template(..., add_generation_prompt=True)` 取代 sft();EOS 饑荒頭號嫌疑(終止反射=chat 框架條件反射,偽模板不觸發;23.3% 自然停=魯棒殘餘)。
2. **採樣對齊**:gen_one 補 presence 1.5/top_k 20/min_p 0(照抄 ifeval_runner.SAMPLING);採樣只改軌跡抽樣不改 KD 條件分佈,軌跡落在裁判抽樣流形=嚴格版 on-manifold。
3. **停止+終止訊號入列**:completions 加 stop `<|im_end|>`;vLLM 回文不含 stop 字串 → tokenize 後**手動補 im_end token 於列尾**(否則換方式重演饑荒);finish_reason==length 列誠實照記不補假 EOS,入手術/淘汰池。
4. **儀器閘+對照臂**:產前 100-200 列探針量分科自然終止率(manifest 記 finish_reason 分佈);模板更換=語料 régime 更換,E58 prereg 設對照臂(考場行為語料 vs cal10 式),官方三科裁,不假設必贏。

附註:偽模板同時是 P1「學生思考不終止」失律的候選遠因(教材無終止示範+框架失配雙因);對照臂勝敗兩頭都是乾淨知識。

## 3. E58:語料規模化 × 梯級複利(prereg DRAFT → 呈議 → FROZEN 才點火)

- 依據:E54 歸因 token 饑荒(現 ~153M 有效 vs BitDistill 10B);A₂→A₃ 證明語料品質放大劑量效率;co-scaling 法則(每輪 tokens ≳1-2× 旁路 244M 參數)。
- 骨架:v3 產線擴產(新題池 × 16k × 手術)→ cycle-3 fill(warm-start res57)→ drain → 斜率量測;判準與劑量在 DRAFT 議定。
- 開放項(DRAFT 時裁):窗長配比(§1 判別結果餵入)、epochs、是否隔輪 drain(排水成本攤提)。

## 4. E55:on-policy 矯正段(DRAFT @fe9e14e,#70)

- 入場點:梯級複利收斂後的最強窗態(非固定 E57 梯尾)。
- 機制:學生快照自生軌跡 @eval 採樣參數 + bf16 教師標註;唯一能照亮學生自有失律區(迴圈/空核)的形態;GKD 文獻正錨。

## 5. Contingency:碼軸(僅當 B₃ ρ 崩塌或梯級斜率死亡)

- E37 式局部重解碼 on 16k 料 / 駐留窗 curriculum / r256(帶 E38 排水稅警告,須在餵飽條件下開庭)。

## 紀律

新臂一律 prereg DRAFT → 呈議 → 用戶核准 FROZEN → 點火;凍結後偏離需核准+誠實入卷;官方三科唯一裁判;.pt/.npz 不入 git。
