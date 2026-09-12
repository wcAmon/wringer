# ROADMAP post-E69(用戶 2026-09-11 21:3x「同意 照這個路線」)

目標:Wringer 成果先以 Hugging Face 社群文章 + 模型發布,再補學術基線與第二模型後出技術報告。

## 已收(W2 鏈 09-12 全收:p3b_w2 comp 0.9468 新冠軍;p3a_w comp 0.9282 @2.571 ρ 0.846;E69 封關)
- W2 鏈(chain_e69_w2.sh):p3b_w 二輪蓄排 → p3b_w2(A₂/B₂/官方,~09-12 10:30)→ p3a 蓄排 → p3a_w(含官方,~09-12 深夜)。判讀:B₂ − 84.15 是否超噪音帶(he 一題 0.61 pp,comp ±0.016);p3a_w 的 ρ 與 2.571 帳面 comp。

## 第一階段:文章 + 模型(W2 收後)
1. **污染檢查** ✅ @afe16d8/388041d(evidence/p1_grouping/corkscrew/contamination_e69.md):GSM8K/IFEval 乾淨;HumanEval S25=10 題(bank 近逐字重現,calib 繼承 4 題);去 S25 he:p3b_w 84.42 vs 全 84.15(不抬分);文章/模型卡同報兩者;第二模型前先去污染題庫。
2. **打包驗證** ✅ @afe16d8/388041d(pack_p3b_w.md;容器 data/wringer_p3b_w.safetensors 1.102 GiB = 2.6553 b/w;R1/R2=0):**R3 2.96% 權重差 1 ulp**(α 存 fp32、帳面 fp16)→ 待辦 2a:W2 收後從容器材化 export → he 閘 → 官方三科(發布用,~3.5 h,需核准);2b:quantize 加 α fp16 投影(W2 收後改)。整 LM bpw:embed bf16 4.69 / int8 3.48 / Q4_K 級 2.95;主數字仍 body 2.655。
3. **目錄改名** p2_anchor/wringer → wringer(git mv + import shim,舊路徑留一版)。
4. **HF 文章**:以 RETRO_wringer.md 為底(敘事版),附 §0 表、配方、死路清單;**發布模型** p3b_w(或 p3b_w2 較高者,p3b_w 留對照),附三科官方分數與材化 bf16 對照。

## 第二階段:技術報告(約 2–3 週機器時間)
5. **學術基線**(同 bpw、同三科、同 A1 官方框架):AQLM / QuIP# / PV-tuning / ParetoQ 式 QAT 中至少兩個,另加 GPTQ/AWQ 2–3 bit 作為弱基線。
6. **第二模型**:純 transformer Qwen3-4B(bf16 起 → P1 八元 g128 → 代價表 → P3 → 蓄排),驗通則性;再加一到兩個標準套件(MMLU 等)。
7. **消融**:α int8 vs fp16、閉合加權、先驗 λ、蓄水步數、擰乾一輪 vs 兩輪。
8. **技術報告**:方法 + 表格 + 消融 + 與 QA-LoRA / LoftQ 的正面比較;敘事留在文章。

## 不做
- 壓到 ~2 bpw(估 2.3–2.4 有機會、2.0 不行;不在本路線)。
- R-alpha(+0.6 在噪音帶內)。
- 三元線(已退為歷史)。

## 規則不變
- 新臂 prereg + 用戶核准才點火;官方三科只跑核准點;只認官方裁判;長鏈 nohup setsid + Monitor;Bash 指令文字不得含程序關鍵字面。
