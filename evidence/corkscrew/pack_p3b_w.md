# Wringer 打包驗證 p3b_w(vs exports/e69_p3b_w)

- 模組 200,本體權重 3,565,158,400;容器 `data/wringer_p3b_w.safetensors` 1.102 GiB = 2.6553 b/w(safetensors 頭與對齊含在內)
- 帳面:碼 2.5882 + α 0.0668 = **2.6551 b/w**(ledger total_fixed 對照見 quant_p3b_w.json)
- R1 碼往返不一致:0(須 0)
- R2 材化 bf16(α fp32)vs export 不一致:0(須 0 = 容器碼 + 狀態 α 完整重建評測模型)
- R3 材化 bf16(α 帳面精度)vs export 不一致:105457626 / 3,565,158,400 = 2.9580%;max|Δ| 1.953e-03,max rel 7.812e-03

| 型別 | 模組 | 權重 | R3 不一致 |
|---|---|---|---|
| g4_a8 | 104 | 1,761,607,680 | 38003585 (2.1573%) |
| g8_a8 | 72 | 1,677,721,600 | 61616460 (3.6726%) |
| g256_a16 | 16 | 41,943,040 | 1557684 (3.7138%) |
| g16_a8 | 8 | 83,886,080 | 4279897 (5.1020%) |

| 整語言模型 bpw(不含視覺塔 333M) | 值 |
|---|---|
| embed_bf16(=評測態) | 4.6877 |
| embed_int8_per_row(假設,未評測) | 3.4794 |
| embed_4.5bpw(假設,GGUF Q4_K 同級,未評測) | 2.9495 |

參數:本體 3,565,158,400 / embed(tied) 635,699,200 / 其餘 bf16 4,893,696 / 合計 4,205,751,296;GGUF U25 對照 bpw_whole 2.8586 / body 2.5584。

R3 最差模組:
- model.language_model.layers.15.self_attn.o_proj 6.1914% max|Δ| 4.883e-04
- model.language_model.layers.3.self_attn.o_proj 5.9069% max|Δ| 9.766e-04
- model.language_model.layers.11.self_attn.o_proj 5.7819% max|Δ| 4.883e-04
- model.language_model.layers.7.self_attn.o_proj 5.6135% max|Δ| 9.766e-04
- model.language_model.layers.19.self_attn.o_proj 4.7637% max|Δ| 4.883e-04
- model.language_model.layers.23.self_attn.o_proj 4.5992% max|Δ| 9.766e-04
- model.language_model.layers.2.mlp.up_proj 4.1406% max|Δ| 2.441e-04
- model.language_model.layers.31.self_attn.k_proj 4.1055% max|Δ| 4.883e-04

## 裁讀(2026-09-11 22:4x)

- **R1/R2 = 0**:容器(碼位元流 + α)加狀態 fp32 α 可位元一致重建評測用 export;容器落地 1.102 GiB = 2.6553 b/w,與帳面 2.6551 相符(safetensors 頭 +0.0002)。帳面不是估算,是實際檔案。
- **R3 = 2.958%**:帳面精度 α(fp16 / int8+fp16 逐列尺度)材化後,2.96% 的 bf16 權重與評測態差 1 ulp(max rel 7.8e-3 = bf16 一個尾數位)。原因:引擎解出的 α 存 fp32,ledger 以 16 bit 記帳但未在解算時把 α 投影到 fp16;int8 路徑的逐列尺度亦未投影。
- **結論**:目前評測的 p3b_w 與帳面模型不是同一組權重(3% 權重差 1 ulp)。發布模型必須是「容器可完整重建」的那組。

### 發布動作
1. 從容器材化 export(α 帳面精度)→ he 閘(與 84.15 比,噪音帶 ±0.9)→ 官方三科,發布用該組分數(GPU,W2 收後,約 3.5 h;用戶核准後跑)。
2. 引擎修正(W2 收後再改,不動運行中的鏈):quantize 解出 α 後投影到 fp16(int8 路徑:s_row 投影 fp16 後重取 q),使狀態 = 帳面,R3 恒為 0。
3. 整語言模型 bpw 三種說法都寫進模型卡:評測態 embed bf16 4.69;若 embed int8 逐列 3.48、Q4_K 同級 2.95(後兩者未評測);GGUF U25 whole 2.86 是 embed Q4_K 帳。文章主數字維持 **body 2.655 b/w**(與 GGUF body 2.558 同帳)。
4. kernel(位元流 GEMM)列 future work;容器格式即 HF 上傳物,附 unpack 腳本。
