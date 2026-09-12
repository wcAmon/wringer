# Wringer 打包驗證 p3b_w2(vs exports/e69_p3b_w2)

- 模組 200,本體權重 3,565,158,400;容器 `data/wringer_p3b_w2.safetensors` 1.102 GiB = 2.6553 b/w(safetensors 頭與對齊含在內)
- 帳面:碼 2.5882 + α 0.0668 = **2.6551 b/w**(ledger total_fixed 對照見 quant_p3b_w2.json)
- R1 碼往返不一致:0(須 0)
- R2 材化 bf16(α fp32)vs export 不一致:0(須 0 = 容器碼 + 狀態 α 完整重建評測模型)
- R3 材化 bf16(α 帳面精度)vs export 不一致:105209061 / 3,565,158,400 = 2.9510%;max|Δ| 1.953e-03,max rel 7.812e-03

| 型別 | 模組 | 權重 | R3 不一致 |
|---|---|---|---|
| g4_a8 | 104 | 1,761,607,680 | 37853097 (2.1488%) |
| g8_a8 | 72 | 1,677,721,600 | 61496251 (3.6655%) |
| g256_a16 | 16 | 41,943,040 | 1583033 (3.7742%) |
| g16_a8 | 8 | 83,886,080 | 4276680 (5.0982%) |

| 整語言模型 bpw(不含視覺塔 333M) | 值 |
|---|---|
| embed_bf16(=評測態) | 4.6877 |
| embed_int8_per_row(假設,未評測) | 3.4794 |
| embed_4.5bpw(假設,GGUF Q4_K 同級,未評測) | 2.9495 |

參數:本體 3,565,158,400 / embed(tied) 635,699,200 / 其餘 bf16 4,893,696 / 合計 4,205,751,296;GGUF U25 對照 bpw_whole 2.8586 / body 2.5584。

R3 最差模組:
- model.language_model.layers.15.self_attn.o_proj 6.0710% max|Δ| 4.883e-04
- model.language_model.layers.3.self_attn.o_proj 5.9272% max|Δ| 1.953e-03
- model.language_model.layers.11.self_attn.o_proj 5.8253% max|Δ| 4.883e-04
- model.language_model.layers.7.self_attn.o_proj 5.6639% max|Δ| 4.883e-04
- model.language_model.layers.23.self_attn.o_proj 4.7458% max|Δ| 9.766e-04
- model.language_model.layers.19.self_attn.o_proj 4.7148% max|Δ| 4.883e-04
- model.language_model.layers.19.self_attn.k_proj 4.3559% max|Δ| 4.883e-04
- model.language_model.layers.27.self_attn.k_proj 4.1548% max|Δ| 4.883e-04
