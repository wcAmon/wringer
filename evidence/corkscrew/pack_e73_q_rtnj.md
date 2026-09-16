# Wringer 打包驗證 e73_q_rtnj(vs exports/e73_q_rtnj)

- 模組 252,本體權重 3,633,315,840;容器 `data/wringer_e73_q_rtnj.safetensors` 1.116 GiB = 2.6385 b/w(safetensors 頭與對齊含在內)
- 帳面:碼 2.5195 + α 0.1188 = **2.6383 b/w**(ledger total_fixed 對照見 quant_e73_q_rtnj.json)
- R1 碼往返不一致:0(須 0)
- R2 材化 bf16(α fp32)vs export 不一致:0(須 0 = 容器碼 + 狀態 α 完整重建評測模型)
- R3 材化 bf16(α 帳面精度)vs export 不一致:0 / 3,633,315,840 = 0.0000%;max|Δ| 0.000e+00,max rel 0.000e+00

| 型別 | 模組 | 權重 | R3 不一致 |
|---|---|---|---|
| g8_a16 | 72 | 754,974,720 | 0 (0.0000%) |
| g256_a16 | 72 | 188,743,680 | 0 (0.0000%) |
| g4_a16 | 108 | 2,689,597,440 | 0 (0.0000%) |

| 整語言模型 bpw(不含視覺塔 333M) | 值 |
|---|---|
| embed_bf16(=評測態) | 2.6383 |
| embed_int8_per_row(假設,未評測) | 2.6383 |
| embed_4.5bpw(假設,GGUF Q4_K 同級,未評測) | 2.6383 |

參數:本體 3,633,315,840 / embed(tied) 0 / 其餘 bf16 0 / 合計 3,633,315,840;GGUF U25 對照 bpw_whole 2.8586 / body 2.5584。

R3 最差模組:
- model.layers.9.self_attn.v_proj 0.0000% max|Δ| 0.000e+00
- model.layers.9.self_attn.q_proj 0.0000% max|Δ| 0.000e+00
- model.layers.9.self_attn.o_proj 0.0000% max|Δ| 0.000e+00
- model.layers.9.self_attn.k_proj 0.0000% max|Δ| 0.000e+00
- model.layers.9.mlp.up_proj 0.0000% max|Δ| 0.000e+00
- model.layers.9.mlp.gate_proj 0.0000% max|Δ| 0.000e+00
- model.layers.9.mlp.down_proj 0.0000% max|Δ| 0.000e+00
- model.layers.8.self_attn.v_proj 0.0000% max|Δ| 0.000e+00
