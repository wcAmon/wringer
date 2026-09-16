# 模型端記憶探針 qwen3_4b(對照 a1)

guided completion:參考文字前 40% 詞當前綴(raw completions,temperature 0),量生成對參考後綴的逐字前綴長(exact_prefix,詞)與最長共同詞串(lcr)。
he_run:官方 HumanEval 回應 vs canonical_solution 的最長共同詞串。

| 模型 | 集合 | n | exact_prefix 均值 | ≥8 | ≥20 | lcr 均值 | ≥13 | ≥25 |
|---|---|---|---|---|---|---|---|---|
| qwen3_4b | ifeval | 539 | 0.33 | 1 | 0 | 2.06 | 2 | 0 |
| qwen3_4b | humaneval | 147 | 8.0 | 61 | 20 | 14.24 | 63 | 18 |
| qwen3_4b | gsm8k | 1303 | 3.54 | 195 | 66 | 14.42 | 772 | 92 |
| a1 | ifeval | 539 | 0.37 | 1 | 0 | 2.09 | 2 | 0 |
| a1 | humaneval | 147 | 10.63 | 71 | 25 | 14.98 | 69 | 22 |
| a1 | gsm8k | 1303 | 3.33 | 189 | 63 | 14.41 | 776 | 102 |

| 模型 | HumanEval 回應 vs canonical | lcr 均值 | ≥13 | ≥25 | ≥50 |
|---|---|---|---|---|---|
| qwen3_4b | n=164 | 8.77 | 31 | 6 | 2 |
| a1 | n=164 | 8.55 | 27 | 5 | 2 |

top exact_prefix(model):
- ifeval: [(8, '2801'), (6, '1592'), (6, '1012'), (5, '2779'), (5, '1531')]
- humaneval: [(70, 'HumanEval/129'), (37, 'HumanEval/72'), (37, 'HumanEval/40'), (28, 'HumanEval/156'), (28, 'HumanEval/143')]
- gsm8k: [(59, '927'), (51, '547'), (48, '937'), (46, '647'), (46, '1076')]

## 裁決(2026-09-12)

- 規則:exact_prefix≥20 / lcr≥25 計數對 A1 對照;Qwen3-4B 三集合均不高於 A1(he 20/18 vs 25/22;gs 66/92 vs 63/102;if 0/0 vs 0/0)→ **全集為主判,不啟用去污染子集**。
- 單題強訊號:HumanEval/129 exact_prefix 70(A1 7)= 逐字重現 canonical;Qwen 獨有 ≥20 題共 7 題 ['HumanEval/12', 'HumanEval/129', 'HumanEval/139', 'HumanEval/143', 'HumanEval/43', 'HumanEval/56', 'HumanEval/61'](A1 獨有 12 題;兩模型共有 13 題屬「題目可猜」型)。
- 敏感度子集:he 全集 0.9695(seed 20260806)、排除 Qwen 獨有 7 題 0.9682(n=157);與 E69 的 S25 子集同性質,報告時附註即可。
- bf16 錨(單 seed):he 0.9695 / IF 0.8133 / GS 0.9484;he 需補 3 seed 均值(第二模型 prereg 錨定段)。
