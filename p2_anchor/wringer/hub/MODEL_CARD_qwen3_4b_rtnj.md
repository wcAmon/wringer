---
license: apache-2.0
base_model: Qwen/Qwen3-4B
tags:
- quantization
- low-bit
- wringer
- qwen3
language:
- en
---

# Wringer — Qwen3-4B at 2.638 bits/weight

**v0.3 (2026-09-16; v0.2 2026-09-14; v0.1 2026-09-13).** Second model produced with Wringer (first: [Agents-A1-4B-Wringer-Q2.6](https://huggingface.co/wcamon/Agents-A1-4B-Wringer-Q2.6)).
**v0.3 changes the finishing solver, not the filled weights:** the same rank-128 "water" is now wrung out with nearest-point codes + joint closed-form scales instead of GPTQ codes + joint scales (research-state comp 0.9066 → 0.9223; see "Why the solver changed"). The previous container (`wringer_e70_p3a_w.safetensors`) is removed from this repo; its numbers stay in the table below.
HumanEval is reported as the mean ± sd over 4 sampling seeds (temperature 1.0); IFEval and GSM8K are single official runs.

A 2.638 bits-per-weight (body) quantization of [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
(36 dense transformer layers, 3.63 B quantized body weights, revision `1cfa9a7`), produced with **Wringer**:
fixed-grid GPTQ codes → one round of low-rank "water" (r128 LoRA, KD to the bf16 parent on the model's own long-trajectory
corpus, 3000 steps, 7.8 h on one GPU) → closed-form "wring" that re-solves codes (nearest point) and scales (joint least squares with a prior) so **no adapter is shipped**.
One fill-and-wring round was run; zero-training codes are the starting point and are reported below.

## Scores (thinking on, 16k max tokens, vLLM with fp8 KV cache)

| Model | body b/w | IFEval (prompt, strict) | HumanEval (4-seed mean ± sd) | GSM8K | comp* |
|---|---|---|---|---|---|
| bf16 parent (anchor) | 16 | 82.53 (2-run mean) | 94.66 ± 0.58 | 95.00 | 1.000 |
| **this model v0.3 (materialized from the container = what you download)** | **2.638** | 78.56 | 82.62 ± 2.46 | 87.87 | **0.9166** |
| this model v0.3, research-state weights (fp32 scales) | 2.638 | 79.11 | 83.23 ± 2.25 | 88.25 | 0.9223 |
| v0.2 (same water, GPTQ finisher), materialized | 2.638 | 75.42 | 83.38 ± 2.07 | 89.16 | 0.9111 |
| v0.2, research-state weights | 2.638 | 78.56 | 78.96 ± 1.17 | 88.70 | 0.9066 |
| same-budget STE-QAT control (codes + scales trained through the quantizer, 3000 steps) | 2.638 | 74.68 | 74.09 ± 3.64 | 84.31 | 0.8583 |
| GPTQ 3-bit g128 (llm-compressor, same calibration rows) | 3.148 | 78.19 | 87.04 ± 2.19 | 91.96 | 0.945 |
| AWQ 4-bit g128, asymmetric (llm-compressor) | 4.156 | 83.36 | 94.51 ± 1.32 | 91.05 | 0.989 |
| GPTQ 2-bit g128 / HQQ 2-bit g64 | 2.141 / 2.500 | — | 0 | — | 0 (collapsed: thinking never terminates) |
| zero-training p3a (same codes, no water) | 2.638 | 74.12 | 72.87 (2 seeds) | 73.62 | 0.8143 |
| llama.cpp GGUF IQ2_S, imatrix + UD-style protection | 2.520 | 53.23 | 46.95 (2 seeds) | 77.41 | 0.652 |
| llama.cpp GGUF IQ2_M, same recipe | 2.727 | 71.90 | 81.40 (2 seeds) | 88.32 | 0.887 |
| llama.cpp GGUF IQ3_XXS, same recipe | 3.066 | 77.26 | 87.20 (2 seeds) | 92.42 | 0.943 |

\*comp = mean of the three retention ratios against the bf16 anchor (HumanEval ratio uses the 4-seed means).
The materialized and research-state rows of each version have bit-identical weights (see `reports/pack_e73_q_rtnj.md`, R2 = R3 = 0); their difference is
sampling noise plus vLLM batch nondeterminism. Body b/w counts the quantized linear layers
only; the tied embedding (389 M) and norms stay bf16 (whole-LM figure 3.93 b/w with bf16 embedding; 2.82 b/w if the
embedding were stored at Q4_K-class 4.5 b/w — not evaluated). GGUF rows are llama.cpp (imatrix on the same calibration
text, attn_v iq3_s + first two ffn_down q4_k protected, llama-server, same prompts and 16k budget; whole-file sizes
2.71 / 2.90 / 3.21 b/w since GGUF also quantizes the embedding). Linear interpolation of the GGUF curve at 2.638 b/w gives
comp ≈ 0.79; the nearest GGUF point above (IQ2_M, +0.09 b/w) reaches 0.887, and it matches or beats this model on
HumanEval while trailing on IFEval.

Compared with the first Wringer model (Agents-A1-4B, comp 0.9465 at 2.655 b/w), Qwen3-4B loses more at the same budget:
the per-module sensitivity probes showed every bit-saving move on this model costs several times to an order of magnitude more HumanEval per bit than on A1 (where two of the moves were nearly free)
(`reports/prereg_e70_qwen3_4b.json`, `stage1_sensitivity.result_cost_table`), so the composition below is close to forced.

## What is in this repo

- `wringer_e73_q_rtnj.safetensors` — the container (1.12 GiB): packed codes + fp16 block scales, metadata `wringer_meta`.
- `wringer_unpack.py` — dependency-free materializer: container + bf16 source export → bf16 HF checkpoint
  (re-run independently for v0.2: 398 tensors, bit-identical to `bf16/`).
- `bf16/` — the materialized bf16 checkpoint (fake-quant weights; loads with `transformers` / vLLM like the parent).
- `reports/` — packing verification (`pack_e73_q_rtnj.md`, v0.2: `pack_e70_p3a_w.md`), contamination check (`contam_probe_qwen3_4b.md`),
  the pre-registrations with results filled in (`prereg_e70_qwen3_4b.json`, `prereg_e71_ste.json`, `prereg_e72_baselines.json`, `prereg_e73_collapse.json`, `prereg_e74_zerortnj.json`).

No custom kernel is provided. The container is a storage format; inference uses the materialized bf16 weights.
(The A1 release documents a lossless conversion of the same code layout into GPTQ 4-bit format for Marlin kernels,
3.84 GiB and 1.9× decode at batch 1; the same converter applies here but was not run for this model.)

## Recipe (per module, 36 layers × 7 linears = 252 modules)

| Module | grid | block | scales |
|---|---|---|---|
| self_attn.k_proj, v_proj | int8 | per row | fp16 |
| self_attn.q_proj, self_attn.o_proj | 8-level {−4..3} | 128 | int8 + fp16 row scale |
| mlp.gate_proj, up_proj, down_proj | 4-level {−2..1} | 128 | int8 + fp16 row scale |

Ledger: code 2.520 + scales 0.119 = 2.638 b/w. Starting codes solved by GPTQ per layer on 128 × 16k self-generated calibration
rows (code / instruction-following / math prompts, thinking mode); scales by a joint least-squares closed form with a
prior (λ = 0.01). Water: r128 LoRA on every quantized module, codes frozen, KD to the bf16 parent with closure-weighted
loss. Wring (v0.3): nearest-point rounding of the watered weights onto the same grid, then the same joint closed-form scale
solve with the watered weights as target. (v0.2 used GPTQ codes in the wring step.)

## Why the solver changed

On the *same* filled weights we compared four finishers (pre-registered as E73): nearest-point codes + block scale search
0.8585; nearest-point + joint closed-form scales **0.9223**; GPTQ + joint scales 0.9066 (v0.2); a same-budget STE-QAT run
lands at 0.8583. The joint scale solve carries the gain on this model (+0.064, outside the ±0.02 noise band); the GPTQ-vs-nearest-point
difference on top of it (−0.016) is inside the band. On the A1 model the matched comparison (same second-round filled target, research state) gives GPTQ + joint
scales 0.941 vs nearest-point + joint scales 0.926, i.e. the opposite sign, so the solver choice is model-dependent and
second-order (both differences are within about 1.5× the respective noise bands). What is not second-order is the water: the nearest-point finisher applied to the bf16 weights *without*
filling scores 0.3347 at the same ledger (E74). Measured on the weights (`reports/instr_gridfriendly_qwen.json`): the bf16
parent sits 0.46 (relative Frobenius) away from its nearest joint-scale grid representation, the filled target 0.07; the two
code solvers disagree on 18 % of codes at zero training and on 1.7 % after filling.

## Honest caveats

- **Sampling noise.** HumanEval at temperature 1.0 swings by several points between seeds (sd 1–2 pp here, up to 2.7 pp
  on A1). IFEval single runs on this model differ by ~1–2 pp between seeds. All HumanEval numbers are 4-seed means;
  IFEval / GSM8K are single runs.
- **Contamination.** The calibration corpus is model-generated. A memorization probe (guided completion on the three test
  sets, compared against A1 as control) found no set where Qwen3-4B exceeds the control; one task (HumanEval/129) is
  reproduced verbatim by the parent. With the 7 Qwen-specific flagged tasks removed the parent scores 96.82 vs 96.95
  (single seed, older protocol) — no effect on conclusions. Details in `reports/contam_probe_qwen3_4b.md`.
- **Three benchmarks only.** IFEval / HumanEval / GSM8K in thinking mode. No MMLU-class or agentic suites yet.
- **Evaluation protocol change.** All Qwen3-4B numbers use vLLM with fp8 KV cache and 128 concurrent requests, validated
  against the bf16 KV baseline on the parent (IFEval 82.81 vs 81.3–85.4 band, GSM8K 94.77 vs 94.84). A1 numbers on the
  first model card use bf16 KV.
- **Baselines.** The STE-QAT control and the GPTQ / AWQ / HQQ points in the table were produced by us on the same calibration
  rows and evaluated under the same protocol; the full zero-training curve (including HQQ 3-bit 0.823 and the AWQ
  *symmetric* 4-bit point, 0.861, whose deficit is a thinking-termination failure rather than a math failure) is in the
  technical report (`reports/TECH_REPORT_wringer.md`).

## How it was made

Article (first model): on the Hugging Face blog. Code, pre-registrations and evidence: https://github.com/wcAmon/wringer.

## License

The parent model's license (Apache-2.0) applies to the weights. Wringer code: Apache-2.0.
