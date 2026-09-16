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

# Wringer p3a_w — Qwen3-4B at 2.638 bits/weight

**v0.2 (2026-09-14; v0.1 2026-09-13).** Second model produced with Wringer (first: [Agents-A1-4B-Wringer-Q2.6](https://huggingface.co/wcamon/Agents-A1-4B-Wringer-Q2.6)).
HumanEval is reported as the mean ± sd over 4 sampling seeds (temperature 1.0); IFEval and GSM8K are single official runs.

A 2.638 bits-per-weight (body) quantization of [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
(36 dense transformer layers, 3.63 B quantized body weights, revision `1cfa9a7`), produced with **Wringer**:
fixed-grid GPTQ codes → one round of low-rank "water" (r128 LoRA, KD to the bf16 parent on the model's own long-trajectory
corpus, 3000 steps, 7.8 h on one GPU) → closed-form "wring" that re-solves scales and codes so **no adapter is shipped**.
One fill-and-wring round was run; zero-training codes are the starting point and are reported below.

## Scores (thinking on, 16k max tokens, vLLM with fp8 KV cache)

| Model | body b/w | IFEval (prompt, strict) | HumanEval (4-seed mean ± sd) | GSM8K | comp* |
|---|---|---|---|---|---|
| bf16 parent (anchor) | 16 | 82.53 (2-run mean) | 94.66 ± 0.58 | 95.00 | 1.000 |
| **this model (materialized from the container = what you download)** | **2.638** | 75.42 | 83.38 ± 2.07 | 89.16 | **0.9111** |
| this model, research-state weights (fp32 scales) | 2.638 | 78.56 | 78.96 ± 1.17 | 88.70 | 0.9066 |
| zero-training p3a (same codes, no water) | 2.638 | 74.12 | 72.87 (2 seeds) | 73.62 | 0.8143 |
| llama.cpp GGUF IQ2_S, imatrix + UD-style protection | 2.520 | 53.23 | 46.95 (2 seeds) | 77.41 | 0.652 |
| llama.cpp GGUF IQ2_M, same recipe | 2.727 | 71.90 | 81.40 (2 seeds) | 88.32 | 0.887 |
| llama.cpp GGUF IQ3_XXS, same recipe | 3.066 | 77.26 | 87.20 (2 seeds) | 92.42 | 0.943 |

\*comp = mean of the three retention ratios against the bf16 anchor (HumanEval ratio uses the 4-seed means).
The two Wringer rows have bit-identical weights (see `reports/pack_e70_p3a_w.md`, R2 = R3 = 0); their difference is
sampling noise plus vLLM batch nondeterminism, and both sit in the same band. Body b/w counts the quantized linear layers
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

- `wringer_e70_p3a_w.safetensors` — the container (1.12 GiB): packed codes + fp16 / int8 block scales, metadata `wringer_meta`.
- `wringer_unpack.py` — dependency-free materializer: container + bf16 source export → bf16 HF checkpoint
  (we re-ran it independently on this repo: 398 tensors, bit-identical to `bf16/`).
- `bf16/` — the materialized bf16 checkpoint (fake-quant weights; loads with `transformers` / vLLM like the parent).
- `reports/` — packing verification (`pack_e70_p3a_w.md`), contamination check (`contam_probe_qwen3_4b.md`),
  the full pre-registration with results filled in (`prereg_e70_qwen3_4b.json`).

No custom kernel is provided. The container is a storage format; inference uses the materialized bf16 weights.
(The A1 release documents a lossless conversion of the same code layout into GPTQ 4-bit format for Marlin kernels,
3.84 GiB and 1.9× decode at batch 1; the same converter applies here but was not run for this model.)

## Recipe (per module, 36 layers × 7 linears = 252 modules)

| Module | grid | block | scales |
|---|---|---|---|
| self_attn.k_proj, v_proj | int8 | per row | fp16 |
| self_attn.q_proj, self_attn.o_proj | 8-level {−4..3} | 128 | int8 + fp16 row scale |
| mlp.gate_proj, up_proj, down_proj | 4-level {−2..1} | 128 | int8 + fp16 row scale |

Ledger: code 2.520 + scales 0.119 = 2.638 b/w. Codes solved by GPTQ per layer on 128 × 16k self-generated calibration
rows (code / instruction-following / math prompts, thinking mode); scales by a joint least-squares closed form with a
prior (λ = 0.01). Water: r128 LoRA on every quantized module, codes frozen, KD to the bf16 parent with closure-weighted
loss; wring: the same GPTQ + closed-form solve with the watered weights as the target. Recovery ratio
ρ = (B − P3)/(A − P3) = 0.870 (A = HumanEval with the adapter still attached, B = after wringing it out).

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
- **Baselines.** Same-budget STE-QAT control and GPTQ / AWQ / HQQ reference points are scheduled and will be added to the
  technical report, not to this card.

## How it was made

Article (first model): on the Hugging Face blog. Code, pre-registrations and evidence: https://github.com/wcAmon/wringer.

## License

The parent model's license (Apache-2.0) applies to the weights. Wringer code: Apache-2.0.
