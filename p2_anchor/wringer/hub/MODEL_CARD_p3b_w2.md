---
license: apache-2.0
base_model: InternScience/Agents-A1-4B
tags:
- quantization
- low-bit
- wringer
- qwen3.5
language:
- en
---

# Wringer p3b_w2 — Agents-A1-4B at 2.655 bits/weight

**Draft v0.2 (2026-09-12).** HumanEval is reported as the mean ± sd over 4 sampling seeds (temperature 1.0); IFEval and GSM8K are single official runs.

A 2.655 bits-per-weight (body) quantization of [InternScience/Agents-A1-4B](https://huggingface.co/InternScience/Agents-A1-4B)
(Qwen3.5 hybrid: 25 GatedDeltaNet linear-attention layers + 7 full-attention layers, 3.57 B quantized body weights),
produced with **Wringer**: fixed-grid GPTQ codes → one round of low-rank "water" (r128 LoRA, KD to the bf16 parent on
the model's own long-trajectory corpus) → closed-form "wring" that re-solves scales and codes so **no adapter is shipped**.
Two fill-and-wring rounds were run.

## Scores (official A1 evaluation framework, thinking on, 16k max tokens)

| Model | body b/w | IFEval (prompt, strict) | HumanEval (4-seed mean ± sd) | GSM8K | comp* |
|---|---|---|---|---|---|
| bf16 parent (anchor) | 16 | 92.79 | 93.75 ± 1.52 | 95.53 | 1.000 |
| **this model (materialized from the container = what you download)** | **2.655** | 88.17 | 85.37 ± 1.32 | 93.48 | **0.9465** |
| this model, research-state weights (fp32 scales) | 2.655 | 86.32 | 86.74 ± 2.74 | 92.49 | 0.9412 |
| single-round control p3b_w | 2.655 | 87.80 | 85.37 ± 2.63 | 92.95 | 0.9433 |
| zero-training p3b (same codes, no water) | 2.655 | 75.97 | 70.73 (1 seed) | 83.70 | 0.8144 |
| llama.cpp GGUF at the same body b/w (U27, IQ2_M-class, 2.763) | 2.763 | 73.38 | 63.41 | 76.65 | 0.7547 |

\*comp = mean of the three retention ratios against the bf16 anchor (HumanEval ratio uses the 4-seed means). The three Wringer rows are within sampling noise of each other (HumanEval single runs swing by up to 5.5 pp at temperature 1.0); the second wring round did not hurt and is what we publish. Body b/w counts the quantized linear layers only;
embeddings, norms and lm_head stay bf16 (whole-LM figure 4.69 b/w with bf16 embeddings; 2.95 b/w if the tied embedding
were stored at Q4_K-class 4.5 b/w — not evaluated).

HumanEval with the 10 tasks flagged by our contamination check removed (see below), seed 20260806: 89.61 (research state), 94.16 (anchor).

## What is in this repo

- `wringer_p3b_w2.safetensors` — the container (1.10 GiB): packed codes + fp16 / int8 block scales, metadata `wringer_meta`.
- `wringer_unpack.py` — dependency-free materializer: container + bf16 source export → bf16 HF checkpoint.
- `bf16/` — the materialized bf16 checkpoint (fake-quant weights; loads with `transformers` / vLLM like the parent).
- `reports/` — packing verification (`pack_p3b_w2.md`), contamination check (`contamination_e69.md`), verdict JSON.

No custom kernel is provided. The container is a storage format; inference uses the materialized bf16 weights.

## Recipe (per module)

| Module | grid | block | scales |
|---|---|---|---|
| self_attn.k_proj, v_proj | int8 | per row | fp16 |
| self_attn.o_proj | 16-level | 128 | int8 + fp16 row scale |
| linear_attn.in_proj_z, in_proj_qkv, mlp.gate_proj, linear_attn.out_proj | 4-level {−2..1} | 128 | int8 + fp16 row scale |
| self_attn.q_proj, mlp.up_proj, mlp.down_proj | 8-level {−4..3} | 128 | int8 + fp16 row scale |

Ledger: code 2.588 + scales 0.067 = 2.655 b/w. Codes solved by GPTQ per layer on 128 × 16k self-generated
calibration rows; scales by a joint least-squares closed form with a prior (λ = 0.01).

## Honest caveats

- **Scale precision.** The research solver stored scales in fp32 while the ledger charged fp16. Materializing at ledger
  precision moves 2.95 % of the bf16 weights by one ulp. We re-scored the container-materialized weights: HumanEval
  85.37 ± 1.32 vs 86.74 ± 2.74 for the research state (difference 1.37 pp, SE 1.52 — not distinguishable), IFEval and
  GSM8K slightly higher. What you download is exactly what the first row measures. The solver now projects scales to fp16
  at solve time.
- **Sampling noise.** HumanEval at temperature 1.0 swings by several points between seeds, even for the bf16 parent
  (91.46–94.51). Single-run HumanEval headlines from this project's earlier write-ups (e.g. 89.02) were lucky draws;
  all HumanEval numbers here are 4-seed means.
- **Contamination.** Our calibration corpus is model-generated. An 8-gram check against the three test sets found
  GSM8K and IFEval clean; 10 HumanEval tasks have ≥25-token overlaps with corpus solutions (list in `reports/`).
  Scores are reported with and without them.
- **Three benchmarks only.** IFEval / HumanEval / GSM8K in thinking mode. No MMLU-class or agentic suites yet.
- **One model, one architecture.** Generality beyond this hybrid Qwen3.5 model is not yet shown.

## How it was made

Article: `[HF article link TBD]`. Code, pre-registrations and evidence: `[repo link TBD]`.

## License

The parent model's license applies to the weights. Wringer code: Apache-2.0.
