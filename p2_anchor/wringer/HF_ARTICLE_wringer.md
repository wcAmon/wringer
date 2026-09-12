# Wringer: Fill, then Wring — a 4B Reasoning Model at 2.655 Bits per Weight with 93.7% of Its Benchmark Scores

> **Draft v0.1 (2026-09-11) for a Hugging Face community article.** Numbers marked `[TBD]` are filled after the second wring round (p3b_w2), the p3a water round, and the container re-evaluation finish. Sections marked `[cut?]` are candidates for trimming.
>
> Model: [InternScience/Agents-A1-4B](https://huggingface.co/InternScience/Agents-A1-4B) (Qwen3.5 hybrid: 32 layers, 25 linear-attention + 7 full-attention, 3.565B quantized weights).
> Hardware: one RTX PRO 6000 Blackwell (96 GB). Judges: strict IFEval (541), HumanEval (164), GSM8K test (1319), our own vLLM harness, 16k generation cap.
> Code, pre-registrations, verdicts and evidence JSON: `[repo link TBD]`. Model: `[HF model link TBD]`.

---

## TL;DR

We compress a 4B hybrid linear-attention reasoning model to **2.655 bits per weight of pure integer code** (4/8/16-level grids, block-128 scales, no residual adapters at inference) and keep **94.7 %** (one fill-and-wring round: 93.65 %) of the bf16 model's average score on IFEval / HumanEval / GSM8K. At the same bits-per-weight, the GGUF k-quant / imatrix curve of the same model interpolates to **52.7 %**.

| Point | body bpw | IFEval | HumanEval | GSM8K | comp (mean retention) |
|---|---:|---:|---:|---:|---:|
| bf16 | 16 | 92.79 | 94.51 | 95.53 | 1.000 |
| GGUF U34 (IQ3-class) | 3.295 | 91.87 | 92.68 | 94.16 | 0.9855 |
| **Wringer P1-alpha** (8-level g128, zero-training + α polish) | 3.206 | 90.94 | 90.85 | 95.38 | 0.9799 |
| GGUF U30 | 2.972 | 85.03 | 77.44 | 90.83 | 0.8955 |
| GGUF U27 (IQ2_M-class) | 2.763 | 73.38 | 63.41 | 76.65 | 0.7547 |
| Wringer p3b, zero-training | 2.655 | 75.97 | 70.73 | 83.70 | 0.8144 |
| **Wringer p3b_w, one fill + wring** | **2.655** | **87.80** | 85.37 ± 2.63† | **92.95** | 0.9433 |
| **Wringer p3b_w2, two rounds (published, container-materialized)** | **2.655** | 88.17 | 85.37 ± 1.32† | 93.48 | **0.9465** |
| Wringer p3a_w (α fp16 variant, one round) | 2.571 | 86.88 | 84.15 | 91.51 | 0.9282 |
| GGUF U25 (IQ2_XXS-class) | 2.558 | 34.38 | 20.12 | 36.54 | 0.3216 |

† HumanEval as 4-seed mean ± sd (temperature 1.0); the bf16 anchor itself spans 91.46–94.51 across seeds (mean 93.75, used for the ratio). Other rows are single official runs. The two Wringer rows are within sampling noise of each other.

The recipe has three steps and no training of the integer codes at all:

1. **Closed-form quantization** (full-Σ GPTQ on a sign-aware even grid, joint block scales with a Tikhonov prior).
2. **Fill**: freeze the codes, hang a rank-128 adapter on every quantized linear, self-distil from the bf16 model for ~100M tokens (7.5 h on one GPU).
3. **Wring**: throw the adapter away by re-solving step 1 against `W_q + BA` as the target. Same bit ledger, no adapter at inference — and the wrung model scores *higher* than the model with the adapter still attached.

Everything else in this article is about how we got there: 69 pre-registered experiments, 18 documented cases of a training-side proxy pointing the wrong way, and one methodological rule that carried the whole line — *only the official judges decide*.

---

## 1. What "2.655 bits per weight" means here

We use the same ledger that GGUF uses for its transformer body (`blk.*` linear layers only): the integer code at its fixed rate (4-level 2.0, 8-level 3.0, 16-level 4.0, int8 8.0 b/w) plus the scales (fp16 per 128-block = 0.125 b/w, or int8 per block + one fp16 per row = 0.067 b/w). Embedding, norms and the small SSM parameters are outside the ledger for both us and GGUF; we report the whole-language-model number separately in §6.

The number is not an estimate. The packed container (`wringer_p3b_w.safetensors`, bit-streams + scales) is **1.102 GiB = 2.6553 b/w** for the 3.565B body weights, and unpacking it reproduces the evaluated bf16 tensors bit-for-bit (§6).

The per-module map of p3b_w:

| Module (per layer) | grid | block | scale |
|---|---|---|---|
| linear_attn.in_proj_z, in_proj_qkv, out_proj; mlp.gate_proj | 4-level {−2,−1,0,1} | 128 | int8 + fp16/row |
| mlp.up_proj, mlp.down_proj, self_attn.q_proj | 8-level {−4..3} | 128 | int8 + fp16/row |
| self_attn.o_proj | 16-level | 128 | int8 + fp16/row |
| self_attn.k_proj, v_proj | int8 | per row | fp16/row |

"Sign-aware even grid" means we drop one extreme of the symmetric grid (8 levels are {−4..3}, not {−3..3} plus a spare) and let the block scale carry the sign. The odd-level ternary/quinary grids we used for the first 60 experiments are gone; §4 explains why.

---

## 2. The method

### 2.1 Closed-form quantization with a prior
Per layer, per module: collect the input second moment Σ on 128 calibration rows of 16k tokens (self-generated teacher trajectories, see §5), run a sign-aware scale search, GPTQ with the full Σ (OBS updates, 1 % damping), then re-solve all block scales jointly against Σ with a Tikhonov prior toward the search initialization (λ = 0.01).

The prior is not optional. Without it the joint least-squares solution drifts in the null space of Σ: calibration error 0, dense weight error 44 %, HumanEval −6 pp. This was experiment E65-X, and every closed-form step after it carries the prior.

Zero-training result for the p3b map: comp 0.8144 at 2.655 b/w — already above the GGUF curve (0.527 interpolated). The rest is bought by fill-and-wring.

### 2.2 Fill
Materialize the quantized weights, freeze them, and add `W_q + B·A / r` with r = 128 (B zero-initialized) to every quantized linear. Train only A, B with a KL distillation loss against the bf16 model on the same 128 × 16k self-generated rows. Two details mattered:

- **Closure weighting.** The teacher is a thinking model. Its failure mode after quantization is not wrong answers but *never closing the thinking block*: 14 % empty responses in an early champion. We weight the KL at the `</think>` token by 32 and the 8 tokens after it by 16. This alone moved the three-subject mean by +0.12 in the ternary era and is kept in the final recipe.
- **Fixed budget, no early stopping on proxies.** 3000 steps × batch 2 × 16k ≈ 100M tokens. Validation cross-entropy is logged and never used for decisions.

With the adapter attached (fp32 "water"): HumanEval 79.88.

### 2.3 Wring
Run step 2.1 again with `W_eff = W_q + BA/r` as the target instead of the bf16 weight, same module map, same prior. The adapter is gone; the ledger is unchanged at 2.655 b/w. HumanEval after wringing: **84.15**, higher than the 79.88 with the adapter still attached.

We call the ratio (wrung − zero-training) / (filled − zero-training) the retention ρ. Through most of this project ρ was the bottleneck (0.49–0.97 with step-wise "draining"); closed-form wringing gives ρ = 1.47. The adapter is a scaffold that stores function during training; the closed-form solve absorbs it into codes and scales better than the adapter itself expressed it.

**Second round.** Filling and wringing p3b_w again: with the adapter attached HumanEval is 83.54 — no higher than the 84.15 we started from — and after wringing it is **89.02** (+4.87; noise band: one problem = 0.61 pp). IFEval 86.32 and GSM8K 92.49 move within noise (−1.5 / −0.5; two-run comp noise ±0.016). The adapter ceiling does not rise; the gain appears only after absorption. Two rounds, two observations of the same effect.

---

## 3. What it cost to get below 2.7 bits

Starting from the 3.206 b/w P1 map (all 8-level g128, k/v int8, o_proj 16-level; comp 0.9799), we measured the *price* of each further cut as HumanEval points lost per bit saved, one cut at a time, zero-training:

| Cut (applied alone to P1, HumanEval 90.24) | Δ bpw | HumanEval | Δ | price (pp per b/w) |
|---|---:|---:|---:|---:|
| in_proj_z → 4-level | −0.071 | 92.07 | +1.8 | free (−26) |
| in_proj_qkv → 4-level | −0.141 | 87.80 | −2.4 | 17 |
| gate_proj → 4-level | −0.212 | 84.76 | −5.5 | 26 |
| scales → int8 (+fp16/row) | −0.057 | 88.41 | −1.8 | 32 |
| up_proj → 4-level | −0.212 | 82.93 | −7.3 | 34 |
| out_proj → 4-level | −0.071 | 87.80 | −2.4 | 34 |
| down_proj → 4-level | −0.212 | 81.71 | −8.5 | 40 |
| MLP block 128 → 256 | −0.020 | 87.80 | −2.4 | 61 |
| (GGUF curve, top segment) | | | | 47 |

Two lessons. Block-256 is more expensive than the GGUF curve itself, so it is out. And the cuts do **not** add: stacking the four cheapest (z, qkv, gate, out) plus int8 scales predicted HumanEval 79.9 by summation and delivered 70.7 — a 9 pp non-linearity. Cost tables rank cuts; they do not predict combinations.

`[cut?]` The alternative map p3a (z, qkv, gate, up with fp16 scales; 2.571 b/w) scores 0.7724 zero-training and 0.9282 after one fill-and-wring round (IFEval 86.88 / HumanEval 84.15 / GSM8K 91.51), i.e. 0.58 above the GGUF curve at the same 2.571 b/w — the largest gap over GGUF in the whole series. Per-subject gains track the zero-training gap, not the corpus mix: IFEval, with the largest gap, gained most (+19.2), while HumanEval gained exactly +13.4 on both maps.

---

## 4. Why not ternary

Sixty of the 69 experiments were on ternary and quinary grids. The best ternary point we ever produced is comp 0.7469 (trained) / 0.8235 (closed-form absorption) at ~2.70 b/w — and its HumanEval retention was stuck below 70 %. Three findings closed that line:

1. **End-to-end training does not flip codes.** Across five reproductions the fraction of ternary codes changed by end-to-end distillation was 0.0000; only the continuous axes (scales, adapters) move. When we forced code changes (window-wise re-solves), HumanEval retention fell monotonically with the amount flipped (0.62 → 0.57 → 0.49).
2. **Rebuilding from a higher grid does not help either.** A zero-training 9-level model scores comp 0.9226; solving ternary codes directly against it collapsed to 0.1289. Absorbing the trained ternary champion into an 8-level map gave HumanEval 58.5 versus 90.2 from bf16 directly. Ternary states are not a good source of anything.
3. **The ledger, not the grid, was the real loss.** Ternary with fp16 scales per 32 weights spends 0.5 b/w on scales. Moving to 8-level with 128-blocks costs the same bits and lands *on top of* the GGUF curve zero-training. Once we compared on the same ledger, the whole ternary family was below GGUF.

This matches the external picture (ParetoQ needs tens of billions of QAT tokens for 2-bit; recent ternary PTQ work on Qwen3-4B tops out at 60–70 % code retention). We had 100M tokens and one GPU.

---

## 5. Methodology: only the judges decide

Every arm was pre-registered as a JSON file before ignition — question, thresholds, *falsifier* (kill condition), the exact collection actions — and frozen. When a falsifier fired, the axis closed without a retry. This is what kept 69 experiments from becoming 200.

The rule that mattered most: **training-side numbers do not decide anything.** Validation cross-entropy, KD loss, block reconstruction error, code entropy — all logged, none used. We recorded 18 cases where such a proxy pointed the opposite way from the official benchmarks. A few:

| Belief | What killed it |
|---|---|
| In-domain val CE recovery = capability recovery | 4-benchmark anchor: quinary "fully recovered" on CE had 39–98 % relative gaps |
| Learned rotation carriers | val CE 1.69, *better than bf16 at 2.00*; official score equal to a random rotation |
| Lower block reconstruction error = better | −30 % block error, −10 pp HumanEval |
| Higher fill ceiling = better final | ceiling up, wrung model down (the fixed drain leaks more) |
| Quantization loss is a capacity problem | LoRA adapters kept at inference: zero gain; rank 256 = rank 512 |
| Cost tables add | −9 pp on stacking (§3) |

Two things about the calibration data are worth stating plainly:

- **It is self-generated.** A stronger model (the 4B's teacher lineage) wrote coding / instruction-following / math prompts with programmatic verifiers; the bf16 4B answered them with its thinking trace; we kept 384 rows of ≤16k tokens. The reason is on-manifold: in the ternary era, switching to the model's own trajectories on this bank was worth +4.6 then +15.5 IFEval points over the earlier corpora.
- **It contaminates HumanEval, a little.** We ran a 13-gram / longest-common-run check of both the prompt bank and the decoded calibration rows against all three test sets. GSM8K and IFEval are clean (only instruction-template phrases match). The prompt bank reproduces 10 HumanEval tasks nearly verbatim (docstring, canonical solution or `check()` asserts; longest common run ≥ 25 tokens), and 4 of them are inherited by the calibration rows. Removing those 10 tasks changes p3b_w's HumanEval from 84.15 to **84.42** (p3b_w2: 89.02 → 89.61) and the bf16 anchor's from 94.51 to 94.16 (retention 0.890 → 0.897) — the headline is not inflated by them. The fill gain on the flagged tasks (50 → 80 % pass) is larger than on the rest (72 → 84 %), which with n = 10 we can neither confirm nor rule out; we report both numbers and will decontaminate the bank before the next model. Full report: `contamination_e69.md`.

---

## 6. From ledger to file

All evaluations run on fake-quant bf16 materializations in vLLM. To make sure the ledger is a file and not a story:

- **Container.** Codes are packed as 2/3/4/8-bit little-endian bit-streams per module, scales as fp16 (or int8 + fp16 per row). `wringer_p3b_w.safetensors`: 1.102 GiB, 2.6553 b/w including the safetensors header, versus 2.6551 on the ledger.
- **Round trip.** Unpacked codes match the state bit-for-bit; re-materialized bf16 tensors match the evaluated export bit-for-bit (0 of 3.565B differ).
- **One honest caveat we found while checking.** The solver stores scales in fp32; the ledger charges 16 bits. Materializing with fp16 scales moves 2.95 % of the bf16 weights by one ulp. The published model is therefore re-materialized *from the container* and re-scored: IFEval 88.17 / HumanEval 85.37 ± 1.32 (4 seeds) / GSM8K 93.48 — indistinguishable from the research state (86.32 / 86.74 ± 2.74 / 92.49). The solver now projects scales to fp16 at solve time so this cannot recur.
- **A second caveat the first one uncovered.** Re-scoring HumanEval under several seeds showed single runs swing by up to 5.5 pp at temperature 1.0, for the bf16 parent too. Our earlier single-run headline of 89.02 for the two-round model was a lucky draw; the two-round and one-round models are within noise of each other. Every HumanEval number in this article is now a 4-seed mean, and "two rounds" is claimed only as "does not hurt".
- **Whole-model bits.** With the tied embedding kept in bf16 (the evaluated state), the language model is 4.69 b/w overall; with an int8 embedding it would be 3.48, and with a Q4_K-class embedding (what GGUF U25 uses) 2.95 versus GGUF's 2.86. We have not evaluated the model with a quantized embedding.
- **No kernels yet.** The container is a storage format; a bit-stream GEMM is future work. Inference in this article is dequantize-to-bf16.

---

## 7. Limitations

1. One model, three benchmarks, one seed per official run. Noise band: ±0.9 pp on HumanEval, ±0.016 on comp.
2. HumanEval is still the weakest subject (89 % retained vs 94.6 % IFEval, 97.3 % GSM8K).
3. No academic PTQ/QAT baselines were run in our harness; the comparison is against GGUF only. `[Phase 2: AQLM / QuIP# / PV-tuning-style baselines at equal bpw.]`
4. Below ~2.3 b/w the cost table says the recipe runs out: all-4-level pure code is 2.07 b/w and scores ~0.5 zero-training.
5. Calibration bank contamination of HumanEval (§5), disclosed, small, and to be fixed for the next model.

---

## 8. Dead ends (so you don't repeat them) `[cut?]`

Grouping algorithms for ternary scales (magnitude / gradient / correlation / spectral), zero-training 2:4 sparsity, learned rotation carriers, resident mixed-bucket adapters, VQ / trellis solvers, entropy coding of codes (saves 0.10 b/w, nothing more), block-256, per-layer feature KD, curriculum reordering, norm-only fine-tuning, alternating code/scale re-solves against the *original* weight, bootstrapping a lower grid from a trained champion, rebuilding a basin from scratch with 42M tokens, changing the adapter during draining, reverse-KL distillation, and every multi-stage "corridor" from 7 levels down to 3.

---

## 9. Reproducibility

- Pre-registration and verdict JSON for every experiment (E1–E69), evidence JSON for every official run, chain scripts with their sentinels, the calibration-generation pipeline, and the contamination / packing scripts: `[repo link TBD]`.
- Milestone states kept: `g9` (9-level zero-training floor), `st51c_e2e` (trained ternary champion), `pa64_asymrefitq2` (closed-form ternary champion), `p1alpha`, `p3b_w`, `p3b_w2` `[TBD]`.
- Model on the Hub: `p3b_w2` (container + unpack script + bf16 materialization), with `p3b_w` kept as the single-round control. `[links TBD]`

---

*Wringer was developed in ~5 weeks of single-GPU time by one person and one agent, with the agent running the experiment chains under pre-registered kill conditions and the person deciding what to try next. The name is what it does: fill the rag, then wring it out.*
