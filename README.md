# Wringer

Low-bit weight quantization by **fill-and-wring**: fixed-grid GPTQ codes → a low-rank "water" adapter trained by
distillation on the model's own long trajectories → a closed-form "wring" that re-solves scales and codes so that no
adapter is shipped. First result: InternScience/Agents-A1-4B (Qwen3.5 hybrid) at 2.655 body bits/weight keeping
~94–95 % of IFEval / HumanEval / GSM8K, versus ~75 % for llama.cpp GGUF at the same size.

- Model + container + reports: https://huggingface.co/wcamon/Agents-A1-4B-Wringer-Q2.6
- Article: Hugging Face blog (link added after publication)
- Method retrospective (zh-TW): `p2_anchor/wringer/RETRO_wringer.md`; experiment history E1–E69: `p2_anchor/wringer/HISTORY_e1_e69.md`
- Pre-registrations: `p2_anchor/wringer/prereg_*.json`; verdicts: `p2_anchor/wringer/verdict_*.json`
- Evidence: `evidence/corkscrew/` (contamination check, packing verification, multi-seed HumanEval, per-experiment JSON);
  `evidence/a1eval/` (per-run summaries and per-task scores of the official three-subject evaluation)

This is a research snapshot of the Wringer code path only (the `corkscrew` name in evidence paths is the project's
earlier codename). It is not the "hoops" training engine, which is a separate project. Scripts assume a single GPU,
`vllm serve` for evaluation, and the paths in `chain_*.sh` (`$REPO`, `$SCRATCH`) adjusted to your machine.

Key entry points: `p2_anchor/wringer/quantize.py` (codes + closed-form scales), `res_fill.py` (water),
`pack_verify.py` / `pack_materialize.py` / `hub/wringer_unpack.py` (container), `p2_anchor/a1eval/` (evaluation).

License: Apache-2.0 (code). Model weights follow the parent model's license.
