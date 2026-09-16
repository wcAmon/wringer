# Wringer

Low-bit weight quantization by **fill-and-wring**: fixed-grid codes → a low-rank "water" adapter trained by
distillation on the model's own long trajectories → a closed-form "wring" that re-solves scales and codes so that no
adapter is shipped. Results: InternScience/Agents-A1-4B (Qwen3.5 hybrid) at 2.655 body bits/weight keeps 0.947 of the
bf16 IFEval / HumanEval / GSM8K mean; Qwen3-4B at 2.638 b/w keeps 0.922. Every zero-training method we measured on the
same models (GPTQ, HQQ, AWQ, llama.cpp GGUF) falls to ≤ 0.65 or collapses between 2.5 and 2.75 b/w.

- **Technical report (2026-09-16):** `p2_anchor/wringer/TECH_REPORT_wringer.md` — two-model tables, the zero-training
  collapse curve, and the attribution experiments E71 (same-budget STE-QAT), E73 (same filled weights, four finishers),
  E74 (same finisher, no fill). Figure: `evidence/corkscrew/fig_collapse_curve.png`.
- Models + containers + reports: https://huggingface.co/wcamon/Agents-A1-4B-Wringer-Q2.6 ,
  https://huggingface.co/wcamon/Qwen3-4B-Wringer-Q2.6 (v0.3: nearest-point + joint-scale finisher)
- Article (first model): Hugging Face blog (link added after publication)
- Method retrospective (zh-TW): `p2_anchor/wringer/RETRO_wringer.md`; experiment history E1–E69: `p2_anchor/wringer/HISTORY_e1_e69.md`;
  E70–E74 are documented in their pre-registrations (`prereg_e70_qwen3_4b.json` … `prereg_e74_zerortnj.json`)
- Pre-registrations: `p2_anchor/wringer/prereg_*.json`; verdicts: `p2_anchor/wringer/verdict_*.json`
- Evidence: `evidence/corkscrew/` (contamination check, packing verification, multi-seed HumanEval, per-experiment JSON);
  `evidence/a1eval/` (per-run summaries and per-task scores of the official three-subject evaluation)

Snapshot 2 (2026-09-16) adds the Qwen3-4B line (E70), the STE control (E71), external baselines via llm-compressor / hqq /
llama.cpp (E72, `baseline_quant.py`, `a1_wrap.py`), the finisher ablations (E73/E74, `quantize.py --solver rtn --rtn-scale joint`),
the grid-friendliness instrument (`instr_gridfriendly.py`) and the figure script (`fig_collapse_curve.py`).

This is a research snapshot of the Wringer code path only (the `corkscrew` name in evidence paths is the project's
earlier codename). It is not the "hoops" training engine, which is a separate project. Scripts assume a single GPU,
`vllm serve` for evaluation, and the paths in `chain_*.sh` (`$REPO`, `$SCRATCH`) adjusted to your machine.

Key entry points: `p2_anchor/wringer/quantize.py` (codes + closed-form scales; `--solver gptq|rtn`, `--rtn-scale search|joint`), `res_fill.py` (water),
`pack_verify.py` / `pack_materialize.py` / `hub/wringer_unpack.py` (container), `p2_anchor/a1eval/` (evaluation).

License: Apache-2.0 (code). Model weights follow the parent model's license.
