#!/bin/bash
# 第二模型 Qwen3-4B(純 transformer)前置:三科去污染探針 + bf16 官方三科錨(用戶 09-12「繼續下一步」)
#   1) 等下載;vLLM 服 Qwen3-4B → guided-completion 探針(三集)→ 官方 humaneval / ifeval / gsm8k(A1 協定,思考模式)
#   2) vLLM 服 A1 bf16 → 同探針(對照)
#   3) 計分:contam_probe_score(model=qwen3_4b, control=a1)
# 哨兵:M2P_START / M2P_DL / M2P_PROBE:<tag> / M2P_OFFICIAL:<subject> .. / M2P_SCORE .. / M2P_FAIL:.. / M2P_CHAIN_DONE
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval
BF16=~/.cache/huggingface/hub/models--InternScience--Agents-A1-4B/snapshots/945c40a4aa6f534d434a353207b8d42ecf7a5293
fail() { echo "M2P_FAIL:$1"; date; exit 1; }
idle() { while pgrep -f "^\S*python -m p2_anchor|^\S*vllm serve|^\S*/llama-server" >/dev/null; do sleep 30; done; }
serve_vllm() {
  PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/vllm serve "$1" --served-model-name a1 --reasoning-parser qwen3 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --port 8000 > "$S/vllm_m2p.log" 2>&1 &
  spid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_m2p.log" && return 0
    grep -qE "initialization failed" "$S/vllm_m2p.log" && fail serve
    sleep 1
  done; fail serve_timeout
}
stop_vllm() { kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 8; }
score() { python3 -c "import json;d=json.load(open('$1/summary.json'));print(d['strict']['prompt_level'] if 'strict' in d else d['score'])"; }
echo "M2P_START"; date
for i in $(seq 1 720); do grep -q DL_DONE "$S/dl_qwen3_4b.log" && break; grep -q Traceback "$S/dl_qwen3_4b.log" && fail download; sleep 10; done
Q=$(grep DL_DONE "$S/dl_qwen3_4b.log" | awk '{print $2}'); [ -f "$Q/config.json" ] || fail no_model
echo "M2P_DL $Q"; date
T=qwen3_4b_bf16
idle; serve_vllm "$Q"
[ -f "$A1/probe_qwen3_4b/gsm8k.jsonl" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.contam_probe_gen --out "$A1/probe_qwen3_4b" > "$S/probe_qwen3_4b.log" 2>&1 || fail probe_qwen
echo "M2P_PROBE:qwen3_4b"; date
[ -f "$A1/humaneval_$T/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --out "$A1/humaneval_$T" > "$S/eval_he_$T.log" 2>&1 || fail eval_he
echo "M2P_OFFICIAL:humaneval $(score $A1/humaneval_$T)"; date
[ -f "$A1/ifeval_$T/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.ifeval_runner --base-url http://127.0.0.1:8000/v1 --model a1 --workers 16 --out "$A1/ifeval_$T" > "$S/eval_if_$T.log" 2>&1 || fail eval_if
echo "M2P_OFFICIAL:ifeval $(score $A1/ifeval_$T)"; date
[ -f "$A1/gsm8k_$T/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject gsm8k --base-url http://127.0.0.1:8000/v1 --model a1 --workers 16 --out "$A1/gsm8k_$T" > "$S/eval_gs_$T.log" 2>&1 || fail eval_gs
echo "M2P_OFFICIAL:gsm8k $(score $A1/gsm8k_$T)"; date
stop_vllm
if [ ! -f "$A1/probe_a1/gsm8k.jsonl" ]; then
  idle; serve_vllm "$BF16"
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.contam_probe_gen --out "$A1/probe_a1" > "$S/probe_a1.log" 2>&1 || fail probe_a1
  stop_vllm
fi
echo "M2P_PROBE:a1"; date
PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.contam_probe_score --tag qwen3_4b --probe "$A1/probe_qwen3_4b" --he-run "$A1/humaneval_$T" \
  --control-tag a1 --control-probe "$A1/probe_a1" --control-he-run "$A1/humaneval_bf16_w128" > "$S/probe_score.log" 2>&1 || fail score
echo "M2P_SCORE $(grep PROBE_SCORE_DONE $S/probe_score.log | cut -c1-300)"; date
echo "M2P_CHAIN_DONE"
