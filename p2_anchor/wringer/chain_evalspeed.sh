#!/bin/bash
# 評測引擎加速量測(用戶 09-12 19:xx「把引擎優化成適應低位元的加速」):四個槓桿各量加速倍數與分數等價
#   A  vLLM Qwen3-4B bf16:IF w64 / w128、GS w64;再 KV fp8:IF w128、GS w128(對照既有 w16:IF 260s 81.33 / GS 1165s 94.84)
#   B  vLLM A1 p3b_w2:gptq4 export(Marlin)IF w64 + he 4 seed;bf16 pack export IF w64(對照既有 w16 IF 1575s 88.17;he 85.37±1.32)
#   C  llama.cpp A1 u27 GGUF:GS --limit 200 於 -np 8 vs -np 32(對照全量 12221s @np8 w16)
# 哨兵:ES_START / ES_POINT:<tag> <score> <gen_s> <usage> / ES_FAIL:<step> / ES_CHAIN_DONE;log $S/evalspeed_chain.log
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval
LC=$HOME/llama.cpp/build/bin
Q3=~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
export PATH=$PWD/.venv-vllm/bin:$PATH
fail() { echo "ES_FAIL:$1"; date; exit 1; }
idle() { while pgrep -f "^\S*python -m p2_anchor|^\S*vllm serve|^\S*/llama-server" >/dev/null; do sleep 15; done; }
serve_vllm() {   # $1 model path, rest extra args
  local m=$1; shift
  .venv-vllm/bin/vllm serve "$m" --served-model-name a1 --reasoning-parser qwen3 --max-model-len 32768 --gpu-memory-utilization 0.85 --port 8000 "$@" > "$S/vllm_es.log" 2>&1 &
  spid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_es.log" && return 0
    grep -qE "initialization failed|Traceback" "$S/vllm_es.log" && fail serve
    sleep 1
  done; fail serve_timeout
}
serve_llama() {  # $1 gguf, $2 np, $3 ctx
  $LC/llama-server -m "$1" -a a1 --host 127.0.0.1 --port 8000 -ngl 99 -c $3 -np $2 -fa on --jinja --reasoning-format none > "$S/llama_es.log" 2>&1 &
  spid=$!
  for i in $(seq 1 600); do
    grep -q "server is listening" "$S/llama_es.log" && return 0
    grep -qiE "error|failed" "$S/llama_es.log" && fail serve_llama
    sleep 1
  done; fail serve_llama_timeout
}
stop_srv() { kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 8; }
score() { python3 -c "import json;d=json.load(open('$1/summary.json'));print(d['strict']['prompt_level'] if 'strict' in d else d['score'], d.get('generation_s'), json.dumps(d.get('usage') or {}))"; }
point() { echo "ES_POINT:$1 $(score $A1/speed_$1)"; date; }
runif() { [ -f "$A1/speed_$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.ifeval_runner --base-url http://127.0.0.1:8000/v1 --model a1 --workers $2 --out "$A1/speed_$1" > "$S/es_$1.log" 2>&1 || fail "$1"; point $1; }
rungs() { [ -f "$A1/speed_$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject gsm8k --base-url http://127.0.0.1:8000/v1 --model a1 --workers $2 ${3:+--limit $3} --out "$A1/speed_$1" > "$S/es_$1.log" 2>&1 || fail "$1"; point $1; }
runhe() { [ -f "$A1/speed_$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --workers 128 --seed $2 --out "$A1/speed_$1" > "$S/es_$1.log" 2>&1 || fail "$1"; point $1; }
echo "ES_START"; date
# ---- A: Qwen3-4B bf16
idle; serve_vllm "$Q3"
runif q3_if_w64 64
runif q3_if_w128 128
rungs q3_gs_w64 64
stop_srv
idle; serve_vllm "$Q3" --kv-cache-dtype fp8
runif q3_if_w128_kv8 128
rungs q3_gs_w128_kv8 128
stop_srv
# ---- B: A1 p3b_w2 gptq4 vs bf16 pack
idle; serve_vllm exports/e69_p3b_w2_gptq4
runif a1g4_if_w64 64
for sd in 20260806 1 2 3; do runhe a1g4_he_s$sd $sd; done
stop_srv
idle; serve_vllm exports/e69_p3b_w2_pack
runif a1bf_if_w64 64
stop_srv
# ---- C: llama.cpp u27 GGUF slots
idle; serve_llama data/gguf/a1-4b-u27.gguf 8 262144
rungs u27_gs200_np8 16 200
stop_srv
idle; serve_llama data/gguf/a1-4b-u27.gguf 32 524288
rungs u27_gs200_np32 32 200
stop_srv
echo "ES_CHAIN_DONE"; date
