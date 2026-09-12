#!/bin/bash
# E68-P 官方三科鏈(用戶核准 2026-09-09 21:2x):等 chain_e68b E68B_CHAIN_DONE →
#   P1-gptq IF+GS(重新 export → vLLM)→ U34 IF+GS(llama-server,補齊 U 曲線三科)→
#   P1-alpha IF+GS(僅當 he(p1alpha) ≥ he(p1gptq))
# 哨兵:E68C_START / E68C_SCORE:<tag> <sub> <score> / E68C_POINT_DONE:<tag> / E68C_SKIP / E68C_FAIL / E68C_CHAIN_DONE
cd $REPO || exit 1
S=$SCRATCH
EVC=evidence/p1_grouping/corkscrew; A1=evidence/p1_grouping/a1eval
LC=$HOME/llama.cpp/build/bin
fail() { echo "E68C_FAIL:$1"; date; exit 1; }
idle() { while pgrep -f "vllm serv[e]|llama-serve[r]|e2e_sof[t]|gsq_bloc[k]|corkscrew.quantiz[e]" >/dev/null; do sleep 30; done; }
serve_vllm() {
  PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/vllm serve "$1" --served-model-name a1 --reasoning-parser qwen3 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --port 8000 > "$S/vllm_e68c.log" 2>&1 &
  spid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_e68c.log" && return 0
    grep -qE "initialization failed" "$S/vllm_e68c.log" && fail serve
    sleep 1
  done; fail serve_timeout
}
serve_llama() {
  $LC/llama-server -m "$1" -a a1 --host 127.0.0.1 --port 8000 -ngl 99 -c 262144 -np 8 -fa on --jinja --reasoning-format none > "$S/llamaserver_e68c.log" 2>&1 &
  spid=$!
  for i in $(seq 1 600); do
    curl -s http://127.0.0.1:8000/health 2>/dev/null | grep -q '"ok"' && return 0
    grep -qiE "error|failed" "$S/llamaserver_e68c.log" && { kill $spid 2>/dev/null; fail serve; }
    sleep 1
  done; kill $spid 2>/dev/null; fail serve_timeout
}
run_ifgs() {   # <sfx> <tag>
  local sfx=$1 TAG=$2
  [ -f "$A1/ifeval_${sfx}/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.ifeval_runner --base-url http://127.0.0.1:8000/v1 --model a1 --workers 16 --out "$A1/ifeval_${sfx}" > "$S/eval_if_$TAG.log" 2>&1 || fail "eval_if_$TAG"
  [ -f "$A1/gsm8k_${sfx}/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject gsm8k --base-url http://127.0.0.1:8000/v1 --model a1 --workers 16 --out "$A1/gsm8k_${sfx}" > "$S/eval_gs_$TAG.log" 2>&1 || fail "eval_gs_$TAG"
}
report() {     # <sfx> <tag>
  local sfx=$1 TAG=$2
  echo "E68C_SCORE:$TAG if $(python3 -c "import json;print(json.load(open('$A1/ifeval_${sfx}/summary.json'))['strict']['prompt_level'])")"
  echo "E68C_SCORE:$TAG gs $(python3 -c "import json;print(json.load(open('$A1/gsm8k_${sfx}/summary.json'))['score'])")"
  echo "E68C_POINT_DONE:$TAG"; date
}
he_of() { python3 -c "import json;print(json.load(open('$A1/humaneval_$1/summary.json'))['score'])"; }
# ours:export 態 → vLLM → IF+GS → 刪 export
ours() {       # <tag>
  local TAG=$1 sfx=e68_$1 OUT=exports/e68_$1
  if [ ! -f "$A1/ifeval_${sfx}/summary.json" ] || [ ! -f "$A1/gsm8k_${sfx}/summary.json" ]; then
    idle
    [ -f "$OUT/config.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export --state-tag $TAG --out $OUT > "$S/export_c_$TAG.log" 2>&1 || fail "export_$TAG"
    pgrep -f "vllm serv[e]|llama-serve[r]" >/dev/null && fail gpu_busy
    serve_vllm "$OUT"
    run_ifgs $sfx $TAG
    kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 8
    rm -rf "$OUT"
  fi
  report $sfx $TAG
}

until grep -qE "E68B_CHAIN_DONE|E68B_FAIL" $S/e68b_chain.log 2>/dev/null; do sleep 60; done
grep -q "E68B_FAIL" $S/e68b_chain.log && echo "E68C_WARN:chain_e68b_failed(續跑 P1-gptq/U34)"
sleep 20; idle
echo "E68C_START"; date

# ---- P1-gptq IF+GS ----
ours p1gptq

# ---- U34 IF+GS(llama.cpp)----
sfx=e67u_u34
if [ ! -f "$A1/ifeval_${sfx}/summary.json" ] || [ ! -f "$A1/gsm8k_${sfx}/summary.json" ]; then
  idle
  pgrep -f "vllm serv[e]|llama-serve[r]" >/dev/null && fail gpu_busy
  serve_llama data/gguf/a1-4b-u34.gguf
  run_ifgs $sfx u34
  kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 5
fi
report $sfx u34

# ---- P1-alpha IF+GS(條件:he ≥ P1-gptq)----
if [ -f "$A1/humaneval_e68_p1alpha/summary.json" ] && [ -f data/qs_p1alpha/layer31.pt ]; then
  if python3 -c "import sys;sys.exit(0 if $(he_of e68_p1alpha) >= $(he_of e68_p1gptq) else 1)"; then
    ours p1alpha
  else
    echo "E68C_SKIP:p1alpha he=$(he_of e68_p1alpha) < p1gptq=$(he_of e68_p1gptq)"
  fi
else
  echo "E68C_SKIP:p1alpha_no_he"
fi
echo "E68C_CHAIN_DONE"; date
