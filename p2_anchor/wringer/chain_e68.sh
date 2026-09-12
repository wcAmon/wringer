#!/bin/bash
# E68-P 帳面對齊臂(prereg FROZEN @8054fc8):P0-rtn → P0-gptq → P1-gptq,每點 quantize(8 級留零 g128,e58r1b 128 列)
#   → state_entropy(帳)→ export → vLLM he → E68_POINT。零訓練。P1-alpha / P2-champ 另鏈。
cd $REPO || exit 1
S=$SCRATCH
EVC=evidence/p1_grouping/corkscrew; A1=evidence/p1_grouping/a1eval
CAL="--calib evidence/p1_grouping/calib_e58r1b_traj.pt --n-calib 128"
MM="self_attn.k_proj:256:row,self_attn.v_proj:256:row,self_attn.o_proj:16:128"
fail() { echo "E68_FAIL:$1"; date; exit 1; }
serve() {
  PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/vllm serve "$1" --served-model-name a1 --reasoning-parser qwen3 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --port 8000 > "$S/vllm_e68.log" 2>&1 &
  vpid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_e68.log" && return 0
    grep -qE "initialization failed" "$S/vllm_e68.log" && fail serve
    sleep 1
  done; fail serve_timeout
}
point() {
  local TAG=$1; shift
  if [ ! -f "data/qs_$TAG/layer31.pt" ]; then
    PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.quantize --tag $TAG $CAL "$@" > "$S/quant_$TAG.log" 2>&1 || fail "quant_$TAG"
  fi
  grep "^QUANT_LEDGER" "$S/quant_$TAG.log" | sed "s/^/E68_LEDGER:$TAG /"
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.state_entropy --state-tag $TAG --out $EVC/entropy_$TAG.json 2>/dev/null | grep STATE_ENTROPY | sed "s/^/E68_ENTROPY:$TAG /"
  echo "E68_QUANT_DONE:$TAG"; date
  local OUT=exports/e68_$TAG sfx=e68_$TAG
  if [ ! -f "$A1/humaneval_${sfx}/summary.json" ]; then
    [ -f "$OUT/config.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export --state-tag $TAG --out $OUT > "$S/export_$TAG.log" 2>&1 || fail "export_$TAG"
    pgrep -f "vllm serv[e]|llama-serve[r]" >/dev/null && fail gpu_busy
    serve "$OUT"
    PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --out "$A1/humaneval_${sfx}" > "$S/eval_he_$TAG.log" 2>&1 || fail "eval_he_$TAG"
    kill $vpid 2>/dev/null; wait $vpid 2>/dev/null; sleep 8
    rm -rf "$OUT"
  fi
  echo "E68_POINT:$TAG he=$(python3 -c "import json;print(json.load(open('$A1/humaneval_${sfx}/summary.json'))['score'])")"; date
}
echo "E68_START"; date
pgrep -f "vllm serv[e]|llama-serve[r]|e2e_sof[t]|gsq_bloc[k]" >/dev/null && fail gpu_busy_at_start
point p0rtn  --grid 8 --block 128 --solver rtn
point p0gptq --grid 8 --block 128 --solver gptq --prior-lam 0.01
point p1gptq --grid 8 --block 128 --solver gptq --prior-lam 0.01 --module-map "$MM"
echo "E68_CHAIN_DONE"; date
