#!/bin/bash
# E67-A 求解器對決(prereg_e67.json):gsq_block A30(H≈3.0)/ A29(H≈2.9),λ 由 gsq_lambda_pick.json;每點 export → vLLM → he
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval
EVC=evidence/p1_grouping/corkscrew
PICK=$EVC/gsq_lambda_pick.json
fail() { echo "E67A_FAIL:$1"; date; exit 1; }
[ -f "$PICK" ] || fail no_lambda_pick
LAM30=$(python3 -c "import json;print(json.load(open('$PICK'))['A30'])")
LAM29=$(python3 -c "import json;print(json.load(open('$PICK'))['A29'])")
EXTRA=$(python3 -c "import json;print(json.load(open('$PICK')).get('extra',''))")
echo "E67A_LAMBDAS A30=$LAM30 A29=$LAM29 extra=$EXTRA"

serve() {
  PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/vllm serve "$1" --served-model-name a1 --reasoning-parser qwen3 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --port 8000 > "$S/vllm_e67a.log" 2>&1 &
  vpid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_e67a.log" && return 0
    grep -qE "initialization failed" "$S/vllm_e67a.log" && fail serve
    sleep 1
  done; fail serve_timeout
}

point() {
  local TAG=$1 LAM=$2
  if [ ! -f "data/qs_$TAG/layer31.pt" ] || [ ! -f "$EVC/gsq_$TAG.json" ]; then
    PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.gsq_block --save-tag $TAG --lam $LAM $EXTRA > "$S/gsq_$TAG.log" 2>&1 || fail "gsq_$TAG"
    grep -q "^GSQ_DONE" "$S/gsq_$TAG.log" || fail "gsq_sentinel_$TAG"
  fi
  grep "^GSQ_LEDGER" "$S/gsq_$TAG.log"
  echo "E67A_GSQ_DONE:$TAG"; date
  local OUT=exports/e67a_$TAG sfx=e67a_$TAG
  if [ ! -f "$A1/humaneval_${sfx}/summary.json" ]; then
    [ -f "$OUT/config.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export --state-tag $TAG --out $OUT --verify > "$S/export_$TAG.log" 2>&1 || fail "export_$TAG"
    pgrep -f "vllm serv[e]|llama-serve[r]" >/dev/null && fail gpu_busy
    serve "$OUT"
    PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --out "$A1/humaneval_${sfx}" > "$S/eval_he_$TAG.log" 2>&1 || fail "eval_he_$TAG"
    kill $vpid 2>/dev/null; wait $vpid 2>/dev/null; sleep 8
    rm -rf "$OUT"
  fi
  echo "E67A_POINT:$TAG he=$(python3 -c "import json;print(json.load(open('$A1/humaneval_${sfx}/summary.json'))['score'])")"; date
}

echo "E67A_START"; date
pgrep -f "vllm serv[e]|llama-serve[r]" >/dev/null && fail gpu_busy_at_start
point gsq_a30 $LAM30
point gsq_a29 $LAM29
PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.verdict_e67 || fail verdict
echo "E67A_CHAIN_DONE"; date
