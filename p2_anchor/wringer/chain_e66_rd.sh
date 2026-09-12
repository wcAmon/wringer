#!/bin/bash
# E66-RD 率失真台階(prereg_e66_rd.json;零訓練;he 單科裁)
#   R1/R2/R3:g9 全模組 GPTQ + 率罰(λ 由 rd_lambda_pick.json:H≈3.0/2.9/2.8)
#   S1g:L0–7 九→三 GPTQ 直解對照(--rate-level 3)
#   每點:ladder → export --verify → vLLM → he → 刪 export(態留 data/qs_rd_*)→ verdict_e66_rd.json
#   R1 he < 80 ⇒ FALSIFIER_RD:仍跑 S1g(對照獨立),跳過 R2/R3
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval
EVC=evidence/p1_grouping/corkscrew
PICK=$EVC/rd_lambda_pick.json
fail() { echo "E66RD_FAIL:$1"; date; exit 1; }
[ -f "$PICK" ] || fail no_lambda_pick
LAM1=$(python3 -c "import json;print(json.load(open('$PICK'))['R1'])")
LAM2=$(python3 -c "import json;print(json.load(open('$PICK'))['R2'])")
LAM3=$(python3 -c "import json;print(json.load(open('$PICK'))['R3'])")
echo "E66RD_LAMBDAS R1=$LAM1 R2=$LAM2 R3=$LAM3"

serve() {
  PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/vllm serve "$1" \
    --served-model-name a1 --reasoning-parser qwen3 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --port 8000 > "$S/vllm_e66rd.log" 2>&1 &
  vpid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_e66rd.log" && return 0
    grep -qE "initialization failed" "$S/vllm_e66rd.log" && fail serve
    sleep 1
  done; fail serve_timeout
}

# point <tag> <ladder args...>
point() {
  local TAG=$1; shift 1
  if [ ! -f "data/qs_$TAG/layer31.pt" ] || [ ! -f "$EVC/ladder_$TAG.json" ]; then
    PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.ladder_e65x --mode rate --save-tag $TAG "$@" > "$S/ladder_$TAG.log" 2>&1 || fail "ladder_$TAG"
    grep -q "^LADDER_DONE" "$S/ladder_$TAG.log" || fail "ladder_sentinel_$TAG"
  fi
  grep "^LADDER_LEDGER" "$S/ladder_$TAG.log"
  echo "E66RD_LADDER_DONE:$TAG"; date
  local OUT=exports/e66rd_$TAG sfx=e66rd_$TAG
  if [ ! -f "$A1/humaneval_${sfx}/summary.json" ]; then
    [ -f "$OUT/config.json" ] || \
      PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export --state-tag $TAG --out $OUT --verify > "$S/export_$TAG.log" 2>&1 || fail "export_$TAG"
    pgrep -f "vllm serve" >/dev/null && fail gpu_busy_vllm
    serve "$OUT"
    PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner \
      --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --out "$A1/humaneval_${sfx}" > "$S/eval_he_$TAG.log" 2>&1 || fail "eval_he_$TAG"
    kill $vpid 2>/dev/null; wait $vpid 2>/dev/null; sleep 8
    rm -rf "$OUT"
  fi
  HE=$(python3 -c "import json;print(json.load(open('$A1/humaneval_${sfx}/summary.json'))['score'])")
  echo "E66RD_POINT:$TAG he=$HE"; date
}

echo "E66RD_START"; date
pgrep -f "vllm serve" >/dev/null && fail gpu_busy_at_start

point rd_r1 --lam $LAM1
HE_R1=$(python3 -c "import json;print(json.load(open('$A1/humaneval_e66rd_rd_r1/summary.json'))['score'])")
if python3 -c "import sys;sys.exit(0 if $HE_R1 >= 0.80 else 1)"; then
  point rd_r2 --lam $LAM2
  point rd_r3 --lam $LAM3
else
  echo "E66RD_FALSIFIER_RD:he_r1=$HE_R1 <0.80,跳過 R2/R3"; date
fi
point rd_s1g --rate-level 3 --rate-layers 0-7

PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.verdict_e66_rd || fail verdict
echo "E66RD_CHAIN_DONE"; date
