#!/bin/bash
# E68-P 第二鏈(prereg FROZEN @8054fc8):等 chain_e68 E68_CHAIN_DONE →
#   P2-champ:pa64 冠軍 W_eff(exports/e64_pa64_asymrefitq2)閉式吸收進 P1 格式(8 級 g128 + k/v int8 + o 16 級)→ he
#   P1-alpha:p1gptq 態 e2e_soft --alpha-only 800 步(碼凍結、帳不動;閉合加權 32/16/8、hidchunk、grad-ckpt)→ 熵帳 → he
# 哨兵:E68B_START / E68B_LEDGER / E68B_ENTROPY / E68B_QUANT_DONE / E68B_E2E_DONE / E68B_POINT / E68B_FAIL / E68B_CHAIN_DONE
cd $REPO || exit 1
S=$SCRATCH
EVC=evidence/p1_grouping/corkscrew; A1=evidence/p1_grouping/a1eval
CAL="--calib evidence/p1_grouping/calib_e58r1b_traj.pt --n-calib 128"
MM="self_attn.k_proj:256:row,self_attn.v_proj:256:row,self_attn.o_proj:16:128"
fail() { echo "E68B_FAIL:$1"; date; exit 1; }
idle() { while pgrep -f "vllm serv[e]|llama-serve[r]|e2e_sof[t]|gsq_bloc[k]|corkscrew.quantiz[e]" >/dev/null; do sleep 30; done; }
serve() {
  PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/vllm serve "$1" --served-model-name a1 --reasoning-parser qwen3 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --port 8000 > "$S/vllm_e68b.log" 2>&1 &
  vpid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_e68b.log" && return 0
    grep -qE "initialization failed" "$S/vllm_e68b.log" && fail serve
    sleep 1
  done; fail serve_timeout
}
he_point() {
  local TAG=$1
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.state_entropy --state-tag $TAG --out $EVC/entropy_$TAG.json 2>/dev/null | grep STATE_ENTROPY | sed "s/^/E68B_ENTROPY:$TAG /"
  local OUT=exports/e68_$TAG sfx=e68_$TAG
  if [ ! -f "$A1/humaneval_${sfx}/summary.json" ]; then
    [ -f "$OUT/config.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export --state-tag $TAG --out $OUT > "$S/export_$TAG.log" 2>&1 || fail "export_$TAG"
    pgrep -f "vllm serv[e]|llama-serve[r]" >/dev/null && fail gpu_busy
    serve "$OUT"
    PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --out "$A1/humaneval_${sfx}" > "$S/eval_he_$TAG.log" 2>&1 || fail "eval_he_$TAG"
    kill $vpid 2>/dev/null; wait $vpid 2>/dev/null; sleep 8
    rm -rf "$OUT"
  fi
  echo "E68B_POINT:$TAG he=$(python3 -c "import json;print(json.load(open('$A1/humaneval_${sfx}/summary.json'))['score'])")"; date
}

until grep -qE "E68_CHAIN_DONE|E68_FAIL" $S/e68_chain.log 2>/dev/null; do sleep 60; done
grep -q "E68_FAIL" $S/e68_chain.log && fail "chain_e68_failed"
sleep 20; idle
echo "E68B_START"; date

# ---- P2-champ ----
TAG=p2champ
if [ ! -f "data/qs_$TAG/layer31.pt" ]; then
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.quantize --tag $TAG $CAL --base-dir exports/e64_pa64_asymrefitq2 \
    --grid 8 --block 128 --solver gptq --prior-lam 0.01 --module-map "$MM" > "$S/quant_$TAG.log" 2>&1 || fail "quant_$TAG"
fi
grep "^QUANT_LEDGER" "$S/quant_$TAG.log" | sed "s/^/E68B_LEDGER:$TAG /"
echo "E68B_QUANT_DONE:$TAG"; date
he_point $TAG

# ---- P1-alpha ----
[ -f data/qs_p1gptq/layer31.pt ] || fail no_p1gptq_state
TAG=p1alpha
if [ ! -f "data/qs_$TAG/layer31.pt" ]; then
  idle
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.e2e_soft --tag $TAG --state-tag p1gptq --alpha-only \
    --data evidence/p1_grouping/calib_e58r1b_traj.pt --lengths evidence/p1_grouping/calib_e58r1b_len.pt \
    --steps 800 --batch 2 --microbatch 1 --gamma 0.0 --kd-impl hidchunk --grad-ckpt \
    --close-weight 32 --close-post-weight 16 --close-post-k 8 > "$S/e2e_$TAG.log" 2>&1 || fail "e2e_$TAG"
  grep -q "^E2E_DONE" "$S/e2e_$TAG.log" || fail "e2e_sentinel_$TAG"
fi
grep -E "val CE|flip" "$S/e2e_$TAG.log" | sed "s/^/E68B_E2E:$TAG /"
echo "E68B_E2E_DONE:$TAG"; date
he_point $TAG
echo "E68B_CHAIN_DONE"; date
