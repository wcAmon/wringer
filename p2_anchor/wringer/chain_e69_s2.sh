#!/bin/bash
# E69 第二段:由 S1 代價表組 P3 候選(prereg_e69.json stage2;點火需用戶核)。等 chain_e69_s1 E69S1_CHAIN_DONE →
#   p3a 深砍:z4 + qkv4 + gate4 + up4(α fp16)            est 2.570  單項加總預測 he ≈ 76.8
#   p3b 小刀:z4 + qkv4 + gate4 + out4 + α int8            est 2.654  單項加總預測 he ≈ 79.9
#   各:閉式排水 → 熵帳 → export → vLLM → he;兩點 he 齊 → he 高者先官方 IF+GS,再另一點
# 哨兵:E69S2_START / E69S2_LEDGER:<tag> / E69S2_POINT:<tag> he=.. / E69S2_SCORE:<tag> <sub> .. / E69S2_OFFICIAL_DONE:<tag> / E69S2_FAIL / E69S2_CHAIN_DONE
cd $REPO || exit 1
S=$SCRATCH
EVC=evidence/p1_grouping/corkscrew; A1=evidence/p1_grouping/a1eval
CAL="--calib evidence/p1_grouping/calib_e58r1b_traj.pt --n-calib 128"
BASE="self_attn.k_proj:256:row,self_attn.v_proj:256:row,self_attn.o_proj:16:128"
MM_A="$BASE,linear_attn.in_proj_z:4:128,linear_attn.in_proj_qkv:4:128,mlp.gate_proj:4:128,mlp.up_proj:4:128"
MM_B="$BASE,linear_attn.in_proj_z:4:128,linear_attn.in_proj_qkv:4:128,mlp.gate_proj:4:128,linear_attn.out_proj:4:128"
fail() { echo "E69S2_FAIL:$1"; date; exit 1; }
idle() { while pgrep -f "vllm serv[e]|llama-serve[r]|e2e_sof[t]|gsq_bloc[k]|corkscrew.quantiz[e]" >/dev/null; do sleep 30; done; }
serve_vllm() {
  PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/vllm serve "$1" --served-model-name a1 --reasoning-parser qwen3 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --port 8000 > "$S/vllm_e69s2.log" 2>&1 &
  spid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_e69s2.log" && return 0
    grep -qE "initialization failed" "$S/vllm_e69s2.log" && fail serve
    sleep 1
  done; fail serve_timeout
}
score() { python3 -c "import json;d=json.load(open('$1/summary.json'));print(d['strict']['prompt_level'] if 'strict' in d else d['score'])"; }
export_of() {   # <tag> → 確保 exports/e69_<tag> 存在
  local TAG=$1 OUT=exports/e69_$1
  [ -f "$OUT/config.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export --state-tag $TAG --out $OUT > "$S/export_$TAG.log" 2>&1 || fail "export_$TAG"
}
# build <tag> <module-map> <extra args...>:閉式排水 → 熵帳 → he(export 保留給官方段與 S3)
build() {
  local TAG=$1 MM=$2; shift 2
  local OUT=exports/e69_$TAG sfx=e69_$TAG
  if [ ! -f "data/qs_$TAG/layer31.pt" ]; then
    idle
    PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.quantize --tag $TAG $CAL --grid 8 --block 128 --solver gptq --prior-lam 0.01 --module-map "$MM" "$@" > "$S/quant_$TAG.log" 2>&1 || fail "quant_$TAG"
  fi
  grep "^QUANT_LEDGER" "$S/quant_$TAG.log" | sed "s/^/E69S2_LEDGER:$TAG /"
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.state_entropy --state-tag $TAG --out $EVC/entropy_$TAG.json 2>/dev/null | grep STATE_ENTROPY | cut -c1-200 | sed "s/^/E69S2_ENTROPY:$TAG /"
  if [ ! -f "$A1/humaneval_${sfx}/summary.json" ]; then
    export_of $TAG
    idle; pgrep -f "vllm serv[e]|llama-serve[r]" >/dev/null && fail gpu_busy
    serve_vllm "$OUT"
    PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --out "$A1/humaneval_${sfx}" > "$S/eval_he_$TAG.log" 2>&1 || fail "eval_he_$TAG"
    kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 8
  fi
  echo "E69S2_POINT:$TAG he=$(score $A1/humaneval_${sfx})"; date
}
official() {   # <tag>:IF + GS
  local TAG=$1 OUT=exports/e69_$1 sfx=e69_$1
  if [ ! -f "$A1/ifeval_${sfx}/summary.json" ] || [ ! -f "$A1/gsm8k_${sfx}/summary.json" ]; then
    export_of $TAG
    idle; pgrep -f "vllm serv[e]|llama-serve[r]" >/dev/null && fail gpu_busy
    serve_vllm "$OUT"
    [ -f "$A1/ifeval_${sfx}/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.ifeval_runner --base-url http://127.0.0.1:8000/v1 --model a1 --workers 16 --out "$A1/ifeval_${sfx}" > "$S/eval_if_$TAG.log" 2>&1 || fail "eval_if_$TAG"
    [ -f "$A1/gsm8k_${sfx}/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject gsm8k --base-url http://127.0.0.1:8000/v1 --model a1 --workers 16 --out "$A1/gsm8k_${sfx}" > "$S/eval_gs_$TAG.log" 2>&1 || fail "eval_gs_$TAG"
    kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 8
  fi
  echo "E69S2_SCORE:$TAG if $(score $A1/ifeval_${sfx})"
  echo "E69S2_SCORE:$TAG gs $(score $A1/gsm8k_${sfx})"
  echo "E69S2_OFFICIAL_DONE:$TAG"; date
}

until grep -qE "E69S1_CHAIN_DONE|E69S1_FAIL" $S/e69s1_chain.log 2>/dev/null; do sleep 60; done
grep -q "E69S1_FAIL" $S/e69s1_chain.log && echo "E69S2_WARN:s1_chain_failed(U26 可能缺;續跑)"
sleep 20; idle
echo "E69S2_START"; date

build p3a "$MM_A"
build p3b "$MM_B" --alpha-bits 8

HA=$(score $A1/humaneval_e69_p3a); HB=$(score $A1/humaneval_e69_p3b)
if python3 -c "import sys;sys.exit(0 if $HA >= $HB else 1)"; then ORDER="p3a p3b"; else ORDER="p3b p3a"; fi
echo "E69S2_ORDER:$ORDER"
for t in $ORDER; do official $t; done
echo "E69S2_CHAIN_DONE"; date
