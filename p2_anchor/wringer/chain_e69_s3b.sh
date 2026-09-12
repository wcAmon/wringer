#!/bin/bash
# E69 第三段續:p3b_w B 閘 84.15 ≫ U(R-flip 條件未觸)⇒ p3b_w 為 p3b 線最終檔,官方三科(prereg「最終檔官方」用戶 Ok 已含)先跑,
# 再接 p3a R-water 同流程(chain_e69_s3w 的 water 函式原樣)。取代 chain_e69_s3w.sh(p3a fill 剛起跑即中止,重跑)。
# 哨兵:E69S3_OFFICIAL:<t> <sub> .. / E69S3_OFFICIAL_DONE:<t> / 其餘同 chain_e69_s3w
cd $REPO || exit 1
S=$SCRATCH
EVC=evidence/p1_grouping/corkscrew; A1=evidence/p1_grouping/a1eval
CAL="--calib evidence/p1_grouping/calib_e58r1b_traj.pt --n-calib 128"
BASE="self_attn.k_proj:256:row,self_attn.v_proj:256:row,self_attn.o_proj:16:128"
MM_A="$BASE,linear_attn.in_proj_z:4:128,linear_attn.in_proj_qkv:4:128,mlp.gate_proj:4:128,mlp.up_proj:4:128"
fail() { echo "E69S3_FAIL:$1"; date; exit 1; }
idle() { while pgrep -f "^\S*python -m p2_anchor|^\S*vllm serve|^\S*/llama-server" >/dev/null; do sleep 30; done; }
serve_vllm() {
  PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/vllm serve "$1" --served-model-name a1 --reasoning-parser qwen3 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --port 8000 > "$S/vllm_e69s3.log" 2>&1 &
  spid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_e69s3.log" && return 0
    grep -qE "initialization failed" "$S/vllm_e69s3.log" && fail serve
    sleep 1
  done; fail serve_timeout
}
score() { python3 -c "import json;d=json.load(open('$1/summary.json'));print(d['strict']['prompt_level'] if 'strict' in d else d['score'])"; }
he_of() {   # <export dir> <sfx> <failtag>
  if [ ! -f "$A1/humaneval_$2/summary.json" ]; then
    idle
    serve_vllm "$1"
    PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --out "$A1/humaneval_$2" > "$S/eval_he_$2.log" 2>&1 || fail "eval_he_$3"
    kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 8
  fi
}
official() {   # <export dir> <sfx>
  local OUT=$1 sfx=$2
  if [ ! -f "$A1/ifeval_${sfx}/summary.json" ] || [ ! -f "$A1/gsm8k_${sfx}/summary.json" ]; then
    idle
    serve_vllm "$OUT"
    [ -f "$A1/ifeval_${sfx}/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.ifeval_runner --base-url http://127.0.0.1:8000/v1 --model a1 --workers 16 --out "$A1/ifeval_${sfx}" > "$S/eval_if_$sfx.log" 2>&1 || fail "eval_if_$sfx"
    [ -f "$A1/gsm8k_${sfx}/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject gsm8k --base-url http://127.0.0.1:8000/v1 --model a1 --workers 16 --out "$A1/gsm8k_${sfx}" > "$S/eval_gs_$sfx.log" 2>&1 || fail "eval_gs_$sfx"
    kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 8
  fi
  echo "E69S3_OFFICIAL:$sfx if $(score $A1/ifeval_${sfx})"
  echo "E69S3_OFFICIAL:$sfx gs $(score $A1/gsm8k_${sfx})"
  echo "E69S3_OFFICIAL_DONE:$sfx"; date
}
water() {
  local T=$1 MM=$2; shift 2
  local RESA=exports/e69_${T}_resA W=${T}_w OUTW=exports/e69_${T}_w
  echo "E69S3_START:$T"; date
  [ -f "exports/e69_$T/config.json" ] || fail "no_export_$T"
  if [ ! -f "$EVC/fill_res69_$T.json" ]; then
    idle
    PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.res_fill \
      --tag res69_$T --state-tag $T --ref-export exports/e69_$T --out $RESA --r-res 128 \
      --data evidence/p1_grouping/calib_e58r1b_traj.pt --lengths evidence/p1_grouping/calib_e58r1b_len.pt \
      --batch 2 --microbatch 1 --kd-impl hidchunk --grad-ckpt \
      --close-weight 32 --close-post-weight 16 --close-post-k 8 \
      --steps 3000 --plateau-eps 0.002 --plateau-min-steps 3000 > "$S/fill_res69_$T.log" 2>&1 || fail "fill_$T"
  fi
  echo "E69S3_FILL_DONE:$T $(grep -E 'pre-fill|post-fill' $S/fill_res69_$T.log | tr '\n' ' ' | cut -c1-160)"; date
  he_of $RESA e69_${T}_resA "${T}_resA"
  echo "E69S3_A:$T he=$(score $A1/humaneval_e69_${T}_resA)"; date
  if [ ! -f "data/qs_$W/layer31.pt" ]; then
    idle
    PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.quantize --tag $W --base-dir $RESA $CAL --grid 8 --block 128 --solver gptq --prior-lam 0.01 --module-map "$MM" "$@" > "$S/quant_$W.log" 2>&1 || fail "quant_$W"
  fi
  grep "^QUANT_LEDGER" "$S/quant_$W.log" | sed "s/^/E69S3_LEDGER:$T /"
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.state_entropy --state-tag $W --out $EVC/entropy_$W.json 2>/dev/null | grep STATE_ENTROPY | cut -c1-200 | sed "s/^/E69S3_ENTROPY:$T /"
  [ -f "$OUTW/config.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export --state-tag $W --out $OUTW > "$S/export_$W.log" 2>&1 || fail "export_$W"
  he_of $OUTW e69_$W "$W"
  echo "E69S3_B:$T he=$(score $A1/humaneval_e69_$W)"; date
}

sleep 15; idle
official exports/e69_p3b_w e69_p3b_w
water p3a "$MM_A"
echo "E69S3_CHAIN_DONE"; date
