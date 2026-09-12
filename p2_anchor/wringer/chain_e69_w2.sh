#!/bin/bash
# E69 續段(用戶 09-11 20:5x 裁方向:p3b 先、p3a 後;點火另核):
#   W2:p3b_w(冠軍 2.655)再蓄一輪水 → A₂ he → 閉式排水回同配置(p3b_w2)→ B₂ he → 官方 IF+GS
#   P3A:p3a(2.571)蓄水 → A he → 排水(p3a_w)→ B he → 官方 IF+GS
# 哨兵:E69W2_START:<t> / E69W2_FILL_DONE:<t> / E69W2_A:<t> he=.. / E69W2_LEDGER:<t> / E69W2_B:<t> he=.. / E69W2_OFFICIAL:<sfx> .. / E69W2_OFFICIAL_DONE:<sfx> / E69W2_FAIL:<t> / E69W2_CHAIN_DONE
cd $REPO || exit 1
S=$SCRATCH
EVC=evidence/p1_grouping/corkscrew; A1=evidence/p1_grouping/a1eval
CAL="--calib evidence/p1_grouping/calib_e58r1b_traj.pt --n-calib 128"
BASE="self_attn.k_proj:256:row,self_attn.v_proj:256:row,self_attn.o_proj:16:128"
MM_A="$BASE,linear_attn.in_proj_z:4:128,linear_attn.in_proj_qkv:4:128,mlp.gate_proj:4:128,mlp.up_proj:4:128"
MM_B="$BASE,linear_attn.in_proj_z:4:128,linear_attn.in_proj_qkv:4:128,mlp.gate_proj:4:128,linear_attn.out_proj:4:128"
fail() { echo "E69W2_FAIL:$1"; date; exit 1; }
idle() { while pgrep -f "^\S*python -m p2_anchor|^\S*vllm serve|^\S*/llama-server" >/dev/null; do sleep 30; done; }
serve_vllm() {
  PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/vllm serve "$1" --served-model-name a1 --reasoning-parser qwen3 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --port 8000 > "$S/vllm_e69w2.log" 2>&1 &
  spid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_e69w2.log" && return 0
    grep -qE "initialization failed" "$S/vllm_e69w2.log" && fail serve
    sleep 1
  done; fail serve_timeout
}
score() { python3 -c "import json;d=json.load(open('$1/summary.json'));print(d['strict']['prompt_level'] if 'strict' in d else d['score'])"; }
he_of() {   # <export dir> <sfx>
  if [ ! -f "$A1/humaneval_$2/summary.json" ]; then
    idle; serve_vllm "$1"
    PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --out "$A1/humaneval_$2" > "$S/eval_he_$2.log" 2>&1 || fail "eval_he_$2"
    kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 8
  fi
}
official() {   # <export dir> <sfx>
  local OUT=$1 sfx=$2
  if [ ! -f "$A1/ifeval_${sfx}/summary.json" ] || [ ! -f "$A1/gsm8k_${sfx}/summary.json" ]; then
    idle; serve_vllm "$OUT"
    [ -f "$A1/ifeval_${sfx}/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.ifeval_runner --base-url http://127.0.0.1:8000/v1 --model a1 --workers 16 --out "$A1/ifeval_${sfx}" > "$S/eval_if_$sfx.log" 2>&1 || fail "eval_if_$sfx"
    [ -f "$A1/gsm8k_${sfx}/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject gsm8k --base-url http://127.0.0.1:8000/v1 --model a1 --workers 16 --out "$A1/gsm8k_${sfx}" > "$S/eval_gs_$sfx.log" 2>&1 || fail "eval_gs_$sfx"
    kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 8
  fi
  echo "E69W2_OFFICIAL:$sfx if $(score $A1/ifeval_${sfx})"
  echo "E69W2_OFFICIAL:$sfx gs $(score $A1/gsm8k_${sfx})"
  echo "E69W2_OFFICIAL_DONE:$sfx"; date
}
# water <src state tag> <out tag> <module-map> <extra quantize args...>
water() {
  local SRC=$1 W=$2 MM=$3; shift 3
  local RESA=exports/e69_${SRC}_resA OUTW=exports/e69_$W
  echo "E69W2_START:$W"; date
  [ -f "exports/e69_$SRC/config.json" ] || fail "no_export_$SRC"
  if [ ! -f "$EVC/fill_res69_$W.json" ]; then
    idle
    PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.res_fill \
      --tag res69_$W --state-tag $SRC --ref-export exports/e69_$SRC --out $RESA --r-res 128 \
      --data evidence/p1_grouping/calib_e58r1b_traj.pt --lengths evidence/p1_grouping/calib_e58r1b_len.pt \
      --batch 2 --microbatch 1 --kd-impl hidchunk --grad-ckpt \
      --close-weight 32 --close-post-weight 16 --close-post-k 8 \
      --steps 3000 --plateau-eps 0.002 --plateau-min-steps 3000 > "$S/fill_res69_$W.log" 2>&1 || fail "fill_$W"
  fi
  echo "E69W2_FILL_DONE:$W $(grep -E 'pre-fill|post-fill' $S/fill_res69_$W.log | tr '\n' ' ' | cut -c1-160)"; date
  he_of $RESA e69_${SRC}_resA
  echo "E69W2_A:$W he=$(score $A1/humaneval_e69_${SRC}_resA)"; date
  if [ ! -f "data/qs_$W/layer31.pt" ]; then
    idle
    PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.quantize --tag $W --base-dir $RESA $CAL --grid 8 --block 128 --solver gptq --prior-lam 0.01 --module-map "$MM" "$@" > "$S/quant_$W.log" 2>&1 || fail "quant_$W"
  fi
  grep "^QUANT_LEDGER" "$S/quant_$W.log" | sed "s/^/E69W2_LEDGER:$W /"
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.state_entropy --state-tag $W --out $EVC/entropy_$W.json 2>/dev/null | grep STATE_ENTROPY | cut -c1-200 | sed "s/^/E69W2_ENTROPY:$W /"
  [ -f "$OUTW/config.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export --state-tag $W --out $OUTW > "$S/export_$W.log" 2>&1 || fail "export_$W"
  he_of $OUTW e69_$W
  echo "E69W2_B:$W he=$(score $A1/humaneval_e69_$W)"; date
  official $OUTW e69_$W
}

sleep 15; idle
water p3b_w p3b_w2 "$MM_B" --alpha-bits 8
water p3a   p3a_w  "$MM_A"
echo "E69W2_CHAIN_DONE"; date
