#!/bin/bash
# E69 第三段 R-water(用戶 09-10 21:xx「同意 點火」):p3b → p3a 各一輪 蓄水 → A 閘 → 閉式排水 → B 閘
#   等 chain_e69_u25gs 收(E69S1_POINT:u25gs|E69S1_FAIL)→ 每點:
#   res_fill(碼凍 r128 旁路 KD,bf16 自身老師,calib_e58r1b 128 列 16k,閉合加權 32/16/8,3000 步 batch 2,固定步數)
#   → exports/e69_<t>_resA → vLLM he(A 閘)→ quantize --base-dir resA 同配置(E64 排水)→ 熵帳 → export → vLLM he(B 閘)
# 哨兵:E69S3_START:<t> / E69S3_FILL_DONE:<t> / E69S3_A:<t> he=.. / E69S3_LEDGER:<t> / E69S3_B:<t> he=.. / E69S3_FAIL:<t> / E69S3_CHAIN_DONE
cd $REPO || exit 1
S=$SCRATCH
EVC=evidence/p1_grouping/corkscrew; A1=evidence/p1_grouping/a1eval
CAL="--calib evidence/p1_grouping/calib_e58r1b_traj.pt --n-calib 128"
BASE="self_attn.k_proj:256:row,self_attn.v_proj:256:row,self_attn.o_proj:16:128"
MM_A="$BASE,linear_attn.in_proj_z:4:128,linear_attn.in_proj_qkv:4:128,mlp.gate_proj:4:128,mlp.up_proj:4:128"
MM_B="$BASE,linear_attn.in_proj_z:4:128,linear_attn.in_proj_qkv:4:128,mlp.gate_proj:4:128,linear_attn.out_proj:4:128"
fail() { echo "E69S3_FAIL:$1"; date; exit 1; }
idle() { while pgrep -f "vllm serv[e]|llama-serve[r]|e2e_sof[t]|res_fil[l]|corkscrew.quantiz[e]" >/dev/null; do sleep 30; done; }
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
    idle; pgrep -f "vllm serv[e]|llama-serve[r]" >/dev/null && fail gpu_busy
    serve_vllm "$1"
    PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --out "$A1/humaneval_$2" > "$S/eval_he_$2.log" 2>&1 || fail "eval_he_$3"
    kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 8
  fi
}
# water <t> <module-map> <extra quantize args...>
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

until grep -qE "E69S1_POINT:u25gs|E69S1_FAIL" $S/e69s1_chain.log 2>/dev/null; do sleep 60; done
sleep 20; idle

water p3b "$MM_B" --alpha-bits 8
water p3a "$MM_A"
echo "E69S3_CHAIN_DONE"; date
