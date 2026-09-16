#!/bin/bash
# E71 STE-QAT 同預算對照(prereg_e71_ste.json;點火需用戶核准):
#   等 E70 打包鏈 CHAIN_DONE + S1U2 重跑的 chain_e70_s1 收尾(第 3 個 E70S1_START 之後出現 CHAIN_DONE/FAIL)→
#   臂 1 Qwen3-4B:e2e_soft --gamma 0(純 STE)3000 步 on e70_p3a → export → he 4 seed + IF + GS(E70 協定)→ comp
#   臂 2 A1:qs_p3b 若缺則閉式重解(chain_e69_s2 同參數)→ he 1 seed 對照 70.73 → e2e_soft 同款 → export → he 4 seed + IF + GS(E69 協定)→ comp
# 哨兵:E71_START / E71_HANDOFF / E71_P3B_LEDGER / E71_P3B_HE / E71_E2E:<arm> ..(post-e2e val CE + flip_vs_input;帳面=起點 P3,翻碼不改位寬)/ E71_B:<arm> he=.. / E71_OFFICIAL:<arm> .. / E71_FAIL:<step> / E71_CHAIN_DONE
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval; EV=evidence/p1_grouping; EVC=evidence/p1_grouping/corkscrew
export PATH=$PWD/.venv-vllm/bin:$PATH
spid=""
fail() { echo "E71_FAIL:$1"; date; [ -n "$spid" ] && kill $spid 2>/dev/null; exit 1; }
idle() { while pgrep -f "^\S*python -m p2_anchor|^\S*vllm serve|^\S*/llama-server|^\S*/llama-quantize|^\S*/llama-imatrix" >/dev/null; do sleep 30; done; }
serve_vllm() {   # $1 export dir, $2 extra vllm args
  .venv-vllm/bin/vllm serve "$1" --served-model-name a1 --reasoning-parser qwen3 --max-model-len 32768 --gpu-memory-utilization 0.85 $2 --port 8000 > "$S/vllm_e71.log" 2>&1 &
  spid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_e71.log" && return 0
    grep -qE "initialization failed|Traceback" "$S/vllm_e71.log" && fail serve
    sleep 1
  done; fail serve_timeout
}
stop_srv() { kill $spid 2>/dev/null; wait $spid 2>/dev/null; spid=""; sleep 8; }
score() { python3 -c "import json;d=json.load(open('$1/summary.json'));print(d['strict']['prompt_level'] if 'strict' in d else d['score'])"; }
runhe() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --workers 128 --seed $2 --out "$A1/$1" > "$S/e71_$1.log" 2>&1 || fail "$1"; }
runif() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.ifeval_runner --base-url http://127.0.0.1:8000/v1 --model a1 --workers $3 --seed $2 --out "$A1/$1" > "$S/e71_$1.log" 2>&1 || fail "$1"; }
rungs() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject gsm8k --base-url http://127.0.0.1:8000/v1 --model a1 --workers $3 --seed $2 --out "$A1/$1" > "$S/e71_$1.log" 2>&1 || fail "$1"; }
he4() { python3 -c "
import json,statistics as st;v=[json.load(open('$A1/humaneval_$1_s%s/summary.json'%s))['score']*100 for s in ('20260806','1','2','3')];print('%.2f %.2f'%(st.mean(v),st.stdev(v)))"; }
comp() { python3 -c "print(round(($1/$4+$2/$5+$3/$6)/3,4))"; }
# ---- e2e_soft 純 STE(同 res_fill 預算):$1 tag $2 state $3 data-prefix $4 last-layer
ste() {
  local TAG=$1 ST=$2 DP=$3 LL=$4
  if [ ! -f "data/qs_$TAG/layer$LL.pt" ]; then
    idle
    PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.e2e_soft --tag $TAG --state-tag $ST \
      --data $EV/${DP}_traj.pt --lengths $EV/${DP}_len.pt \
      --steps 3000 --batch 2 --microbatch 1 --gamma 0.0 --lora-r 128 --kd-impl hidchunk --grad-ckpt \
      --close-weight 32 --close-post-weight 16 --close-post-k 8 > "$S/e2e_$TAG.log" 2>&1 || fail "e2e_$TAG"
    grep -q "^E2E_DONE" "$S/e2e_$TAG.log" || fail "e2e_sentinel_$TAG"
  fi
  echo "E71_E2E:$TAG $(grep -E 'val CE|flip' $S/e2e_$TAG.log | tail -n 4 | tr '\n' ' ' | cut -c1-400)"; date
  [ -f "exports/$TAG/config.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export --state-tag $TAG --out exports/$TAG > "$S/export_$TAG.log" 2>&1 || fail "export_$TAG"
}
echo "E71_START"; date
# ---- 等 GPU:打包鏈收 + S1U2 重跑收(第 3 個 E70S1_START 之後的 CHAIN_DONE/FAIL)
s1u_end() { awk '/^E70S1_START/{n=NR} END{print n+0}' "$S/e70s1_chain.log" 2>/dev/null | xargs -I{} tail -n +{} "$S/e70s1_chain.log" | grep -E "^E70S1_(CHAIN_DONE|FAIL)"; }
for i in $(seq 1 17280); do
  grep -q "E70PK_CHAIN_DONE\|E70PK_FAIL" "$S/e70pk_chain.log" 2>/dev/null && [ "$(grep -c '^E70S1_START' $S/e70s1_chain.log 2>/dev/null)" -ge 3 ] && s1u_end >/dev/null 2>&1 && break
  sleep 10
done
echo "E71_HANDOFF pk=$(grep -oE 'E70PK_(CHAIN_DONE|FAIL[^ ]*)' $S/e70pk_chain.log | tail -n 1) s1=$(s1u_end | tail -n 1)"; date
sleep 30; idle

# ================= 臂 1:Qwen3-4B =================
export WRINGER_MODEL=Qwen/Qwen3-4B WRINGER_REV=1cfa9a7208912126459214e8b04321603b3df60c WRINGER_CALIB_VAL=$EV/calib_e70_val.pt
T=e71_ste_q
ste $T e70_p3a calib_e70 35
if [ ! -f "$A1/gsm8k_$T/summary.json" ]; then
  idle; serve_vllm exports/$T "--kv-cache-dtype fp8"
  for sd in 20260806 1 2 3; do runhe humaneval_${T}_s$sd $sd; done
  echo "E71_B:$T he4=$(he4 $T)"; date
  runif ifeval_$T 20260806 128
  rungs gsm8k_$T 20260806 128
  stop_srv
fi
read HM HS <<<"$(he4 $T)"; IF=$(python3 -c "print($(score $A1/ifeval_$T)*100)"); GS=$(python3 -c "print($(score $A1/gsm8k_$T)*100)")
echo "E71_OFFICIAL:$T if=$IF gs=$GS he=$HM±$HS comp=$(comp $IF $GS $HM 82.53 94.9962 94.66) vs wringer 0.9066 / P3 0.8143"; date

# ================= 臂 2:A1 =================
unset WRINGER_MODEL WRINGER_REV WRINGER_CALIB_VAL
CAL="--calib $EV/calib_e58r1b_traj.pt --n-calib 128"
BASE="self_attn.k_proj:256:row,self_attn.v_proj:256:row,self_attn.o_proj:16:128"
MM_B="$BASE,linear_attn.in_proj_z:4:128,linear_attn.in_proj_qkv:4:128,mlp.gate_proj:4:128,linear_attn.out_proj:4:128"
if [ ! -f "data/qs_p3b/layer31.pt" ]; then
  idle
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.quantize --tag p3b $CAL --grid 8 --block 128 --solver gptq --prior-lam 0.01 --module-map "$MM_B" --alpha-bits 8 > "$S/quant_p3b_e71.log" 2>&1 || fail quant_p3b
fi
echo "E71_P3B_LEDGER $(grep '^QUANT_LEDGER' $S/quant_p3b_e71.log 2>/dev/null | tail -n 1 | cut -c1-300)"
if [ ! -f "exports/e69_p3b/config.json" ]; then
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export --state-tag p3b --out exports/e69_p3b > "$S/export_p3b_e71.log" 2>&1 || fail export_p3b
  idle; serve_vllm exports/e69_p3b ""
  runhe humaneval_e71_p3b_redo_s20260806 20260806
  stop_srv
  HE0=$(score $A1/humaneval_e71_p3b_redo_s20260806)
  echo "E71_P3B_HE redo=$HE0 vs E69 0.7073"; date
  python3 -c "import sys; sys.exit(0 if abs(float('$HE0')-0.7073) <= 0.055 else 1)" || fail p3b_redo_mismatch
fi
T=e71_ste_a
ste $T p3b calib_e58r1b 31
if [ ! -f "$A1/gsm8k_$T/summary.json" ]; then
  idle; serve_vllm exports/$T ""
  for sd in 20260806 1 2 3; do runhe humaneval_${T}_s$sd $sd; done
  echo "E71_B:$T he4=$(he4 $T)"; date
  runif ifeval_$T 20260806 16
  rungs gsm8k_$T 20260806 16
  stop_srv
fi
read HM HS <<<"$(he4 $T)"; IF=$(python3 -c "print($(score $A1/ifeval_$T)*100)"); GS=$(python3 -c "print($(score $A1/gsm8k_$T)*100)")
echo "E71_OFFICIAL:$T if=$IF gs=$GS he=$HM±$HS comp=$(comp $IF $GS $HM 92.79 95.53 93.75) vs wringer 0.9433 / P3 0.8144"; date
echo "E71_CHAIN_DONE"; date
