#!/bin/bash
# E74 零訓練 rtnj 對照(prereg_e74_zerortnj.json;用戶已核「把零訓練rtnj拉上排程」):等 E73 收尾 →
#   Qwen3-4B:bf16 原模型(無 --base-dir)→ quantize --solver rtn --rtn-scale joint(配置同 e73_q_rtnj)→ export → 三科(E70 協定)
#   A1:bf16 原模型 → 同上(配置同 e73_a_rtnj)→ E69 協定
# 哨兵:E74_START / E74_HANDOFF / E74_LEDGER:<t> .. / E74_B:<t> he4=.. / E74_OFFICIAL:<t> if= gs= he= comp= vs .. / E74_FAIL:<step> / E74_CHAIN_DONE
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval; EV=evidence/p1_grouping; EVC=evidence/p1_grouping/corkscrew
export PATH=$PWD/.venv-vllm/bin:$PATH
spid=""
fail() { echo "E74_FAIL:$1"; date; [ -n "$spid" ] && kill $spid 2>/dev/null; exit 1; }
idle() { while pgrep -f "python -m p2_anchor|bin/vllm serve|/llama-server" >/dev/null; do sleep 30; done; }
serve_vllm() {
  .venv-vllm/bin/vllm serve "$1" --served-model-name a1 --reasoning-parser qwen3 --max-model-len 32768 --gpu-memory-utilization 0.85 $2 --port 8000 > "$S/vllm_e74.log" 2>&1 &
  spid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_e74.log" && return 0
    grep -qE "initialization failed|Traceback" "$S/vllm_e74.log" && return 1
    sleep 1
  done; return 1
}
stop_srv() { kill $spid 2>/dev/null; wait $spid 2>/dev/null; spid=""; sleep 8; }
score() { python3 -c "import json;d=json.load(open('$1/summary.json'));print(d['strict']['prompt_level'] if 'strict' in d else d['score'])"; }
runhe() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --workers 128 --seed $2 --out "$A1/$1" > "$S/e74_$1.log" 2>&1; }
runif() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.ifeval_runner --base-url http://127.0.0.1:8000/v1 --model a1 --workers $3 --seed $2 --out "$A1/$1" > "$S/e74_$1.log" 2>&1; }
rungs() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject gsm8k --base-url http://127.0.0.1:8000/v1 --model a1 --workers $3 --seed $2 --out "$A1/$1" > "$S/e74_$1.log" 2>&1; }
he4() { python3 -c "
import json,statistics as st;v=[json.load(open('$A1/humaneval_$1_s%s/summary.json'%s))['score']*100 for s in ('20260806','1','2','3')];print('%.2f %.2f'%(st.mean(v),st.stdev(v)))"; }
# variant <tag> <rtn-scale> <calib-args> <module-map> <alpha-bits> <expect-total-fixed> <vllm-extra> <ifgs-workers> <denIF> <denGS> <denHE> <ste-comp> <wringer-comp>
variant() {
  local T=$1 RS=$2 CAL=$3 MM=$4 AB=$5 EXP=$6 VX=$7 W=$8 DIF=$9 DGS=${10} DHE=${11} P3=${12} WR=${13}
  local OUT=exports/$T
  if [ ! -f "data/qs_$T/layer00.pt" ]; then
    idle
    PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.quantize --tag $T $CAL --grid 8 --block 128 --solver rtn --rtn-scale $RS --prior-lam 0.01 --alpha-bits $AB --module-map "$MM" > "$S/quant_$T.log" 2>&1 || fail "quant_$T"
  fi
  local LED=$(grep '^QUANT_LEDGER' $S/quant_$T.log | tail -n 1 | cut -c1-300)
  echo "E74_LEDGER:$T $LED"; date
  local TF=$(python3 -c "import json,re;print(json.loads(re.sub(r'^QUANT_LEDGER ','','''$LED'''))['total_fixed'])")
  python3 -c "import sys;sys.exit(0 if abs($TF-$EXP)<0.002 else 1)" || fail "ledger_${T}_${TF}_ne_${EXP}"
  [ -f "$OUT/config.json" ] || { idle; PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export --state-tag $T --out $OUT > "$S/export_$T.log" 2>&1 || fail "export_$T"; }
  if [ ! -f "$A1/gsm8k_$T/summary.json" ]; then
    idle
    serve_vllm $OUT "$VX" || fail "serve_$T"
    for sd in 20260806 1 2 3; do runhe humaneval_${T}_s$sd $sd; done
    runif ifeval_$T 20260806 $W
    rungs gsm8k_$T 20260806 $W
    stop_srv
  fi
  [ -f "$A1/gsm8k_$T/summary.json" ] && [ -f "$A1/ifeval_$T/summary.json" ] && [ -f "$A1/humaneval_${T}_s3/summary.json" ] || fail "eval_$T"
  read HM HS <<<"$(he4 $T)"; echo "E74_B:$T he4=$HM $HS"
  local IF=$(python3 -c "print($(score $A1/ifeval_$T)*100)") GS=$(python3 -c "print($(score $A1/gsm8k_$T)*100)")
  echo "E74_OFFICIAL:$T if=$IF gs=$GS he=$HM±$HS comp=$(python3 -c "print(round(($IF/$DIF+$GS/$DGS+$HM/$DHE)/3,4))") vs P3 $P3 / wringer $WR"; date
  rm -rf "$OUT"
}
echo "E74_START"; date
# 等 E73 收尾;只看最後一次 E73_START 之後
e73_end() { awk '/^E73_START/{n=NR} END{print n+0}' "$S/e73_chain.log" 2>/dev/null | xargs -I{} tail -n +{} "$S/e73_chain.log" | grep -E "^E73_(CHAIN_DONE|FAIL)"; }
for i in $(seq 1 17280); do [ -n "$(e73_end)" ] && break; sleep 10; done
echo "E74_HANDOFF $(e73_end | tail -n 1)"; date
sleep 30; idle

# ===== Qwen3-4B(E70 協定;配置同 e73_q_rtnj,零訓練)=====
export WRINGER_MODEL=Qwen/Qwen3-4B WRINGER_REV=1cfa9a7208912126459214e8b04321603b3df60c WRINGER_CALIB_VAL=$EV/calib_e70_val.pt
CALQ="--calib $EV/calib_e70_traj.pt --n-calib 128"
MMQ="self_attn.k_proj:256:row,self_attn.v_proj:256:row,self_attn.o_proj:8:128,mlp.gate_proj:4:128,mlp.up_proj:4:128,mlp.down_proj:4:128"
QD="82.53 94.9962 94.66"
variant e74_q_rtnj0 joint "$CALQ" "$MMQ" 16 2.6383 "--kv-cache-dtype fp8" 128 $QD 0.8143 0.9066
unset WRINGER_MODEL WRINGER_REV WRINGER_CALIB_VAL

# ===== A1(E69 協定;配置同 e73_a_rtnj,零訓練)=====
CALA="--calib $EV/calib_e58r1b_traj.pt --n-calib 128"
MMA="self_attn.k_proj:256:row,self_attn.v_proj:256:row,self_attn.o_proj:16:128,linear_attn.in_proj_z:4:128,linear_attn.in_proj_qkv:4:128,mlp.gate_proj:4:128,linear_attn.out_proj:4:128"
AD="92.79 95.53 93.75"
variant e74_a_rtnj0 joint "$CALA" "$MMA" 8 2.6551 "" 16 $AD 0.8144 0.9433
echo "E74_CHAIN_DONE"; date
