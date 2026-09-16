#!/bin/bash
# E73 收尾歸因探針(prereg_e73_collapse.json;用戶已核「排在 E72 之後」):等 E72 收尾 →
#   Qwen3-4B:exports/e70_p3a_resA(含水)→ quantize --solver rtn(search / joint 兩變體,配置同 p3a_w 排水)→ export → he 4 seed + IF + GS(E70 協定)
#   A1:exports/e69_p3b_w_resA → 同上(配置同 p3b_w 排水,α int8)→ E69 協定
# 哨兵:E73_START / E73_HANDOFF / E73_LEDGER:<t> .. / E73_B:<t> he4=.. / E73_OFFICIAL:<t> if= gs= he= comp= vs .. / E73_FAIL:<step> / E73_CHAIN_DONE
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval; EV=evidence/p1_grouping; EVC=evidence/p1_grouping/corkscrew
export PATH=$PWD/.venv-vllm/bin:$PATH
spid=""
fail() { echo "E73_FAIL:$1"; date; [ -n "$spid" ] && kill $spid 2>/dev/null; exit 1; }
idle() { while pgrep -f "python -m p2_anchor|bin/vllm serve|/llama-server" >/dev/null; do sleep 30; done; }
serve_vllm() {
  .venv-vllm/bin/vllm serve "$1" --served-model-name a1 --reasoning-parser qwen3 --max-model-len 32768 --gpu-memory-utilization 0.85 $2 --port 8000 > "$S/vllm_e73.log" 2>&1 &
  spid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_e73.log" && return 0
    grep -qE "initialization failed|Traceback" "$S/vllm_e73.log" && return 1
    sleep 1
  done; return 1
}
stop_srv() { kill $spid 2>/dev/null; wait $spid 2>/dev/null; spid=""; sleep 8; }
score() { python3 -c "import json;d=json.load(open('$1/summary.json'));print(d['strict']['prompt_level'] if 'strict' in d else d['score'])"; }
runhe() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --workers 128 --seed $2 --out "$A1/$1" > "$S/e73_$1.log" 2>&1; }
runif() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.ifeval_runner --base-url http://127.0.0.1:8000/v1 --model a1 --workers $3 --seed $2 --out "$A1/$1" > "$S/e73_$1.log" 2>&1; }
rungs() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject gsm8k --base-url http://127.0.0.1:8000/v1 --model a1 --workers $3 --seed $2 --out "$A1/$1" > "$S/e73_$1.log" 2>&1; }
he4() { python3 -c "
import json,statistics as st;v=[json.load(open('$A1/humaneval_$1_s%s/summary.json'%s))['score']*100 for s in ('20260806','1','2','3')];print('%.2f %.2f'%(st.mean(v),st.stdev(v)))"; }
# variant <tag> <base-dir> <rtn-scale> <calib-args> <module-map> <alpha-bits> <expect-total-fixed> <vllm-extra> <ifgs-workers> <denIF> <denGS> <denHE> <ste-comp> <wringer-comp>
variant() {
  local T=$1 BASE=$2 RS=$3 CAL=$4 MM=$5 AB=$6 EXP=$7 VX=$8 W=$9 DIF=${10} DGS=${11} DHE=${12} STE=${13} WR=${14}
  local OUT=exports/$T
  if [ ! -f "data/qs_$T/layer00.pt" ]; then
    idle
    PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.quantize --tag $T --base-dir $BASE $CAL --grid 8 --block 128 --solver rtn --rtn-scale $RS --prior-lam 0.01 --alpha-bits $AB --module-map "$MM" > "$S/quant_$T.log" 2>&1 || fail "quant_$T"
  fi
  local LED=$(grep '^QUANT_LEDGER' $S/quant_$T.log | tail -n 1 | cut -c1-300)
  echo "E73_LEDGER:$T $LED"; date
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
  read HM HS <<<"$(he4 $T)"; echo "E73_B:$T he4=$HM $HS"
  local IF=$(python3 -c "print($(score $A1/ifeval_$T)*100)") GS=$(python3 -c "print($(score $A1/gsm8k_$T)*100)")
  echo "E73_OFFICIAL:$T if=$IF gs=$GS he=$HM±$HS comp=$(python3 -c "print(round(($IF/$DIF+$GS/$DGS+$HM/$DHE)/3,4))") vs ste $STE / wringer $WR"; date
  rm -rf "$OUT"
}
echo "E73_START"; date
# 等 E72 falsifier 查證鏈(chain_e72p_awq.sh,本身等 E72 收尾)收尾;只看最後一次 E72P_START 之後
e72_end() { awk '/^E72P_START/{n=NR} END{print n+0}' "$S/e72p_chain.log" 2>/dev/null | xargs -I{} tail -n +{} "$S/e72p_chain.log" | grep -E "^E72P_(CHAIN_DONE|FAIL)"; }
for i in $(seq 1 17280); do [ -n "$(e72_end)" ] && break; sleep 10; done
echo "E73_HANDOFF $(e72_end | tail -n 1)"; date
sleep 30; idle

# ===== Qwen3-4B(E70 協定;配置同 p3a_w 排水)=====
export WRINGER_MODEL=Qwen/Qwen3-4B WRINGER_REV=1cfa9a7208912126459214e8b04321603b3df60c WRINGER_CALIB_VAL=$EV/calib_e70_val.pt
[ -f exports/e70_p3a_resA/config.json ] || fail no_resA_qwen
CALQ="--calib $EV/calib_e70_traj.pt --n-calib 128"
MMQ="self_attn.k_proj:256:row,self_attn.v_proj:256:row,self_attn.o_proj:8:128,mlp.gate_proj:4:128,mlp.up_proj:4:128,mlp.down_proj:4:128"
QD="82.53 94.9962 94.66"
variant e73_q_rtn  exports/e70_p3a_resA search "$CALQ" "$MMQ" 16 2.6383 "--kv-cache-dtype fp8" 128 $QD 0.8583 0.9066
variant e73_q_rtnj exports/e70_p3a_resA joint  "$CALQ" "$MMQ" 16 2.6383 "--kv-cache-dtype fp8" 128 $QD 0.8583 0.9066
unset WRINGER_MODEL WRINGER_REV WRINGER_CALIB_VAL

# ===== A1(E69 協定;配置同 p3b_w 排水)=====
[ -f exports/e69_p3b_w_resA/config.json ] || fail no_resA_a1
CALA="--calib $EV/calib_e58r1b_traj.pt --n-calib 128"
MMA="self_attn.k_proj:256:row,self_attn.v_proj:256:row,self_attn.o_proj:16:128,linear_attn.in_proj_z:4:128,linear_attn.in_proj_qkv:4:128,mlp.gate_proj:4:128,linear_attn.out_proj:4:128"
AD="92.79 95.53 93.75"
variant e73_a_rtn  exports/e69_p3b_w_resA search "$CALA" "$MMA" 8 2.6551 "" 16 $AD 0.9263 0.9433
variant e73_a_rtnj exports/e69_p3b_w_resA joint  "$CALA" "$MMA" 8 2.6551 "" 16 $AD 0.9263 0.9433
echo "E73_CHAIN_DONE"; date
