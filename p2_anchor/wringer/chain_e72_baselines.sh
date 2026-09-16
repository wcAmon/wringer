#!/bin/bash
# E72 外部基線(prereg_e72_baselines.json;點火需用戶核准):等 E71_CHAIN_DONE →
#   Qwen3-4B 5 點(gptq2/gptq3/awq4/hqq2/hqq3)量化(.venv-base)→ he 4 seed + IF + GS(E70 協定)→ comp
#   A1 5 點同上(E69 協定:KV bf16、IF/GS w16)
#   單點失敗:E72_POINT_FAIL 記錄後繼續(prereg risk:A1 混合架構 AWQ 可能缺席)
# 哨兵:E72_START / E72_HANDOFF / E72_LEDGER:<m>_<pt> .. / E72_COLLAPSED:<m>_<pt> / E72_POINT_FAIL:<m>_<pt> <step> / E72_B:<m>_<pt> he4=.. / E72_OFFICIAL:<m>_<pt> if= gs= he= comp= bpw= / E72_FAIL:<step> / E72_CHAIN_DONE
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval; EV=evidence/p1_grouping; EVC=evidence/p1_grouping/corkscrew
export PATH=$PWD/.venv-vllm/bin:$PATH
spid=""
fail() { echo "E72_FAIL:$1"; date; [ -n "$spid" ] && kill $spid 2>/dev/null; exit 1; }
idle() { while pgrep -f "python -m p2_anchor|bin/vllm serve|/llama-server|/llama-quantize|/llama-imatrix" >/dev/null; do sleep 30; done; }
serve_vllm() {
  .venv-vllm/bin/vllm serve "$1" --served-model-name a1 --reasoning-parser qwen3 --max-model-len 32768 --gpu-memory-utilization 0.85 $2 --port 8000 > "$S/vllm_e72.log" 2>&1 &
  spid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_e72.log" && return 0
    grep -qE "initialization failed|Traceback" "$S/vllm_e72.log" && return 1
    sleep 1
  done; return 1
}
stop_srv() { kill $spid 2>/dev/null; wait $spid 2>/dev/null; spid=""; sleep 8; }
score() { python3 -c "import json;d=json.load(open('$1/summary.json'));print(d['strict']['prompt_level'] if 'strict' in d else d['score'])"; }
runhe() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --workers 128 --seed $2 --out "$A1/$1" > "$S/e72_$1.log" 2>&1; }
runif() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.ifeval_runner --base-url http://127.0.0.1:8000/v1 --model a1 --workers $3 --seed $2 --out "$A1/$1" > "$S/e72_$1.log" 2>&1; }
rungs() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject gsm8k --base-url http://127.0.0.1:8000/v1 --model a1 --workers $3 --seed $2 --out "$A1/$1" > "$S/e72_$1.log" 2>&1; }
# amendment_1/1b:四 seed he 全 0 ⇒ 崩塌,IF/GS 不跑
collapsed() { python3 -c "
import json,sys
for s in ('20260806','1','2','3'):
    d=json.load(open('$A1/humaneval_$1_s%s/summary.json'%s))
    if d['score']>0: sys.exit(1)   # amendment_1b:四 seed 全 0 即崩塌,截斷比例不再要求
sys.exit(0)" 2>/dev/null; }
he4() { python3 -c "
import json,statistics as st;v=[json.load(open('$A1/humaneval_$1_s%s/summary.json'%s))['score']*100 for s in ('20260806','1','2','3')];print('%.2f %.2f'%(st.mean(v),st.stdev(v)))"; }
# point <prefix> <pt> <method> <bits> <group> <extra-quant-args> <model> <rev> <calibprefix> <vllm-extra> <ifgs-workers> <denIF> <denGS> <denHE>
point() {
  local PFX=$1 PT=$2 M=$3 B=$4 G=$5 X=$6 MODEL=$7 REV=$8 CP=$9 VX=${10} W=${11} DIF=${12} DGS=${13} DHE=${14}
  local T=${PFX}_${PT} OUT=exports/e72_${PFX}_${PT} LED=$EVC/baseline_e72_${PFX}_${PT}.json
  if [ ! -f "$LED" ]; then
    idle
    local CAL=""; [ "$M" != "hqq" ] && CAL="--calib $EV/${CP}_traj.pt --lengths $EV/${CP}_len.pt --n-calib 128 --max-len 16384"
    .venv-base/bin/python -m p2_anchor.wringer.baseline_quant --method $M --bits $B --group $G $X --model $MODEL --revision $REV $CAL --device cuda --out $OUT --ledger $LED > "$S/bq_$T.log" 2>&1 \
      || { echo "E72_POINT_FAIL:$T quant $(grep -E 'Error|assert' $S/bq_$T.log | tail -n 1 | cut -c1-200)"; date; rm -rf "$OUT"; return 0; }
  fi
  echo "E72_LEDGER:$T $(grep '^BASELINE_DONE' $S/bq_$T.log | cut -c15-300)"; date
  if [ ! -f "$A1/gsm8k_e72_$T/summary.json" ] && ! { [ -f "$A1/humaneval_e72_${T}_s3/summary.json" ] && collapsed e72_$T; }; then
    idle
    serve_vllm $OUT "$VX" || { echo "E72_POINT_FAIL:$T serve"; stop_srv; return 0; }
    for sd in 20260806 1 2 3; do runhe humaneval_e72_${T}_s$sd $sd; done
    if collapsed e72_$T; then echo "E72_COLLAPSED:$T he 四 seed 全 0,IF/GS 依 amendment_1b 不跑"; date
    else runif ifeval_e72_$T 20260806 $W; rungs gsm8k_e72_$T 20260806 $W; fi
    stop_srv
  fi
  [ -f "$A1/humaneval_e72_${T}_s3/summary.json" ] || { echo "E72_POINT_FAIL:$T eval"; return 0; }
  read HM HS <<<"$(he4 e72_$T)"
  local BPW=$(python3 -c "import json;print(round(json.load(open('$LED'))['bpw_body'],4))")
  if collapsed e72_$T; then
    echo "E72_OFFICIAL:$T if=NA gs=NA he=$HM±$HS comp=0.0 bpw=$BPW COLLAPSED"; date
  else
    [ -f "$A1/gsm8k_e72_$T/summary.json" ] && [ -f "$A1/ifeval_e72_$T/summary.json" ] || { echo "E72_POINT_FAIL:$T eval"; return 0; }
    local IF=$(python3 -c "print($(score $A1/ifeval_e72_$T)*100)") GS=$(python3 -c "print($(score $A1/gsm8k_e72_$T)*100)")
    echo "E72_OFFICIAL:$T if=$IF gs=$GS he=$HM±$HS comp=$(python3 -c "print(round(($IF/$DIF+$GS/$DGS+$HM/$DHE)/3,4))") bpw=$BPW"; date
  fi
  # 評測後刪 export 省磁碟(ledger + a1eval 保留)
  rm -rf "$OUT"
}
echo "E72_START"; date
# 只看最後一次 E71_START 之後的收尾哨兵(E71 重點火會續寫同一 log;舊 E71_FAIL 不得觸發交接)
e71_end() { awk '/^E71_START/{n=NR} END{print n+0}' "$S/e71_chain.log" 2>/dev/null | xargs -I{} tail -n +{} "$S/e71_chain.log" | grep -E "^E71_(CHAIN_DONE|FAIL)"; }
for i in $(seq 1 17280); do [ -n "$(e71_end)" ] && break; sleep 10; done
echo "E72_HANDOFF $(e71_end | tail -n 1)"; date
sleep 30; idle

# ===== Qwen3-4B(E70 協定)=====
Q=Qwen/Qwen3-4B; QR=1cfa9a7208912126459214e8b04321603b3df60c; QV="--kv-cache-dtype fp8"; QD="82.53 94.9962 94.66"
point q gptq2 gptq 2 128 ""            $Q $QR calib_e70 "$QV" 128 $QD
point q gptq3 gptq 3 128 ""            $Q $QR calib_e70 "$QV" 128 $QD
point q awq4  awq  4 128 "--symmetric" $Q $QR calib_e70 "$QV" 128 $QD
point q hqq2  hqq  2 64  ""            $Q $QR calib_e70 "$QV" 128 $QD
point q hqq3  hqq  3 64  ""            $Q $QR calib_e70 "$QV" 128 $QD

# ===== A1(E69 協定)=====
A=InternScience/Agents-A1-4B; AR=$(PYTHONPATH=. .venv/bin/python -c "from p1_grouping.modelio import MODEL_REVISION;print(MODEL_REVISION)"); AD="92.79 95.53 93.75"
point a gptq2 gptq 2 128 ""            $A $AR calib_e58r1b "" 16 $AD
point a gptq3 gptq 3 128 ""            $A $AR calib_e58r1b "" 16 $AD
point a awq4  awq  4 128 "--symmetric" $A $AR calib_e58r1b "" 16 $AD
point a hqq2  hqq  2 64  ""            $A $AR calib_e58r1b "" 16 $AD
point a hqq3  hqq  3 64  ""            $A $AR calib_e58r1b "" 16 $AD
echo "E72_CHAIN_DONE"; date
