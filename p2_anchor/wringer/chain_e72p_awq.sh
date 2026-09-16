#!/bin/bash
# E72 falsifier 查證(prereg_e72 falsifiers:「AWQ 4-bit comp < 0.90 ⇒ 疑工具鏈,先查再裁,不入表」):等 E72 收尾 →
#   A:重做 awq4(symmetric,同 E72 配置)→ vLLM → 探針 think(抓 reasoning_content:content 空是否 = 思考段 EOS)+ nothink(數學能力儀器)
#   B:awq4 asymmetric(llm-compressor 預設 zero-point;AWQ 論文預設)→ 官方三科(E70 協定)→ E72P_OFFICIAL:q_awq4a
# 哨兵:E72P_START / E72P_HANDOFF / E72P_LEDGER / E72P_PROBE:<mode> .. / E72P_B:.. / E72P_OFFICIAL:q_awq4a .. / E72P_FAIL:<step> / E72P_CHAIN_DONE
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval; EV=evidence/p1_grouping; EVC=evidence/p1_grouping/corkscrew
export PATH=$PWD/.venv-vllm/bin:$PATH
spid=""
fail() { echo "E72P_FAIL:$1"; date; [ -n "$spid" ] && kill $spid 2>/dev/null; exit 1; }
idle() { while pgrep -f "python -m p2_anchor|bin/vllm serve|/llama-server" >/dev/null; do sleep 30; done; }
serve_vllm() {
  .venv-vllm/bin/vllm serve "$1" --served-model-name a1 --reasoning-parser qwen3 --max-model-len 32768 --gpu-memory-utilization 0.85 $2 --port 8000 > "$S/vllm_e72p.log" 2>&1 &
  spid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_e72p.log" && return 0
    grep -qE "initialization failed|Traceback" "$S/vllm_e72p.log" && return 1
    sleep 1
  done; return 1
}
stop_srv() { kill $spid 2>/dev/null; wait $spid 2>/dev/null; spid=""; sleep 8; }
score() { python3 -c "import json;d=json.load(open('$1/summary.json'));print(d['strict']['prompt_level'] if 'strict' in d else d['score'])"; }
runhe() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --workers 128 --seed $2 --out "$A1/$1" > "$S/e72p_$1.log" 2>&1; }
runif() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.ifeval_runner --base-url http://127.0.0.1:8000/v1 --model a1 --workers 128 --seed $2 --out "$A1/$1" > "$S/e72p_$1.log" 2>&1; }
rungs() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject gsm8k --base-url http://127.0.0.1:8000/v1 --model a1 --workers 128 --seed $2 --out "$A1/$1" > "$S/e72p_$1.log" 2>&1; }
he4() { python3 -c "
import json,statistics as st;v=[json.load(open('$A1/humaneval_$1_s%s/summary.json'%s))['score']*100 for s in ('20260806','1','2','3')];print('%.2f %.2f'%(st.mean(v),st.stdev(v)))"; }
Q=Qwen/Qwen3-4B; QR=1cfa9a7208912126459214e8b04321603b3df60c; QV="--kv-cache-dtype fp8"
CAL="--calib $EV/calib_e70_traj.pt --lengths $EV/calib_e70_len.pt --n-calib 128 --max-len 16384"
DIF=82.53; DGS=94.9962; DHE=94.66
echo "E72P_START"; date
e72_end() { awk '/^E72_START/{n=NR} END{print n+0}' "$S/e72_chain.log" 2>/dev/null | xargs -I{} tail -n +{} "$S/e72_chain.log" | grep -E "^E72_(CHAIN_DONE|FAIL)"; }
for i in $(seq 1 17280); do [ -n "$(e72_end)" ] && break; sleep 10; done
echo "E72P_HANDOFF $(e72_end | tail -n 1)"; date
sleep 30; idle

# ===== A:awq4 symmetric 重做 + 探針 =====
OUT=exports/e72p_q_awq4s
if [ ! -f "$EVC/e72p_awq4s_nothink.json" ]; then
  [ -f "$OUT/config.json" ] || .venv-base/bin/python -m p2_anchor.wringer.baseline_quant --method awq --bits 4 --group 128 --symmetric --model $Q --revision $QR $CAL --device cuda --out $OUT --ledger $S/bq_e72p_awq4s_ledger.json > "$S/bq_e72p_awq4s.log" 2>&1 || fail quant_awq4s
  echo "E72P_LEDGER:awq4s $(grep '^BASELINE_DONE' $S/bq_e72p_awq4s.log | cut -c15-160)"; date
  serve_vllm $OUT "$QV" || fail serve_awq4s
  for md in think nothink; do
    PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.e72_awq_probe --mode $md --n 150 --out $EVC/e72p_awq4s_$md.json > "$S/e72p_probe_$md.log" 2>&1 || { stop_srv; fail probe_$md; }
    echo "E72P_PROBE:awq4s $(grep '^PROBE_RESULT' $S/e72p_probe_$md.log | cut -c14-400)"; date
  done
  stop_srv
  rm -rf "$OUT"
fi

# ===== B:awq4 asymmetric 官方三科 =====
T=q_awq4a; OUT=exports/e72_$T; LED=$EVC/baseline_e72_$T.json
if [ ! -f "$LED" ]; then
  idle
  .venv-base/bin/python -m p2_anchor.wringer.baseline_quant --method awq --bits 4 --group 128 --model $Q --revision $QR $CAL --device cuda --out $OUT --ledger $LED > "$S/bq_$T.log" 2>&1 || fail quant_$T
fi
echo "E72P_LEDGER:$T $(grep '^BASELINE_DONE' $S/bq_$T.log | cut -c15-160)"; date
if [ ! -f "$A1/gsm8k_e72_$T/summary.json" ]; then
  idle
  serve_vllm $OUT "$QV" || fail serve_$T
  for sd in 20260806 1 2 3; do runhe humaneval_e72_${T}_s$sd $sd; done
  runif ifeval_e72_$T 20260806
  rungs gsm8k_e72_$T 20260806
  stop_srv
fi
[ -f "$A1/gsm8k_e72_$T/summary.json" ] && [ -f "$A1/ifeval_e72_$T/summary.json" ] && [ -f "$A1/humaneval_e72_${T}_s3/summary.json" ] || fail eval_$T
read HM HS <<<"$(he4 e72_$T)"; echo "E72P_B:$T he4=$HM $HS"
IF=$(python3 -c "print($(score $A1/ifeval_e72_$T)*100)"); GS=$(python3 -c "print($(score $A1/gsm8k_e72_$T)*100)")
BPW=$(python3 -c "import json;print(round(json.load(open('$LED'))['bpw_body'],4))")
EMP=$(python3 -c "import json;print(sum(1 for l in open('$A1/gsm8k_e72_$T/responses.jsonl') if not json.loads(l)['response'].strip()))")
echo "E72P_OFFICIAL:$T if=$IF gs=$GS he=$HM±$HS comp=$(python3 -c "print(round(($IF/$DIF+$GS/$DGS+$HM/$DHE)/3,4))") bpw=$BPW gs_empty=$EMP"; date
rm -rf "$OUT"
echo "E72P_CHAIN_DONE"; date
