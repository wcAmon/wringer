#!/bin/bash
# E70 第二段(prereg stage2_compose;用戶 09-12 核准 E70 全鏈自主接續):等 E70S1_CHAIN_DONE|FAIL →
#   讀 prereg stage2_compose.candidates(compose_e70 產):p3a / p3b 各零訓練閉式 → export → vLLM(KV fp8)→ he 2 seed
#   → he 高者官方 IF(seed 20260806)+ GS(seed 20260806)→ comp = mean(if/IF, he/HE, gs/GS)(分母 prereg anchor.comp_denominators)
#   輸家刪 export(state dir data/qs_e70_<tag> 兩者皆留,S3 排水基底)
# 哨兵:E70S2_START / E70S2_HANDOFF .. / E70S2_LEDGER:<tag> .. / E70S2_POINT:<tag> he=<s0>,<s1> / E70S2_WINNER <tag> / E70S2_OFFICIAL:<tag> if=<> gs=<> he=<> comp=<> / E70S2_FAIL:<step> / E70S2_CHAIN_DONE
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval; EV=evidence/p1_grouping
PRE=p2_anchor/wringer/prereg_e70_qwen3_4b.json
export WRINGER_MODEL=Qwen/Qwen3-4B WRINGER_REV=1cfa9a7208912126459214e8b04321603b3df60c WRINGER_CALIB_VAL=$EV/calib_e70_val.pt
export PATH=$PWD/.venv-vllm/bin:$PATH
CAL="--calib $EV/calib_e70_traj.pt --n-calib 128"
fail() { echo "E70S2_FAIL:$1"; date; exit 1; }
idle() { while pgrep -f "^\S*python -m p2_anchor|^\S*vllm serve|^\S*/llama-server|^\S*/llama-quantize|^\S*/llama-imatrix" >/dev/null; do sleep 15; done; }
serve_vllm() {
  .venv-vllm/bin/vllm serve "$1" --served-model-name a1 --reasoning-parser qwen3 --max-model-len 32768 --gpu-memory-utilization 0.85 --kv-cache-dtype fp8 --port 8000 > "$S/vllm_e70s2.log" 2>&1 &
  spid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_e70s2.log" && return 0
    grep -qE "initialization failed|Traceback" "$S/vllm_e70s2.log" && fail serve
    sleep 1
  done; fail serve_timeout
}
stop_srv() { kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 8; }
score() { python3 -c "import json;d=json.load(open('$1/summary.json'));print(d['strict']['prompt_level'] if 'strict' in d else d['score'])"; }
runhe() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --workers 128 --seed $2 --out "$A1/$1" > "$S/e70_$1.log" 2>&1 || fail "$1"; }
runif() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.ifeval_runner --base-url http://127.0.0.1:8000/v1 --model a1 --workers 128 --seed $2 --out "$A1/$1" > "$S/e70_$1.log" 2>&1 || fail "$1"; }
rungs() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject gsm8k --base-url http://127.0.0.1:8000/v1 --model a1 --workers 128 --seed $2 --out "$A1/$1" > "$S/e70_$1.log" 2>&1 || fail "$1"; }
cand() { python3 -c "import json;c=json.load(open('$PRE'))['stage2_compose']['candidates']['$1'];print(c['$2'])"; }
# compose <tag>
compose() {
  local TAG=e70_$1 AB=$(cand $1 alpha_bits) MM=$(cand $1 module_map) OUT=exports/e70_$1
  if [ ! -f "data/qs_$TAG/layer35.pt" ]; then
    idle
    PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.quantize --tag $TAG $CAL --grid 8 --block 128 --solver gptq --prior-lam 0.01 --alpha-bits $AB --module-map "$MM" > "$S/quant_$TAG.log" 2>&1 || fail "quant_$TAG"
  fi
  echo "E70S2_LEDGER:$1 $(grep '^QUANT_LEDGER' $S/quant_$TAG.log | tail -n 1 | cut -c1-300) escal=$(grep -c ESCALATE $S/quant_$TAG.log)"
  [ -f "$OUT/config.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export --state-tag $TAG --out $OUT > "$S/export_$TAG.log" 2>&1 || fail "export_$TAG"
  if [ ! -f "$A1/humaneval_${TAG}_s1/summary.json" ]; then
    idle; serve_vllm "$OUT"
    runhe humaneval_${TAG}_s20260806 20260806
    runhe humaneval_${TAG}_s1 1
    stop_srv
  fi
  echo "E70S2_POINT:$1 he=$(score $A1/humaneval_${TAG}_s20260806),$(score $A1/humaneval_${TAG}_s1)"; date
}
echo "E70S2_START"; date
for i in $(seq 1 8640); do grep -q "E70S1_CHAIN_DONE\|E70S1_FAIL" "$S/e70s1_chain.log" 2>/dev/null && break; sleep 10; done
echo "E70S2_HANDOFF $(grep -E 'E70S1_CHAIN_DONE|E70S1_FAIL' $S/e70s1_chain.log | head -n 1)"; date
sleep 20
CANDS="p3a"
[ "$(cand p3b module_map)$(cand p3b alpha_bits)" != "$(cand p3a module_map)$(cand p3a alpha_bits)" ] && CANDS="p3a p3b"
for c in $CANDS; do compose $c; done
# 勝者:he 兩 seed 均值高者
WIN=$(python3 - "$A1" $CANDS <<'PY'
import json,sys
a1=sys.argv[1]; best=None
for c in sys.argv[2:]:
    m=sum(json.load(open(f"{a1}/humaneval_e70_{c}_s{s}/summary.json"))["score"] for s in ("20260806","1"))/2
    if best is None or m>best[1]: best=(c,m)
print(best[0])
PY
)
echo "E70S2_WINNER $WIN"; date
for c in $CANDS; do [ "$c" = "$WIN" ] || rm -rf "exports/e70_$c"; done
TAG=e70_$WIN
if [ ! -f "$A1/gsm8k_${TAG}/summary.json" ]; then
  idle; serve_vllm "exports/$TAG"
  runif ifeval_${TAG} 20260806
  rungs gsm8k_${TAG} 20260806
  stop_srv
fi
python3 - "$A1" "$TAG" "$PRE" <<'PY'
import json,sys
a1,tag,pre=sys.argv[1:4]
den=json.load(open(pre))["anchor"]["comp_denominators"]
he=sum(json.load(open(f"{a1}/humaneval_{tag}_s{s}/summary.json"))["score"] for s in ("20260806","1"))/2*100
iff=json.load(open(f"{a1}/ifeval_{tag}/summary.json"))["strict"]["prompt_level"]*100
gs=json.load(open(f"{a1}/gsm8k_{tag}/summary.json"))["score"]*100
comp=(iff/den["IF"]+he/den["HE"]+gs/den["GS"])/3
print(f"E70S2_OFFICIAL:{tag} if={iff:.2f} gs={gs:.2f} he={he:.2f} comp={comp:.4f}")
PY
date
echo "E70S2_CHAIN_DONE"; date
