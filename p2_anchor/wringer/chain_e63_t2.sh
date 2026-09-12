#!/bin/bash
# E63 步驟 3(prereg_e63_t2 FROZEN):T2 探針(煙測→全模型→compose)→ 等 S1 鏈收 → 官方 3 臂 → 裁
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval
EVC=evidence/p1_grouping/corkscrew
fail() { echo "T2_FAIL:$1"; date; exit 1; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
echo "T2_WAIT"; date
while :; do   # 前向探針:S1 訓練已起(記憶體先佔定)且已用 < 72 GB(留 ≥ 24 GB;st63a 訓練 61 GB)、無 vLLM、無 hold
  if grep -q "^S1_TRAIN_START" "$S/e63_s1_chain.log" 2>/dev/null && ! pgrep -f "vllm serve" >/dev/null \
     && [ "$(gpu_used)" -lt 72000 ] && [ ! -f "$S/T2_HOLD" ]; then sleep 300; [ "$(gpu_used)" -lt 72000 ] && break; fi
  sleep 60
done
echo "T2_PROBE_SMOKE_START"; date
if [ ! -f "$EVC/probe_t2_smoke.json" ] || ! grep -q PROBE_T2_DONE "$S/probe_t2_smoke.log" 2>/dev/null; then
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.probe_t2 --smoke > "$S/probe_t2_smoke.log" 2>&1 || fail probe_smoke
fi
grep -q PROBE_T2_DONE "$S/probe_t2_smoke.log" || fail probe_smoke_sentinel
echo "T2_PROBE_SMOKE_DONE"; date
if [ ! -f data/qs_pt63_t2gptq/layer31.pt ]; then
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.probe_t2 --n-calib 128 --batch 4 \
    --save-arms ctrl,t2beta,t2gptq --save-prefix pt63 > "$S/probe_t2_full.log" 2>&1 || fail probe_full
fi
grep -q PROBE_T2_DONE "$S/probe_t2_full.log" || fail probe_full_sentinel
grep PROBE_T2_SUMMARY "$S/probe_t2_full.log" | cut -c1-600
echo "T2_PROBE_DONE"; date
[ -f data/qs_pt63_t2beta_dp_asym/layer31.pt ] || \
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.probe_t2 --compose --save-prefix pt63 || fail compose
echo "T2_COMPOSE_DONE"; date
while :; do   # 官方:等 S1 鏈收卷、GPU 全空
  if grep -q "^S1_CHAIN_DONE\|^S1_FAIL" "$S/e63_s1_chain.log" 2>/dev/null && ! pgrep -f "vllm serve" >/dev/null \
     && ! pgrep -f "train_res" >/dev/null && [ ! -f "$S/T2_HOLD" ]; then break; fi
  sleep 60
done
echo "T2_OFFICIAL_START"; date
serve() {
  PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/vllm serve "$1" \
    --served-model-name a1 --reasoning-parser qwen3 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --port 8000 > "$S/vllm_t2.log" 2>&1 &
  vpid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_t2.log" && return 0
    grep -qE "initialization failed" "$S/vllm_t2.log" && fail serve
    sleep 1
  done; fail serve_timeout
}
for arm in t2beta_dp t2gptq_dp t2beta_dp_asym; do
  TAG=pt63_$arm; OUT=exports/e63_$TAG; sfx=e63_$TAG
  [ -f "$OUT/config.json" ] || \
    PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export --state-tag $TAG --out $OUT --verify || fail "export_$arm"
  echo "T2_EXPORT_DONE:$arm"; date
  need=0
  for sub in ifeval humaneval gsm8k; do [ -f "$A1/${sub}_${sfx}/summary.json" ] || need=1; done
  if [ "$need" = 1 ]; then
    serve "$OUT"
    [ -f "$A1/ifeval_${sfx}/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.ifeval_runner \
      --base-url http://127.0.0.1:8000/v1 --model a1 --out "$A1/ifeval_${sfx}" || fail "eval_if_$arm"
    for sub in humaneval gsm8k; do
      [ -f "$A1/${sub}_${sfx}/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner \
        --subject "$sub" --base-url http://127.0.0.1:8000/v1 --model a1 --out "$A1/${sub}_${sfx}" || fail "eval_${sub}_$arm"
    done
    kill $vpid 2>/dev/null; wait $vpid 2>/dev/null; sleep 8
  fi
  echo "T2_EVAL_DONE:$arm"; date
done
python3 - <<'PYEOF'
import json
A1="evidence/p1_grouping/a1eval"
anchors={"ifeval":92.79,"humaneval":94.51,"gsm8k":95.53}; A4=0.7038
def rd(sfx):
    sc,emp={},{}
    for sub in anchors:
        d=json.load(open(f"{A1}/{sub}_{sfx}/summary.json"))
        sc[sub]=round((d["strict"]["prompt_level"] if sub=="ifeval" else d["score"])*100,2)
        emp[sub]=sum(1 for l in open(f"{A1}/{sub}_{sfx}/responses.jsonl") if not json.loads(l)["response"].strip())
    comp=sum(sc[s]/anchors[s] for s in anchors)/3
    return sc,comp,emp
res={}
for arm in ("pa63_ctrl","pa63_asymblkq2","pt63_t2beta_dp","pt63_t2gptq_dp","pt63_t2beta_dp_asym"):
    sc,comp,emp=rd(f"e63_{arm}"); res[arm]=comp
    print(f"T2_B:{arm} if={sc['ifeval']} he={sc['humaneval']} gs={sc['gsm8k']} comp={comp:.4f} rho={comp/A4:.3f} empties={emp}")
def band(d): return "POS" if d>0.011 else "NEG" if d<-0.011 else "NULL"
a=res["pt63_t2beta_dp"]-res["pa63_ctrl"]; b=res["pt63_t2beta_dp_asym"]-res["pa63_asymblkq2"]; c=res["pt63_t2gptq_dp"]-res["pt63_t2beta_dp"]
print(f"T2_DELTA:a(t2beta_dp-ctrl)={a:+.4f} {band(a)} | b(t2beta_dp_asym-asymblkq2)={b:+.4f} {band(b)} | c(t2gptq_dp-t2beta_dp)={c:+.4f} {band(c)}")
print(f"T2_VERDICT:{band(a)} (a 主判;帶 ±0.011)")
PYEOF
echo "T2_CHAIN_DONE"; date
