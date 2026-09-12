#!/bin/bash
# E64-PL S1(prereg_e64_probes FROZEN 2026-09-08):r 上冠軍台,零訓練三臂官方配對
#   探針(probe_asym_mix,冠軍 st51c_e2e 碼 + res62s2 水,128 calib)→ 存 pa64_{ctrl,asymblkq2,asymrefitq2}
#   → 三臂 export(--verify)→ vLLM → 官方三科 → verdict_e64_s1.json(J1–J3)
#   Hold:存在 $S/E64_HOLD 則探針後不點官方(每 60s 重查)
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval
EVC=evidence/p1_grouping/corkscrew
fail() { echo "E64S1_FAIL:$1"; date; exit 1; }

echo "E64S1_START"; date
if [ ! -f data/qs_pa64_asymrefitq2/layer31.pt ] || [ ! -f "$EVC/probe_asym_e64_champ.json" ]; then
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.probe_asym_mix \
    --state-tag st51c_e2e --res-tag res62s2 --n-calib 128 --batch 4 --alt 3 \
    --save-arms ctrl,asymblkq2,asymrefitq2 --save-prefix pa64 \
    --out "$EVC/probe_asym_e64_champ.json" > "$S/probe_asym_e64.log" 2>&1 || fail probe
  grep -q "^PROBE_ASYM_DONE" "$S/probe_asym_e64.log" || fail probe_sentinel
fi
echo "E64S1_PROBE_DONE"; date
grep "^PROBE_ASYM_SUMMARY" "$S/probe_asym_e64.log" | cut -c1-600

while [ -f "$S/E64_HOLD" ]; do sleep 60; done
pgrep -f "vllm serve" >/dev/null && fail gpu_busy_vllm

serve() {
  PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/vllm serve "$1" \
    --served-model-name a1 --reasoning-parser qwen3 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --port 8000 > "$S/vllm_e64s1.log" 2>&1 &
  vpid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_e64s1.log" && return 0
    grep -qE "initialization failed" "$S/vllm_e64s1.log" && fail serve
    sleep 1
  done; fail serve_timeout
}

for arm in ctrl asymblkq2 asymrefitq2; do
  TAG=pa64_$arm; OUT=exports/e64_$TAG; sfx=e64_$TAG
  [ -f "$OUT/config.json" ] || \
    PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export \
      --state-tag $TAG --out $OUT --verify > "$S/export_$TAG.log" 2>&1 || fail "export_$arm"
  echo "E64S1_EXPORT_DONE:$arm"; date
  need=0
  for sub in ifeval humaneval gsm8k; do [ -f "$A1/${sub}_${sfx}/summary.json" ] || need=1; done
  if [ "$need" = 1 ]; then
    serve "$OUT"
    [ -f "$A1/ifeval_${sfx}/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.ifeval_runner \
      --base-url http://127.0.0.1:8000/v1 --model a1 --out "$A1/ifeval_${sfx}" > "$S/eval_if_$TAG.log" 2>&1 || fail "eval_if_$arm"
    for sub in humaneval gsm8k; do
      [ -f "$A1/${sub}_${sfx}/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner \
        --subject "$sub" --base-url http://127.0.0.1:8000/v1 --model a1 --out "$A1/${sub}_${sfx}" > "$S/eval_${sub}_$TAG.log" 2>&1 || fail "eval_${sub}_$arm"
    done
    kill $vpid 2>/dev/null; wait $vpid 2>/dev/null; sleep 8
  fi
  echo "E64S1_EVAL_DONE:$arm"; date
done

python3 - <<'PYEOF'
import json
A1="evidence/p1_grouping/a1eval"
old={"ifeval":92.79,"humaneval":94.51,"gsm8k":95.53}
new={"ifeval":90.57,"humaneval":93.90,"gsm8k":96.21}
res={}
for arm in ("ctrl","asymblkq2","asymrefitq2"):
    sfx=f"e64_pa64_{arm}"; sc={}; emp={}
    for sub in old:
        d=json.load(open(f"{A1}/{sub}_{sfx}/summary.json"))
        sc[sub]=round((d["strict"]["prompt_level"] if sub=="ifeval" else d["score"])*100,2)
        emp[sub]=sum(1 for l in open(f"{A1}/{sub}_{sfx}/responses.jsonl") if not json.loads(l)["response"].strip())
    res[arm]={"scores":sc,"comp_old":sum(sc[s]/old[s] for s in old)/3,"comp_new":sum(sc[s]/new[s] for s in new)/3,"empties":emp}
    print(f"E64S1_B:{arm} if={sc['ifeval']} he={sc['humaneval']} gs={sc['gsm8k']} comp={res[arm]['comp_old']:.4f} (new {res[arm]['comp_new']:.4f}) empties={emp}")
c=res["ctrl"]["comp_old"]; a=res["asymblkq2"]["comp_old"]; f=res["asymrefitq2"]["comp_old"]
J1="POS" if a-c>0.016 else ("NEG" if a-c<-0.016 else "NULL")
J2="NEW_DELIVERABLE_CHAMPION" if a>=0.7629 else ("PAR" if a>=0.7388 else "TYPE11_ALERT")
J3="CTRL_HARMFUL" if c<0.72 else "CTRL_OK"
J2b="REFIT_POS" if f-a>0.016 else ("REFIT_NEG" if f-a<-0.016 else "REFIT_NULL")
print(f"E64S1_DELTA:asym-ctrl={a-c:+.4f} refit-asym={f-a:+.4f} J1={J1} J2={J2} J3={J3} refit={J2b}")
json.dump({"experiment":"E64-PL S1 r 上冠軍台(零訓練)","arms":res,"anchors":{"old":old,"new":new},
           "judges":{"J1":J1,"J2":J2,"J3":J3,"refit":J2b,"delta_asym_ctrl":round(a-c,4),"delta_refit_asym":round(f-a,4)}},
          open("p2_anchor/wringer/verdict_e64_s1.json","w"),ensure_ascii=False,indent=1)
PYEOF
echo "E64S1_CHAIN_DONE"; date
