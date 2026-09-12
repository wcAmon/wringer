#!/bin/bash
# E63 S0 官方配對(prereg_e63_asym FROZEN;用戶 2026-09-06 裁 Q1「允許尺度側 2 bit/block」):
#   等 S2 鏈(蓄水 + 官方 A)收 S2_CHAIN_DONE 釋出 GPU → 兩臂 export(--verify)→ vLLM → 三科
#   臂:pa63_ctrl(st54r 碼固定 + 對稱 α 閉式)vs pa63_asymblkq2(同碼 + 逐 block 2 bit r + α 閉式)
#   判準:Δcomp = asymblkq2 − ctrl;噪音帶 ±0.011;POS > +0.011 / NEG < −0.011 / NULL 其間
#   Hold:存在 $S/E63_HOLD 則不點火(每 60s 重查)
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval
S2LOG=$S/diag_codes_s2_chain.log
fail() { echo "E63S0_FAIL:$1"; date; exit 1; }

echo "E63S0_WAIT"; date
while :; do
  if grep -q "^S2_CHAIN_DONE\|^S2_FAIL" "$S2LOG" 2>/dev/null && [ -f data/qs_pa63_asymblkq2/layer31.pt ] \
     && [ -f data/qs_pa63_ctrl/layer31.pt ] && [ ! -f "$S/E63_HOLD" ] \
     && ! pgrep -f "tag res62s2" >/dev/null && ! pgrep -f "vllm serve" >/dev/null; then break; fi
  sleep 60
done
echo "E63S0_START"; date

serve() {
  PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/vllm serve "$1" \
    --served-model-name a1 --reasoning-parser qwen3 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --port 8000 > "$S/vllm_e63s0.log" 2>&1 &
  vpid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_e63s0.log" && return 0
    grep -qE "initialization failed" "$S/vllm_e63s0.log" && fail serve
    sleep 1
  done; fail serve_timeout
}

for arm in ctrl asymblkq2; do
  TAG=pa63_$arm; OUT=exports/e63_$TAG; sfx=e63_$TAG
  [ -f "$OUT/config.json" ] || \
    PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export \
      --state-tag $TAG --out $OUT --verify || fail "export_$arm"
  echo "E63S0_EXPORT_DONE:$arm"; date
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
  echo "E63S0_EVAL_DONE:$arm"; date
done

python3 - <<'PYEOF'
import json
A1="evidence/p1_grouping/a1eval"
anchors={"ifeval":92.79,"humaneval":94.51,"gsm8k":95.53}
A4=0.7038
res={}
for arm in ("ctrl","asymblkq2"):
    sfx=f"e63_pa63_{arm}"; sc={}; emp={}
    for sub in anchors:
        d=json.load(open(f"{A1}/{sub}_{sfx}/summary.json"))
        sc[sub]=round((d["strict"]["prompt_level"] if sub=="ifeval" else d["score"])*100,2)
        emp[sub]=sum(1 for l in open(f"{A1}/{sub}_{sfx}/responses.jsonl") if not json.loads(l)["response"].strip())
    comp=sum(sc[s]/anchors[s] for s in anchors)/3
    res[arm]={"scores":sc,"comp":comp,"rho":comp/A4,"empties":emp}
    print(f"E63S0_B:{arm} if={sc['ifeval']} he={sc['humaneval']} gs={sc['gsm8k']} comp={comp:.4f} rho={comp/A4:.3f} empties={emp}")
d=res["asymblkq2"]["comp"]-res["ctrl"]["comp"]
ds={s:round(res["asymblkq2"]["scores"][s]-res["ctrl"]["scores"][s],2) for s in anchors}
v="POS" if d>0.011 else "NEG" if d<-0.011 else "NULL"
print(f"E63S0_DELTA:comp={d:+.4f} by={ds}")
print(f"E63S0_VERDICT:{v} (band ±0.011)")
PYEOF
echo "E63S0_CHAIN_DONE"; date
