#!/bin/bash
# E62b 診斷(用戶 2026-09-05「先診斷再S2」核准):st62b 碼+M 去 L(L=I)→ export --verify → 官方三科
#   H_overfit(L 推論有害):he ≥ 34 回到 st61c 帶  / H_lazy(碼被練弱):he ≤ 32.5 / 兩者:he < 31
#   完成後自動接 s2_fill_chain.sh(S2 蓄水鏈,st61c 引擎;先讀冠軍 A 閘)
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval
OUT=exports/e62_st62b_codes
sfx=e62_st62b_codes
fail() { echo "NOC_FAIL:$1"; date; exit 1; }
echo "NOC_START"; date
[ -f "$OUT/config.json" ] || \
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export \
    --state-tag st62b --out $OUT --strip-lmix --strip-mix --verify || fail export
echo "NOC_EXPORT_DONE"; date
serve() {
  PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/vllm serve "$1" \
    --served-model-name a1 --reasoning-parser qwen3 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --port 8000 > "$S/vllm_noc.log" 2>&1 &
  vpid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_noc.log" && return 0
    grep -qE "initialization failed" "$S/vllm_noc.log" && fail serve
    sleep 1
  done; fail serve_timeout
}
need=0
for sub in ifeval humaneval gsm8k; do [ -f "$A1/${sub}_${sfx}/summary.json" ] || need=1; done
if [ "$need" = 1 ]; then
  serve "$OUT"
  [ -f "$A1/ifeval_${sfx}/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.ifeval_runner \
    --base-url http://127.0.0.1:8000/v1 --model a1 --out "$A1/ifeval_${sfx}" || fail eval_if
  for sub in humaneval gsm8k; do
    [ -f "$A1/${sub}_${sfx}/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner \
      --subject "$sub" --base-url http://127.0.0.1:8000/v1 --model a1 --out "$A1/${sub}_${sfx}" || fail "eval_${sub}"
  done
  kill $vpid 2>/dev/null; wait $vpid 2>/dev/null; sleep 8
fi
echo "NOC_EVAL_DONE"; date
SFX=$sfx python3 - <<'PYEOF'
import json, os
A1="evidence/p1_grouping/a1eval"; sfx=os.environ["SFX"]
anchors={"ifeval":92.79,"humaneval":94.51,"gsm8k":95.53}; A4=0.7038
ref={"noL(碼+M)":{"rho":0.916,"ifeval":75.05,"humaneval":40.24,"gsm8k":66.87},
     "st61c(碼+M)":{"rho":0.889,"ifeval":75.23,"humaneval":35.37,"gsm8k":66.19},
     "st61b(純碼)":{"rho":0.872,"ifeval":73.20,"humaneval":33.54,"gsm8k":66.57}}
sc,emp={},{}
for sub in anchors:
    d=json.load(open(f"{A1}/{sub}_{sfx}/summary.json"))
    sc[sub]=round((d["strict"]["prompt_level"] if sub=="ifeval" else d["score"])*100,2)
    emp[sub]=sum(1 for l in open(f"{A1}/{sub}_{sfx}/responses.jsonl") if not json.loads(l)["response"].strip())
comp=sum(sc[s]/anchors[s] for s in anchors)/3; rho=comp/A4
print(f"NOC_B:if={sc['ifeval']} he={sc['humaneval']} gs={sc['gsm8k']} comp={comp:.4f} rho={rho:.3f} empties={emp}")
for k,r in ref.items():
    print(f"NOC_DELTA:vs_{k}=rho{rho-r['rho']:+.3f} if{sc['ifeval']-r['ifeval']:+.2f} he{sc['humaneval']-r['humaneval']:+.2f} gs{sc['gsm8k']-r['gsm8k']:+.2f}")
v = "M_ALSO_SCAFFOLD(純碼≈碼+M,M 可拆)" if rho>=0.916-0.016 else "M_CARRIES(去 M 傷>噪音,M 為真載體)" if rho<0.916-0.016 and rho>=0.872 else "M_CARRIES_HARD(純碼低於 st61b 純碼)"
print(f"NOC_VERDICT:{v} vs_st61b_codes={'POS' if rho>0.872+0.016 else 'NEG' if rho<0.872-0.016 else 'NULL'}")
PYEOF
echo "NOC_CHAIN_DONE"; date
echo "=== 接 S2 蓄水鏈 ==="
bash "$S/s2_fill_chain.sh"
