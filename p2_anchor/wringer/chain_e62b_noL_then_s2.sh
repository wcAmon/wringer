#!/bin/bash
# E62b 診斷(用戶 2026-09-05「先診斷再S2」核准):st62b 碼+M 去 L(L=I)→ export --verify → 官方三科
#   H_overfit(L 推論有害):he ≥ 34 回到 st61c 帶  / H_lazy(碼被練弱):he ≤ 32.5 / 兩者:he < 31
#   完成後自動接 s2_fill_chain.sh(S2 蓄水鏈,st61c 引擎;先讀冠軍 A 閘)
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval
OUT=exports/e62_st62b_noL
sfx=e62_st62b_noL
fail() { echo "NOL_FAIL:$1"; date; exit 1; }
echo "NOL_START"; date
[ -f "$OUT/config.json" ] || \
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export \
    --state-tag st62b --out $OUT --strip-lmix --verify || fail export
echo "NOL_EXPORT_DONE"; date
serve() {
  PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/vllm serve "$1" \
    --served-model-name a1 --reasoning-parser qwen3 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --port 8000 > "$S/vllm_nol.log" 2>&1 &
  vpid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_nol.log" && return 0
    grep -qE "initialization failed" "$S/vllm_nol.log" && fail serve
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
echo "NOL_EVAL_DONE"; date
SFX=$sfx python3 - <<'PYEOF'
import json, os
A1="evidence/p1_grouping/a1eval"; sfx=os.environ["SFX"]
anchors={"ifeval":92.79,"humaneval":94.51,"gsm8k":95.53}; A4=0.7038
s61c={"rho":0.889,"ifeval":75.23,"humaneval":35.37,"gsm8k":66.19}
s62b={"rho":0.869,"ifeval":75.23,"humaneval":31.71,"gsm8k":65.73}
sc,emp={},{}
for sub in anchors:
    d=json.load(open(f"{A1}/{sub}_{sfx}/summary.json"))
    sc[sub]=round((d["strict"]["prompt_level"] if sub=="ifeval" else d["score"])*100,2)
    emp[sub]=sum(1 for l in open(f"{A1}/{sub}_{sfx}/responses.jsonl") if not json.loads(l)["response"].strip())
comp=sum(sc[s]/anchors[s] for s in anchors)/3; rho=comp/A4
he=sc["humaneval"]
verdict="H_overfit(L 推論有害)" if he>=34 else "H_lazy(碼被練弱)" if he>32.5 and he>=31 else "H_lazy(碼被練弱)" if he<=32.5 and he>=31 else "BOTH(L 在補碼的洞)"
print(f"NOL_B:if={sc['ifeval']} he={he} gs={sc['gsm8k']} comp={comp:.4f} rho={rho:.3f} empties={emp}")
print(f"NOL_DELTA:vs_st62b={{'ifeval':{sc['ifeval']-s62b['ifeval']:.2f},'humaneval':{he-s62b['humaneval']:.2f},'gsm8k':{sc['gsm8k']-s62b['gsm8k']:.2f}}} vs_st61c={{'ifeval':{sc['ifeval']-s61c['ifeval']:.2f},'humaneval':{he-s61c['humaneval']:.2f},'gsm8k':{sc['gsm8k']-s61c['gsm8k']:.2f}}}")
print(f"NOL_VERDICT:{verdict}")
PYEOF
echo "NOL_CHAIN_DONE"; date
echo "=== 接 S2 蓄水鏈 ==="
bash "$S/s2_fill_chain.sh"
