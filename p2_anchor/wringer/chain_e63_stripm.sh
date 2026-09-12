#!/bin/bash
# 步驟 1(用戶 2026-09-06 17:4x 裁「1-3 步跑」):st61b 去 M 零訓練官方 —— 修正純碼訓練基線
#   st61b 位面含 .mix(M 近單位陣,偏 I 0.0013);去 M 後官方 ρ 對照 st61b 0.872 / 甲 0.777
#   讀法:ρ_stripM ≥ 0.856(0.872−帶)⇒ M 近單位陣不承載,純碼訓練基線上修為 ~0.87;
#         ρ_stripM ≤ 0.793(甲+帶)⇒ M 承載 ≥ 0.08,純碼基線=甲 0.777 維持;其間=部分承載
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval
OUT=exports/e63_st61b_stripm
sfx=e63_st61b_stripm
fail() { echo "STRIPM_FAIL:$1"; date; exit 1; }
echo "STRIPM_START"; date
[ -f "$OUT/config.json" ] || \
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export \
    --state-tag st61b --out $OUT --strip-mix --verify || fail export
echo "STRIPM_EXPORT_DONE"; date
serve() {
  PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/vllm serve "$1" \
    --served-model-name a1 --reasoning-parser qwen3 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --port 8000 > "$S/vllm_stripm.log" 2>&1 &
  vpid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_stripm.log" && return 0
    grep -qE "initialization failed" "$S/vllm_stripm.log" && fail serve
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
echo "STRIPM_EVAL_DONE"; date
SFX=$sfx python3 - <<'PYEOF'
import json, os
A1="evidence/p1_grouping/a1eval"; sfx=os.environ["SFX"]
anchors={"ifeval":92.79,"humaneval":94.51,"gsm8k":95.53}; A4=0.7038
ref={"st61b(碼+M)":{"rho":0.872,"ifeval":73.20,"humaneval":33.54,"gsm8k":66.57},
     "jia_st58rb(純碼)":{"rho":0.777,"ifeval":68.39,"humaneval":25.61,"gsm8k":60.50},
     "asym0(零訓練非對稱)":{"rho":0.820,"ifeval":69.69,"humaneval":34.15,"gsm8k":59.14}}
sc,emp={},{}
for sub in anchors:
    d=json.load(open(f"{A1}/{sub}_{sfx}/summary.json"))
    sc[sub]=round((d["strict"]["prompt_level"] if sub=="ifeval" else d["score"])*100,2)
    emp[sub]=sum(1 for l in open(f"{A1}/{sub}_{sfx}/responses.jsonl") if not json.loads(l)["response"].strip())
comp=sum(sc[s]/anchors[s] for s in anchors)/3; rho=comp/A4
print(f"STRIPM_B:if={sc['ifeval']} he={sc['humaneval']} gs={sc['gsm8k']} comp={comp:.4f} rho={rho:.3f} empties={emp}")
for k,r in ref.items():
    print(f"STRIPM_DELTA:{k} drho={rho-r['rho']:+.3f} by={{'if':{sc['ifeval']-r['ifeval']:+.2f},'he':{sc['humaneval']-r['humaneval']:+.2f},'gs':{sc['gsm8k']-r['gsm8k']:+.2f}}}")
v="M_NOT_CARRYING(純碼基線上修~0.87)" if rho>=0.856 else "M_CARRIES(純碼基線=甲 0.777)" if rho<=0.793 else "M_PARTIAL"
print(f"STRIPM_VERDICT:{v}")
PYEOF
echo "STRIPM_CHAIN_DONE"; date
