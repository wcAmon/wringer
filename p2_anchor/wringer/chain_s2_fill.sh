#!/bin/bash
# S2 第一步:冠軍 st51c_e2e 蓄水 r128(鏡射 res58r1 配方,fresh 初始化)→ 合併 fp 模型 → 官方三科 → A 閘
#   A 閘語義:碼凍結 + 滿水庫 = 排水典範天花板;B ≈ A × ρ。北極星 student+M+L comp ≥ 0.90。
#   判讀:A ≥ 0.90 排水可行 / 0.80–0.90 天花板不足(M/L 常駐容量從蓄水段共訓)/ <0.80 轉 QAT with 常駐載體
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval
EVC=evidence/p1_grouping/corkscrew
TAG=res62s2
OUT=exports/e62_s2_resA
fail() { echo "S2_FAIL:$1"; date; exit 1; }
echo "S2_START"; date
[ -f "$EVC/fill_$TAG.json" ] || \
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.res_fill \
    --tag $TAG --state-tag st51c_e2e --ref-export exports/st51c_e2e_x \
    --out $OUT --r-res 128 \
    --data evidence/p1_grouping/calib_e58r1_traj.pt --lengths evidence/p1_grouping/calib_e58r1_len.pt \
    --batch 2 --microbatch 1 --kd-impl hidchunk --grad-ckpt \
    --steps 5741 --plateau-eps 0.002 --plateau-min-steps 5741 || fail fill
echo "S2_FILL_DONE"; date
serve() {
  PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/vllm serve "$1" \
    --served-model-name a1 --reasoning-parser qwen3 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --port 8000 > "$S/vllm_s2.log" 2>&1 &
  vpid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_s2.log" && return 0
    grep -qE "initialization failed" "$S/vllm_s2.log" && fail serve
    sleep 1
  done; fail serve_timeout
}
sfx=e62_s2_A; need=0
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
echo "S2_EVAL_DONE"; date
SFX=$sfx python3 - <<'PYEOF'
import json, os
A1="evidence/p1_grouping/a1eval"; sfx=os.environ["SFX"]
anchors={"ifeval":92.79,"humaneval":94.51,"gsm8k":95.53}
champ={"comp":0.7469,"ifeval":75.97,"humaneval":54.88,"gsm8k":80.36}
sc,emp={},{}
for sub in anchors:
    d=json.load(open(f"{A1}/{sub}_{sfx}/summary.json"))
    sc[sub]=round((d["strict"]["prompt_level"] if sub=="ifeval" else d["score"])*100,2)
    emp[sub]=sum(1 for l in open(f"{A1}/{sub}_{sfx}/responses.jsonl") if not json.loads(l)["response"].strip())
comp=sum(sc[s]/anchors[s] for s in anchors)/3
band="DRAIN_OK" if comp>=0.90 else "CEILING_SHORT" if comp>=0.80 else "PIVOT_QAT_CARRIER"
print(f"S2_A:if={sc['ifeval']} he={sc['humaneval']} gs={sc['gsm8k']} comp={comp:.4f} empties={emp}")
print(f"S2_A_DELTA:vs_champion={comp-champ['comp']:+.4f} gate(>0.7629)={'PASS' if comp>champ['comp']+0.016 else 'FAIL'} by={{'ifeval':{sc['ifeval']-champ['ifeval']:+.2f},'humaneval':{sc['humaneval']-champ['humaneval']:+.2f},'gsm8k':{sc['gsm8k']-champ['gsm8k']:+.2f}}}")
print(f"S2_A_VERDICT:{band} headroom_to_0.90={0.90-comp:+.4f}")
PYEOF
echo "S2_CHAIN_DONE"; date
