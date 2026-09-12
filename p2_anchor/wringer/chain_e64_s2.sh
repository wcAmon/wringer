#!/bin/bash
# E64-PL S2(prereg_e64_probes FROZEN;用戶裁 Q3 權重占比對齊):位元放置——T2β 深層讀者端 vs down_proj 全層
#   等 S1 鏈 E64S1_CHAIN_DONE + GPU 空 → probe_t2(A4 st54r+res58r1,arms ctrl,t2beta,存 pt64_t2beta 全模組)
#   → compose_sel:gate/up/in_proj_qkv/in_proj_z 且層 ≥ 21 ← t2beta(權重占比 0.2162 ≈ down_proj 0.2118),其餘 ← qs_pa63_asymblkq2
#   → export → 官方三科 → verdict_e64_s2.json(J4:Δ_place = readers_asym − pt63_t2beta_dp_asym 0.6112;帶 ±0.016)
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval
EVC=evidence/p1_grouping/corkscrew
fail() { echo "E64S2_FAIL:$1"; date; exit 1; }

echo "E64S2_WAIT"; date
while :; do
  if grep -q "^E64S1_CHAIN_DONE\|^E64S1_FAIL" "$S/e64_s1_chain.log" 2>/dev/null && [ ! -f "$S/E64_HOLD" ] \
     && ! pgrep -f "vllm serve" >/dev/null && ! pgrep -f "probe_asym_mix" >/dev/null; then break; fi
  sleep 60
done
echo "E64S2_START"; date
if [ ! -f data/qs_pt64_t2beta/layer31.pt ]; then
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.probe_t2 \
    --n-calib 128 --batch 4 --alt 3 --arms ctrl,t2beta --save-arms t2beta --save-prefix pt64 \
    --out "$EVC/probe_t2_e64.json" > "$S/probe_t2_e64.log" 2>&1 || fail probe
  grep -q "^PROBE_T2_DONE" "$S/probe_t2_e64.log" || fail probe_sentinel
fi
echo "E64S2_PROBE_DONE"; date
NAME=pt64_t2beta_readers_asym
[ -f data/qs_$NAME/layer31.pt ] || \
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.probe_t2 --compose --save-prefix pt64 \
    --compose-arm t2beta --compose-families gate_proj,up_proj,in_proj_qkv,in_proj_z --compose-layer-min 21 \
    --compose-others qs_pa63_asymblkq2 --compose-name $NAME > "$S/compose_$NAME.log" 2>&1 || fail compose
cat "$S/compose_$NAME.log"
echo "E64S2_COMPOSE_DONE"; date

serve() {
  PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/vllm serve "$1" \
    --served-model-name a1 --reasoning-parser qwen3 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --port 8000 > "$S/vllm_e64s2.log" 2>&1 &
  vpid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_e64s2.log" && return 0
    grep -qE "initialization failed" "$S/vllm_e64s2.log" && fail serve
    sleep 1
  done; fail serve_timeout
}

TAG=$NAME; OUT=exports/e64_$TAG; sfx=e64_$TAG
[ -f "$OUT/config.json" ] || \
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export \
    --state-tag $TAG --out $OUT --verify > "$S/export_$TAG.log" 2>&1 || fail export
echo "E64S2_EXPORT_DONE"; date
need=0
for sub in ifeval humaneval gsm8k; do [ -f "$A1/${sub}_${sfx}/summary.json" ] || need=1; done
if [ "$need" = 1 ]; then
  serve "$OUT"
  [ -f "$A1/ifeval_${sfx}/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.ifeval_runner \
    --base-url http://127.0.0.1:8000/v1 --model a1 --out "$A1/ifeval_${sfx}" > "$S/eval_if_$TAG.log" 2>&1 || fail eval_if
  for sub in humaneval gsm8k; do
    [ -f "$A1/${sub}_${sfx}/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner \
      --subject "$sub" --base-url http://127.0.0.1:8000/v1 --model a1 --out "$A1/${sub}_${sfx}" > "$S/eval_${sub}_$TAG.log" 2>&1 || fail "eval_$sub"
  done
  kill $vpid 2>/dev/null; wait $vpid 2>/dev/null; sleep 8
fi
echo "E64S2_EVAL_DONE"; date

python3 - <<'PYEOF'
import json
A1="evidence/p1_grouping/a1eval"
old={"ifeval":92.79,"humaneval":94.51,"gsm8k":95.53}
A4=0.7038
ref=json.load(open("p2_anchor/wringer/verdict_e63_t2.json"))["arms_official"]
sfx="e64_pt64_t2beta_readers_asym"; sc={}; emp={}
for sub in old:
    d=json.load(open(f"{A1}/{sub}_{sfx}/summary.json"))
    sc[sub]=round((d["strict"]["prompt_level"] if sub=="ifeval" else d["score"])*100,2)
    emp[sub]=sum(1 for l in open(f"{A1}/{sub}_{sfx}/responses.jsonl") if not json.loads(l)["response"].strip())
comp=sum(sc[s]/old[s] for s in old)/3
dp=ref["pt63_t2beta_dp_asym"]; asym=ref["pa63_asymblkq2"]
d_place=comp-dp["comp"]; d_asym=comp-asym["comp"]
J4="POS" if d_place>0.016 else ("NEG" if d_place<-0.016 else "NULL")
by={s:round(sc[s]-dp[s],2) for s in old}
print(f"E64S2_B:readers_asym if={sc['ifeval']} he={sc['humaneval']} gs={sc['gsm8k']} comp={comp:.4f} rho={comp/A4:.3f} empties={emp}")
print(f"E64S2_DELTA:place={d_place:+.4f} vs_asym={d_asym:+.4f} by={by} J4={J4}")
json.dump({"experiment":"E64-PL S2 位元放置(A4 零訓練):T2β 深層讀者端 L21–31(占比 0.2162)+ r vs down_proj 全層(0.2118)+ r",
           "arm":{"scores":sc,"comp":comp,"rho":comp/A4,"empties":emp},
           "refs":{"pt63_t2beta_dp_asym":dp,"pa63_asymblkq2":asym},
           "judges":{"J4":J4,"delta_place":round(d_place,4),"delta_vs_asym":round(d_asym,4),"by_subject_vs_dp":by}},
          open("p2_anchor/wringer/verdict_e64_s2.json","w"),ensure_ascii=False,indent=1)
PYEOF
echo "E64S2_CHAIN_DONE"; date
