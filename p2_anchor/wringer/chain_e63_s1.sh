#!/bin/bash
# E63 S1(用戶 2026-09-06 17:4x 裁「1-3 步跑」;prereg_e63_s1 FROZEN):
#   等步驟 1(STRIPM_CHAIN_DONE)釋出 GPU → 煙測 st63smk(--smoke --asym-r-bits 2)→ 閘 →
#   正式 st63a(甲 st58rb hparams + --asym-r-bits 2)→ export(--verify)→ vLLM 三科 → 裁 ρ 對階梯
#   Hold:$S/S1_HOLD 存在則不點火(每 60s 重查)
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval
EVC=evidence/p1_grouping/corkscrew
fail() { echo "S1_FAIL:$1"; date; exit 1; }
JIA="--state-tag st54r --res-tag res58r1 --n-calib 128 \
  --calib evidence/p1_grouping/calib_e58r1b_traj.pt --lengths evidence/p1_grouping/calib_e58r1b_len.pt \
  --batch 4 --win 2 --stride 1 --gsteps 5 --tau 0.5 --w-max 4 --comp-rounds 3 --comp-cap 0.03 \
  --tail-soft 10 --tail-ste 4 --s0 30 --s-plateau 2 --lora-r 64 --lora-lr 6e-4 --factor-lr 3e-3 \
  --clip 1.0 --learn-theta --holdout-frac 0.05 --x-cpu"
ASYM="--asym-r-bits 2 --asym-alt 2"
echo "S1_WAIT"; date
while :; do
  if grep -q "^STRIPM_CHAIN_DONE\|^STRIPM_FAIL" "$S/e63_stripm.log" 2>/dev/null && [ ! -f "$S/S1_HOLD" ] \
     && ! pgrep -f "vllm serve" >/dev/null && ! pgrep -f "train_res" >/dev/null; then break; fi
  sleep 60
done
echo "S1_SMOKE_START"; date
if [ ! -f "$EVC/train_st63smk.json" ]; then
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.train_res --tag st63smk --smoke $JIA $ASYM \
    > "$S/train_st63smk.log" 2>&1 || fail smoke_run
fi
python3 - <<'PYEOF' || exit 1
import json
d=json.load(open("evidence/p1_grouping/corkscrew/train_st63smk.json"))
evs=[e for w in d.get("windows",[]) for e in w.get("drain_events",[])] if "windows" in d else d.get("drain_events",[])
als=[a for e in evs for a in e.get("asym_ls",[])]
ok = len(als)>0 and all(a["gain"]>-1e-6 for a in als)
print(f"S1_SMOKE_GATE:{'PASS' if ok else 'FAIL'} events={len(evs)} asym_ls={len(als)} "
      f"gain_mean={sum(a['gain'] for a in als)/max(len(als),1):.4f} "
      f"refit_flip_mean={sum(a['refit_flip'] for a in als)/max(len(als),1):.5f} "
      f"val_ce={d.get('post_drain_val_ce')}")
raise SystemExit(0 if ok else 1)
PYEOF
[ $? -eq 0 ] || fail smoke_gate
echo "S1_SMOKE_DONE"; date
[ -f "$S/S1_HOLD" ] && fail hold_after_smoke
echo "S1_TRAIN_START"; date
if [ ! -f "$EVC/train_st63a.json" ]; then
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.train_res --tag st63a --resume $JIA $ASYM \
    > "$S/train_st63a.log" 2>&1 || fail train_run
fi
grep -q "post-drain val CE" "$S/train_st63a.log" || fail train_sentinel
echo "S1_TRAIN_DONE"; date
OUT=exports/e63_st63a; sfx=e63_st63a
[ -f "$OUT/config.json" ] || \
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export --state-tag st63a --out $OUT --verify || fail export
echo "S1_EXPORT_DONE"; date
serve() {
  PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/vllm serve "$1" \
    --served-model-name a1 --reasoning-parser qwen3 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --port 8000 > "$S/vllm_s1.log" 2>&1 &
  vpid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_s1.log" && return 0
    grep -qE "initialization failed" "$S/vllm_s1.log" && fail serve
    sleep 1
  done; fail serve_timeout
}
need=0
for sub in ifeval humaneval gsm8k; do [ -f "$A1/${sub}_${sfx}/summary.json" ] || need=1; done
if [ "$need" = 1 ]; then
  while pgrep -f "vllm serve" >/dev/null; do sleep 60; done
  serve "$OUT"
  [ -f "$A1/ifeval_${sfx}/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.ifeval_runner \
    --base-url http://127.0.0.1:8000/v1 --model a1 --out "$A1/ifeval_${sfx}" || fail eval_if
  for sub in humaneval gsm8k; do
    [ -f "$A1/${sub}_${sfx}/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner \
      --subject "$sub" --base-url http://127.0.0.1:8000/v1 --model a1 --out "$A1/${sub}_${sfx}" || fail "eval_${sub}"
  done
  kill $vpid 2>/dev/null; wait $vpid 2>/dev/null; sleep 8
fi
echo "S1_EVAL_DONE"; date
SFX=$sfx python3 - <<'PYEOF'
import json, os
A1="evidence/p1_grouping/a1eval"; sfx=os.environ["SFX"]
anchors={"ifeval":92.79,"humaneval":94.51,"gsm8k":95.53}; A4=0.7038
ladder={"ctrl0":0.744,"jia_st58rb":0.777,"asym0":0.820,"st61b(碼+M)":0.872,"st61c(碼+M)":0.889}
try:
    sm={}
    for sub in anchors:
        d=json.load(open(f"{A1}/{sub}_e63_st61b_stripm/summary.json"))
        sm[sub]=(d["strict"]["prompt_level"] if sub=="ifeval" else d["score"])*100
    ladder["st61b_stripM"]=round(sum(sm[s]/anchors[s] for s in anchors)/3/A4,3)
except Exception: pass
sc,emp={},{}
for sub in anchors:
    d=json.load(open(f"{A1}/{sub}_{sfx}/summary.json"))
    sc[sub]=round((d["strict"]["prompt_level"] if sub=="ifeval" else d["score"])*100,2)
    emp[sub]=sum(1 for l in open(f"{A1}/{sub}_{sfx}/responses.jsonl") if not json.loads(l)["response"].strip())
comp=sum(sc[s]/anchors[s] for s in anchors)/3; rho=comp/A4; rho_he=sc["humaneval"]/45.12
print(f"S1_B:if={sc['ifeval']} he={sc['humaneval']} gs={sc['gsm8k']} comp={comp:.4f} rho={rho:.3f} rho_he={rho_he:.3f} empties={emp}")
for k,r in ladder.items(): print(f"S1_DELTA:{k} drho={rho-r:+.3f}")
band="PASS" if rho>=0.90 else "PARTIAL" if rho>=0.836 else "FAIL"
print(f"S1_VERDICT:{band} (PASS≥0.90 / PARTIAL≥0.836 / FAIL)")
PYEOF
echo "S1_CHAIN_DONE"; date
