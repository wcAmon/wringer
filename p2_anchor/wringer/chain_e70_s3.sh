#!/bin/bash
# E70 第三段(prereg stage3_recovery;用戶 09-12 核准 E70 全鏈自主接續):等 E70S2_CHAIN_DONE →
#   勝者 T(E70S2_WINNER)→ R-water:碼凍 r128 KD 蓄水 3000 步(hidchunk、閉合加權 32/16/8、batch 2、grad-ckpt、bf16 自身老師、calib_e70 128 窗)
#   → A 閘 he 2 seed → E64 排水:同 P3 配置閉式重解(--base-dir 蓄水材化)→ B 閘 he 4 seed → 官方 IF+GS → comp、ρ=(B−P3)/(A−P3)
# 哨兵:E70S3_START / E70S3_HANDOFF .. / E70S3_FILL_DONE .. / E70S3_A he=.. / E70S3_LEDGER .. / E70S3_B he=s0,s1,s2,s3 / E70S3_OFFICIAL if= gs= he= comp= rho= / E70S3_FAIL:<step> / E70S3_CHAIN_DONE
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval; EV=evidence/p1_grouping; EVC=evidence/p1_grouping/corkscrew
PRE=p2_anchor/wringer/prereg_e70_qwen3_4b.json
export WRINGER_MODEL=Qwen/Qwen3-4B WRINGER_REV=1cfa9a7208912126459214e8b04321603b3df60c WRINGER_CALIB_VAL=$EV/calib_e70_val.pt
export PATH=$PWD/.venv-vllm/bin:$PATH
CAL="--calib $EV/calib_e70_traj.pt --n-calib 128"
fail() { echo "E70S3_FAIL:$1"; date; exit 1; }
idle() { while pgrep -f "^\S*python -m p2_anchor|^\S*vllm serve|^\S*/llama-server|^\S*/llama-quantize|^\S*/llama-imatrix" >/dev/null; do sleep 15; done; }
serve_vllm() {
  .venv-vllm/bin/vllm serve "$1" --served-model-name a1 --reasoning-parser qwen3 --max-model-len 32768 --gpu-memory-utilization 0.85 --kv-cache-dtype fp8 --port 8000 > "$S/vllm_e70s3.log" 2>&1 &
  spid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_e70s3.log" && return 0
    grep -qE "initialization failed|Traceback" "$S/vllm_e70s3.log" && fail serve
    sleep 1
  done; fail serve_timeout
}
stop_srv() { kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 8; }
score() { python3 -c "import json;d=json.load(open('$1/summary.json'));print(d['strict']['prompt_level'] if 'strict' in d else d['score'])"; }
runhe() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --workers 128 --seed $2 --out "$A1/$1" > "$S/e70_$1.log" 2>&1 || fail "$1"; }
runif() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.ifeval_runner --base-url http://127.0.0.1:8000/v1 --model a1 --workers 128 --seed $2 --out "$A1/$1" > "$S/e70_$1.log" 2>&1 || fail "$1"; }
rungs() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject gsm8k --base-url http://127.0.0.1:8000/v1 --model a1 --workers 128 --seed $2 --out "$A1/$1" > "$S/e70_$1.log" 2>&1 || fail "$1"; }
cand() { python3 -c "import json;c=json.load(open('$PRE'))['stage2_compose']['candidates']['$1'];print(c['$2'])"; }
echo "E70S3_START"; date
for i in $(seq 1 8640); do grep -q "E70S2_CHAIN_DONE\|E70S2_FAIL" "$S/e70s2_chain.log" 2>/dev/null && break; sleep 10; done
echo "E70S3_HANDOFF $(grep -E 'E70S2_CHAIN_DONE|E70S2_FAIL' $S/e70s2_chain.log | head -n 1)"; date
grep -q "E70S2_CHAIN_DONE" "$S/e70s2_chain.log" || fail s2_not_done
T=$(grep "^E70S2_WINNER" "$S/e70s2_chain.log" | tail -n 1 | awk '{print $2}')
[ -n "$T" ] || fail no_winner
AB=$(cand $T alpha_bits); MM=$(cand $T module_map)
P3=e70_$T; RESA=exports/${P3}_resA; W=${P3}_w; OUTW=exports/$W
echo "E70S3_TARGET $T alpha_bits=$AB mm=$MM"; date
[ -f "exports/$P3/config.json" ] || fail "no_export_$P3"
sleep 20
# ---- 蓄水
if [ ! -f "$EVC/fill_res70_$T.json" ]; then
  idle
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.res_fill \
    --tag res70_$T --state-tag $P3 --ref-export exports/$P3 --out $RESA --r-res 128 \
    --data $EV/calib_e70_traj.pt --lengths $EV/calib_e70_len.pt \
    --batch 2 --microbatch 1 --kd-impl hidchunk --grad-ckpt \
    --close-weight 32 --close-post-weight 16 --close-post-k 8 \
    --steps 3000 --plateau-eps 0.002 --plateau-min-steps 3000 > "$S/fill_res70_$T.log" 2>&1 || fail "fill_$T"
fi
echo "E70S3_FILL_DONE $(grep -E 'pre-fill|post-fill' $S/fill_res70_$T.log | tr '\n' ' ' | cut -c1-200)"; date
# ---- A 閘
if [ ! -f "$A1/humaneval_${P3}_resA_s1/summary.json" ]; then
  idle; serve_vllm "$RESA"
  runhe humaneval_${P3}_resA_s20260806 20260806
  runhe humaneval_${P3}_resA_s1 1
  stop_srv
fi
echo "E70S3_A he=$(score $A1/humaneval_${P3}_resA_s20260806),$(score $A1/humaneval_${P3}_resA_s1)"; date
# ---- 排水(同配置閉式重解,基底=蓄水材化)
if [ ! -f "data/qs_$W/layer35.pt" ]; then
  idle
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.quantize --tag $W --base-dir $RESA $CAL --grid 8 --block 128 --solver gptq --prior-lam 0.01 --alpha-bits $AB --module-map "$MM" > "$S/quant_$W.log" 2>&1 || fail "quant_$W"
fi
echo "E70S3_LEDGER $(grep '^QUANT_LEDGER' $S/quant_$W.log | tail -n 1 | cut -c1-300) escal=$(grep -c ESCALATE $S/quant_$W.log)"
[ -f "$OUTW/config.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export --state-tag $W --out $OUTW > "$S/export_$W.log" 2>&1 || fail "export_$W"
# ---- B 閘 4 seed + 官方
if [ ! -f "$A1/gsm8k_$W/summary.json" ]; then
  idle; serve_vllm "$OUTW"
  for sd in 20260806 1 2 3; do runhe humaneval_${W}_s$sd $sd; done
  echo "E70S3_B he=$(score $A1/humaneval_${W}_s20260806),$(score $A1/humaneval_${W}_s1),$(score $A1/humaneval_${W}_s2),$(score $A1/humaneval_${W}_s3)"; date
  runif ifeval_$W 20260806
  rungs gsm8k_$W 20260806
  stop_srv
fi
python3 - "$A1" "$P3" "$W" "$PRE" <<'PY'
import json,sys,statistics as st
a1,p3,w,pre=sys.argv[1:5]
den=json.load(open(pre))["anchor"]["comp_denominators"]
sc=lambda d: json.load(open(f"{a1}/{d}/summary.json"))["score"]*100
heB=[sc(f"humaneval_{w}_s{s}") for s in ("20260806","1","2","3")]
heA=[sc(f"humaneval_{p3}_resA_s{s}") for s in ("20260806","1")]
heP=[sc(f"humaneval_{p3}_s{s}") for s in ("20260806","1")]
iff=json.load(open(f"{a1}/ifeval_{w}/summary.json"))["strict"]["prompt_level"]*100
gs=sc(f"gsm8k_{w}")
B,A,P=st.mean(heB),st.mean(heA),st.mean(heP)
rho=(B-P)/(A-P) if abs(A-P)>1e-9 else float("nan")
comp=(iff/den["IF"]+B/den["HE"]+gs/den["GS"])/3
print(f"E70S3_OFFICIAL:{w} if={iff:.2f} gs={gs:.2f} he={B:.2f}±{st.stdev(heB):.2f} comp={comp:.4f} A={A:.2f} P3={P:.2f} rho={rho:.3f}")
PY
date
echo "E70S3_CHAIN_DONE"; date
