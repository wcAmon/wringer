#!/bin/bash
# E67-B 第一階(prereg_e67.json B.start/stage1;用戶「自動接」到 he1 為止):
#   等 A 鏈收 → J_A PASS/GREY ⇒ B0=gsq_b0(λ0.58,H≈2.65)/ FAIL ⇒ B0=rd_b0(率傾斜 λ0.23,H≈2.66)
#   → export+vLLM he0 → e2e_soft KD(閉合加權 32/16/8 + 率項 λ0.05 保熵,800 步,e58r1b 16k on-manifold 語料)
#   → 熵帳 H0/H1(state_entropy)→ export+vLLM he1 → e67b_stage1.json;falsifier he1 ≤ he0+3 ⇒ 標 STOP(目標改 2.9–3.0),蓄水由用戶裁
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval
EVC=evidence/p1_grouping/corkscrew
fail() { echo "E67B_FAIL:$1"; date; exit 1; }
serve() {
  PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/vllm serve "$1" --served-model-name a1 --reasoning-parser qwen3 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --port 8000 > "$S/vllm_e67b.log" 2>&1 &
  vpid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_e67b.log" && return 0
    grep -qE "initialization failed" "$S/vllm_e67b.log" && fail serve
    sleep 1
  done; fail serve_timeout
}
he_of() {   # he_of <state-tag> <sfx>
  local TAG=$1 sfx=$2 OUT=exports/e67b_$1
  if [ ! -f "$A1/humaneval_${sfx}/summary.json" ]; then
    [ -f "$OUT/config.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export --state-tag $TAG --out $OUT --verify > "$S/export_$TAG.log" 2>&1 || fail "export_$TAG"
    pgrep -f "vllm serv[e]|llama-serve[r]" >/dev/null && fail gpu_busy
    serve "$OUT"
    PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --out "$A1/humaneval_${sfx}" > "$S/eval_he_$TAG.log" 2>&1 || fail "eval_he_$TAG"
    kill $vpid 2>/dev/null; wait $vpid 2>/dev/null; sleep 8
    rm -rf "$OUT"
  fi
  python3 -c "import json;print(json.load(open('$A1/humaneval_${sfx}/summary.json'))['score'])"
}

echo "E67B_WAIT_A"; date
until grep -q "E67A_WRAPPER_EXIT" $S/e67a_wrapper.log 2>/dev/null; do sleep 120; done
until grep -q "B0_PRESOLVE_DONE" $S/b0_presolve.log 2>/dev/null; do grep -q "B0_PRESOLVE_ABORT" $S/b0_presolve.log 2>/dev/null && fail presolve_abort; sleep 60; done
while pgrep -f "vllm serv[e]|llama-serve[r]|e2e_sof[t]" >/dev/null; do sleep 30; done
grep -q "^E2E_DONE" $S/e2e_rate_smoke3.log 2>/dev/null || fail e2e_rate_smoke_not_passed
[ -f p2_anchor/wringer/verdict_e67.json ] || fail no_verdict_e67
JA=$(python3 -c "import json;print(json.load(open('p2_anchor/wringer/verdict_e67.json'))['judges']['J_A'])")
case "$JA" in FAIL*) B0=rd_b0;; *) B0=gsq_b0;; esac
echo "E67B_START J_A=$JA B0=$B0"; date
[ -f data/qs_$B0/layer31.pt ] || fail "no_state_$B0"
H0=$(PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.state_entropy --state-tag $B0 --out $EVC/entropy_$B0.json | grep -oE '"H_joint": [0-9.]+' | grep -oE '[0-9.]+$')
HE0=$(he_of $B0 e67b_$B0)
echo "E67B_HE0:$B0 he0=$HE0 H0=$H0"; date

TAG=e67b_s1
if [ ! -f data/qs_$TAG/layer31.pt ]; then
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.e2e_soft --tag $TAG --state-tag $B0 \
    --data evidence/p1_grouping/calib_e58r1b_traj.pt --lengths evidence/p1_grouping/calib_e58r1b_len.pt \
    --steps 800 --batch 2 --microbatch 1 --gamma 0.8 --kd-impl hidchunk --full-ckpt --grad-ckpt \
    --close-weight 32 --close-post-weight 16 --close-post-k 8 \
    --rate-lam 0.05 > "$S/e2e_$TAG.log" 2>&1 || fail "e2e_$TAG"
  grep -q "^E2E_DONE" "$S/e2e_$TAG.log" || fail "e2e_sentinel_$TAG"
fi
echo "E67B_E2E_DONE:$TAG"; date
H1=$(PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.state_entropy --state-tag $TAG --out $EVC/entropy_$TAG.json | grep -oE '"H_joint": [0-9.]+' | grep -oE '[0-9.]+$')
HE1=$(he_of $TAG e67b_$TAG)
python3 - <<PY
import json
he0,he1,H0,H1=float("$HE0"),float("$HE1"),float("$H0"),float("$H1")
d={"experiment":"E67-B stage1","J_A":"$JA","B0":"$B0","he0":he0,"he1":he1,"H0":H0,"H1":H1,"dhe_pp":round(100*(he1-he0),2),"dH":round(H1-H0,4),
   "falsifier_he1_le_he0p3": he1<=he0+0.03, "entropy_kept": H1<=H0+0.05,
   "verdict": ("STOP(訓練換不回 he;目標改 2.9–3.0)" if he1<=he0+0.03 else "GO(he1 過閘;蓄水待用戶裁)") + ("" if H1<=H0+0.05 else " ⚠ 熵漂移 >0.05,he 增益含位元回流"),
   "recipe":{"e2e":"800 步 batch2 γ0.8 hidchunk full_ckpt close 32/16/8 rate_lam 0.05 corpus e58r1b"}}
json.dump(d,open("$EVC/e67b_stage1.json","w"),ensure_ascii=False,indent=1); print("E67B_STAGE1 "+json.dumps(d,ensure_ascii=False))
PY
echo "E67B_S1_DONE"; date
