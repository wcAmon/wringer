#!/bin/bash
# E65-X 兩端夾擊階梯(prereg_e65_x.json;零訓練;全部 he 單科裁)
#   U 臂:g9 逐段折 T2(ladder_e65x --mode fold,循序學生輸入,未折層不動)U0(=g9 同日對照)/U1 L0–15(cross vs w9 目標 A/B)/U2 L0–19/U3 L0–23/U4 L0–27
#         U1 he < 0.70 ⇒ falsifier_U,跳過 U2–U4
#   D 臂:pa64 逐段加 T2β(--mode t2add)D1 down_proj L0–15 / D2 +全模組 L8–15 / D3 全模組 L0–15
#   A 臂:U1/U2 的 g128 變體(折層 α/r 在 g128 下解;--fold-block 128)裁 he;pa64 事後粗化 α 已否證(煙測),α8 量化器待設計
#   每點:ladder → export --verify → vLLM → 評測 → 刪 export(態保留於 data/qs_ux_*)→ verdict_e65_x.json
#   Hold:$S/E65X_HOLD 存在則暫停點火下一點(每 60 s 重查)
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval
EVC=evidence/p1_grouping/corkscrew
fail() { echo "E65X_FAIL:$1"; date; exit 1; }
hold() { while [ -f "$S/E65X_HOLD" ]; do sleep 60; done; }

serve() {
  PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/vllm serve "$1" \
    --served-model-name a1 --reasoning-parser qwen3 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --port 8000 > "$S/vllm_e65x.log" 2>&1 &
  vpid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_e65x.log" && return 0
    grep -qE "initialization failed" "$S/vllm_e65x.log" && fail serve
    sleep 1
  done; fail serve_timeout
}

# point <tag> <subjects:he|he,if> <ladder args...>
point() {
  local TAG=$1 SUBJ=$2; shift 2
  hold
  if [ ! -f "data/qs_$TAG/layer31.pt" ] || [ ! -f "$EVC/ladder_$TAG.json" ]; then
    PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.ladder_e65x --save-tag $TAG "$@" > "$S/ladder_$TAG.log" 2>&1 || fail "ladder_$TAG"   # 不用 eval:t2-spec 含 ;
    grep -q "^LADDER_DONE" "$S/ladder_$TAG.log" || fail "ladder_sentinel_$TAG"
  fi
  grep "^LADDER_LEDGER" "$S/ladder_$TAG.log"
  echo "E65X_LADDER_DONE:$TAG"; date
  local OUT=exports/e65x_$TAG sfx=e65x_$TAG need=0
  [ -f "$A1/humaneval_${sfx}/summary.json" ] || need=1
  [ "$SUBJ" = "he,if" ] && { [ -f "$A1/ifeval_${sfx}/summary.json" ] || need=1; }
  if [ "$need" = 1 ]; then
    [ -f "$OUT/config.json" ] || \
      PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export --state-tag $TAG --out $OUT --verify > "$S/export_$TAG.log" 2>&1 || fail "export_$TAG"
    pgrep -f "vllm serve" >/dev/null && fail gpu_busy_vllm
    serve "$OUT"
    [ -f "$A1/humaneval_${sfx}/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner \
      --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --out "$A1/humaneval_${sfx}" > "$S/eval_he_$TAG.log" 2>&1 || fail "eval_he_$TAG"
    if [ "$SUBJ" = "he,if" ]; then
      [ -f "$A1/ifeval_${sfx}/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.ifeval_runner \
        --base-url http://127.0.0.1:8000/v1 --model a1 --out "$A1/ifeval_${sfx}" > "$S/eval_if_$TAG.log" 2>&1 || fail "eval_if_$TAG"
    fi
    kill $vpid 2>/dev/null; wait $vpid 2>/dev/null; sleep 8
    rm -rf "$OUT"
  fi
  HE=$(python3 -c "import json;print(json.load(open('$A1/humaneval_${sfx}/summary.json'))['score'])")
  echo "E65X_POINT:$TAG he=$HE"; date
}

echo "E65X_START"; date
pgrep -f "vllm serve" >/dev/null && fail gpu_busy_at_start

# ---- U 臂(amendment_2:未折層不重擬;U1 交叉目標 vs w9 目標 A/B,勝者定 TARGET)----
point ux_u0 he --mode fold --state-tag g9 --fold-layers none --no-refit-others
point ux_u1 he --mode fold --state-tag g9 --fold-layers 0-15 --no-refit-others --target cross
point ux_u1w he --mode fold --state-tag g9 --fold-layers 0-15 --no-refit-others --target w9
HE_U1=$(python3 -c "import json;print(json.load(open('$A1/humaneval_e65x_ux_u1/summary.json'))['score'])")
HE_U1W=$(python3 -c "import json;print(json.load(open('$A1/humaneval_e65x_ux_u1w/summary.json'))['score'])")
TARGET=$(python3 -c "print('cross' if $HE_U1 >= $HE_U1W else 'w9')")
HE_BEST=$(python3 -c "print(max($HE_U1,$HE_U1W))")
echo "E65X_TARGET:$TARGET he_cross=$HE_U1 he_w9=$HE_U1W"; date
if python3 -c "import sys;sys.exit(0 if $HE_BEST >= 0.70 else 1)"; then
  point ux_u2 he --mode fold --state-tag g9 --fold-layers 0-19 --no-refit-others --target $TARGET
  point ux_u3 he --mode fold --state-tag g9 --fold-layers 0-23 --no-refit-others --target $TARGET
  point ux_u4 he --mode fold --state-tag g9 --fold-layers 0-27 --no-refit-others --target $TARGET
else
  echo "E65X_FALSIFIER_U:he_u1=$HE_BEST <0.70,跳過 U2–U4"; date
fi

# ---- D 臂 ----
point ux_d1 he --mode t2add --t2-spec "down_proj:0-15" --target-state-tag st51c_e2e
point ux_d2 he --mode t2add --t2-spec "down_proj:0-15;all:8-15" --target-state-tag st51c_e2e
point ux_d3 he --mode t2add --t2-spec "all:0-15" --target-state-tag st51c_e2e

# ---- A 臂(g128)撤銷:falsifier_U 成立(U1 0.0 / U1w 0.61),折層 g128 變體無可比對象(amendment_3)----
# ---- D 臂 amendment_3:pa64 已含水,目標基底改 dense(st51c_e2e)+水(舊 D1 雙水 he 55.49 作廢,存 *_bad_doublewater)----

PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.verdict_e65_x || fail verdict
echo "E65X_CHAIN_DONE"; date
