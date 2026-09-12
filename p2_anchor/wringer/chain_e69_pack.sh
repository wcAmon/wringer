#!/bin/bash
# E69 第一階段 2a(用戶 09-12 核准):p3b_w2 打包驗證 → 容器材化 export → he 閘 → 官方 IF+GS
#   閘:he ≥ 89.02 − 1.22(兩題噪音帶),不過則停並報。
# 哨兵:E69PK_START / E69PK_PACK / E69PK_MAT / E69PK_HE / E69PK_GATE_FAIL / E69PK_OFFICIAL:.. / E69PK_OFFICIAL_DONE / E69PK_FAIL:.. / E69PK_CHAIN_DONE
cd $REPO || exit 1
S=$SCRATCH
EVC=evidence/p1_grouping/corkscrew; A1=evidence/p1_grouping/a1eval
TAG=p3b_w2; SRC=exports/e69_$TAG; OUT=exports/e69_${TAG}_pack; SFX=e69_${TAG}_pack
CONT=data/wringer_$TAG.safetensors
fail() { echo "E69PK_FAIL:$1"; date; exit 1; }
idle() { while pgrep -f "^\S*python -m p2_anchor|^\S*vllm serve|^\S*/llama-server" >/dev/null; do sleep 30; done; }
serve_vllm() {
  PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/vllm serve "$1" --served-model-name a1 --reasoning-parser qwen3 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --port 8000 > "$S/vllm_e69pk.log" 2>&1 &
  spid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_e69pk.log" && return 0
    grep -qE "initialization failed" "$S/vllm_e69pk.log" && fail serve
    sleep 1
  done; fail serve_timeout
}
score() { python3 -c "import json;d=json.load(open('$1/summary.json'));print(d['strict']['prompt_level'] if 'strict' in d else d['score'])"; }
echo "E69PK_START:$TAG"; date
# 1. 打包驗證(CPU)
if [ ! -f "$EVC/pack_$TAG.json" ]; then
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.pack_verify --state-tag $TAG --export $SRC --out $CONT > "$S/pack_$TAG.log" 2>&1 || fail pack
fi
echo "E69PK_PACK:$TAG $(grep PACK_DONE $S/pack_$TAG.log | cut -c1-200)"; date
# 2. 容器材化 export
if [ ! -f "$OUT/model.safetensors" ]; then
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.pack_materialize --container $CONT --src-export $SRC --out $OUT > "$S/mat_$TAG.log" 2>&1 || fail materialize
fi
echo "E69PK_MAT:$TAG $(grep MATERIALIZE_DONE $S/mat_$TAG.log | cut -c1-200)"; date
# 3. he 閘
if [ ! -f "$A1/humaneval_$SFX/summary.json" ]; then
  idle; serve_vllm "$OUT"
  PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --out "$A1/humaneval_$SFX" > "$S/eval_he_$SFX.log" 2>&1 || fail eval_he
  kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 8
fi
HE=$(score $A1/humaneval_$SFX)
echo "E69PK_HE:$TAG he=$HE"; date
python3 -c "import sys; sys.exit(0 if float('$HE') >= 0.8902-0.0122 else 1)" || { echo "E69PK_GATE_FAIL:$TAG he=$HE < 0.8780"; echo "E69PK_CHAIN_DONE"; exit 0; }
# 4. 官方 IF + GS
if [ ! -f "$A1/ifeval_$SFX/summary.json" ] || [ ! -f "$A1/gsm8k_$SFX/summary.json" ]; then
  idle; serve_vllm "$OUT"
  [ -f "$A1/ifeval_$SFX/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.ifeval_runner --base-url http://127.0.0.1:8000/v1 --model a1 --workers 16 --out "$A1/ifeval_$SFX" > "$S/eval_if_$SFX.log" 2>&1 || fail eval_if
  [ -f "$A1/gsm8k_$SFX/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject gsm8k --base-url http://127.0.0.1:8000/v1 --model a1 --workers 16 --out "$A1/gsm8k_$SFX" > "$S/eval_gs_$SFX.log" 2>&1 || fail eval_gs
  kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 8
fi
echo "E69PK_OFFICIAL:$SFX if $(score $A1/ifeval_$SFX)"
echo "E69PK_OFFICIAL:$SFX gs $(score $A1/gsm8k_$SFX)"
echo "E69PK_OFFICIAL_DONE:$SFX"; date
echo "E69PK_CHAIN_DONE"
