#!/bin/bash
# E69 2a 續段:噪音判別後(評測態 vs 材化 he 均值差 1.37 ± 1.52 SE = 無可辨效應)
#   1) 材化 export 官方 IF + GS(2a 核准範圍)
#   2) he 多 seed 重測(發布表改用 4 seed 均值):bf16 錨、p3b_w 各補 seed 1..3
# 哨兵:E69PK2_START / E69PK2_OFFICIAL:<sfx> .. / E69PK2_OFFICIAL_DONE / E69PK2_HE:<sfx>:<seed> he=.. / E69PK2_FAIL:.. / E69PK2_CHAIN_DONE
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval
PACK=exports/e69_p3b_w2_pack; SFX=e69_p3b_w2_pack
BF16=~/.cache/huggingface/hub/models--InternScience--Agents-A1-4B/snapshots/945c40a4aa6f534d434a353207b8d42ecf7a5293
fail() { echo "E69PK2_FAIL:$1"; date; exit 1; }
idle() { while pgrep -f "^\S*python -m p2_anchor|^\S*vllm serve|^\S*/llama-server" >/dev/null; do sleep 30; done; }
serve_vllm() {
  PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/vllm serve "$1" --served-model-name a1 --reasoning-parser qwen3 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --port 8000 > "$S/vllm_e69pk2.log" 2>&1 &
  spid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_e69pk2.log" && return 0
    grep -qE "initialization failed" "$S/vllm_e69pk2.log" && fail serve
    sleep 1
  done; fail serve_timeout
}
stop_vllm() { kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 8; }
score() { python3 -c "import json;d=json.load(open('$1/summary.json'));print(d['strict']['prompt_level'] if 'strict' in d else d['score'])"; }
echo "E69PK2_START"; date
# 1. 官方 IF + GS(材化 export)
if [ ! -f "$A1/ifeval_$SFX/summary.json" ] || [ ! -f "$A1/gsm8k_$SFX/summary.json" ]; then
  idle; serve_vllm "$PACK"
  [ -f "$A1/ifeval_$SFX/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.ifeval_runner --base-url http://127.0.0.1:8000/v1 --model a1 --workers 16 --out "$A1/ifeval_$SFX" > "$S/eval_if_$SFX.log" 2>&1 || fail eval_if
  [ -f "$A1/gsm8k_$SFX/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject gsm8k --base-url http://127.0.0.1:8000/v1 --model a1 --workers 16 --out "$A1/gsm8k_$SFX" > "$S/eval_gs_$SFX.log" 2>&1 || fail eval_gs
  stop_vllm
fi
echo "E69PK2_OFFICIAL:$SFX if $(score $A1/ifeval_$SFX)"
echo "E69PK2_OFFICIAL:$SFX gs $(score $A1/gsm8k_$SFX)"
echo "E69PK2_OFFICIAL_DONE:$SFX"; date
# 2. he 多 seed:bf16 錨、p3b_w
for pair in "$BF16:bf16_w128" "exports/e69_p3b_w:e69_p3b_w"; do
  EXP=${pair%%:*}; T=${pair##*:}
  need=0; for seed in 1 2 3; do [ -f "$A1/humaneval_${T}_s$seed/summary.json" ] || need=1; done
  [ $need = 1 ] || continue
  idle; serve_vllm "$EXP"
  for seed in 1 2 3; do
    OUT="$A1/humaneval_${T}_s$seed"
    [ -f "$OUT/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --seed $seed --out "$OUT" > "$S/eval_he_${T}_s$seed.log" 2>&1 || fail "eval_${T}_s$seed"
    echo "E69PK2_HE:$T:$seed he=$(score $OUT)"; date
  done
  stop_vllm
done
echo "E69PK2_CHAIN_DONE"
