#!/bin/bash
# E69 2a 閘失敗後的儀器判別:材化 export(容器 α)vs 評測態 export 各跑 3 個額外 seed 的 he,
#   以雙均值判定 −2.44 pp 是採樣噪音還是 α 精度效應。
# 哨兵:E69PN_START / E69PN_HE:<sfx>:<seed> he=.. / E69PN_FAIL:.. / E69PN_CHAIN_DONE
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval
fail() { echo "E69PN_FAIL:$1"; date; exit 1; }
idle() { while pgrep -f "^\S*python -m p2_anchor|^\S*vllm serve|^\S*/llama-server" >/dev/null; do sleep 30; done; }
serve_vllm() {
  PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/vllm serve "$1" --served-model-name a1 --reasoning-parser qwen3 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --port 8000 > "$S/vllm_e69pn.log" 2>&1 &
  spid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_e69pn.log" && return 0
    grep -qE "initialization failed" "$S/vllm_e69pn.log" && fail serve
    sleep 1
  done; fail serve_timeout
}
score() { python3 -c "import json;d=json.load(open('$1/summary.json'));print(d['score'])"; }
echo "E69PN_START"; date
for pair in "exports/e69_p3b_w2:e69_p3b_w2" "exports/e69_p3b_w2_pack:e69_p3b_w2_pack"; do
  EXP=${pair%%:*}; SFX=${pair##*:}
  idle; serve_vllm "$EXP"
  for seed in 1 2 3; do
    OUT="$A1/humaneval_${SFX}_s$seed"
    [ -f "$OUT/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --seed $seed --out "$OUT" > "$S/eval_he_${SFX}_s$seed.log" 2>&1 || fail "eval_${SFX}_s$seed"
    echo "E69PN_HE:$SFX:$seed he=$(score $OUT)"; date
  done
  kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 8
done
echo "E69PN_CHAIN_DONE"
