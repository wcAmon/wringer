#!/bin/bash
# E70 發布打包(用戶 09-13 裁「第二模型也上 HF」):Qwen3-4B p3a_w
#   1. 打包驗證(CPU)→ 2. 容器材化 export(CPU)→ [等 E70S1_CHAIN_DONE 讓出 GPU] → 3. 材化 he 4 seed(E70 協定 KV fp8 + w128)→ 4. 官方 IF + GS
#   閘:材化 he 4 seed 均值 ≥ 研究態 78.96 − 2.7(單跑噪音帶);不過則停並報(不上傳)。
# 哨兵:E70PK_START / E70PK_PACK / E70PK_MAT / E70PK_HANDOFF / E70PK_HE / E70PK_GATE_FAIL / E70PK_OFFICIAL:.. / E70PK_FAIL:.. / E70PK_CHAIN_DONE
cd $REPO || exit 1
S=$SCRATCH
EVC=evidence/p1_grouping/corkscrew; A1=evidence/p1_grouping/a1eval
export WRINGER_MODEL=Qwen/Qwen3-4B WRINGER_REV=1cfa9a7208912126459214e8b04321603b3df60c
TAG=e70_p3a_w; SRC=exports/$TAG; OUT=exports/${TAG}_pack; SFX=${TAG}_pack
CONT=data/wringer_$TAG.safetensors
fail() { echo "E70PK_FAIL:$1"; date; kill $spid 2>/dev/null; exit 1; }
idle() { while pgrep -f "^\S*python -m p2_anchor|^\S*vllm serve|^\S*/llama-server|^\S*/llama-quantize|^\S*/llama-imatrix" >/dev/null; do sleep 30; done; }
serve_vllm() {
  PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/vllm serve "$1" --served-model-name a1 --reasoning-parser qwen3 --max-model-len 32768 --gpu-memory-utilization 0.85 --kv-cache-dtype fp8 --port 8000 > "$S/vllm_e70pk.log" 2>&1 &
  spid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_e70pk.log" && return 0
    grep -qE "initialization failed|Traceback" "$S/vllm_e70pk.log" && fail serve
    sleep 1
  done; fail serve_timeout
}
stop_srv() { kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 8; }
score() { python3 -c "import json;d=json.load(open('$1/summary.json'));print(d['strict']['prompt_level'] if 'strict' in d else d['score'])"; }
runhe() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --workers 128 --seed $2 --out "$A1/$1" > "$S/e70_$1.log" 2>&1 || fail "$1"; }
runif() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.ifeval_runner --base-url http://127.0.0.1:8000/v1 --model a1 --workers 128 --seed $2 --out "$A1/$1" > "$S/e70_$1.log" 2>&1 || fail "$1"; }
rungs() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject gsm8k --base-url http://127.0.0.1:8000/v1 --model a1 --workers 128 --seed $2 --out "$A1/$1" > "$S/e70_$1.log" 2>&1 || fail "$1"; }
echo "E70PK_START:$TAG"; date
# 1. 打包驗證(CPU)
if [ ! -f "$EVC/pack_$TAG.json" ]; then
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.pack_verify --state-tag $TAG --export $SRC --out $CONT > "$S/pack_$TAG.log" 2>&1 || fail pack
fi
echo "E70PK_PACK:$TAG $(grep PACK_DONE $S/pack_$TAG.log | cut -c1-200)"; date
# 2. 容器材化 export(CPU)
if [ ! -f "$OUT/model.safetensors" ]; then
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.pack_materialize --container $CONT --src-export $SRC --out $OUT > "$S/mat_$TAG.log" 2>&1 || fail materialize
fi
echo "E70PK_MAT:$TAG $(grep MATERIALIZE_DONE $S/mat_$TAG.log | cut -c1-200)"; date
# 等 S1U 讓出 GPU
# S1U 追加寫入同一日誌,07:47 有舊 FAIL 行 → 只看最後一個 E70S1_START 之後的段落
s1u_end() { awk '/^E70S1_START/{n=NR} END{print n+0}' "$S/e70s1_chain.log" 2>/dev/null | xargs -I{} tail -n +{} "$S/e70s1_chain.log" | grep -E "^E70S1_(CHAIN_DONE|FAIL)"; }
for i in $(seq 1 8640); do s1u_end >/dev/null 2>&1 && break; sleep 10; done
echo "E70PK_HANDOFF $(s1u_end | tail -n 1)"; date
sleep 30; idle
# 3. 材化 he 4 seed
if [ ! -f "$A1/humaneval_${SFX}_s3/summary.json" ]; then
  serve_vllm "$OUT"
  for sd in 20260806 1 2 3; do runhe humaneval_${SFX}_s$sd $sd; done
  stop_srv
fi
HE=$(python3 -c "
import json;print(sum(json.load(open('$A1/humaneval_${SFX}_s%s/summary.json'%s))['score'] for s in ('20260806','1','2','3'))/4)")
echo "E70PK_HE:$TAG he4=$(score $A1/humaneval_${SFX}_s20260806),$(score $A1/humaneval_${SFX}_s1),$(score $A1/humaneval_${SFX}_s2),$(score $A1/humaneval_${SFX}_s3) mean=$HE"; date
python3 -c "import sys; sys.exit(0 if float('$HE') >= 0.7896-0.027 else 1)" || { echo "E70PK_GATE_FAIL:$TAG he=$HE < 0.7626"; echo "E70PK_CHAIN_DONE"; exit 0; }
# 4. 官方 IF + GS
if [ ! -f "$A1/gsm8k_$SFX/summary.json" ]; then
  idle; serve_vllm "$OUT"
  runif ifeval_$SFX 20260806
  rungs gsm8k_$SFX 20260806
  stop_srv
fi
echo "E70PK_OFFICIAL:$SFX if=$(score $A1/ifeval_$SFX) gs=$(score $A1/gsm8k_$SFX) he=$HE"; date
echo "E70PK_CHAIN_DONE"; date
