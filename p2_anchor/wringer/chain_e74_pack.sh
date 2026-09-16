#!/bin/bash
# E74 發布打包(用戶 09-15 裁「Qwen 排水預設 rtnj 及 HF 模型更新」):Qwen3-4B e73_q_rtnj(含水權重 + 最近點碼 + 閉式尺度,研究態 comp 0.9223)
#   0. 由 qs 狀態重 export(E73 鏈評測後已刪 export)→
#   1. 打包驗證(CPU)→ 2. 容器材化 export(CPU)→ [等 E70S1_CHAIN_DONE 讓出 GPU] → 3. 材化 he 4 seed(E70 協定 KV fp8 + w128)→ 4. 官方 IF + GS
#   閘:材化 he 4 seed 均值 ≥ 研究態 83.23 − 2.7 = 80.53;不過則停並報(不上傳)。上傳另行手動(模型卡 v0.3)。
# 哨兵:E74PK_START / E74PK_PACK / E74PK_MAT / E74PK_HANDOFF / E74PK_HE / E74PK_GATE_FAIL / E74PK_OFFICIAL:.. / E74PK_FAIL:.. / E74PK_CHAIN_DONE
cd $REPO || exit 1
S=$SCRATCH
EVC=evidence/p1_grouping/corkscrew; A1=evidence/p1_grouping/a1eval
export WRINGER_MODEL=Qwen/Qwen3-4B WRINGER_REV=1cfa9a7208912126459214e8b04321603b3df60c
TAG=e73_q_rtnj; SRC=exports/$TAG; OUT=exports/${TAG}_pack; SFX=${TAG}_pack
CONT=data/wringer_$TAG.safetensors
fail() { echo "E74PK_FAIL:$1"; date; kill $spid 2>/dev/null; exit 1; }
idle() { while pgrep -f "python -m p2_anchor|bin/vllm serve|/llama-server|/llama-quantize|/llama-imatrix" >/dev/null; do sleep 30; done; }
serve_vllm() {
  PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/vllm serve "$1" --served-model-name a1 --reasoning-parser qwen3 --max-model-len 32768 --gpu-memory-utilization 0.85 --kv-cache-dtype fp8 --port 8000 > "$S/vllm_e74pk.log" 2>&1 &
  spid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_e74pk.log" && return 0
    grep -qE "initialization failed|Traceback" "$S/vllm_e74pk.log" && fail serve
    sleep 1
  done; fail serve_timeout
}
stop_srv() { kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 8; }
score() { python3 -c "import json;d=json.load(open('$1/summary.json'));print(d['strict']['prompt_level'] if 'strict' in d else d['score'])"; }
runhe() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --workers 128 --seed $2 --out "$A1/$1" > "$S/e74pk_$1.log" 2>&1 || fail "$1"; }
runif() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.ifeval_runner --base-url http://127.0.0.1:8000/v1 --model a1 --workers 128 --seed $2 --out "$A1/$1" > "$S/e74pk_$1.log" 2>&1 || fail "$1"; }
rungs() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject gsm8k --base-url http://127.0.0.1:8000/v1 --model a1 --workers 128 --seed $2 --out "$A1/$1" > "$S/e74pk_$1.log" 2>&1 || fail "$1"; }
echo "E74PK_START:$TAG"; date
e74_end() { awk '/^E74_START/{n=NR} END{print n+0}' "$S/e74_chain.log" 2>/dev/null | xargs -I{} tail -n +{} "$S/e74_chain.log" | grep -E "^E74_(CHAIN_DONE|FAIL)"; }
for i in $(seq 1 8640); do [ -n "$(e74_end)" ] && break; sleep 10; done
echo "E74PK_HANDOFF $(e74_end | tail -n 1)"; date
sleep 30; idle
# 0. 由 qs 狀態重 export
export WRINGER_CALIB_VAL=evidence/p1_grouping/calib_e70_val.pt
[ -f "$SRC/config.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export --state-tag $TAG --out $SRC > "$S/export_${TAG}_pk.log" 2>&1 || fail export
echo "E74PK_EXPORT:$TAG"; date
# 1. 打包驗證(CPU)
if [ ! -f "$EVC/pack_$TAG.json" ]; then
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.pack_verify --state-tag $TAG --export $SRC --out $CONT > "$S/pack_$TAG.log" 2>&1 || fail pack
fi
echo "E74PK_PACK:$TAG $(grep PACK_DONE $S/pack_$TAG.log | cut -c1-200)"; date
# 2. 容器材化 export(CPU)
if [ ! -f "$OUT/model.safetensors" ]; then
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.pack_materialize --container $CONT --src-export $SRC --out $OUT > "$S/mat_$TAG.log" 2>&1 || fail materialize
fi
echo "E74PK_MAT:$TAG $(grep MATERIALIZE_DONE $S/mat_$TAG.log | cut -c1-200)"; date
# 3. 材化 he 4 seed
idle
if [ ! -f "$A1/humaneval_${SFX}_s3/summary.json" ]; then
  serve_vllm "$OUT"
  for sd in 20260806 1 2 3; do runhe humaneval_${SFX}_s$sd $sd; done
  stop_srv
fi
HE=$(python3 -c "
import json;print(sum(json.load(open('$A1/humaneval_${SFX}_s%s/summary.json'%s))['score'] for s in ('20260806','1','2','3'))/4)")
echo "E74PK_HE:$TAG he4=$(score $A1/humaneval_${SFX}_s20260806),$(score $A1/humaneval_${SFX}_s1),$(score $A1/humaneval_${SFX}_s2),$(score $A1/humaneval_${SFX}_s3) mean=$HE"; date
python3 -c "import sys; sys.exit(0 if float('$HE') >= 0.8323-0.027 else 1)" || { echo "E74PK_GATE_FAIL:$TAG he=$HE < 0.8053"; echo "E74PK_CHAIN_DONE"; exit 0; }
# 4. 官方 IF + GS
if [ ! -f "$A1/gsm8k_$SFX/summary.json" ]; then
  idle; serve_vllm "$OUT"
  runif ifeval_$SFX 20260806
  rungs gsm8k_$SFX 20260806
  stop_srv
fi
echo "E74PK_OFFICIAL:$SFX if=$(score $A1/ifeval_$SFX) gs=$(score $A1/gsm8k_$SFX) he=$HE"; date
echo "E74PK_CHAIN_DONE"; date
