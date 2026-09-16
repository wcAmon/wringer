#!/bin/bash
# 2c 第二段:真 2/3-bit GPTQ 版面(native 升位保值)——套 vLLM 補丁 → fp16 煙測(gptq4 fp16 對照 + gptq23)→ latency/throughput
# 哨兵:G23_START / G23_PATCH .. / G23_SMOKE:<tag> .. / G23_LAT:gptq23:<cfg> .. / G23_TP:gptq23 .. / G23_FAIL:<step> / G23_CHAIN_DONE
cd $REPO || exit 1
S=$SCRATCH
O=evidence/p1_grouping/corkscrew/bench_2c
mkdir -p "$O"
fail() { echo "G23_FAIL:$1"; date; exit 1; }
idle() { while pgrep -f "^\S*python -m p2_anchor|^\S*vllm " >/dev/null; do sleep 15; done; }
export PATH=$PWD/.venv-vllm/bin:$PATH
VB=.venv-vllm/bin/vllm
PY=.venv-vllm/bin/python
echo "G23_START"; date
idle
$PY -m p2_anchor.wringer.vllm_patch_23bit apply > "$S/vllm_patch.log" 2>&1 || fail patch
echo "G23_PATCH $(grep VLLM_PATCH $S/vllm_patch.log)"; date
# fp16 對照:gptq4 走 fp16 激活(Marlin),量 fp16 本身對輸出的影響
[ -f "$O/smoke_gptq4_fp16.json" ] || PYTHONPATH=. $PY -m p2_anchor.wringer.gptq_smoke --model exports/e69_p3b_w2_gptq4 --dtype float16 --out "$O/smoke_gptq4_fp16.json" --compare "$O/smoke_bf16.json" > "$S/smoke_gptq4_fp16.log" 2>&1 || fail smoke_gptq4_fp16
echo "G23_SMOKE:gptq4_fp16 $(grep SMOKE_DONE $S/smoke_gptq4_fp16.log | cut -c1-400)"; date
idle
[ -f "$O/smoke_gptq23.json" ] || PYTHONPATH=. $PY -m p2_anchor.wringer.gptq_smoke --model exports/e69_p3b_w2_gptq23 --dtype float16 --out "$O/smoke_gptq23.json" --compare "$O/smoke_bf16.json" > "$S/smoke_gptq23.log" 2>&1 || fail smoke_gptq23
echo "G23_SMOKE:gptq23 $(grep SMOKE_DONE $S/smoke_gptq23.log | cut -c1-400)"; date
grep -hE "Model loading took|KV cache memory|Using .* for AutoGPTQLinearMethod" "$S/smoke_gptq23.log" | sed 's/^/  /' | sort -u | head -8
m=exports/e69_p3b_w2_gptq23; tag=gptq23
for cfg in "1 512 128" "1 2048 128" "8 512 128" "32 512 128"; do
  set -- $cfg; bs=$1; il=$2; ol=$3; name="bs${bs}_in${il}_out${ol}"
  idle
  $VB bench latency --model "$m" --dtype float16 --batch-size $bs --input-len $il --output-len $ol --num-iters 8 --num-iters-warmup 3 \
    --max-model-len 4096 --gpu-memory-utilization 0.6 --output-json "$O/lat_${tag}_$name.json" > "$S/lat_${tag}_$name.log" 2>&1 || fail "lat_${tag}_$name"
  echo "G23_LAT:$tag:$name $($PY -c "import json;d=json.load(open('$O/lat_${tag}_$name.json'));print(round(d['avg_latency'],4),'s')")"; date
done
idle
$VB bench throughput --model "$m" --dtype float16 --dataset-name random --input-len 1024 --output-len 256 --num-prompts 256 --max-num-seqs 64 \
  --max-model-len 4096 --gpu-memory-utilization 0.6 --output-json "$O/tp_$tag.json" > "$S/tp_$tag.log" 2>&1 || fail "tp_$tag"
echo "G23_TP:$tag $(grep -E "Throughput" $S/tp_$tag.log | tail -n 1)"; date
echo "G23_CHAIN_DONE"; date
