#!/bin/bash
# 2c:p3b_w2 GPTQ(marlin4 升位保值)vs bf16 材化 export——vLLM 離線煙測(greedy 一致)+ latency + throughput + 記憶體
# 哨兵:G4_START / G4_SMOKE:<tag> .. / G4_LAT:<tag>:<cfg> .. / G4_TP:<tag> .. / G4_FAIL:<step> / G4_CHAIN_DONE
cd $REPO || exit 1
S=$SCRATCH
O=evidence/p1_grouping/corkscrew/bench_2c
mkdir -p "$O"
fail() { echo "G4_FAIL:$1"; date; exit 1; }
idle() { while pgrep -f "^\S*python -m p2_anchor|^\S*vllm " >/dev/null; do sleep 15; done; }
export PATH=$PWD/.venv-vllm/bin:$PATH
VB=.venv-vllm/bin/vllm
PY=.venv-vllm/bin/python
echo "G4_START"; date
declare -A MODEL=([bf16]=exports/e69_p3b_w2_pack [gptq4]=exports/e69_p3b_w2_gptq4)
for tag in bf16 gptq4; do
  m=${MODEL[$tag]}
  idle
  cmp=""; [ "$tag" = gptq4 ] && cmp="--compare $O/smoke_bf16.json"
  PYTHONPATH=. $PY -m p2_anchor.wringer.gptq_smoke --model "$m" --out "$O/smoke_$tag.json" $cmp > "$S/smoke_$tag.log" 2>&1 || fail "smoke_$tag"
  echo "G4_SMOKE:$tag $(grep SMOKE_DONE $S/smoke_$tag.log | cut -c1-400)"; date
  grep -hE "Model loading took|KV cache memory|GPU KV cache size|Maximum concurrency" "$S/smoke_$tag.log" | sed 's/^/  /' | head -6
  for cfg in "1 512 128" "1 2048 128" "8 512 128" "32 512 128"; do
    set -- $cfg; bs=$1; il=$2; ol=$3; name="bs${bs}_in${il}_out${ol}"
    idle
    $VB bench latency --model "$m" --batch-size $bs --input-len $il --output-len $ol --num-iters 8 --num-iters-warmup 3 \
      --max-model-len 4096 --gpu-memory-utilization 0.6 --output-json "$O/lat_${tag}_$name.json" > "$S/lat_${tag}_$name.log" 2>&1 || fail "lat_${tag}_$name"
    echo "G4_LAT:$tag:$name $($PY -c "import json;d=json.load(open('$O/lat_${tag}_$name.json'));print(round(d['avg_latency'],4),'s')")"; date
  done
  idle
  $VB bench throughput --model "$m" --dataset-name random --input-len 1024 --output-len 256 --num-prompts 256 --max-num-seqs 64 \
    --max-model-len 4096 --gpu-memory-utilization 0.6 --output-json "$O/tp_$tag.json" > "$S/tp_$tag.log" 2>&1 || fail "tp_$tag"
  echo "G4_TP:$tag $(grep -E "Throughput" $S/tp_$tag.log | tail -n 1)"; date
done
echo "G4_CHAIN_DONE"; date
