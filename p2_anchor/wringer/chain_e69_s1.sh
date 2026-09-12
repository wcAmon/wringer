#!/bin/bash
# E69 第一段:敏感度探針(prereg_e69.json stage1;點火需用戶核)。等 chain_e68c E68C_CHAIN_DONE →
#   8 個探針:P1 配方各單獨施加一個帳面動作 → 閉式排水 → 熵帳 → export → vLLM → he
#   最後 U26:GGUF IQ2_XS+保護 → he,if,gs(U 曲線低端錨)
# 哨兵:E69S1_START / E69S1_LEDGER:<tag> / E69S1_ENTROPY:<tag> / E69S1_POINT:<tag> he=.. / E69S1_U26 .. / E69S1_FAIL / E69S1_CHAIN_DONE
cd $REPO || exit 1
S=$SCRATCH
EVC=evidence/p1_grouping/corkscrew; A1=evidence/p1_grouping/a1eval
LC=$HOME/llama.cpp/build/bin; IM=data/gguf/imatrix-a1-4b.gguf
CAL="--calib evidence/p1_grouping/calib_e58r1b_traj.pt --n-calib 128"
BASE="self_attn.k_proj:256:row,self_attn.v_proj:256:row,self_attn.o_proj:16:128"
fail() { echo "E69S1_FAIL:$1"; date; exit 1; }
idle() { while pgrep -f "vllm serv[e]|llama-serve[r]|e2e_sof[t]|gsq_bloc[k]|corkscrew.quantiz[e]" >/dev/null; do sleep 30; done; }
serve_vllm() {
  PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/vllm serve "$1" --served-model-name a1 --reasoning-parser qwen3 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --port 8000 > "$S/vllm_e69s1.log" 2>&1 &
  spid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_e69s1.log" && return 0
    grep -qE "initialization failed" "$S/vllm_e69s1.log" && fail serve
    sleep 1
  done; fail serve_timeout
}
serve_llama() {
  $LC/llama-server -m "$1" -a a1 --host 127.0.0.1 --port 8000 -ngl 99 -c 262144 -np 8 -fa on --jinja --reasoning-format none > "$S/llamaserver_e69s1.log" 2>&1 &
  spid=$!
  for i in $(seq 1 600); do
    curl -s http://127.0.0.1:8000/health 2>/dev/null | grep -q '"ok"' && return 0
    grep -qiE "error|failed" "$S/llamaserver_e69s1.log" && { kill $spid 2>/dev/null; fail serve; }
    sleep 1
  done; kill $spid 2>/dev/null; fail serve_timeout
}
score() { python3 -c "import json;d=json.load(open('$1/summary.json'));print(d['strict']['prompt_level'] if 'strict' in d else d['score'])"; }
# probe <tag> <extra quantize args...>
probe() {
  local TAG=$1; shift
  local OUT=exports/e69_$TAG sfx=e69_$TAG
  if [ ! -f "data/qs_$TAG/layer31.pt" ]; then
    idle
    PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.quantize --tag $TAG $CAL --grid 8 --block 128 --solver gptq --prior-lam 0.01 "$@" > "$S/quant_$TAG.log" 2>&1 || fail "quant_$TAG"
  fi
  grep "^QUANT_LEDGER" "$S/quant_$TAG.log" | sed "s/^/E69S1_LEDGER:$TAG /"
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.state_entropy --state-tag $TAG --out $EVC/entropy_$TAG.json 2>/dev/null | grep STATE_ENTROPY | cut -c1-200 | sed "s/^/E69S1_ENTROPY:$TAG /"
  if [ ! -f "$A1/humaneval_${sfx}/summary.json" ]; then
    [ -f "$OUT/config.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export --state-tag $TAG --out $OUT > "$S/export_$TAG.log" 2>&1 || fail "export_$TAG"
    pgrep -f "vllm serv[e]|llama-serve[r]" >/dev/null && fail gpu_busy
    serve_vllm "$OUT"
    PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --out "$A1/humaneval_${sfx}" > "$S/eval_he_$TAG.log" 2>&1 || fail "eval_he_$TAG"
    kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 8
    rm -rf "$OUT"
  fi
  echo "E69S1_POINT:$TAG he=$(score $A1/humaneval_${sfx})"; date
}

until grep -qE "E68C_CHAIN_DONE|E68C_FAIL" $S/e68c_chain.log 2>/dev/null; do sleep 60; done
sleep 20; idle
echo "E69S1_START"; date

probe s1_a8    --alpha-bits 8 --module-map "$BASE"
probe s1_down4 --module-map "$BASE,mlp.down_proj:4:128"
probe s1_up4   --module-map "$BASE,mlp.up_proj:4:128"
probe s1_gate4 --module-map "$BASE,mlp.gate_proj:4:128"
probe s1_qkv4  --module-map "$BASE,linear_attn.in_proj_qkv:4:128"
probe s1_z4    --module-map "$BASE,linear_attn.in_proj_z:4:128"
probe s1_b256  --module-map "$BASE,mlp.gate_proj:8:256,mlp.up_proj:8:256,mlp.down_proj:8:256"
probe s1_out4  --module-map "$BASE,linear_attn.out_proj:4:128"

# ---- U26:GGUF IQ2_XS + 保護 → he,if,gs ----
G=data/gguf/a1-4b-u26.gguf; sfx=e67u_u26
if [ ! -f "$G" ]; then
  $LC/llama-quantize --imatrix $IM --token-embedding-type q4_k --output-tensor-type q6_k --tensor-type 'ssm_out=iq3_s' --tensor-type 'attn_v=iq3_s' \
    --tensor-type 'blk\.[01]\.ffn_down=q4_k' data/gguf/a1-4b-bf16.gguf "$G" IQ2_XS 8 > "$S/quant_u26.log" 2>&1 || fail quantize_u26
fi
PYTHONPATH=.:$HOME/llama.cpp/gguf-py .venv/bin/python -m p2_anchor.wringer.gguf_ledger "$G" > "$EVC/gguf_ledger_u26.txt" 2>&1
echo "E69S1_U26_LEDGER $(head -6 $EVC/gguf_ledger_u26.txt | tr -d '\n')"
if [ ! -f "$A1/gsm8k_${sfx}/summary.json" ]; then
  idle; pgrep -f "vllm serv[e]|llama-serve[r]" >/dev/null && fail gpu_busy
  serve_llama "$G"
  [ -f "$A1/humaneval_${sfx}/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --workers 16 --out "$A1/humaneval_${sfx}" > "$S/eval_he_u26.log" 2>&1 || fail eval_he_u26
  [ -f "$A1/ifeval_${sfx}/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.ifeval_runner --base-url http://127.0.0.1:8000/v1 --model a1 --workers 16 --out "$A1/ifeval_${sfx}" > "$S/eval_if_u26.log" 2>&1 || fail eval_if_u26
  PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject gsm8k --base-url http://127.0.0.1:8000/v1 --model a1 --workers 16 --out "$A1/gsm8k_${sfx}" > "$S/eval_gs_u26.log" 2>&1 || fail eval_gs_u26
  kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 5
fi
echo "E69S1_U26 he=$(score $A1/humaneval_${sfx}) if=$(score $A1/ifeval_${sfx}) gs=$(score $A1/gsm8k_${sfx})"
echo "E69S1_CHAIN_DONE"; date
