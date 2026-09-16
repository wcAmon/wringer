#!/bin/bash
# E70 第一段(prereg_e70 stage1 + gguf_U_curve;用戶 09-12 核准):等 E70S0_CHAIN_DONE →
#   七探針:P1 配方各單獨施加一個帳面動作 → 閉式 → export → vLLM(KV fp8)→ he 2 seed → 刪 export
#   GGUF U 三點:convert bf16 → imatrix(calib_e70 文字)→ dry-run 選最近 2.5/2.7/3.1 主幹 bpw 的三個預設 → quantize(UD 式保護)→ llama-server → he 2 seed + IF + GS
# 哨兵:E70S1_START / E70S1_LEDGER:<tag> / E70S1_POINT:<tag> he=<s0>,<s1> / E70S1_U_TABLE .. / E70S1_U_LEDGER:<tag> / E70S1_U_SCORE:<tag> <sub> <score> / E70S1_FAIL / E70S1_CHAIN_DONE
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval; EV=evidence/p1_grouping; EVC=evidence/p1_grouping/corkscrew
LC=$HOME/llama.cpp/build/bin
Q3=~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
export WRINGER_MODEL=Qwen/Qwen3-4B WRINGER_REV=1cfa9a7208912126459214e8b04321603b3df60c WRINGER_CALIB_VAL=$EV/calib_e70_val.pt
export PATH=$PWD/.venv-vllm/bin:$PATH
CAL="--calib $EV/calib_e70_traj.pt --n-calib 128"
BASEKV="self_attn.k_proj:256:row,self_attn.v_proj:256:row"
BASE="$BASEKV,self_attn.o_proj:16:128"
NP=${E70_LLAMA_NP:-16}          # llama-server 並行槽(評測加速鏈 C 組裁定後可改 32)
fail() { echo "E70S1_FAIL:$1"; date; exit 1; }
idle() { while pgrep -f "^\S*python -m p2_anchor|^\S*vllm serve|^\S*/llama-server" >/dev/null; do sleep 15; done; }
serve_vllm() {
  .venv-vllm/bin/vllm serve "$1" --served-model-name a1 --reasoning-parser qwen3 --max-model-len 32768 --gpu-memory-utilization 0.85 --kv-cache-dtype fp8 --port 8000 > "$S/vllm_e70s1.log" 2>&1 &
  spid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_e70s1.log" && return 0
    grep -qE "initialization failed|Traceback" "$S/vllm_e70s1.log" && fail serve
    sleep 1
  done; fail serve_timeout
}
serve_llama() {
  $LC/llama-server -m "$1" -a a1 --host 127.0.0.1 --port 8000 -ngl 99 -c $((NP * 16384)) -np $NP -fa on --jinja --reasoning-format none > "$S/llama_e70s1.log" 2>&1 &
  spid=$!
  for i in $(seq 1 600); do
    curl -s http://127.0.0.1:8000/health 2>/dev/null | grep -q '"ok"' && return 0
    grep -qiE "error|failed" "$S/llama_e70s1.log" && { kill $spid 2>/dev/null; fail serve_llama; }
    sleep 1
  done; kill $spid 2>/dev/null; fail serve_llama_timeout
}
stop_srv() { kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 8; }
score() { python3 -c "import json;d=json.load(open('$1/summary.json'));print(d['strict']['prompt_level'] if 'strict' in d else d['score'])"; }
runhe() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --workers $3 --seed $2 --out "$A1/$1" > "$S/e70_$1.log" 2>&1 || fail "$1"; }
runif() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.ifeval_runner --base-url http://127.0.0.1:8000/v1 --model a1 --workers $3 --seed $2 --out "$A1/$1" > "$S/e70_$1.log" 2>&1 || fail "$1"; }
rungs() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject gsm8k --base-url http://127.0.0.1:8000/v1 --model a1 --workers $3 --seed $2 --out "$A1/$1" > "$S/e70_$1.log" 2>&1 || fail "$1"; }
# probe <tag> <extra quantize args...>
probe() {
  local TAG=$1; shift
  local OUT=exports/e70_$TAG sfx=e70_$TAG
  if [ ! -f "data/qs_$TAG/layer35.pt" ]; then
    idle
    PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.quantize --tag $TAG $CAL --grid 8 --block 128 --solver gptq --prior-lam 0.01 "$@" > "$S/quant_$TAG.log" 2>&1 || fail "quant_$TAG"
  fi
  grep "^QUANT_LEDGER" "$S/quant_$TAG.log" | tail -n 1 | cut -c1-300 | sed "s/^/E70S1_LEDGER:$TAG /"
  if [ ! -f "$A1/humaneval_${sfx}_s1/summary.json" ]; then
    [ -f "$OUT/config.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export --state-tag $TAG --out $OUT > "$S/export_$TAG.log" 2>&1 || fail "export_$TAG"
    idle; serve_vllm "$OUT"
    runhe humaneval_${sfx}_s20260806 20260806 128
    runhe humaneval_${sfx}_s1 1 128
    stop_srv
    rm -rf "$OUT"
  fi
  echo "E70S1_POINT:$TAG he=$(score $A1/humaneval_${sfx}_s20260806),$(score $A1/humaneval_${sfx}_s1)"; date
}
echo "E70S1_START"; date
for i in $(seq 1 4320); do grep -q "E70S0_CHAIN_DONE\|E70S0_FAIL" "$S/e70s0_chain.log" 2>/dev/null && break; sleep 10; done
grep -q "E70S0_CHAIN_DONE" "$S/e70s0_chain.log" || fail s0_not_done
sleep 20
probe s1_a8    --alpha-bits 8 --module-map "$BASE"
probe s1_down4 --module-map "$BASE,mlp.down_proj:4:128"
probe s1_up4   --module-map "$BASE,mlp.up_proj:4:128"
probe s1_gate4 --module-map "$BASE,mlp.gate_proj:4:128"
probe s1_q4    --module-map "$BASE,self_attn.q_proj:4:128"
probe s1_o8    --module-map "$BASEKV,self_attn.o_proj:8:128"
probe s1_b256  --module-map "$BASE,mlp.gate_proj:8:256,mlp.up_proj:8:256,mlp.down_proj:8:256"
# ---- GGUF U 曲線三點 ----
BF=data/gguf/qwen3-4b-bf16.gguf; IM=data/gguf/imatrix-qwen3-4b.gguf
[ -f "$BF" ] || .venv/bin/python $HOME/llama.cpp/convert_hf_to_gguf.py "$Q3" --outtype bf16 --outfile "$BF" > "$S/e70_convert.log" 2>&1 || fail convert
[ -f "$S/imatrix_calib_e70.txt" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.calib_to_text --calib $EV/calib_e70_traj.pt --len $EV/calib_e70_len.pt --out "$S/imatrix_calib_e70.txt" > "$S/e70_calibtext.log" 2>&1 || fail calibtext
idle
[ -f "$IM" ] || $LC/llama-imatrix -m "$BF" -f "$S/imatrix_calib_e70.txt" -o "$IM" -ngl 99 -c 2048 > "$S/e70_imatrix.log" 2>&1 || fail imatrix
PROT=(--tensor-type 'attn_v=iq3_s' --tensor-type 'blk\.[01]\.ffn_down=q4_k')
: > "$S/e70_u_table.txt"
for P in IQ2_XXS IQ2_XS IQ2_S IQ2_M IQ3_XXS IQ3_XS IQ3_S; do
  GGUF_BF16=$BF .venv/bin/python p2_anchor/wringer/gguf_dryrun_bpw.py $P "${PROT[@]}" 2>/dev/null | tr '\n' ' ' >> "$S/e70_u_table.txt"; echo >> "$S/e70_u_table.txt"
done
echo "E70S1_U_TABLE $(cut -c1-120 $S/e70_u_table.txt | tr '\n' '|')"
PICK=$(python3 - "$S/e70_u_table.txt" <<'PY'
import re,sys
rows=[]
for l in open(sys.argv[1]):
    m=re.search(r"^(\S+).*bpw_body=([\d.]+)",l)
    if m: rows.append((m.group(1),float(m.group(2))))
out=[]
for tgt in (2.5,2.7,3.1):
    best=min(rows,key=lambda r:abs(r[1]-tgt))
    if best[0] not in out: out.append(best[0])
print(" ".join(out))
PY
)
echo "E70S1_U_PICK $PICK"
for P in $PICK; do
  TAG=u_$(echo $P | tr 'A-Z' 'a-z'); G=data/gguf/qwen3-4b-$TAG.gguf; sfx=e70_$TAG
  [ -f "$G" ] || $LC/llama-quantize --imatrix $IM --token-embedding-type q4_k --output-tensor-type q6_k "${PROT[@]}" "$BF" "$G" $P 8 > "$S/quant_$TAG.log" 2>&1 || fail "quantize_$TAG"
  PYTHONPATH=.:$HOME/llama.cpp/gguf-py .venv/bin/python -m p2_anchor.wringer.gguf_ledger "$G" > "$EVC/gguf_ledger_e70_$TAG.txt" 2>&1
  echo "E70S1_U_LEDGER:$TAG $(head -6 $EVC/gguf_ledger_e70_$TAG.txt | tr -d '\n' | cut -c1-300)"
  if [ ! -f "$A1/gsm8k_${sfx}/summary.json" ]; then
    idle; serve_llama "$G"
    runhe humaneval_${sfx}_s20260806 20260806 $NP
    runhe humaneval_${sfx}_s1 1 $NP
    runif ifeval_${sfx} 20260806 $NP
    rungs gsm8k_${sfx} 20260806 $NP
    stop_srv
  fi
  echo "E70S1_U_SCORE:$TAG he $(score $A1/humaneval_${sfx}_s20260806),$(score $A1/humaneval_${sfx}_s1) if $(score $A1/ifeval_${sfx}) gs $(score $A1/gsm8k_${sfx})"; date
done
echo "E70S1_CHAIN_DONE"; date
