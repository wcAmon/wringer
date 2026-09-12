#!/bin/bash
# E67-U 對照曲線(prereg_e67.json):UD 式 GGUF 三點(IQ2_S+保護 ≈2.76 / IQ2_M+保護 ≈2.97 / IQ3_XXS+保護 ≈3.40 主幹 bpw)
#   每點:llama-quantize(imatrix)→ gguf_ledger → llama-server(OpenAI 端點,-a a1)→ a1eval he(U27/U30 另跑 IF+GS,用戶已核准)
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval
EVC=evidence/p1_grouping/corkscrew
LC=$HOME/llama.cpp/build/bin
IM=data/gguf/imatrix-a1-4b.gguf
fail() { echo "E67U_FAIL:$1"; date; exit 1; }
[ -f "$IM" ] || fail no_imatrix
PROT=(--tensor-type 'ssm_out=iq3_s' --tensor-type 'attn_v=iq3_s')

serve() {
  $LC/llama-server -m "$1" -a a1 --host 127.0.0.1 --port 8000 -ngl 99 -c 262144 -np 8 -fa on --jinja --reasoning-format none > "$S/llamaserver.log" 2>&1 &
  spid=$!
  for i in $(seq 1 600); do
    curl -s http://127.0.0.1:8000/health 2>/dev/null | grep -q '"ok"' && return 0
    grep -qiE "error|failed" "$S/llamaserver.log" && { kill $spid 2>/dev/null; fail serve; }
    sleep 1
  done; kill $spid 2>/dev/null; fail serve_timeout
}

# point <tag> <preset> <subjects: he | he,if,gs> <extra quantize args...>
point() {
  local TAG=$1 PRESET=$2 SUBJ=$3; shift 3
  local G=data/gguf/a1-4b-$TAG.gguf sfx=e67u_$TAG
  if [ ! -f "$G" ]; then
    $LC/llama-quantize --imatrix $IM --token-embedding-type q4_k --output-tensor-type q6_k "${PROT[@]}" "$@" data/gguf/a1-4b-bf16.gguf "$G" $PRESET 8 > "$S/quant_$TAG.log" 2>&1 || fail "quantize_$TAG"
  fi
  PYTHONPATH=.:$HOME/llama.cpp/gguf-py .venv/bin/python -m p2_anchor.wringer.gguf_ledger "$G" > "$EVC/gguf_ledger_$TAG.txt" 2>&1
  head -6 "$EVC/gguf_ledger_$TAG.txt" | tr -d '\n'; echo
  local need=0
  for sub in ${SUBJ//,/ }; do [ -f "$A1/${sub/if/ifeval}_${sfx}/summary.json" ] || need=1; done
  if [ "$need" = 1 ]; then
    pgrep -f "vllm serv[e]|llama-serve[r]" >/dev/null && fail gpu_busy
    serve "$G"
    for sub in ${SUBJ//,/ }; do
      case $sub in
        he) [ -f "$A1/humaneval_${sfx}/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --workers 16 --out "$A1/humaneval_${sfx}" > "$S/eval_he_$TAG.log" 2>&1 || fail "eval_he_$TAG";;
        gs) [ -f "$A1/gsm8k_${sfx}/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject gsm8k --base-url http://127.0.0.1:8000/v1 --model a1 --workers 16 --out "$A1/gsm8k_${sfx}" > "$S/eval_gs_$TAG.log" 2>&1 || fail "eval_gs_$TAG";;
        if) [ -f "$A1/ifeval_${sfx}/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.ifeval_runner --base-url http://127.0.0.1:8000/v1 --model a1 --workers 16 --out "$A1/ifeval_${sfx}" > "$S/eval_if_$TAG.log" 2>&1 || fail "eval_if_$TAG";;
      esac
    done
    kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 5
  fi
  for sub in ${SUBJ//,/ }; do
    d=$A1/${sub/if/ifeval}_${sfx}; [ $sub = he ] && d=$A1/humaneval_${sfx}; [ $sub = gs ] && d=$A1/gsm8k_${sfx}
    echo "E67U_SCORE:$TAG $sub $(python3 -c "import json;d=json.load(open('$d/summary.json'));print(d['strict']['prompt_level'] if 'strict' in d else d['score'])")"
  done
  echo "E67U_POINT_DONE:$TAG"; date
}

echo "E67U_START"; date
point u27 IQ2_S he,if,gs
point u30 IQ2_M he,if,gs --tensor-type 'blk\.[01]\.ffn_down=q4_k'
point u34 IQ3_XXS he --tensor-type 'blk\.[01]\.ffn_down=q4_k'
echo "E67U_CHAIN_DONE"; date
