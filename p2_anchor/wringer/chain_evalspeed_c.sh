#!/bin/bash
# 評測加速鏈 C 組重跑(原 chain_evalspeed.sh 哨兵字串不符出局):llama.cpp u27 GGUF GS-200 於 -np 8 vs -np 32
#   先 SIGSTOP S0 鏈(避免其 idle() 在兩次伺服器切換空檔搶 GPU),收完 SIGCONT
#   沿用原鏈遺留的 np8 伺服器(若 /health ok)
# 哨兵:ESC_START / ES_POINT:<tag> .. / ES_FAIL:<step> / ES_CHAIN_DONE;log $S/evalspeed_c_chain.log
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval
LC=$HOME/llama.cpp/build/bin
S0PID=$(pgrep -f "^bash p2_anchor/wringer/chain_e70_s0" | head -n 1)
resume() { [ -n "$S0PID" ] && kill -CONT $S0PID 2>/dev/null; }
fail() { echo "ES_FAIL:$1"; date; resume; exit 1; }
trap resume EXIT
[ -n "$S0PID" ] && kill -STOP $S0PID && echo "ESC_S0_PAUSED $S0PID"
serve_llama() {  # $1 gguf, $2 np, $3 ctx
  $LC/llama-server -m "$1" -a a1 --host 127.0.0.1 --port 8000 -ngl 99 -c $3 -np $2 -fa on --jinja --reasoning-format none > "$S/llama_esc_np$2.log" 2>&1 &
  spid=$!
  for i in $(seq 1 600); do
    curl -s http://127.0.0.1:8000/health 2>/dev/null | grep -q '"ok"' && return 0
    grep -qiE "error|failed" "$S/llama_esc_np$2.log" && { kill $spid 2>/dev/null; fail serve_llama; }
    sleep 1
  done; kill $spid 2>/dev/null; fail serve_llama_timeout
}
stop_srv() { kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 8; }
score() { python3 -c "import json;d=json.load(open('$1/summary.json'));print(d['strict']['prompt_level'] if 'strict' in d else d['score'], d.get('generation_s'), json.dumps(d.get('usage') or {}))"; }
point() { echo "ES_POINT:$1 $(score $A1/speed_$1)"; date; }
rungs() { [ -f "$A1/speed_$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject gsm8k --base-url http://127.0.0.1:8000/v1 --model a1 --workers $2 ${3:+--limit $3} --out "$A1/speed_$1" > "$S/es_$1.log" 2>&1 || fail "$1"; point $1; }
echo "ESC_START"; date
# np8:沿用遺留伺服器
spid=$(pgrep -f "^\S*/llama-server .* -np 8 " | head -n 1)
if [ -n "$spid" ] && curl -s http://127.0.0.1:8000/health 2>/dev/null | grep -q '"ok"'; then
  echo "ESC_REUSE_NP8 $spid"
else
  pkill -f "^\S*/llama-server " 2>/dev/null; sleep 8
  serve_llama data/gguf/a1-4b-u27.gguf 8 262144
fi
rungs u27_gs200_np8 16 200
stop_srv
serve_llama data/gguf/a1-4b-u27.gguf 32 524288
rungs u27_gs200_np32 32 200
stop_srv
echo "ES_CHAIN_DONE"; date
