#!/bin/bash
# E69 U25 GSM8K 補收:等 S2 全鏈收尾(E69S2_CHAIN_DONE|E69S2_FAIL)→ llama-server U25 → gsm8k → E69S1_POINT:u25gs
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval; LC=$HOME/llama.cpp/build/bin; LOG=$S/e69s1_chain.log
G=data/gguf/a1-4b-u25.gguf; sfx=e67u_u25
fail() { echo "E69S1_FAIL:$1" >> $LOG; date >> $LOG; exit 1; }
score() { python3 -c "import json;d=json.load(open('$1/summary.json'));print(d['strict']['prompt_level'] if 'strict' in d else d['score'])"; }
until grep -qE "E69S2_CHAIN_DONE|E69S2_FAIL" $S/e69s2_chain.log 2>/dev/null; do sleep 60; done
sleep 20
while pgrep -f "vllm serv[e]|llama-serve[r]|e2e_sof[t]|corkscrew.quantiz[e]" >/dev/null; do sleep 30; done
if [ ! -f "$A1/gsm8k_${sfx}/summary.json" ]; then
  $LC/llama-server -m "$G" -a a1 --host 127.0.0.1 --port 8000 -ngl 99 -c 262144 -np 8 -fa on --jinja --reasoning-format none > "$S/llamaserver_u25gs.log" 2>&1 &
  spid=$!
  for i in $(seq 1 600); do
    curl -s http://127.0.0.1:8000/health 2>/dev/null | grep -q '"ok"' && break
    grep -qiE "error|failed" "$S/llamaserver_u25gs.log" && { kill $spid 2>/dev/null; fail serve_u25gs; }
    sleep 1
  done
  curl -s http://127.0.0.1:8000/health 2>/dev/null | grep -q '"ok"' || { kill $spid 2>/dev/null; fail serve_timeout_u25gs; }
  PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject gsm8k --base-url http://127.0.0.1:8000/v1 --model a1 --workers 16 --out "$A1/gsm8k_${sfx}" > "$S/eval_gs_u25.log" 2>&1 || { kill $spid 2>/dev/null; fail eval_gs_u25; }
  kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 5
fi
echo "E69S1_POINT:u25gs he=$(score $A1/humaneval_${sfx}) if=$(score $A1/ifeval_${sfx}) gs=$(score $A1/gsm8k_${sfx})" >> $LOG
date >> $LOG
