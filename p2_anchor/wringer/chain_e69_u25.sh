#!/bin/bash
# E69 S1 尾段替換:U26(IQ2_XS+保護)實測主幹 2.744 ≈ U27,不符「2.5-2.6 錨」目的 → 改 U25(IQ2_XXS+保護,主幹 2.558)三科。
# 追加寫入 $S/e69s1_chain.log(沿用 S1 哨兵;結尾補 E69S1_CHAIN_DONE 供 chain_e69_s2 接棒)
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval; LC=$HOME/llama.cpp/build/bin
LOG=$S/e69s1_chain.log
fail() { echo "E69S1_FAIL:$1" >> $LOG; date >> $LOG; exit 1; }
G=data/gguf/a1-4b-u25.gguf; sfx=e67u_u25
[ -f "$G" ] || fail no_u25_gguf
score() { python3 -c "import json;d=json.load(open('$1/summary.json'));print(d['strict']['prompt_level'] if 'strict' in d else d['score'])"; }
while pgrep -f "vllm serv[e]|llama-serve[r]|corkscrew.quantiz[e]" >/dev/null; do sleep 30; done
$LC/llama-server -m "$G" -a a1 --host 127.0.0.1 --port 8000 -ngl 99 -c 262144 -np 8 -fa on --jinja --reasoning-format none > "$S/llamaserver_u25.log" 2>&1 &
spid=$!
for i in $(seq 1 600); do
  curl -s http://127.0.0.1:8000/health 2>/dev/null | grep -q '"ok"' && break
  grep -qiE "error|failed" "$S/llamaserver_u25.log" && { kill $spid 2>/dev/null; fail serve; }
  sleep 1
done
curl -s http://127.0.0.1:8000/health 2>/dev/null | grep -q '"ok"' || { kill $spid 2>/dev/null; fail serve_timeout; }
[ -f "$A1/humaneval_${sfx}/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --workers 16 --out "$A1/humaneval_${sfx}" > "$S/eval_he_u25.log" 2>&1 || fail eval_he_u25
[ -f "$A1/ifeval_${sfx}/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.ifeval_runner --base-url http://127.0.0.1:8000/v1 --model a1 --workers 16 --out "$A1/ifeval_${sfx}" > "$S/eval_if_u25.log" 2>&1 || fail eval_if_u25
[ -f "$A1/gsm8k_${sfx}/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject gsm8k --base-url http://127.0.0.1:8000/v1 --model a1 --workers 16 --out "$A1/gsm8k_${sfx}" > "$S/eval_gs_u25.log" 2>&1 || fail eval_gs_u25
kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 5
echo "E69S1_POINT:u25 he=$(score $A1/humaneval_${sfx}) if=$(score $A1/ifeval_${sfx}) gs=$(score $A1/gsm8k_${sfx})" >> $LOG
echo "E69S1_CHAIN_DONE" >> $LOG; date >> $LOG
