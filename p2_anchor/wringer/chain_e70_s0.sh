#!/bin/bash
# E70 第 0 段(用戶 09-12 核准「可行的期間就發射」):等評測加速鏈 ES_CHAIN_DONE →
#   錨補跑(新協定:vLLM KV fp8 + IF/GS 128 併發;he 4 seed、IF 3 seed、GS 2 seed)→ 語料自產(gen_e58 協定,Qwen3-4B 自身)
#   → 分層抽 128 窗 + val → P1 八元 g128 閉式(k/v int8 逐列、o_proj 16 級)→ export → he 2 seed(閘:≥ bf16 − 5 pp)
# 哨兵:E70S0_START / E70S0_ANCHOR:<tag> <score> / E70S0_CORPUS .. / E70S0_CALIB .. / E70S0_P1_LEDGER .. / E70S0_P1:he_s<seed> .. / E70S0_FAIL:<step> / E70S0_CHAIN_DONE
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval; EV=evidence/p1_grouping; EVC=evidence/p1_grouping/corkscrew
Q3=~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
export WRINGER_MODEL=Qwen/Qwen3-4B WRINGER_REV=1cfa9a7208912126459214e8b04321603b3df60c WRINGER_CALIB_VAL=$EV/calib_e70_val.pt
export PATH=$PWD/.venv-vllm/bin:$PATH
fail() { echo "E70S0_FAIL:$1"; date; exit 1; }
idle() { while pgrep -f "^\S*python -m p2_anchor|^\S*vllm serve|^\S*/llama-server" >/dev/null; do sleep 15; done; }
serve_vllm() {
  .venv-vllm/bin/vllm serve "$1" --served-model-name a1 --reasoning-parser qwen3 --max-model-len 32768 --gpu-memory-utilization 0.85 --kv-cache-dtype fp8 --port 8000 > "$S/vllm_e70s0.log" 2>&1 &
  spid=$!
  for i in $(seq 1 900); do
    grep -qE "Application startup complete|Uvicorn running" "$S/vllm_e70s0.log" && return 0
    grep -qE "initialization failed|Traceback" "$S/vllm_e70s0.log" && fail serve
    sleep 1
  done; fail serve_timeout
}
stop_srv() { kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 8; }
score() { python3 -c "import json;d=json.load(open('$1/summary.json'));print(d['strict']['prompt_level'] if 'strict' in d else d['score'])"; }
runhe() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject humaneval --base-url http://127.0.0.1:8000/v1 --model a1 --workers 128 --seed $2 --out "$A1/$1" > "$S/e70_$1.log" 2>&1 || fail "$1"; }
runif() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.ifeval_runner --base-url http://127.0.0.1:8000/v1 --model a1 --workers 128 --seed $2 --out "$A1/$1" > "$S/e70_$1.log" 2>&1 || fail "$1"; }
rungs() { [ -f "$A1/$1/summary.json" ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.a1eval.subject_runner --subject gsm8k --base-url http://127.0.0.1:8000/v1 --model a1 --workers 128 --seed $2 --out "$A1/$1" > "$S/e70_$1.log" 2>&1 || fail "$1"; }
echo "E70S0_START"; date
for i in $(seq 1 1440); do grep -q "ES_CHAIN_DONE\|ES_FAIL" "$S/evalspeed_chain.log" 2>/dev/null && break; sleep 10; done
echo "E70S0_HANDOFF $(grep -E 'ES_CHAIN_DONE|ES_FAIL' $S/evalspeed_chain.log | head -n 1)"; date
# ---- 錨(新協定)
idle; serve_vllm "$Q3"
for sd in 20260806 1 2 3; do runhe humaneval_e70_bf16_s$sd $sd; echo "E70S0_ANCHOR:he_s$sd $(score $A1/humaneval_e70_bf16_s$sd)"; done
for sd in 1 2; do runif ifeval_e70_bf16_s$sd $sd; echo "E70S0_ANCHOR:if_s$sd $(score $A1/ifeval_e70_bf16_s$sd)"; done
rungs gsm8k_e70_bf16_s1 1; echo "E70S0_ANCHOR:gs_s1 $(score $A1/gsm8k_e70_bf16_s1)"; date
# ---- 語料自產(examroom 協定,Qwen3-4B 自身)
if [ ! -f "$EV/calib_e70r0_traj.pt" ]; then
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.gen_e58_examroom --domains ifshape,code,math --per-domain-counts code:260,ifshape:180,math:480 --pack --workers 128 --out-tag e70r0 > "$S/e70_corpus.log" 2>&1 || fail corpus
fi
echo "E70S0_CORPUS $(grep -E '^(ifshape|code|math):' $S/e70_corpus.log | tr '\n' ' ' | cut -c1-300)"; date
# ifshape 補產(Qwen3-4B ifshape 短:180 題僅打包 10 列 < 配額 37;產線 seed 同序,前 180 題即 r0 那批 → ifshape 全改用 r1)
if [ ! -f "$EV/calib_e70r1_traj.pt" ]; then
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.gen_e58_examroom --domains ifshape --per-domain-counts ifshape:1080 --pack --workers 128 --out-tag e70r1 > "$S/e70_corpus_r1.log" 2>&1 || fail corpus_r1
fi
echo "E70S0_CORPUS_R1 $(grep -E '^ifshape:' $S/e70_corpus_r1.log | tr '\n' ' ' | cut -c1-300) rows=$(grep -o 'packed_rows[^,}]*' $EV/E58_TRAJ_MANIFEST_e70r1.json)"; date
stop_srv
PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.build_calib_e70 --src e70r0=code,math --src e70r1=ifshape --out e70 > "$S/e70_calib.log" 2>&1 || fail calib
echo "E70S0_CALIB $(grep CALIB_E70_DONE $S/e70_calib.log | cut -c1-300)"; date
# falsifier:code 閉合率 < 0.70 不點訓練鏈(S1/S2 零訓練仍可跑;此處只記錄)
python3 -c "import json;m=json.load(open('$EV/E70_CALIB_MANIFEST_e70.json'));c=m['closure']['code'];print('E70S0_CLOSURE code',c,'TRAIN_OK' if c>=0.70 else 'TRAIN_BLOCKED')"
# ---- P1 八元 g128(k/v int8 逐列、o_proj 16 級)閉式
BASE="self_attn.k_proj:256:row,self_attn.v_proj:256:row,self_attn.o_proj:16:128"
idle
[ -f data/qs_e70_p1/layer35.pt ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.quantize --tag e70_p1 --calib $EV/calib_e70_traj.pt --n-calib 128 --grid 8 --block 128 --solver gptq --prior-lam 0.01 --module-map "$BASE" > "$S/quant_e70_p1.log" 2>&1 || fail quant_p1
echo "E70S0_P1_LEDGER $(grep QUANT_LEDGER $S/quant_e70_p1.log | tail -n 1 | tr '\n' ' ' | cut -c1-300)"; date
[ -f exports/e70_p1/config.json ] || PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export --state-tag e70_p1 --out exports/e70_p1 > "$S/export_e70_p1.log" 2>&1 || fail export_p1
idle; serve_vllm exports/e70_p1
for sd in 20260806 1; do runhe humaneval_e70_p1_s$sd $sd; echo "E70S0_P1:he_s$sd $(score $A1/humaneval_e70_p1_s$sd)"; done
stop_srv
echo "E70S0_CHAIN_DONE"; date
