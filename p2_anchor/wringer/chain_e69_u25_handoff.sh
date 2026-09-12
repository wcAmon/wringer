#!/bin/bash
# E69 排程調整(09-10 16:2x):U25(崩塌錨,he 20.12)IFEval/GSM8K 幾乎每題撞 16k 上限,GS 估 ~7h。
# U25 三科只供最後 J3 comp 內插,不影響 S2 的 he/官方量測 → IF 收完即交棒 S2,GS 延到 S2 收尾後(chain_e69_u25gs.sh)。
cd $REPO || exit 1
S=$SCRATCH
A1=evidence/p1_grouping/a1eval; LOG=$S/e69s1_chain.log; sfx=e67u_u25
score() { python3 -c "import json;d=json.load(open('$1/summary.json'));print(d['strict']['prompt_level'] if 'strict' in d else d['score'])"; }
until [ -f "$A1/ifeval_${sfx}/summary.json" ] || grep -q "E69S1_CHAIN_DONE\|E69S1_FAIL" $LOG; do sleep 30; done
grep -q "E69S1_CHAIN_DONE\|E69S1_FAIL" $LOG && exit 0     # 原鏈已自行收尾
sleep 5
pkill -f "chain_e69_u25.sh" 2>/dev/null
pkill -f "a1-4b-u25.gguf" 2>/dev/null; sleep 8
pgrep -f "a1-4b-u25.gguf" >/dev/null && pkill -9 -f "a1-4b-u25.gguf"
echo "E69S1_POINT:u25 he=$(score $A1/humaneval_${sfx}) if=$(score $A1/ifeval_${sfx}) gs=deferred(chain_e69_u25gs)" >> $LOG
echo "E69S1_CHAIN_DONE" >> $LOG; date >> $LOG
