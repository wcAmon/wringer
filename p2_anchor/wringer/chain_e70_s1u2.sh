#!/bin/bash
# E70 S1 GGUF U 三點第二次補跑(第一次補跑死於 ifeval_runner dump_usage 對 server_error None 排序,已修並改先落盤回應):
#   等打包鏈 E70PK_CHAIN_DONE(材化 he/IF/GS 先走)→ 原樣重跑 chain_e70_s1.sh(檔案守衛)
# 哨兵:E70S1U2_START / E70S1U2_HANDOFF / 其餘為 E70S1_* 寫入 $S/e70s1_chain.log
cd $REPO || exit 1
S=$SCRATCH
echo "E70S1U2_START"; date
for i in $(seq 1 17280); do grep -q "E70PK_CHAIN_DONE" "$S/e70pk_chain.log" 2>/dev/null && break; sleep 10; done
grep -q "E70PK_CHAIN_DONE" "$S/e70pk_chain.log" || { echo "E70S1U2_FAIL:timeout"; exit 1; }
echo "E70S1U2_HANDOFF"; date
sleep 30
exec bash p2_anchor/wringer/chain_e70_s1.sh >> "$S/e70s1_chain.log" 2>&1
