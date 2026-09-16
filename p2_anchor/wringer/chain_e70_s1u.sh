#!/bin/bash
# E70 S1 GGUF U 三點補跑(S1 於 IQ2_S IFEval 因 llama.cpp peg-native 500 出局;runner 已改為空答計 0):
#   等 E70S3_CHAIN_DONE(主線 S2/S3 先走,不搶 GPU)→ 原樣重跑 chain_e70_s1.sh(探針/轉檔/imatrix/已收點皆有檔案守衛)
# 哨兵:E70S1U_START / E70S1U_HANDOFF / 其餘為 chain_e70_s1 的 E70S1_* 寫入 $S/e70s1_chain.log
cd $REPO || exit 1
S=$SCRATCH
echo "E70S1U_START"; date
for i in $(seq 1 17280); do grep -q "E70S3_CHAIN_DONE" "$S/e70s3_chain.log" 2>/dev/null && break; sleep 10; done
grep -q "E70S3_CHAIN_DONE" "$S/e70s3_chain.log" || { echo "E70S1U_FAIL:timeout"; exit 1; }
echo "E70S1U_HANDOFF"; date
sleep 30
exec bash p2_anchor/wringer/chain_e70_s1.sh >> "$S/e70s1_chain.log" 2>&1
