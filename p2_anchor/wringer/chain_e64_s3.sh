#!/bin/bash
# E64-PL S3(prereg_e64_probes FROZEN;用戶裁 Q4 工程先起):VQ 求解器 A/B 煙測(不跑官方)
#   等 S2 鏈 E64S2_CHAIN_DONE + GPU 空 → train_res --smoke(A4 st54r+res58r1,st63a 配置含 asym r)三臂:
#   G greedy / J joint d=4 全格 / V vq d=4 K=64 → 逐事件 obj_after、窗末水位、flips、conform、holdout
#   → verdict_e64_s3.json(J6:V vs J 的 obj 相對改善 ≥10% B1 / 5–10% 灰帶 V8 / <5% B 封盤;J7:J vs G ≥5% 併入引擎)
cd $REPO || exit 1
S=$SCRATCH
EVC=evidence/p1_grouping/corkscrew
fail() { echo "E64S3_FAIL:$1"; date; exit 1; }
JIA="--state-tag st54r --res-tag res58r1 --n-calib 128 \
  --calib evidence/p1_grouping/calib_e58r1b_traj.pt --lengths evidence/p1_grouping/calib_e58r1b_len.pt \
  --batch 4 --win 2 --stride 1 --gsteps 5 --tau 0.5 --w-max 4 --comp-rounds 3 --comp-cap 0.03 \
  --tail-soft 10 --tail-ste 4 --s0 30 --s-plateau 2 --lora-r 64 --lora-lr 6e-4 --factor-lr 3e-3 \
  --clip 1.0 --learn-theta --holdout-frac 0.05 --x-cpu --asym-r-bits 2 --asym-alt 2"

echo "E64S3_WAIT"; date
while :; do
  if grep -q "^E64S2_CHAIN_DONE\|^E64S2_FAIL" "$S/e64_s2_chain.log" 2>/dev/null && [ ! -f "$S/E64_HOLD" ] \
     && ! pgrep -f "vllm serve" >/dev/null && ! pgrep -f "probe_t2" >/dev/null; then break; fi
  sleep 60
done
echo "E64S3_START"; date
for arm in g j v; do
  case $arm in g) SOLV="--comp-solver greedy";; j) SOLV="--comp-solver joint --vq-d 4";; v) SOLV="--comp-solver vq --vq-d 4 --vq-k 64";; esac
  TAG=st64smk_$arm
  if [ ! -f "$EVC/train_$TAG.json" ]; then
    PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.train_res --tag $TAG --smoke $JIA $SOLV \
      > "$S/train_$TAG.log" 2>&1 || fail "run_$arm"
    grep -q "post-drain val CE" "$S/train_$TAG.log" || fail "sentinel_$arm"
  fi
  echo "E64S3_ARM_DONE:$arm"; date
done

python3 - <<'PYEOF'
import json, statistics
EVC="evidence/p1_grouping/corkscrew"
res={}
for arm in ("g","j","v"):
    d=json.load(open(f"{EVC}/train_st64smk_{arm}.json"))
    ws=d["windows"]; evs=[e for w in ws for e in w["drain_events"]]
    obj=[o for e in evs for o in e.get("obj",[])]
    res[arm]={"n_events":len(evs),"obj_sum":sum(obj),"obj_last_per_window":[w["drain_events"][-1]["obj"] for w in ws],
              "final_level":[w["final_level"] for w in ws],"flips_vs_p0":[w["flips_vs_p0"] for w in ws],
              "conform":[c for e in evs for c in e.get("conform",[])],
              "loss_holdout_last":[ (w["loss_holdout"][-1] if isinstance(w.get("loss_holdout"),list) and w["loss_holdout"] else w.get("loss_holdout")) for w in ws],
              "runtime_s":d.get("runtime_s"),"post_drain_val_ce":d.get("post_drain_val_ce")}
def rel(a,b):  # 相對改善 (b−a)/b,>0 = a 較好
    return (b-a)/max(b,1e-30)
lvl={a:sum(res[a]["final_level"]) for a in res}
J7=rel(lvl["j"],lvl["g"]); J6=rel(lvl["v"],lvl["j"])
hold={a:statistics.mean(res[a]["loss_holdout_last"]) for a in res if all(x is not None for x in res[a]["loss_holdout_last"])}
verdict={"J7_joint_vs_greedy_level_rel":round(J7,4),"J6_vq_vs_joint_level_rel":round(J6,4),
         "J7":"ENGINE_UPGRADE" if J7>=0.05 else "GREEDY_SUFFICES",
         "J6":"B1" if J6>=0.10 else ("GREY_V8" if J6>=0.05 else "B_CLOSED"),
         "holdout_last_mean":{a:round(v,8) for a,v in hold.items()},
         "runtime_s":{a:res[a]["runtime_s"] for a in res}}
print("E64S3_LEVELS "+json.dumps(lvl))
print("E64S3_VERDICT "+json.dumps(verdict,ensure_ascii=False))
json.dump({"experiment":"E64-PL S3 VQ 求解器 A/B(train_res --smoke 2 窗,A4 st54r+res58r1,asym r 開)","arms":res,"verdict":verdict},
          open("p2_anchor/wringer/verdict_e64_s3.json","w"),ensure_ascii=False,indent=1)
PYEOF
echo "E64S3_CHAIN_DONE"; date
