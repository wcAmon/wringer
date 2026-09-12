"""E58 k-rollout 探針(FROZEN 附件;用戶 2026-08-30 裁「探針先行」)。

問題:同題第二軌(temp 1.0 考場採樣、不同 seed)是否為有效新樣本?
若是,梯級題需求減半(R2+ 題池經濟主槓桿)。量測=50 code 題 ×2 軌,
軌跡對 12-gram Jaccard(對齊 bank 去重粒度);對照=E42 dup 飽和前科是
題面 dup,非軌跡 dup。非閘門讀數:照實入卷,R2 sizing 時引用。

判讀基準(記錄用):median Jaccard <0.2=軌跡自然分化(k=2 可用);
>0.5=軌跡近重演(k-rollout 否決)。中間帶回用戶。
"""
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median, quantiles

from transformers import AutoTokenizer

from p1_grouping.modelio import MODEL_ID, MODEL_REVISION
from p2_anchor.wringer.gen_e58_examroom import (
    MAX_TOK, SEED, bank_code, gen_one_exam, wrap_exam)

EV = Path("evidence/p1_grouping")
N_Q = 50


def grams(text, n=12):
    ws = text.split()
    return {" ".join(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    qs = bank_code()
    random.Random(SEED + 999).shuffle(qs)
    qs = qs[:N_Q]
    prompts = [wrap_exam(tok, q) for q in qs]
    base = "http://127.0.0.1:8000/v1"

    def one(i):
        p = prompts[i % N_Q]
        seed = SEED + 7000 + (i % N_Q) + (i // N_Q) * 50021
        return gen_one_exam(base, "a1", p, MAX_TOK, seed)

    with ThreadPoolExecutor(max_workers=64) as ex:
        outs = list(ex.map(one, range(2 * N_Q)))

    jac, stops = [], 0
    for i in range(N_Q):
        (a, fa), (b, fb) = outs[i], outs[i + N_Q]
        stops += (fa == "stop") + (fb == "stop")
        ga, gb = grams(a), grams(b)
        u = len(ga | gb)
        jac.append(len(ga & gb) / u if u else 1.0)
    qrt = [round(x, 4) for x in quantiles(jac, n=4)]
    out = {"n_q": N_Q, "gram_n": 12,
           "jaccard_median": round(median(jac), 4),
           "jaccard_quartiles": qrt,
           "frac_over_0.5": round(sum(j > 0.5 for j in jac) / N_Q, 4),
           "frac_under_0.2": round(sum(j < 0.2 for j in jac) / N_Q, 4),
           "natural_stop_rate": round(stops / (2 * N_Q), 4),
           "verdict_rule": "<0.2 中位=k=2 可用;>0.5=否決;中間帶回用戶"}
    (EV / "E58_KROLL.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print("E58_KROLL_DONE:" + json.dumps(out))


if __name__ == "__main__":
    main()
